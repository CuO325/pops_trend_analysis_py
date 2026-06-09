import numpy as np
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.random as random

def breakpoint1_tempered_model(x, y, beta):
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    b1 = numpyro.sample("beta1", dist.Normal(0, 1))
    b2 = numpyro.sample("beta2", dist.Normal(0, 1))
    tau = numpyro.sample("tau", dist.Uniform(x.min(), x.max()))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = alpha + b1 * x + b2 * jnp.maximum(0, x - tau)
    loglik = dist.Normal(mu, sigma).log_prob(y).sum()
    numpyro.factor("tempered_loglik", beta * loglik)

def breakpoint2_tempered_model(x, y, beta):
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    b1 = numpyro.sample("beta1", dist.Normal(0, 1))
    b2 = numpyro.sample("beta2", dist.Normal(0, 1))
    b3 = numpyro.sample("beta3", dist.Normal(0, 1))
    tau1 = numpyro.sample("tau1", dist.Uniform(x.min(), x.max()))
    tau2 = numpyro.sample("tau2", dist.Uniform(tau1, x.max()))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    h1 = jnp.maximum(0, x - tau1)
    h2 = jnp.maximum(0, x - tau2)
    mu = alpha + b1 * x + b2 * h1 + b3 * h2
    loglik = dist.Normal(mu, sigma).log_prob(y).sum()
    numpyro.factor("tempered_loglik", beta * loglik)

def compute_loglik(samples, x, y, model="1break"):
    alpha = samples["alpha"]
    b1 = samples["beta1"]
    sigma = samples["sigma"]

    if model == "1break":
        b2 = samples["beta2"]
        tau = samples["tau"]
        mu = alpha[:, None] + b1[:, None] * x + b2[:, None] * np.maximum(0, x - tau[:, None])
    else:
        b2 = samples["beta2"]
        b3 = samples["beta3"]
        tau1 = samples["tau1"]
        tau2 = samples["tau2"]
        h1 = np.maximum(0, x - tau1[:, None])
        h2 = np.maximum(0, x - tau2[:, None])
        mu = alpha[:, None] + b1[:, None] * x + b2[:, None] * h1 + b3[:, None] * h2

    sigma_eff = np.maximum(sigma, 1e-3)
    loglik = -0.5 * (((y - mu) / sigma_eff[:, None]) ** 2 + 2 * np.log(sigma_eff[:, None]) + np.log(2 * np.pi))
    return loglik.sum(axis=1)

def run_ti_numpyro_breakpoint_ok(x, y, model="1break", n_beta=20, num_warmup=500, num_samples=1000):
    beta_grid = np.linspace(0.01, 1.0, n_beta)
    E_logL = []

    for i, beta in enumerate(beta_grid):
        if model == "1break":
            kernel = NUTS(lambda x_, y_: breakpoint1_tempered_model(x_, y_, beta))
        else:
            kernel = NUTS(lambda x_, y_: breakpoint2_tempered_model(x_, y_, beta))

        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(random.PRNGKey(i), x, y)

        #mcmc.run(numpyro.random.PRNGKey(i), x, y)
        samples = mcmc.get_samples()

        loglik = compute_loglik(samples, x, y, model=model)
        E_logL.append(loglik.mean())

    logZ = np.trapz(E_logL, beta_grid)
    return logZ, beta_grid, np.array(E_logL)

import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS

def run_ti_numpyro_breakpoint(x, y, model="1break", n_beta=20, num_warmup=500, num_samples=1000, beta_schedule="Power-law (推奨)",  progress_container=None):
    """
    熱力学的積分(TI)を動的なパラメータで実行する関数
    
    Parameters:
    - beta_schedule: "Linear (等間隔)" または "Power-law (推奨)"
    """
    # 1. βのスケジュールを動的に生成
    if beta_schedule == "Power-law (推奨)":
        # 0.0 から 1.0 に向かって、後半にいくほど間隔が広くなる（0付近が密になる）
        # 累乗の数（ここでは4）を大きくするほど、0付近がより細かくなります
        beta_grid = np.linspace(0.0, 1.0, n_beta) ** 4
        # 万が一、モデルの制約で完全に0.0がダメな場合は微小値を足す
        beta_grid = np.clip(beta_grid, 0.0001, 1.0)
    else:
        # 従来の等間隔配置
        beta_grid = np.linspace(0.0001, 1.0, n_beta)
        
    E_logL = []

    for i, beta in enumerate(beta_grid):

        #─── 💡 ここでStreamlitのプログレスバーを更新 ───
        if progress_container is not None:
            current_progress = (i + 1) / n_beta
            progress_container.progress(current_progress, text=f"β = {beta:.4f} を計算中 ({i+1}/{n_beta})")
        # ─────────────────────────────────────────────
        if model == "1break":
            kernel = NUTS(lambda x_, y_: breakpoint1_tempered_model(x_, y_, beta))
        else:
            kernel = NUTS(lambda x_, y_: breakpoint2_tempered_model(x_, y_, beta))

        # 引数から受け取った num_warmup, num_samples を動的に適用
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
        mcmc.run(random.PRNGKey(i), x, y)

        samples = mcmc.get_samples()
        loglik = compute_loglik(samples, x, y, model=model)
        E_logL.append(loglik.mean())

    # 2. 台形公式(np.trapz)は不等間隔の beta_grid にも自動で対応します
    logZ = np.trapz(E_logL, beta_grid)
    
    return logZ, beta_grid, np.array(E_logL)