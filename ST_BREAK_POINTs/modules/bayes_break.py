import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import numpy as np
import jax.random as random



def breakpoint1_model(x, y):
    x_min, x_max = x.min(), x.max()
    alpha = numpyro.sample("alpha", dist.Normal(0, 5))
    beta1 = numpyro.sample("beta1", dist.Normal(0, 5))
    beta2 = numpyro.sample("beta2", dist.Normal(0, 5))
    tau_raw = numpyro.sample("tau_raw", dist.Normal(0, 1))
    tau = x_min + (x_max - x_min) * jnp.exp(tau_raw) / (1 + jnp.exp(tau_raw))
    numpyro.deterministic("tau", tau)
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    h = jnp.maximum(0, x - tau)
    mu = alpha + beta1 * x + beta2 * h
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

def run_breakpoint1_mcmc(x, y, num_warmup=1000, num_samples=2000):
    kernel = NUTS(breakpoint1_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(0), x=x, y=y)
    #mcmc.run(numpyro.random.PRNGKey(0), x=x, y=y)
    return mcmc

def breakpoint2_model(x, y):
    x_min, x_max = x.min(), x.max()
    alpha = numpyro.sample("alpha", dist.Normal(0, 5))
    beta1 = numpyro.sample("beta1", dist.Normal(0, 5))
    beta2 = numpyro.sample("beta2", dist.Normal(0, 5))
    beta3 = numpyro.sample("beta3", dist.Normal(0, 5))
    tau1 = numpyro.sample("tau1", dist.Uniform(x_min, x_max))
    tau2 = numpyro.sample("tau2", dist.Uniform(tau1, x_max))
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    h1 = jnp.maximum(0, x - tau1)
    h2 = jnp.maximum(0, x - tau2)
    mu = alpha + beta1 * x + beta2 * h1 + beta3 * h2
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

def run_breakpoint2_mcmc(x, y, num_warmup=1000, num_samples=2000):
    kernel = NUTS(breakpoint2_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(1), x=x, y=y)
    #mcmc.run(numpyro.random.PRNGKey(1), x=x, y=y)
    return mcmc
def plot_posterior_trend(x, y, samples):
    alpha = samples["alpha"]
    beta1 = samples["beta1"]
    beta2 = samples["beta2"]
    tau = samples["tau"]

    mu_samples = []
    for a, b1, b2, t in zip(alpha, beta1, beta2, tau):
        h = np.maximum(0, x - t)
        mu_samples.append(a + b1 * x + b2 * h)

    mu_samples = np.array(mu_samples)  # shape: (n_samples, n_time)
    mu_mean = mu_samples.mean(axis=0)
    mu_lower = np.percentile(mu_samples, 2.5, axis=0)
    mu_upper = np.percentile(mu_samples, 97.5, axis=0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, "o", markersize=3, color="black", alpha=0.5, label="STL trend")
    ax.plot(x, mu_mean, color="red", label="Posterior mean")
    ax.fill_between(x, mu_lower, mu_upper, color="red", alpha=0.2, label="95% CI")
    ax.legend()
    return fig

def plot_posterior_trend_old(x, y, samples):
    alpha = samples["alpha"]
    beta1 = samples["beta1"]
    beta2 = samples["beta2"]
    tau = samples["tau"]

    mu_samples = []
    for a, b1, b2, t in zip(alpha, beta1, beta2, tau):
        h = np.maximum(0, x - t)
        mu_samples.append(a + b1 * x + b2 * h)

    mu_samples = np.array(mu_samples)
    mu_mean = mu_samples.mean(axis=0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, "o", markersize=3)
    ax.plot(x, mu_mean, color="red")
    return fig

def plot_posterior_trend_2break(x, y, samples):
    alpha = samples["alpha"]
    beta1 = samples["beta1"]
    beta2 = samples["beta2"]
    beta3 = samples["beta3"]
    tau1 = samples["tau1"]
    tau2 = samples["tau2"]

    mu_samples = []
    for a, b1, b2, b3, t1, t2 in zip(alpha, beta1, beta2, beta3, tau1, tau2):
        h1 = np.maximum(0, x - t1)
        h2 = np.maximum(0, x - t2)
        mu_samples.append(a + b1 * x + b2 * h1 + b3 * h2)

    mu_samples = np.array(mu_samples)
    mu_mean = mu_samples.mean(axis=0)
    mu_lower = np.percentile(mu_samples, 2.5, axis=0)
    mu_upper = np.percentile(mu_samples, 97.5, axis=0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, "o", markersize=3, color="black", alpha=0.5, label="STL trend")
    ax.plot(x, mu_mean, color="blue", label="Posterior mean (2-break)")
    ax.fill_between(x, mu_lower, mu_upper, color="blue", alpha=0.2, label="95% CI")
    ax.legend()
    return fig

def plot_posterior_trend_2break_old(x, y, samples):
    alpha = samples["alpha"]
    beta1 = samples["beta1"]
    beta2 = samples["beta2"]
    beta3 = samples["beta3"]
    tau1 = samples["tau1"]
    tau2 = samples["tau2"]

    mu_samples = []
    for a, b1, b2, b3, t1, t2 in zip(alpha, beta1, beta2, beta3, tau1, tau2):
        h1 = np.maximum(0, x - t1)
        h2 = np.maximum(0, x - t2)
        mu_samples.append(a + b1 * x + b2 * h1 + b3 * h2)

    mu_samples = np.array(mu_samples)
    mu_mean = mu_samples.mean(axis=0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, "o", markersize=3)
    ax.plot(x, mu_mean, color="blue")
    return fig
#-----------------------under construction below-----------n_beta=20 → 30 に増やす必要がある可能性--------------強い事前分布（例：tau1 < tau2 < tau3 の幅を狭める）を検討-----------------------------------------------
def breakpoint3_tempered_model(x, y, beta):
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    b1 = numpyro.sample("beta1", dist.Normal(0, 1))
    b2 = numpyro.sample("beta2", dist.Normal(0, 1))
    b3 = numpyro.sample("beta3", dist.Normal(0, 1))
    b4 = numpyro.sample("beta4", dist.Normal(0, 1))

    tau1 = numpyro.sample("tau1", dist.Uniform(x.min(), x.max()))
    tau2 = numpyro.sample("tau2", dist.Uniform(tau1, x.max()))
    tau3 = numpyro.sample("tau3", dist.Uniform(tau2, x.max()))

    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    h1 = jnp.maximum(0, x - tau1)
    h2 = jnp.maximum(0, x - tau2)
    h3 = jnp.maximum(0, x - tau3)

    mu = alpha + b1 * x + b2 * h1 + b3 * h2 + b4 * h3

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

    elif model == "2break":
        b2 = samples["beta2"]
        b3 = samples["beta3"]
        tau1 = samples["tau1"]
        tau2 = samples["tau2"]
        h1 = np.maximum(0, x - tau1[:, None])
        h2 = np.maximum(0, x - tau2[:, None])
        mu = alpha[:, None] + b1[:, None] * x + b2[:, None] * h1 + b3[:, None] * h2

    elif model == "3break":
        b2 = samples["beta2"]
        b3 = samples["beta3"]
        b4 = samples["beta4"]
        tau1 = samples["tau1"]
        tau2 = samples["tau2"]
        tau3 = samples["tau3"]
        h1 = np.maximum(0, x - tau1[:, None])
        h2 = np.maximum(0, x - tau2[:, None])
        h3 = np.maximum(0, x - tau3[:, None])
        mu = alpha[:, None] + b1[:, None] * x + b2[:, None] * h1 + b3[:, None] * h2 + b4[:, None] * h3

    sigma_eff = np.maximum(sigma, 1e-3)
    loglik = -0.5 * (((y - mu) / sigma_eff[:, None]) ** 2 + 2 * np.log(sigma_eff[:, None]) + np.log(2 * np.pi))
    return loglik.sum(axis=1)

def run_ti_numpyro_breakpoint(x, y, model="1break", n_beta=20, num_warmup=500, num_samples=1000):
    beta_grid = np.linspace(0.01, 1.0, n_beta)
    E_logL = []

    for i, beta in enumerate(beta_grid):
        if model == "1break":
            kernel = NUTS(lambda x_, y_: breakpoint1_tempered_model(x_, y_, beta))
        elif model == "2break":
            kernel = NUTS(lambda x_, y_: breakpoint2_tempered_model(x_, y_, beta))
        elif model == "3break":
            kernel = NUTS(lambda x_, y_: breakpoint3_tempered_model(x_, y_, beta))

        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(random.PRNGKey(i), x, y)
        samples = mcmc.get_samples()

        loglik = compute_loglik(samples, x, y, model=model)
        E_logL.append(loglik.mean())

    logZ = np.trapz(E_logL, beta_grid)
    return logZ, beta_grid, np.array(E_logL)

