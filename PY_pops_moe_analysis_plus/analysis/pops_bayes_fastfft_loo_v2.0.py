# pops_bayes_fastfft_loo_v2.0.py
# FFT-KDE 高速ベイズ（MCMC + TI）BayesConfig 対応版

version = "pops_bayes_fastfft_loo_v2.0  by H.Shiraishi."

from datetime import datetime
import os
import time
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
import emcee

from analysis.config import BayesConfig
from analysis.fft_kde_common_engine3 import (
    prepare_fft_kde_lookup,
    fast_kde_log_likelihood_from_residuals,
)


# ---------------------------
# 事前分布
# ---------------------------

def log_prior_informative(params, data_scale=None):
    a, b = params
    if not (-100 < a < 100 and -100 < b < 100):
        return -np.inf

    if data_scale is None:
        return - (a**2) / (2 * 2.0**2) - (b**2) / (2 * 2.0**2)
    else:
        x_range = data_scale.get("x_range", 1.0)
        y_std = data_scale.get("y_std", 1.0)
        a_sd = max(0.1, y_std / max(1e-6, x_range) * 2.0)
        b_sd = max(0.1, y_std * 2.0)
        return - (a**2) / (2 * a_sd**2) - (b**2) / (2 * b_sd**2)


# ---------------------------
# 事後分布（通常 / tempered）
# ---------------------------

def log_posterior_fast(params, x_centered, y, lookup, data_scale=None):
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + ll


def tempered_log_posterior(params, beta, x_centered, y, lookup, data_scale=None):
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + beta * ll


# ---------------------------
# β ごとの E_beta[logL] 推定
# ---------------------------

def estimate_E_loglike_for_beta(
    beta,
    x_centered,
    y,
    lookup,
    cfg: BayesConfig,
    init_center=None,
    data_scale=None,
    show_progress=False,
):
    ndim = 2

    if init_center is None:
        init_center = np.array([0.0, np.mean(y)])

    rng = np.random.default_rng()
    initial_pos = init_center + 1e-5 * rng.normal(size=(cfg.n_walkers, ndim))

    def _logprob(theta):
        return tempered_log_posterior(
            theta, beta, x_centered, y, lookup, data_scale=data_scale
        )

    sampler = emcee.EnsembleSampler(cfg.n_walkers, ndim, _logprob)
    pos, _, _ = sampler.run_mcmc(initial_pos, cfg.n_steps, progress=show_progress)
    chain = sampler.get_chain(discard=int(cfg.n_steps * cfg.burn_frac), flat=True)

    ll_vals = np.array([
        fast_kde_log_likelihood_from_residuals(th[0], th[1], x_centered, y, lookup)
        for th in chain
    ])
    return np.mean(ll_vals), chain, sampler


# ---------------------------
# 熱力学積分による logZ
# ---------------------------

def _build_beta_grid(cfg: BayesConfig):
    if cfg.beta_schedule == "default":
        betas = np.concatenate([
            np.linspace(0, 0.9, 30, endpoint=False),
            np.linspace(0.9, 1.0, 21),
        ])
    elif cfg.beta_schedule == "coarse":
        betas = np.linspace(0.0, 1.0, 21)
    else:
        # 必要なら他のスケジュールも追加
        betas = np.concatenate([
            np.linspace(0, 0.1, 15, endpoint=False),
            np.linspace(0.1, 0.9, 30, endpoint=False),
            np.linspace(0.9, 1.0, 25),
        ])
    return np.unique(np.sort(betas))


def compute_log_marginal_TL(
    df,
    model="linear",
    cfg: BayesConfig = None,
    verbose=True,
):
    if cfg is None:
        cfg = BayesConfig()

    years = df["year"].values.astype(float)
    x_base = years.min()
    x_centered = years - x_base
    y = df["log_conc"].values.astype(float)

    # MLE による残差 → KDE lookup 作成
    if model == "linear":
        res = minimize(
            neg_log_likelihood,
            [-0.01, np.mean(y)],
            args=(x_centered, y),
            method="L-BFGS-B",
        )
        a_mle, b_mle = res.x
        residuals = y - (a_mle * x_centered + b_mle)
    else:
        res0 = minimize(
            neg_log_likelihood_0,
            [np.mean(y)],
            args=(x_centered, y),
            method="L-BFGS-B",
        )
        b0 = res0.x[0]
        a_mle, b_mle = 0.0, b0
        residuals = y - b0

    lookup = prepare_fft_kde_lookup(
        residuals,
        bw_factor=cfg.bw_factor,
        grid_size=cfg.grid_size_lookup,
        use_loo=cfg.mode_loo,
    )
    if verbose:
        print("DEBUG lookup@MLE:", type(lookup))

    betas = _build_beta_grid(cfg)

    if verbose:
        print(f"\n[TI] model={model}")
        print(" betas =", np.round(betas, 3))

    E_logL = []

    data_scale = {
        "x_range": float(x_centered.max() - x_centered.min()),
        "y_std": float(np.nanstd(y)),
    }

    init_center = np.array([a_mle, b_mle])

    for beta in betas:
        if verbose:
            print(f" β={beta:.3f} ... ", end="")
        E, chain, sampler = estimate_E_loglike_for_beta(
            beta,
            x_centered,
            y,
            lookup,
            cfg=cfg,
            init_center=init_center,
            data_scale=data_scale,
            show_progress=False,
        )
        if verbose:
            print(f"E[logL]={E:.3f}")
        E_logL.append(E)

    logZ = trapezoid(E_logL, betas)
    return {
        "model": model,
        "betas": betas,
        "E_logL": np.array(E_logL),
        "logZ": logZ,
        "a_mle": a_mle,
        "b_mle": b_mle,
        "lookup": lookup,
    }


# ---------------------------
# モデル比較
# ---------------------------

def compare_models_logZ(res1, res0, prior_ratio=1.0):
    logBF = res1["logZ"] - res0["logZ"] + np.log(prior_ratio)
    BF = np.exp(logBF)
    p_M1 = BF / (1 + BF)
    return {
        "logZ1": res1["logZ"],
        "logZ0": res0["logZ"],
        "logBF": logBF,
        "BF": BF,
        "p_M1": p_M1,
    }


# ---------------------------
# Rhat
# ---------------------------

def compute_Rhat(chains, axis=0):
    try:
        n_chains, n_samples, n_params = chains.shape
        chain_means = np.mean(chains, axis=1)
        chain_vars = np.var(chains, axis=1, ddof=1)
        W = np.mean(chain_vars, axis=0)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)
        var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        Rhat = np.sqrt(var_hat / W)
        return Rhat
    except Exception:
        return np.array([np.nan] * chains.shape[-1])


# ---------------------------
# MLE 用 KDE 尤度
# ---------------------------

def kde_log_likelihood_from_residuals_plain(a, b, x_centered, y):
    res = y - (a * x_centered + b)
    try:
        kde = gaussian_kde(res)
        pdf_vals = kde.pdf(res)
        return np.sum(np.log(pdf_vals + 1e-12))
    except Exception:
        return -1e20


def neg_log_likelihood(params, x, y):
    a, b = params
    residuals = y - (a * x + b)
    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10


def neg_log_likelihood_0(params, x, y):
    b0 = params[0]
    residuals = y - b0
    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10


# ---------------------------
# 高速 MCMC（auto-stop）
# ---------------------------

def run_bayes_kde_auto_stop_fast(
    df,
    cfg: BayesConfig,
    fixed_lookup=None,
    verbose=True,
    ci_width_threshold=0.02,
    stable_rounds=2,
):
    years = df["year"].values.astype(float)
    x_base = years.min()
    x_centered = years - x_base
    y = df["log_conc"].values.astype(float)
    ndim = 2

    try:
        mle_res = minimize(
            lambda p: -kde_log_likelihood_from_residuals_plain(
                p[0], p[1], x_centered, y
            ),
            x0=[0.0, np.mean(y)],
            method="L-BFGS-B",
        )
        a_mle, b_mle = mle_res.x
    except Exception:
        a_mle, b_mle = 0.0, np.mean(y)

    if fixed_lookup is None:
        residuals_mle = y - (a_mle * x_centered + b_mle)
        lookup = prepare_fft_kde_lookup(
            residuals_mle,
            bw_factor=cfg.bw_factor,
            grid_size=cfg.grid_size_lookup,
            use_loo=cfg.mode_loo,
        )
        if verbose:
            print("DEBUG lookup@run_bayes_kde_auto_stop_fast", type(lookup))
    else:
        lookup = fixed_lookup

    data_scale = {
        "x_range": x_centered.max() - x_centered.min(),
        "y_std": np.nanstd(y),
    }

    rng = np.random.default_rng(int(time.time() % 2**31))
    n_walkers = max(2 * ndim, cfg.n_walkers)
    initial_pos = np.vstack([
        np.array([a_mle, b_mle]) + rng.normal(scale=[0.02, 0.1], size=2)
        for _ in range(n_walkers)
    ])

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_posterior_fast,
        args=(x_centered, y, lookup, data_scale),
    )

    pos = initial_pos
    chunk = 100
    n_chunks = max(1, cfg.n_steps // chunk)
    stable_count = 0
    total_steps_done = 0

    for i in range(n_chunks):
        pos, _, _ = sampler.run_mcmc(pos, chunk, progress=False)
        total_steps_done += chunk

        chain_partial = sampler.get_chain(flat=True)
        if chain_partial.size == 0:
            a_partial = np.array([])
        else:
            a_partial = chain_partial[:, 0]

        if a_partial.size == 0:
            ci = np.array([np.nan, np.nan])
            a_mean = np.nan
            p_neg = np.nan
        else:
            ci = np.percentile(a_partial, [2.5, 97.5])
            a_mean = np.mean(a_partial)
            p_neg = np.mean(a_partial < 0)

        if verbose:
            print(
                f"[FastMCMC {min(total_steps_done, cfg.n_steps)}/{cfg.n_steps}] "
                f"a_mean={a_mean if not np.isnan(a_mean) else 'nan'} "
                f"CI={ci} P(a<0)={p_neg if not np.isnan(p_neg) else 'nan'}"
            )

        ci_width = ci[1] - ci[0] if not np.any(np.isnan(ci)) else np.inf
        ci_is_stable = ci_width < ci_width_threshold
        direction_stable = (
            (not np.any(np.isnan(ci))) and ((ci[1] < 0) or (ci[0] > 0))
        )

        if ci_is_stable and direction_stable:
            stable_count += 1
            if verbose:
                print(f"  → 収束候補 ({stable_count}/{stable_rounds})")
        else:
            stable_count = 0

        if stable_count >= stable_rounds:
            if verbose:
                print("######## 自動早期収束判定 により MCMC 終了 ########")
            break

        if total_steps_done >= cfg.n_steps:
            break

    total_steps = sampler.iteration
    discard = int(cfg.burn_frac * total_steps)
    samples = sampler.get_chain(discard=discard, flat=True)

    if samples.size == 0:
        samples = sampler.get_chain(flat=True)

    a_samps = samples[:, 0] if samples.size != 0 else np.array([np.nan])
    b_samps = samples[:, 1] if samples.size != 0 else np.array([np.nan])

    chain3d = np.transpose(sampler.get_chain()[:, :, :], axes=(1, 0, 2))
    Rhat = compute_Rhat(chain3d)

    result = {
        "samples": samples,
        "a_mean": np.mean(a_samps) if a_samps.size > 0 else np.nan,
        "b_mean": np.mean(b_samps) if b_samps.size > 0 else np.nan,
        "a_ci": np.percentile(a_samps, [2.5, 97.5])
        if a_samps.size > 0
        else np.array([np.nan, np.nan]),
        "b_ci": np.percentile(b_samps, [2.5, 97.5])
        if b_samps.size > 0
        else np.array([np.nan, np.nan]),
        "p_neg_slope": np.mean(a_samps < 0) if a_samps.size > 0 else np.nan,
        "steps_completed": total_steps,
        "stopped_early": stable_count >= stable_rounds,
        "Rhat": Rhat,
        "lookup": lookup,
    }
    return result


# ---------------------------
# ここから下は v1.1b の既存ロジック（AIC, binomial, etc.）をそのまま利用
# （省略せずに使いたければ、v1.1b からそのままコピペで OK）
# ---------------------------

# ... run_mle_and_aic, binomial_trend_analyze, clean_value_user,
#     is_data_acceptable, bayes_trend_judgement, interpret_bayes_factor
#     などは v1.1b のまま ...


if __name__ == "__main__":
    cfg = BayesConfig(
        mode_loo=False,
        grid_size_lookup=1024,
        bw_factor=1.0,
        n_walkers=32,
        n_steps=2000,
        burn_frac=0.5,
        beta_schedule="default",
    )
    print("BayesConfig:", cfg)

    filenames = ["HCB大気温暖期2013"]
    for filename in filenames:
        print(f"\n\n==================== ファイル処理開始: {filename} ====================")
        filepath = f"data/{filename}.csv"
        if not os.path.exists(filepath):
            print(f"ファイルが見つかりません: {filepath}")
            continue

        df_all = pd.read_csv(
            filepath,
            usecols=["substance", "year", "site_id", "concentration", "detection_limit"],
        )
        from pops_bayes_fastfft_loo_v1.1b import clean_value_user  # 既存関数を流用するなら

        df_all["concentration"] = df_all["concentration"].apply(clean_value_user)
        df_all["detection_limit"] = df_all["detection_limit"].apply(clean_value_user)

        is_nondetect = df_all["concentration"] <= df_all["detection_limit"]
        substitute_value = df_all["detection_limit"] / np.sqrt(2)
        df_all["concentration_sub"] = df_all["concentration"].copy()
        df_all.loc[is_nondetect, "concentration_sub"] = substitute_value[is_nondetect]
        df_all["log_conc"] = np.log10(df_all["concentration_sub"])
        df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["log_conc", "detection_limit", "concentration"]
        )

        # ここで TI / MCMC / モデル比較などを呼び出す
        res1 = compute_log_marginal_TL(df_all, model="linear", cfg=cfg, verbose=True)
        res0 = compute_log_marginal_TL(df_all, model="constant", cfg=cfg, verbose=True)
        cmp = compare_models_logZ(res1, res0)

        print("logZ1 =", res1["logZ"])
        print("logZ0 =", res0["logZ"])
        print("logBF =", cmp["logBF"], " BF =", cmp["BF"], " p_M1 =", cmp["p_M1"])
