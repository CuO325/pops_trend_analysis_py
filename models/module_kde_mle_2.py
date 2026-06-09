"""
models/kde_mle.py

KDE-based maximum likelihood estimation (KDE-MLE) for linear trend analysis.

This module provides:
  - KDE-based log-likelihood for a linear model and a constant model
  - A simple KDE-MLE + AIC comparison (linear vs constant)
  - An iterative KDE-MLE variant that updates the residual KDE until convergence
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize


def neg_log_likelihood(params, x, y):
    """
    Negative log-likelihood for a linear model y = a*x + b
    using a Gaussian KDE for the residual distribution.

    Parameters
    ----------
    params : array-like, shape (2,)
        [a, b] where a is slope and b is intercept.
    x : array-like
        Predictor (centered year).
    y : array-like
        Response (log concentration).

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    a, b = params
    residuals = y - (a * x + b)

    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        return -log_likelihood
    except np.linalg.LinAlgError:
        # Return a large penalty when KDE fails
        return 1e10


def neg_log_likelihood_0(params, x, y):
    """
    Negative log-likelihood for a constant model y = b0
    using a Gaussian KDE for the residual distribution.

    Parameters
    ----------
    params : array-like, shape (1,)
        [b0] where b0 is the constant mean level.
    x : array-like
        Predictor (not used, kept for interface consistency).
    y : array-like
        Response (log concentration).

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    b0 = params[0]
    residuals = y - b0

    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10



def neg_log_likelihood_1(params, x, y, kde_fixed):
    """
    修正版：引数として渡された『固定されたKDE（kde_fixed）』を使って
    そのパラメータの尤度を計算する。
    """
    a, b = params
    # 今回のパラメータでの残差を計算
    residuals = y - (a * x + b)

    try:
        # 【修正】自分でKDEを作らず、前のステップで固定された kde_fixed を使って確率密度を計算する
        pdf_values = kde_fixed.pdf(residuals)
        log_likelihood = np.sum(np.log(pdf_values + 1e-10))
        return -log_likelihood
    except Exception:
        return 1e10


import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize


# ---------------------------------------------------------
# 固定 KDE を使った負の対数尤度
# ---------------------------------------------------------
def neg_log_likelihood_fixed_kde(params, x, y, kde_fixed):
    a, b = params
    residuals = y - (a * x + b)
    try:
        pdf_values = kde_fixed.pdf(residuals)
        log_likelihood = np.sum(np.log(pdf_values + 1e-10))
        return -log_likelihood
    except Exception:
        return 1e10


# ---------------------------------------------------------
# simple 版（初期値として利用）
# ---------------------------------------------------------
def run_mle_and_aic_simple(df):
    x_raw = df["year"].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df["log_conc"].values

    # 線形モデル
    def nll(params):
        a, b = params
        residuals = y - (a * x_centered + b)
        kde = gaussian_kde(residuals)
        return -np.sum(np.log(kde.pdf(residuals) + 1e-10))

    res = minimize(nll, [0.0, np.mean(y)], method="L-BFGS-B")
    a1, b1 = res.x
    ll1 = -res.fun

    # 定数モデル
    def nll0(params):
        b0 = params[0]
        residuals = y - b0
        kde = gaussian_kde(residuals)
        return -np.sum(np.log(kde.pdf(residuals) + 1e-10))

    res0 = minimize(nll0, [np.mean(y)], method="L-BFGS-B")
    ll0 = -res0.fun

    aic1 = -2 * ll1 + 4
    aic0 = -2 * ll0 + 2
    delta = aic1 - aic0
    p1 = np.exp(-0.5 * delta) / (1 + np.exp(-0.5 * delta))

    return a1, b1, aic0, aic1, p1, x_base


# ---------------------------------------------------------
# tolerance 自動推定
# ---------------------------------------------------------
def estimate_tolerance(a0, b0, x, y, warmup=3):
    a, b = a0, b0
    deltas = []

    for _ in range(warmup):
        residuals = y - (a * x + b)
        kde = gaussian_kde(residuals)

        res = minimize(
            lambda p: neg_log_likelihood_fixed_kde(p, x, y, kde),
            [a, b],
            method="L-BFGS-B"
        )
        a_new, b_new = res.x
        deltas.append(np.linalg.norm([a_new - a, b_new - b]))

        a, b = a_new, b_new

    med = np.median(deltas)
    return max(med * 5, 1e-4)   # 下限を 1e-4 にして暴走防止


# ---------------------------------------------------------
# iterative 版（完成版）
# ---------------------------------------------------------
def run_mle_and_aic_iterative(df, max_iter=50):

    x_raw = df["year"].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df["log_conc"].values

    # simple 版の結果を初期値にする（最重要）
    a_current, b_current, _, _, _, _ = run_mle_and_aic_simple(df)

    # tolerance を自動推定
    tolerance = estimate_tolerance(a_current, b_current, x_centered, y)

    prev_a, prev_b = a_current, b_current
    converged_status = "未収束"

    for i in range(max_iter):

        residuals = y - (a_current * x_centered + b_current)
        kde_new = gaussian_kde(residuals)

        res = minimize(
            lambda p: neg_log_likelihood_fixed_kde(p, x_centered, y, kde_new),
            [a_current, b_current],
            method="L-BFGS-B"
        )
        a_new, b_new = res.x
        ll1 = -res.fun

        # iteration 間の変化で収束判定
        delta = np.linalg.norm([a_new - prev_a, b_new - prev_b])
        if delta < tolerance:
            converged_status = "収束"
            final_iter = i + 1
            a1, b1 = a_new, b_new
            break

        prev_a, prev_b = a_current, b_current
        a_current, b_current = a_new, b_new

    else:
        converged_status = "最大回数到達"
        final_iter = max_iter
        a1, b1 = a_current, b_current

    # 定数モデル
    res0 = minimize(
        lambda p: neg_log_likelihood_fixed_kde(p, x_centered, y, gaussian_kde(y - p[0])),
        [np.mean(y)],
        method="L-BFGS-B"
    )
    ll0 = -res0.fun

    aic1 = -2 * ll1 + 4
    aic0 = -2 * ll0 + 2
    delta = aic1 - aic0
    p1 = np.exp(-0.5 * delta) / (1 + np.exp(-0.5 * delta))

    return a1, b1, aic0, aic1, p1, x_base, converged_status, final_iter
