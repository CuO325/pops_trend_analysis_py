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


def run_mle_and_aic_simple(df):
    """
    Simple KDE-MLE + AIC comparison between:
      - linear model: y = a*x + b
      - constant model: y = b0

    KDE is built once per model from the residuals at the MLE solution.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'year' and 'log_conc'.

    Returns
    -------
    a1 : float
        MLE slope for the linear model.
    b1 : float
        MLE intercept for the linear model.
    aic0 : float
        AIC for the constant model.
    aic1 : float
        AIC for the linear model.
    p1 : float
        AIC-based posterior weight for the linear model.
    x_base : float
        Minimum year used for centering.
    """
    x_raw = df["year"].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df["log_conc"].values

    # Linear model M1
    res1 = minimize(
        neg_log_likelihood,
        x0=[0.0, np.mean(y)],
        args=(x_centered, y),
        method="L-BFGS-B",
    )
    a1, b1 = res1.x
    max_log_likelihood_1 = -res1.fun

    # Constant model M0
    res0 = minimize(
        neg_log_likelihood_0,
        x0=[np.mean(y)],
        args=(x_centered, y),
        method="L-BFGS-B",
    )
    max_log_likelihood_0 = -res0.fun

    # AIC and AIC-based weight
    aic1 = -2 * max_log_likelihood_1 + 2 * 2  # 2 parameters: a, b
    aic0 = -2 * max_log_likelihood_0 + 2 * 1  # 1 parameter: b0
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator
    p1 = numerator / denominator if denominator != 0 else (0.0 if delta_aic > 0 else 1.0)

    return a1, b1, aic0, aic1, p1, x_base


def run_mle_and_aic_iterative(df, max_iter=50, tolerance=1e-2):
    """
    Iterative KDE-MLE + AIC comparison between:
      - linear model: y = a*x + b
      - constant model: y = b0

    This version updates the residual KDE at each iteration:
      1. Fit a linear model (a, b) given a current KDE.
      2. Rebuild KDE from the new residuals.
      3. Repeat until parameter change is small or max_iter is reached.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'year' and 'log_conc'.
    max_iter : int, optional
        Maximum number of iterations for the linear model refinement.
    tolerance : float, optional
        Convergence threshold on parameter change (L2 norm).

    Returns
    -------
    a1 : float
        Final slope estimate for the linear model.
    b1 : float
        Final intercept estimate for the linear model.
    aic0 : float
        AIC for the constant model.
    aic1 : float
        AIC for the linear model.
    p1 : float
        AIC-based posterior weight for the linear model.
    x_base : float
        Minimum year used for centering.
    """
    x_raw = df["year"].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df["log_conc"].values

    # Initial linear model from ordinary least squares
    a_current, b_current = np.polyfit(x_centered, y, 1)

    # Iterative refinement for the linear model (M1)
    for i in range(max_iter):
        residuals = y - (a_current * x_centered + b_current)

        try:
            kde_new = gaussian_kde(residuals)
        except np.linalg.LinAlgError:
            # If KDE fails, stop refinement and keep current parameters
            print(f"Warning: KDE construction failed at iteration {i}.")
            break

        def neg_log_likelihood_fixed(params, x, y):
            # Note: we still rebuild KDE inside neg_log_likelihood_1,
            # so kde_new is not directly used here. This wrapper is kept
            # for potential future extensions where a fixed KDE is passed.
            return neg_log_likelihood_1(params, x, y, kde_new)

        result = minimize(
            neg_log_likelihood_fixed,
            x0=[a_current, b_current],
            args=(x_centered, y),
            method="L-BFGS-B",
        )
        a_new, b_new = result.x
        max_log_likelihood_1 = -result.fun

        # Convergence check
        param_change = np.linalg.norm(
            np.array([a_new, b_new]) - np.array([a_current, b_current])
        )
        if param_change < tolerance:
            print(f"Converged at iteration {i + 1}.")
            a1, b1 = a_new, b_new
            break

        a_current, b_current = a_new, b_new
    else:
        # Reached max_iter without convergence
        print(f"Warning: reached max_iter={max_iter} without convergence.")
        a1, b1 = a_current, b_current

    # Constant model M0 (no iteration needed)
    try:
        res0 = minimize(
            neg_log_likelihood_0,
            x0=[np.mean(y)],
            args=(x_centered, y),
            method="L-BFGS-B",
        )
        max_log_likelihood_0 = -res0.fun
    except Exception:
        print("Constant model optimization failed.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_base

    # AIC and AIC-based weight
    aic1 = -2 * max_log_likelihood_1 + 2 * 2  # 2 parameters: a, b
    aic0 = -2 * max_log_likelihood_0 + 2 * 1  # 1 parameter: b0
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator
    p1 = numerator / denominator if denominator != 0 else (0.0 if delta_aic > 0 else 1.0)

    return a1, b1, aic0, aic1, p1, x_base

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

def neg_log_likelihood_1_old_bug(params, x, y, kde_fixed):
    """
    Variant of the negative log-likelihood for a linear model y = a*x + b.

    Currently, this function rebuilds a KDE from residuals internally,
    and the argument `kde_fixed` is not used in the computation.
    It is kept for potential future extensions where a precomputed KDE
    is passed and reused.

    Parameters
    ----------
    params : array-like, shape (2,)
        [a, b] where a is slope and b is intercept.
    x : array-like
        Predictor (centered year).
    y : array-like
        Response (log concentration).
    kde_fixed : object
        Placeholder for a precomputed KDE (not used at the moment).

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
        return 1e10
