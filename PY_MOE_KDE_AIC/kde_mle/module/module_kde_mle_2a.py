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
# AIC-weight の計算
# ---------------------------------------------------------
def model_weight(ic0: float, ic1: float) -> float:
    """情報量基準（AIC / AICc）からモデル1（線形モデル）の重みを計算する."""
    diff = float(np.clip(ic0 - ic1, -500.0, 500.0))
    return float(1.0 / (1.0 + np.exp(-0.5 * diff)))


def calc_aicc(aic: float, k: int, n_samples: int) -> float:
    """小標本補正を加えた AICc を計算する."""
    denom = n_samples - k - 1
    if denom <= 0:
        return np.inf
    return float(aic + (2.0 * k * (k + 1.0)) / denom)

# ---------------------------------------------------------
# simple 版（初期値として利用）
# ---------------------------------------------------------
def run_mle_and_aic_simple(df):
    x_raw = df["year"].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df["log_conc"].values

    n_samples = len(y)

    # 線形モデル
    def nll(params):
        a, b = params
        residuals = y - (a * x_centered + b)

        if np.std(residuals) < 1e-6:
            return 1e10

        try:
            kde = gaussian_kde(residuals)
            return -np.sum(np.log(kde.pdf(residuals) + 1e-10))
        except Exception:
            return 1e10

    res = minimize(nll, [0.0, np.mean(y)], method="L-BFGS-B")
    a1, b1 = res.x
    ll1 = -res.fun

    # 定数モデル
    def nll0(params):
        b0 = params[0]
        residuals = y - b0

        if np.std(residuals) < 1e-6:
            return 1e10

        try:
            kde = gaussian_kde(residuals)
            return -np.sum(np.log(kde.pdf(residuals) + 1e-10))
        except Exception:
            return 1e10

    res0 = minimize(nll0, [np.mean(y)], method="L-BFGS-B")
    ll0 = -res0.fun

    res_opt = y - (a1 * x_centered + b1)
    if np.std(res_opt) < 1e-6:
        ci_a_s, ci_b_s = (np.nan, np.nan), (np.nan, np.nan)
    else:
        kde_fixed = gaussian_kde(res_opt)

        def nll_fixed(params):
            a, b = params
            residuals = y - (a * x_centered + b)
            return -np.sum(np.log(kde_fixed.pdf(residuals) + 1e-10))

        ci_a_s, ci_b_s = calc_coef_ci_hessian(nll_fixed, a1, b1)

    # --------------------
    # AIC
    # --------------------
    k0 = 1
    k1 = 2

    aic1 = -2 * ll1 + 2 * k1
    aic0 = -2 * ll0 + 2 * k0
    p1 = model_weight(aic0, aic1)
    #delta = aic1 - aic0
    #p1 = np.exp(-0.5 * delta) / (1 + np.exp(-0.5 * delta))

    # --------------------
    # AICc
    # --------------------
    aicc1 = aic1 + 2 * k1 * (k1 + 1) / (n_samples - k1 - 1)
    aicc0 = aic0 + 2 * k0 * (k0 + 1) / (n_samples - k0 - 1)

    #delta_c = aicc1 - aicc0
    #p1_c = np.exp(-0.5 * delta_c) / (1 + np.exp(-0.5 * delta_c))


    p1_c = model_weight(aicc0, aicc1)

    return a1, b1, ci_a_s, ci_b_s, aic0, aic1, p1, aicc0, aicc1, p1_c, x_base

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


def run_mle_and_aic_iterative(df,a_current,b_current, max_iter=50):

    x_raw = df["year"].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df["log_conc"].values
    n_samples = len(y)
    # ---------------------------------------------------------
    # 1. simple 版の結果を初期値にする（最重要）
    # ---------------------------------------------------------
    #a_current, b_current, _, _, _, _ = run_mle_and_aic_simple(df)

    # ---------------------------------------------------------
    # 2. tolerance を自動推定
    # ---------------------------------------------------------
    tolerance = estimate_tolerance(a_current, b_current, x_centered, y)

    prev_a, prev_b = a_current, b_current
    converged_status = "未収束"

    # ---------------------------------------------------------
    # 3. iterative MLE
    # ---------------------------------------------------------
    for i in range(max_iter):

        residuals = y - (a_current * x_centered + b_current)


        if np.std(residuals) < 1e-6:
            # この年は解析不能
            return None, None, (np.nan, np.nan), (np.nan, np.nan),None, None, None, None, None, None, x_base, "残差ゼロ（KDE不可）", 0,tolerance,

        kde_new = gaussian_kde(residuals)

        # 固定 KDE を使った NLL
        def neg_log_likelihood_fixed(params, x, y):
            return neg_log_likelihood_fixed_kde(params, x, y, kde_new)

        # 最適化
        res = minimize(
            neg_log_likelihood_fixed,
            [a_current, b_current],
            args=(x_centered, y),
            method='L-BFGS-B'
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

    # ---------------------------------------------------------
    # 4. simple 版と同じ M0（定数モデル）
    # ---------------------------------------------------------
    def nll0(params):
        b0 = params[0]
        residuals = y - b0
        try:
            kde0 = gaussian_kde(residuals)
            return -np.sum(np.log(kde0.pdf(residuals) + 1e-10))
        except Exception:
            return 1e10

    res0 = minimize(nll0, [np.mean(y)], method="L-BFGS-B")
    ll0 = -res0.fun

    # ---------------------------------------------------------
    # 5. AIC と p1
    # ---------------------------------------------------------
    k0 = 1
    k1 = 2

    aic1 = -2.0 * ll1 + 2.0 * k1
    aic0 = -2.0 * ll0 + 2.0 * k0
    p1 = model_weight(aic0, aic1)

    aicc1 = calc_aicc(aic1, k1, n_samples)
    aicc0 = calc_aicc(aic0, k0, n_samples)
    p1_c = model_weight(aicc0, aicc1)

    res_final = y - (a1 * x_centered + b1)
    if np.std(res_final) < 1e-6:
        ci_a, ci_b = (np.nan, np.nan), (np.nan, np.nan)
    else:
        final_kde = gaussian_kde(res_final)
        ci_a, ci_b = calc_coef_ci_bootstrap_kde(x_centered, y, a1, b1, final_kde)

    return (
        a1,
        b1,
        ci_a,
        ci_b,
        aic0,
        aic1,
        p1,
        aicc0,
        aicc1,
        p1_c,
        x_base,
        converged_status,
        final_iter,
        tolerance,
    )




from scipy.stats import norm


def calc_coef_ci_hessian(
    nll_func,
    a_hat: float,
    b_hat: float,
    alpha: float = 0.05,
    eps: float = 1e-4,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """負の対数尤度関数の数値ヘッセ行列から傾き a と切片 b の漸近信頼区間を算出する."""
    theta = np.array([a_hat, b_hat], dtype=float)
    f0 = nll_func(theta)

    H = np.zeros((2, 2))
    for i in range(2):
        ei = np.zeros(2)
        ei[i] = eps
        H[i, i] = (nll_func(theta + ei) - 2.0 * f0 + nll_func(theta - ei)) / (eps ** 2)

    e0 = np.array([eps, 0.0])
    e1 = np.array([0.0, eps])
    H[0, 1] = H[1, 0] = (
        nll_func(theta + e0 + e1)
        - nll_func(theta + e0 - e1)
        - nll_func(theta - e0 + e1)
        + nll_func(theta - e0 - e1)
    ) / (4.0 * eps ** 2)

    try:
        cov = np.linalg.inv(H)
        if np.any(np.diag(cov) <= 0):
            return (np.nan, np.nan), (np.nan, np.nan)
        se_a, se_b = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return (np.nan, np.nan), (np.nan, np.nan)

    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci_a = (float(a_hat - z * se_a), float(a_hat + z * se_a))
    ci_b = (float(b_hat - z * se_b), float(b_hat + z * se_b))
    return ci_a, ci_b

# ---------------------------------------------------------
#ヘッセ行列版（simple 用）と残差ブートストラップ版（iterative 用）
# ---------------------------------------------------------
from scipy.stats import norm


def calc_coef_ci_hessian(
    nll_func,
    a_hat: float,
    b_hat: float,
    alpha: float = 0.05,
    eps: float = 1e-4,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
    """負の対数尤度関数の数値ヘッセ行列から傾き a と切片 b の漸近信頼区間を算出する."""
    theta = np.array([a_hat, b_hat], dtype=float)
    f0 = nll_func(theta)

    H = np.zeros((2, 2))
    for i in range(2):
        ei = np.zeros(2)
        ei[i] = eps
        H[i, i] = (nll_func(theta + ei) - 2.0 * f0 + nll_func(theta - ei)) / (eps ** 2)

    e0 = np.array([eps, 0.0])
    e1 = np.array([0.0, eps])
    H[0, 1] = H[1, 0] = (
        nll_func(theta + e0 + e1)
        - nll_func(theta + e0 - e1)
        - nll_func(theta - e0 + e1)
        + nll_func(theta - e0 - e1)
    ) / (4.0 * eps ** 2)

    try:
        cov = np.linalg.inv(H)
        if np.any(np.diag(cov) <= 0):
            return (np.nan, np.nan), (np.nan, np.nan)
        se_a, se_b = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return (np.nan, np.nan), (np.nan, np.nan)

    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci_a = (float(a_hat - z * se_a), float(a_hat + z * se_a))
    ci_b = (float(b_hat - z * se_b), float(b_hat + z * se_b))
    return ci_a, ci_b


def calc_coef_ci_bootstrap_kde(
    x_centered: np.ndarray,
    y: np.ndarray,
    a_hat: float,
    b_hat: float,
    kde_obj,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
    """収束済み KDE を用いた残差ブートストラップにより傾き a と切片 b の信頼区間を算出する."""
    residuals = y - (a_hat * x_centered + b_hat)
    if kde_obj is None or np.std(residuals) < 1e-6:
        return (np.nan, np.nan), (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    y_fitted = a_hat * x_centered + b_hat
    n = len(y)

    a_samples = []
    b_samples = []

    def _make_fast_logpdf_from_kde(kde_obj, residuals: np.ndarray, grid_size: int = 2048):
        """固定 KDE の対数密度をグリッド補間関数に変換する（高速化用）."""
        margin = max(1.0, 3.0 * float(np.std(residuals)))
        grid = np.linspace(float(residuals.min()) - margin, float(residuals.max()) + margin, grid_size)
        log_pdf_grid = np.log(kde_obj.pdf(grid) + 1e-10)
        edge_val = float(np.log(1e-10))

        def eval_logpdf(x: np.ndarray) -> np.ndarray:
            return np.interp(x, grid, log_pdf_grid, left=edge_val, right=edge_val)

        return eval_logpdf
    
    fast_logpdf = _make_fast_logpdf_from_kde(kde_obj, residuals)

    for _ in range(n_boot):
        boot_res = rng.choice(residuals, size=n, replace=True)
        y_boot = y_fitted + boot_res

        def nll_boot(params):
            a, b = params
            res_b = y_boot - (a * x_centered + b)
            return -float(np.sum(fast_logpdf(res_b)))

        def nll_boot_slow(params):
            a, b = params
            res_b = y_boot - (a * x_centered + b)
            return -np.sum(np.log(kde_obj.pdf(res_b) + 1e-10))

        res = minimize(nll_boot, [a_hat, b_hat], method="L-BFGS-B")
        if res.success:
            a_samples.append(res.x[0])
            b_samples.append(res.x[1])

    if len(a_samples) < 10:
        return (np.nan, np.nan), (np.nan, np.nan)

    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)
    ci_a = (float(np.percentile(a_samples, low_q)), float(np.percentile(a_samples, high_q)))
    ci_b = (float(np.percentile(b_samples, low_q)), float(np.percentile(b_samples, high_q)))
    return ci_a, ci_b