# analysis/fft_kde_common_engine2.py

import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d


def prepare_fft_kde_lookup(residuals, bw_factor=1.0, grid_size=4096, use_loo=True):
    """
    残差から FFT-KDE による密度推定を行い、
    LOO 補正込みの log(pdf) 補間関数を含む lookup 辞書を返す。
    """
    residuals = np.asarray(residuals)
    n = len(residuals)

    # 範囲設定
    res_min, res_max = np.min(residuals), np.max(residuals)
    res_range = res_max - res_min
    grid_min = res_min - 0.5 * res_range
    grid_max = res_max + 0.5 * res_range

    # Silverman bandwidth
    std_res = np.std(residuals)
    bw = 1.06 * std_res * (n ** -0.2) * bw_factor

    """
    # グリッド
    grid = np.linspace(grid_min, grid_max, grid_size)
    dx = grid[1] - grid[0]

    # ビン化
    counts, _ = np.histogram(residuals, bins=grid_size, range=(grid_min, grid_max))

    # ガウスカーネル
    kernel_len = int(6 * bw / dx)

    # --- エラー処理（カーネル長が 1 未満なら、最低 3 点にする） ---
    if kernel_len < 3:
        kernel_len = 3

    kernel_x = np.linspace(-3 * bw, 3 * bw, int(6 * bw / dx))
    kernel = np.exp(-0.5 * (kernel_x / bw) ** 2)
    kernel /= (np.sqrt(2 * np.pi) * bw)
    """
    # グリッド
    grid = np.linspace(grid_min, grid_max, grid_size)
    dx = grid[1] - grid[0]

    # ビン化
    counts, _ = np.histogram(residuals, bins=grid_size, range=(grid_min, grid_max))

    # --- bw の安全処理 ---
    if bw <= 0 or np.isnan(bw) or np.isinf(bw):
        bw = dx * 1.0   # 最低限の帯域幅を確保

    # --- カーネル長の安全処理 ---
    kernel_len = int(6 * bw / dx)
    if kernel_len < 3 or np.isinf(kernel_len) or np.isnan(kernel_len):
        kernel_len = 3

    # ガウスカーネル
    kernel_x = np.linspace(-3 * bw, 3 * bw, kernel_len)
    kernel = np.exp(-0.5 * (kernel_x / bw) ** 2)
    kernel /= (np.sqrt(2 * np.pi) * bw)

    # FFT 畳み込み
    density = fftconvolve(counts / (n * dx), kernel * dx, mode="same")

    # LOO 補正（density 空間で一度だけ）
    if use_loo:
        k0 = 1.0 / np.sqrt(2 * np.pi)
        loo_correction = k0 / (n * bw)
        density = np.clip(density - loo_correction, 1e-12, None)
    else:
        loo_correction = 0.0
        density = np.clip(density, 1e-12, None)

    # log(pdf)
    log_density = np.log(density)

    # 補間関数（log(pdf)）
    interp_logpdf = interp1d(
        grid,
        log_density,
        kind="linear",
        fill_value=-100.0,
        bounds_error=False,
    )

    return {
        "grid": grid,
        "pdf": density,
        "logpdf": log_density,
        "interp": interp_logpdf,        # log(pdf) を返す
        "loo_correction": loo_correction,
        "bw": bw,
        "dx": dx,
    }


def fast_kde_log_likelihood(params, x, y, lookup):
    """
    FFT-KDE lookup（辞書）を使って log-likelihood を計算する。
    LOO 補正は lookup 作成時にすでに適用済みとみなす。
    """
    a, b = params
    residuals = y - (a * x + b)

    # log(pdf) を補間
    logpdf = lookup["interp"](residuals)
    logpdf = np.clip(logpdf, -1e6, None)

    return np.sum(logpdf)


def fast_kde_log_likelihood_from_residuals_old(a, b, x_centered, y, lookup):
    """
    (a, b) を個別に受け取る版。MCMC/TI から直接呼びやすい形。
    """
    residuals = y - (a * x_centered + b)
    logpdf = lookup["interp"](residuals)
    logpdf = np.clip(logpdf, -1e6, None)
    return np.sum(logpdf)

def fast_kde_log_likelihood_from_residuals(
    a, b,
    x_centered,
    y,
    lookup
):

    residuals = y - (a * x_centered + b)

    logpdf = lookup["interp"](residuals)

    # NaN / inf 防止
    logpdf = np.nan_to_num(
        logpdf,
        nan=-1e6,
        neginf=-1e6,
        posinf=0.0
    )

    # 下限制限
    logpdf = np.clip(logpdf, -1e6, 0)

    return np.sum(logpdf)
