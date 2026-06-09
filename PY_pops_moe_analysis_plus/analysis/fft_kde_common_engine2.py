from matplotlib.pyplot import grid
import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

def prepare_fft_kde_lookup(residuals, bw_factor=1.0, grid_size=4096, use_loo=True):
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

    # グリッド
    grid = np.linspace(grid_min, grid_max, grid_size)
    dx = grid[1] - grid[0]

    # ビン化
    counts, _ = np.histogram(residuals, bins=grid_size, range=(grid_min, grid_max))

    # ガウスカーネル
    kernel_x = np.linspace(-3*bw, 3*bw, int(6*bw/dx))
    kernel = np.exp(-0.5 * (kernel_x / bw)**2)
    kernel /= (np.sqrt(2*np.pi) * bw)

    # FFT 畳み込み
    density = fftconvolve(counts / (n * dx), kernel * dx, mode='same')

    # LOO 補正
    if use_loo:
        k0 = 1.0 / np.sqrt(2*np.pi)
        loo_correction = k0 / (n * bw)
        density = np.clip(density - loo_correction, 1e-12, None)
    else:
        loo_correction = 0.0
        density = np.clip(density, 1e-12, None)

    # log(pdf)
    log_density = np.log(density)

    # 補間関数（log(pdf)）
    interp_logpdf = interp1d(
        grid, log_density,
        kind='linear',
        fill_value=-100.0,
        bounds_error=False
    )

    # ★★★ 高速な辞書形式で返す ★★★
    return {
        "grid": grid,
        "pdf": density,
        "logpdf": log_density,
        "interp": interp_logpdf,
        "loo_correction": loo_correction,
        "bw": bw,
        "dx": dx,
    }

def fast_kde_log_likelihood(params, x, y, lookup, use_loo=True):
    #use_looは、lookup 側で制御」さてているため、ここでは特に制御しない
    a, b = params
    residuals = y - (a * x + b)

    # すでに「LOO 済み density の log」を補間している
    logpdf = lookup['interp'](residuals)

    # ここで loo_correction をいじらない（もう適用済み）
    # クリップだけ数値安定化として残すなら：
    logpdf = np.clip(logpdf, -1e6, None)

    return np.sum(logpdf)


def fast_kde_log_likelihood_old(params, x, y, lookup, use_loo=True):
    a, b = params
    residuals = y - (a * x + b)

    # pdf from lookup
    f_raw = lookup['interp'](residuals)

    # LOO correction
    if use_loo and 'loo_correction' in lookup:
        f_final = f_raw - lookup['loo_correction']
    else:
        f_final = f_raw

    # avoid negative or zero
    f_final = np.clip(f_final, 1e-12, None)

    return np.sum(np.log(f_final))

