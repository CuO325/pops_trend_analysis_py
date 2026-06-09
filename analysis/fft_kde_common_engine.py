from matplotlib.pyplot import grid
import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

def prepare_fft_kde_lookup_x(residuals, bw_factor=1.0, grid_size=4096, use_loo=True):
    """
    FFT-KDE による密度推定と LOO 補正を行い、
    ベイズ推定で使用する lookup 辞書を返す。
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

    # ★★★ 重要：辞書形式で返す ★★★
    return {
        "grid": grid,
        "pdf": density,
        "logpdf": log_density,
        "interp": interp_logpdf,
        "loo_correction": loo_correction,
        "bw": bw,
        "dx": dx,
        "min": grid_min,
        "max": grid_max,
    }



def prepare_fft_kde_lookup(residuals, bw_factor=1.0, grid_size=4096, use_loo=True):
    #def prepare_fft_kde_lookup(residuals, bw_factor=1.0, grid_size=1024):
    """
    残差からFFTを用いた高速な尤度計算用のルックアップテーブルを作成する
    """
    res_min, res_max = np.min(residuals), np.max(residuals)
    res_range = res_max - res_min
    # グリッドの範囲を少し広めに設定（マージン）
    grid_min = res_min - 0.5 * res_range
    grid_max = res_max + 0.5 * res_range
    
    # KDEの帯域幅（Bandwidth）の計算
    std_res = np.std(residuals)
    n = len(residuals)
    bw = 1.06 * std_res * (n ** -0.2) * bw_factor # シルバーマンの公式
    
    # グリッド上でのKDE計算（FFT用）
    grid = np.linspace(grid_min, grid_max, grid_size)
    dx = grid[1] - grid[0]
    
    # データ点をグリッドに割り当て（ビン化）
    counts, _ = np.histogram(residuals, bins=grid_size, range=(grid_min, grid_max))
    
    # カーネル（ガウス）の作成
    kernel_x = np.linspace(-3*bw, 3*bw, int(6*bw/dx))
    kernel = np.exp(-0.5 * (kernel_x / bw)**2)
    kernel /= (np.sqrt(2 * np.pi) * bw)
    
    # FFT畳み込みによる密度推定
    #density = fftconvolve(counts / (n * dx), kernel * dx, mode='same')
    
    # 対数尤度用の補間関数（ルックアップテーブル）を作成
    # 密度が0にならないよう微小な値を加える（ログ爆発防止）
    #log_density = np.log(density + 1e-10)


    # ... (前述のFFT-KDE計算) ...

    # 密度推定結果 (density) に対して
    # 1. まずベースとなる密度推定（density）を計算する
    # 例: density = fftconvolve(counts, kernel, mode='same') / (n * bw) 
    # ここで必ず 'density' という変数に値を入れる必要があります
    density = fftconvolve(counts / (n * dx), kernel * dx, mode='same')

    if use_loo:
        # LOO補正項: K(0) / (n * h)
        # ガウスカーネルの K(0) = 1/sqrt(2π)
        bw = 1.06 * np.std(residuals) * (n ** -0.2) * bw_factor
        k0 = 1.0 / np.sqrt(2 * np.pi)
        loo_correction = k0 / (n * bw)
        density = np.clip(density - loo_correction, 1e-12, None)
    else:
        density = np.clip(density, 1e-12, None)
        
    log_density = np.log(density)
    # fft_kde_common_engine.py 内の修正

    
    return interp1d(grid, log_density, kind='linear', 
                fill_value=-100.0,  # -inf ではなく非常に小さな有限値を返す
                bounds_error=False)
    #return interp1d(grid, log_density, kind='linear', fill_value=-np.inf, bounds_error=False)
    """
    # 密度計算
    density = fftconvolve(counts / (n * dx), kernel * dx, mode='same')
    
    if use_loo:
        # LOO補正: K(0)/(n*h) を引く
        k0 = 1 / np.sqrt(2 * np.pi)
        # bw はシルバーマン等で計算された値
        loo_correction = k0 / (len(residuals) * bw)
        density = np.clip(density - loo_correction, 1e-12, None)
    else:
        density = np.clip(density, 1e-12, None)
    # 密度が0にならないよう微小な値を加える（ログ爆発防止）
    log_density = np.log(density + 1e-10)

    return interp1d(grid, log_density, kind='linear', fill_value=-np.inf, bounds_error=False)
    """
def fast_kde_log_likelihood_x(params, x, y, lookup, use_loo=True):
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

def fast_kde_log_likelihood(params, x, y, lookup_func):
    """
    パラメータ a, b に対する高速な対数尤度計算
    """
    a, b = params
    current_residuals = y - (a * x + b)
    log_ls = lookup_func(current_residuals)
    
    # 無効な値（範囲外）のチェック
    if np.any(np.isinf(log_ls)):
        return -1e18
    return np.sum(log_ls)

def log_prior_standard(params):
    """
    標準的な事前分布（必要に応じて変更可能）
    """
    a, b = params
    # 緩やかな正規分布を想定
    lp_a = -0.5 * (a / 1.0)**2
    lp_b = -0.5 * (b / 10.0)**2
    return lp_a + lp_b