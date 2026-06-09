import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

def prepare_fft_kde_lookup(residuals, bw_factor=1.0, grid_size=1024):
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
    density = fftconvolve(counts / (n * dx), kernel * dx, mode='same')
    
    # 対数尤度用の補間関数（ルックアップテーブル）を作成
    # 密度が0にならないよう微小な値を加える（ログ爆発防止）
    log_density = np.log(density + 1e-10)
    return interp1d(grid, log_density, kind='linear', fill_value=-np.inf, bounds_error=False)

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