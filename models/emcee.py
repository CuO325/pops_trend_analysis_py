
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import math
#import japanize_matplotlib
#from sklearn.utils import resample
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import statsmodels.api as sm
#import warnings
#from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from sklearn.metrics import r2_score
#import os
import emcee
import time
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

#from tqdm import tqdm  # tqdm がない場合は進捗表示を自作してください


def build_fft_kde_lookup(values, grid_min=None, grid_max=None, grid_n=4096, bandwidth=None, bw_factor=1.0):
    values = np.asarray(values).astype(float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("build_fft_kde_lookup: 入力 values が空です。")

    vmin = values.min() if grid_min is None else grid_min
    vmax = values.max() if grid_max is None else grid_max
    rng = vmax - vmin
    if rng == 0:
        rng = max(1.0, abs(vmin))
    #
    if bandwidth is None:
        sigma = np.std(values)
        n = len(values) # <--- n をここで定義
        # Silverman に近いが少し保守的に
        bw = sigma * (n ** (-1.0/5.0))
        if bw <= 0 or np.isnan(bw):
            bw = max(1e-3, sigma if sigma>0 else 1.0)
    else:
        n = len(values) # <--- n をここで定義
        bw = float(bandwidth)
    bw *= bw_factor

    # ---- パディングを大きめにとる（事前サンプルにも耐えるように） 「外側パディングは bandwidth の12倍程度必要」という KDE の実務的経験則----
    #pad = max(rng * 2.0, 3.0 * np.std(values), 1e-3)
    pad = max(rng * 1.0, 4.0 * np.std(values), bw * 12)

    gmin = vmin - pad
    gmax = vmax + pad
    grid = np.linspace(gmin, gmax, grid_n)
    dx = grid[1] - grid[0]

    counts, _ = np.histogram(values, bins=grid_n, range=(gmin, gmax))
    nvals = len(values)
    counts = counts.astype(float) / (nvals * dx)

    # kernel grid（対称に作る）
    kx = np.linspace(- (grid_n//2)*dx, (grid_n//2)*dx, grid_n)
    kernel = np.exp(-0.5 * (kx / bw)**2)
    kernel /= (np.sum(kernel) * dx)

    conv = fftconvolve(counts, kernel, mode='same')
    pdf_on_grid = conv
    # clip small negatives
    pdf_on_grid = np.clip(pdf_on_grid, 0.0, None)

    # 最小正値を見つけて、補間時の fill_value を決める（極小すぎないように）
    positive = pdf_on_grid[pdf_on_grid > 0]
    if positive.size == 0:
        min_pos = 1e-12
    else:
        min_pos = float(np.min(positive))
    # 補間の外挿値（無限遠扱い）は min_pos の 1e-3 程度にしておく
    fill_value = max(min_pos * 1e-3, 1e-12)

    bin_centers = np.linspace(gmin + dx/2.0, gmax - dx/2.0, grid_n)
    interp = interp1d(bin_centers, pdf_on_grid, bounds_error=False, fill_value=fill_value)

    # ==========================================================
    # ★★★ LOO 補正項の計算と追加 ★★★
    
    # ガウスカーネルの K(0) = 1 / sqrt(2 * pi)
    K0 = 1 / np.sqrt(2 * np.pi) 
    
    # LOO補正項: K(0) / (n * h)
    # n は len(values) または nvals を使用（ここでは nvals がデータ数）
    loo_correction = (1 / (nvals * bw)) * K0 
    
    # ==========================================================

    return {'grid': bin_centers,
            'pdf': pdf_on_grid,
            'interp': interp,
            'bw': bw,
            'dx': dx,
            'gmin': gmin,
            'gmax': gmax,
            'min_pos_pdf': min_pos,
            'fill_value': fill_value,
            # ★★★ 辞書に追加 ★★★
            'loo_correction': loo_correction}

def log_prior_informative(params, data_scale=None):

    a, b = params
    if not (-100 < a < 100 and -100 < b < 100):
        return -np.inf
    if data_scale is None:
        #return - (a**2) / (2 * 0.5**2) - (b**2) / (2 * 2.0**2)
        # 修正案（例：sigma_aを0.5から2.0に拡大）
        return - (a**2) / (2 * 2.0**2) - (b**2) / (2 * 2.0**2)
    else:
        x_range = data_scale.get('x_range', 1.0)
        y_std = data_scale.get('y_std', 1.0)
        #a_sd = max(0.01, y_std / max(1e-6, x_range) * 2.0)
        # 修正案（例：a_sdの最小値を0.01から0.1に拡大）
        a_sd = max(0.1, y_std / max(1e-6, x_range) * 2.0)
        b_sd = max(0.1, y_std * 2.0)

        return - (a**2) / (2 * a_sd**2) - (b**2) / (2 * b_sd**2)
    

def log_posterior_fast(params, x_centered, y, lookup, data_scale=None):

    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + ll

def run_bayes_kde_auto_stop_fast(
    df,
    fixed_lookup=None,
    grid_n=4096,
    bw_factor=1.0,
    n_walkers=16,
    n_steps=2000,
    chunk=100,
    burn_frac=0.5,
    verbose=True,
    ci_width_threshold=0.02,
    stable_rounds=2
):
 

    years = df['year'].values.astype(float)
    x_base = years.min()
    x_centered = years - x_base
    y = df['log_conc'].values.astype(float)
    ndim = 2

    try:
        mle_res = minimize(
            lambda p: -kde_log_likelihood_from_residuals_plain(p[0], p[1], x_centered, y),
            x0=[0.0, np.mean(y)],
            method='L-BFGS-B'
        )
        a_mle, b_mle = mle_res.x
    except Exception:
        a_mle, b_mle = 0.0, np.mean(y)

    if fixed_lookup is None:
        residuals_mle = y - (a_mle * x_centered + b_mle)
        lookup = build_fft_kde_lookup(residuals_mle, grid_n=grid_n, bandwidth=None, bw_factor=bw_factor)
    else:
        lookup = fixed_lookup

    data_scale = {'x_range': x_centered.max() - x_centered.min(), 'y_std': np.nanstd(y)}

    rng = np.random.default_rng(int(time.time() % 2**31))
    if n_walkers < 2*ndim:
        n_walkers = max(2*ndim, n_walkers)
    initial_pos = np.vstack([np.array([a_mle, b_mle]) + rng.normal(scale=[0.02, 0.1], size=2) for _ in range(n_walkers)])

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_fast, args=(x_centered, y, lookup, data_scale))

    pos = initial_pos
    n_chunks = max(1, n_steps // chunk)
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
            print(f"[FastMCMC {min(total_steps_done, n_steps)}/{n_steps}] a_mean={a_mean if not np.isnan(a_mean) else 'nan'} CI={ci} P(a<0)={p_neg if not np.isnan(p_neg) else 'nan'}")

        ci_width = ci[1] - ci[0] if not np.any(np.isnan(ci)) else np.inf
        ci_is_stable = (ci_width < ci_width_threshold)
        direction_stable = (not np.any(np.isnan(ci))) and ((ci[1] < 0) or (ci[0] > 0))

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

        if total_steps_done >= n_steps:
            break

    total_steps = sampler.iteration
    discard = int(burn_frac * total_steps)
    samples = sampler.get_chain(discard=discard, flat=True)

    if samples.size == 0:
        samples = sampler.get_chain(flat=True)

    a_samps = samples[:, 0] if samples.size != 0 else np.array([np.nan])
    b_samps = samples[:, 1] if samples.size != 0 else np.array([np.nan])

    chain3d = np.transpose(sampler.get_chain()[:,:,:], axes=(1,0,2))
    Rhat = compute_Rhat(chain3d)

    result = {
        "samples": samples,
        "a_mean": np.mean(a_samps) if a_samps.size>0 else np.nan,
        "b_mean": np.mean(b_samps) if b_samps.size>0 else np.nan,
        "a_ci": np.percentile(a_samps, [2.5, 97.5]) if a_samps.size>0 else np.array([np.nan, np.nan]),
        "b_ci": np.percentile(b_samps, [2.5, 97.5]) if b_samps.size>0 else np.array([np.nan, np.nan]),
        "p_neg_slope": np.mean(a_samps < 0) if a_samps.size>0 else np.nan,
        "steps_completed": total_steps,
        "stopped_early": stable_count >= stable_rounds,
        "Rhat": Rhat,
        "lookup": lookup
    }
    return result
def safe_gaussian_kde(values, bw_method=None):


    try:
        return gaussian_kde(values, bw_method=bw_method)
    except Exception:
        jittered = values + 1e-8 * np.random.randn(len(values))
        try:
            return gaussian_kde(jittered, bw_method=bw_method)
        except Exception:
            std = np.std(values)
            if std <= 0:
                std = 1e-6
            class SimpleKDE:
                def __init__(self, mu, sigma):
                    self.mu = mu
                    self.sigma = sigma
                def pdf(self, x):
                    return (1.0 / (np.sqrt(2*np.pi)*self.sigma)) * np.exp(-0.5 * ((x - self.mu)/self.sigma)**2)
            return SimpleKDE(np.mean(values), std)
        
# --- 置き換え用：fast_kde_log_likelihood_from_residuals（改良版） ---
def fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup, use_loo_correction=True):


    # 1. 残差の生成
    residuals = y - (a * x_centered + b)

    # 2. 補間テーブルから自己評価バイアスのある密度値 (f_raw) を取得
    f_raw = lookup['interp'](residuals)
    
    # 3. 尤度値 f_final の決定
    if use_loo_correction:
        # LOO補正項を取得
        # lookup に 'loo_correction' が含まれていることが前提
        loo_correction = lookup['loo_correction'] 
        
        # LOO 密度を計算 (f_loo = f_raw - 補正項)
        f_final = f_raw - loo_correction
        
        # 「TL積分の数値的安定性を確保するため、尤度の最小値を e-12 に制限した」
        f_final = np.clip(f_final, 1e-12, None) 

    else:
        # LOO補正なしの場合、生の値を使用
        f_final = f_raw
        # log の underflow を避ける
        f_final = np.clip(f_final, 1e-12, None) 


    return np.sum(np.log(f_final))

def kde_log_likelihood_from_residuals_plain(a, b, x_centered, y):

    res = y - (a * x_centered + b)
    try:
        kde = gaussian_kde(res)
        pdf_vals = kde.pdf(res)
        return np.sum(np.log(pdf_vals + 1e-12))
    except Exception:
        return -1e20

def compute_Rhat(chains, axis=0):

    try:
        n_chains, n_samples, n_params = chains.shape
        chain_means = np.mean(chains, axis=1)
        chain_vars = np.var(chains, axis=1, ddof=1)
        W = np.mean(chain_vars, axis=0)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)
        var_hat = ( (n_samples - 1)/n_samples ) * W + (1/n_samples) * B
        Rhat = np.sqrt(var_hat / W)
        return Rhat
    except Exception:
        return np.array([np.nan]*chains.shape[-1])
