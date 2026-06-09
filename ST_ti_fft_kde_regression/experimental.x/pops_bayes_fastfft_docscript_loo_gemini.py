# pops_bayes_fastfft_pM1_update.py
# FFT-KDE 高速ベイズ（MCMC）を統合した修正版

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import japanize_matplotlib
from sklearn.utils import resample
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from sklearn.metrics import r2_score
import os
import emcee
import time
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

from tqdm import tqdm  # tqdm がない場合は進捗表示を自作してください

# -----------------------------------------------------
#  固定 KDE（FFT）を用いた対数尤度を β で温度付けしたもの
#  FFT-KDEによる対数尤度を温度βでスケーリングした事後分布（log posterior）を返す
# -----------------------------------------------------
def tempered_log_posterior(params, beta, x_centered, y, lookup, data_scale=None):
    """
    tempered posterior = log_prior + beta * log_likelihood
    温度付きの対数事後分布（log posterior）を計算します。

    これは、ベイズ推定における「事前分布 + β倍の対数尤度」に相当します。
    熱力学積分（TI）で使用される温度パラメータ β により、尤度の影響を調整します。

    Parameters:
        - params: パラメータ [a, b]（傾きと切片）
        - beta: 温度パラメータ（0〜1）
        - x_centered: 中心化された説明変数（年）
        - y: 応答変数（対数濃度など）
        - lookup: FFT-KDE による尤度のルックアップテーブル
        - data_scale: スケーリング情報（事前分布に使用）

    Returns:
        - tempered log posterior（スカラー値）

    課題:
        周辺尤度（logZ）の推定精度と温度スケジューリング。熱力学積分（Thermodynamic Integration; TI）による周辺尤度 logZ の推定精度は以下の要素に依存する。「logZ の絶対値」よりも モデル比較時の差（ΔlogZ） に主眼を置き、TI の数値安定性が分析結果に影響する可能性がある。 

            1. β（inverse temperature）グリッドの分解能
                - 0 < β < 1 の区間で密なグリッドが必要
                - 尤度が急に変化する領域では特に細分化が必要
            2. 各 β での MCMC 収束性
                - R̂（収束診断）
                - ESS（実効サンプルサイズ）を確認する必要がある
            3. E[log L] の数値積分誤差
                - trapz（台形公式）を使うが、計算点不足で logZ が不安定になる
            4. 複数ランによる再現性確認
                - logZ を 2〜3 回推定し、標準偏差（TI ノイズ）を報告するのが望ましい    
    """
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf

    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + beta * ll


# -----------------------------------------------------
#  β（温度）ごとに MCMC を回して E_beta[logL] を推定　。n_steps=600, burn_frac=0.4 は試行回数としては比較的少
# 指定された温度βに対してMCMCを実行し、E_β[logL]（期待対数尤度）を推定
# -----------------------------------------------------


def estimate_E_loglike_for_beta(
    beta, x_centered, y, lookup,
    n_walkers=20, n_steps=2000, burn_frac=0.4,
    init_center=None, data_scale=None, show_progress=False
):
    """
    指定された温度 β に対して MCMC を実行し、log尤度の期待値 E[logL] を推定します。

    熱力学積分に必要な E[logL] を各 β ごとに求めるための関数です。

    Parameters:
        - beta: 温度パラメータ（0〜1）
        - x_centered, y: データ
        - lookup: FFT-KDE 尤度テーブル
        - n_walkers: MCMCチェーンの本数
        - n_steps: MCMCのステップ数
        - burn_frac: バーンイン比率（初期ステップの破棄割合）
        - init_center: MCMC初期値（[a, b]）
        - data_scale: 事前分布のスケーリング情報
        - show_progress: 進捗表示の有無

    Returns:
        - E_logL: log尤度の平均
        - chain: MCMCサンプル（burn-in後）
        - sampler: emceeのサンプラーオブジェクト
    """

    ndim = 2

    if init_center is None:
        init_center = np.array([0.0, np.mean(y)])

    rng = np.random.default_rng()
    initial_pos = init_center + 1e-3 * rng.normal(size=(n_walkers, ndim))

    def _logprob(theta):
        return tempered_log_posterior(
            theta, beta, x_centered, y, lookup, data_scale=data_scale
        )

    sampler = emcee.EnsembleSampler(n_walkers, ndim, _logprob)

    pos, _, _ = sampler.run_mcmc(initial_pos, n_steps, progress=show_progress)
    chain = sampler.get_chain(discard=int(n_steps * burn_frac), flat=True)

    # 非 tempered で logL を評価（TI のため）
    ll_vals = np.array([
        fast_kde_log_likelihood_from_residuals(th[0], th[1], x_centered, y, lookup)
        for th in chain
    ])
    return np.mean(ll_vals), chain, sampler


# -----------------------------------------------------
#  熱力学積分により logZ（周辺尤度）を計算 。n_steps=600, burn_frac=0.4 は試行回数としては比較的少い
#  熱力学積分（TI）により周辺尤度 logZ を推定（FFT-KDEによる非パラメトリック尤度）
# -----------------------------------------------------
def compute_log_marginal_TL(
    df, model='linear',
    betas=None,
    n_walkers=20, n_steps=600, burn_frac=0.4,
    grid_n_lookup=1024, bw_factor=1.0,
    verbose=True
):
    """
    model: 'linear' (a,b) or 'constant' (b only, a=0固定)
        
    熱力学積分（Thermodynamic Integration）により、モデルの周辺尤度 logZ を推定します。

    線形モデルと定数モデルのいずれかを選択し、FFT-KDEを用いた非パラメトリックな尤度で評価します。

    Parameters:
        - df: データフレーム（'year', 'log_conc' を含む）
        - model: 'linear'（傾きあり）または 'constant'（傾きなし）
        - betas: 温度パラメータの配列（省略時は自動生成）
        - n_walkers, n_steps, burn_frac: MCMC設定
        - grid_n_lookup: FFT-KDEのグリッド数
        - bw_factor: KDEのバンド幅調整係数
        - verbose: ログ出力の有無

    Returns:
        - dict:
        - 'logZ': 周辺尤度（logスケール）
        - 'E_logL': 各βにおけるlog尤度の平均
        - 'betas': 使用したβの配列
        - 'a_mle', 'b_mle': MLE推定値（初期値として使用）
        - 'lookup': FFT-KDEテーブル
    """
    
    years = df['year'].values.astype(float)
    x_base = years.min()
    x_centered = years - x_base
    y = df['log_conc'].values.astype(float)

    # -------------------
    # MLE による残差 → KDE lookup 作成（モデルごと）
    # -------------------
    if model == 'linear':
        res = minimize(
            neg_log_likelihood,
            [-0.01, np.mean(y)],
            args=(x_centered, y),
            method="L-BFGS-B"
        )
        a_mle, b_mle = res.x
        residuals = y - (a_mle * x_centered + b_mle)

    else:  # constant model
        res0 = minimize(
            neg_log_likelihood_0,
            [np.mean(y)],
            args=(x_centered, y),
            method="L-BFGS-B"
        )
        b0 = res0.x[0]
        a_mle, b_mle = 0.0, b0
        residuals = y - b0

    lookup = build_fft_kde_lookup(
        residuals, grid_n=grid_n_lookup, bw_factor=bw_factor
    )
    """
    # ----------------------------
    # β grid（速度優先：12点）
    # ----------------------------
    if betas is None:
        betas = np.concatenate([
            np.linspace(0, 0.4, 5),
            np.linspace(0.5, 1.0, 7)
        ])

    betas = np.unique(np.sort(betas))
    """
    # ----------------------------
    # β grid（安定性優先：51点以上）
    # ----------------------------
    if betas is None:
        # 均等に51点（デフォルトの12点より大幅に精度向上）
        # betas = np.linspace(0.0, 1.0, 51) 
        
        # さらに安定性を求める場合は、以下のように1.0付近を密にする設定も有効です
        betas = np.concatenate([
            np.linspace(0, 0.9, 30, endpoint=False), # 0.9までを30点
            np.linspace(0.9, 1.0, 21)               # 0.9から1.0までを21点
        ])

    betas = np.unique(np.sort(betas))


    if verbose:
        print(f"\n[TI] model={model}")
        print(" betas =", np.round(betas, 3))

    E_logL = []
    data_scale = {
        'x_range': x_centered.max() - x_centered.min(),
        'y_std': np.nanstd(y)
    }

    init_center = np.array([a_mle, b_mle])

    for beta in betas:
        if verbose:
            print(f" β={beta:.3f} ... ", end="")
        E, chain, sampler = estimate_E_loglike_for_beta(
            beta, x_centered, y, lookup,
            n_walkers=n_walkers,
            n_steps=n_steps,
            burn_frac=burn_frac,
            init_center=init_center,
            data_scale=data_scale,
            show_progress=False
        )
        if verbose:
            print(f"E[logL]={E:.3f}")

        E_logL.append(E)

    # ----------------------------
    # TI 積分：台形公式
    # ----------------------------
    logZ = np.trapz(E_logL, x=betas)

    return {
        'model': model,
        'betas': betas,
        'E_logL': np.array(E_logL),
        'logZ': logZ,
        'a_mle': a_mle,
        'b_mle': b_mle,
        'lookup': lookup,
    }


# -----------------------------------------------------
# 0次 vs 1次 の Bayes 因子・事後確率
# 周辺尤度 logZ に基づいてベイズ因子と事後確率を計算し、モデル比較を行う
# -----------------------------------------------------
def compare_models_logZ(res1, res0, prior_ratio=1.0):
    """
    prior_ratio = p(M1)/p(M0) （通常1）
    
    Compare two models using thermodynamic integration logZ values.

    Parameters:
        - res1: dict containing logZ for model 1
        - res0: dict containing logZ for model 0
        - prior_ratio: prior odds p(M1)/p(M0), default is 1

    Returns:
        - dict with keys:
        - logZ1: log marginal likelihood for model 1
        - logZ0: log marginal likelihood for model 0
        - logBF: log Bayes factor
        - BF: Bayes factor
        - p_M1: posterior probability of model 1
    
       
    2つのモデルの周辺尤度 logZ を比較し、ベイズ因子と事後確率を計算します。

    Parameters:
        - res1: compute_log_marginal_TL() の結果（モデル1）
        - res0: 同上（モデル0）
        - prior_ratio: モデル1とモデル0の事前確率比（通常は1）

    Returns:
        - dict:
        - 'logZ1', 'logZ0': 各モデルのlogZ
        - 'logBF': logベイズ因子
        - 'BF': ベイズ因子
        - 'p_M1': モデル1の事後確率
    """
    
    
    logBF = res1['logZ'] - res0['logZ'] + np.log(prior_ratio)
    BF = np.exp(logBF)
    p_M1 = BF / (1 + BF)

    return {
        'logZ1': res1['logZ'],
        'logZ0': res0['logZ'],
        'logBF': logBF,
        'BF': BF,
        'p_M1': p_M1,
    }


#--------------------------------既存-------------------------------------------------------
# ---------------------------
# 高速 FFT-KDE ユーティリティ
# ---------------------------

def build_fft_kde_lookup_loo(values, grid_min=None, grid_max=None, grid_n=4096, bandwidth=None, bw_factor=1.0):
    """
    高速なカーネル密度推定（KDE）を行うためのルックアップテーブルを構築します。
    FFT（高速フーリエ変換）を用いて、指定されたデータの密度分布を効率的に近似します。

    Parameters:
        values (array-like): KDEを行う対象のデータ（数値の配列）
        grid_min (float, optional): グリッドの最小値（省略時はデータの最小値から自動設定）
        grid_max (float, optional): グリッドの最大値（省略時はデータの最大値から自動設定）
        grid_n (int, optional): グリッドの分割数（デフォルトは4096）
        bandwidth (float, optional): KDEのバンド幅（省略時はSilverman's ruleで自動計算）
        bw_factor (float, optional): バンド幅の調整係数（デフォルトは1.0）

    Returns:
        dict: 以下のキーを含む辞書
            - 'grid': グリッドの中心値（配列）
            - 'pdf': KDEで得られた密度（配列）
            - 'interp': 密度関数の補間関数（scipy.interpolate.interp1d）
            - 'bw': 使用されたバンド幅
            - 'dx': グリッド間隔
            - 'gmin', 'gmax': グリッドの範囲（最小・最大）

    Notes:
        - 欠損値（NaN）は自動的に除去されます。
        - KDEは正規分布カーネルを使用します。
        - FFTにより計算が高速化されています。
    """

    values = np.asarray(values).astype(float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("build_fft_kde_lookup: 入力 values が空です。")
    vmin = values.min() if grid_min is None else grid_min
    vmax = values.max() if grid_max is None else grid_max
    rng = vmax - vmin
    if rng == 0:
        rng = max(1.0, abs(vmin))
    pad = rng * 0.5 + 1e-6
    gmin = vmin - pad
    gmax = vmax + pad
    grid = np.linspace(gmin, gmax, grid_n)
    dx = grid[1] - grid[0]

    if bandwidth is None:
        sigma = np.std(values)
        n = len(values)
        bw = sigma * (n ** (-1.0/5.0))
        if bw <= 0 or np.isnan(bw):
            bw = max(1e-3, sigma if sigma>0 else 1.0)
    else:
        bw = float(bandwidth)
    bw *= bw_factor

    counts, _ = np.histogram(values, bins=grid_n, range=(gmin, gmax))
    nvals = len(values)
    counts = counts.astype(float) / (nvals * dx)

    kx = np.linspace(- (grid_n//2)*dx, (grid_n//2)*dx, grid_n)
    kernel = np.exp(-0.5 * (kx / bw)**2)
    kernel /= (np.sum(kernel) * dx)

    conv = fftconvolve(counts, kernel, mode='same')
    pdf_on_grid = conv
    bin_centers = np.linspace(gmin + dx/2.0, gmax - dx/2.0, grid_n)
    pdf_on_grid = np.clip(pdf_on_grid, 0.0, None)
    interp = interp1d(bin_centers, pdf_on_grid, bounds_error=False, fill_value=1e-300)
    return {'grid': bin_centers, 'pdf': pdf_on_grid, 'interp': interp, 'bw': bw, 'dx': dx, 'gmin': gmin, 'gmax': gmax}
# --- 置き換え用：build_fft_kde_lookup（改良版） ---
def build_fft_kde_lookup_nonloo(values, grid_min=None, grid_max=None, grid_n=4096, bandwidth=None, bw_factor=1.0):
    values = np.asarray(values).astype(float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("build_fft_kde_lookup: 入力 values が空です。")

    vmin = values.min() if grid_min is None else grid_min
    vmax = values.max() if grid_max is None else grid_max
    rng = vmax - vmin
    if rng == 0:
        rng = max(1.0, abs(vmin))

    # ---- パディングを大きめにとる（事前サンプルにも耐えるように） ----
    # 以前は pad = rng * 0.5、ここでは経験則で 2.0 倍以上の余裕を持たせる
    pad = max(rng * 2.0, 3.0 * np.std(values), 1e-3)

    gmin = vmin - pad
    gmax = vmax + pad
    grid = np.linspace(gmin, gmax, grid_n)
    dx = grid[1] - grid[0]

    if bandwidth is None:
        sigma = np.std(values)
        n = len(values)
        # Silverman に近いが少し保守的に
        bw = sigma * (n ** (-1.0/5.0))
        if bw <= 0 or np.isnan(bw):
            bw = max(1e-3, sigma if sigma>0 else 1.0)
    else:
        bw = float(bandwidth)
    bw *= bw_factor

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
        #min_pos = 1e-12
        min_pos = 1e-6
    else:
        min_pos = float(np.min(positive))
    # 補間の外挿値（無限遠扱い）は min_pos の 1e-3 程度にしておく
    fill_value = max(min_pos * 1e-3, 1e-12)

    bin_centers = np.linspace(gmin + dx/2.0, gmax - dx/2.0, grid_n)
    interp = interp1d(bin_centers, pdf_on_grid, bounds_error=False, fill_value=fill_value)

    return {'grid': bin_centers,
            'pdf': pdf_on_grid,
            'interp': interp,
            'bw': bw,
            'dx': dx,
            'gmin': gmin,
            'gmax': gmax,
            'min_pos_pdf': min_pos,
            'fill_value': fill_value}

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

# --- 置き換え用：fast_kde_log_likelihood_from_residuals（改良版） ---
#def fast_kde_log_likelihood_from_residuals_loo_corrected(a, b, x_centered, y, lookup):
def fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup, use_loo_correction=True):
    """
    残差に基づく log-likelihood を FFT-KDE 補間テーブルから高速に推定する。

    本関数は、回帰残差に対して FFT-KDE による密度推定を行い、対数尤度を計算します。
    Leave-One-Out (LOO) 補正を適用することで、自己評価バイアスを除去した密度推定が可能です。
    数値的安定性のため、尤度値の下限を exp(-12) に制限しています。

    Parameters
    ----------
    a : float
        回帰直線の傾き。
    b : float
        回帰直線の切片。
    x_centered : ndarray
        中心化された説明変数（平均0）。
    y : ndarray
        目的変数。
    lookup : dict
        FFT-KDE 補間テーブル。以下のキーを含む必要があります:
        - 'interp': 残差に対する密度補間関数
        - 'loo_correction': LOO補正項（use_loo_correction=True の場合）
    use_loo_correction : bool, optional
        True の場合、LOO補正を適用して密度推定を行う（デフォルト: True）。

    Returns
    -------
    log_likelihood : float
        残差に基づく log-likelihood の合計値。

    Notes
    -----
        - 熱力学積分の数値的ロバスト性を確保するため、尤度の最小値(floor)  を e^-12 に設定した。
        - これは、より小さな値 (e^-000 ～　e-^30) で見られた pM1 の過度な変動を抑制するためである。
        - 尤度値の下限を `1e-12` に設定することで、log の underflow を防ぎ、熱力学積分における数値的ロバスト性を確保しています。
        - LOO補正を適用することで、自己評価バイアスによる過度な pM1 の変動を抑制できます。
    """

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

def fast_kde_log_likelihood_from_residuals_work(a, b, x_centered, y, lookup, use_loo_correction=True):
    """
    互換性を維持しつつ、lookup に基づく適応 floor を用いる。
    - lookup は build_fft_kde_lookup の戻り値を想定
    - use_loo_correction: True のとき LOO 補正（K0/(n*h)）を適用する
    """
    residuals = y - (a * x_centered + b)
    # interp で評価（fill_value は lookup によって適切に設定済み）
    pdf_vals = lookup['interp'](residuals)

    # LOO 補正（もし使うなら）
    if use_loo_correction:
        # n と h を lookup から推定（lookup は元データで構築されている前提）
        # 注意: lookup が元のサンプル数を直接持たない場合は n を呼び出し元で保持して渡すべき。
        # ここでは近似的に n を求められなければ補正しない。
        n_est = getattr(lookup, 'n', None)
        if n_est is None:
            # try to infer n from pdf normalization: sum(pdf)*dx ~ 1 => n_est unknown -> skip
            n_est = None

        if (n_est is not None) and ('bw' in lookup):
            K0 = 1.0 / np.sqrt(2.0 * np.pi)
            corr = (1.0 / (n_est * lookup['bw'])) * K0
            pdf_vals = pdf_vals - corr

    # 数値安定化：lookup 提供の fill_value / min_pos_pdf を使う
    floor = lookup.get('fill_value', 1e-12)
    # 最低でも小さすぎない下限を確保
    floor = max(floor, 1e-12)
    pdf_vals = np.clip(pdf_vals, floor, None)

    return np.sum(np.log(pdf_vals))

def fast_kde_log_likelihood_from_residuals_no_loo(a, b, x_centered, y, lookup):

    """
    FFT-KDEに基づいて、回帰残差の対数尤度（log-likelihood）を計算します。

    Parameters:
    a (float): 回帰直線の傾き
    b (float): 回帰直線の切片
    x_centered (array-like): 中心化された説明変数（例：年 - 最小年）
    y (array-like): 応答変数（例：対数濃度）
    lookup (dict): build_fft_kde_lookup() によって生成されたKDEルックアップ辞書

    Returns:
    float: 残差に対する対数尤度の合計値

    Notes:
        - KDE補間関数を使って、残差の密度を評価します。
        - log(密度)の合計が尤度スコアになります。
        - 密度がゼロにならないよう、微小値（1e-12）を加算しています。

    課題：
        - floor の選択が logZ に影響し、密度 floor（1e-12）について、以下の感度解析が望ましい
        - 最小の非ゼロ密度の 1e−k 倍を floor とする自動設定。
        - 複数の floor 値（1e-6, 1e-9, 1e-12, 1e-15）で周辺尤度を比較
        - logsumexp ベースの安定化手法を使う
    """

    residuals = y - (a * x_centered + b)
    pdf_vals = lookup['interp'](residuals)
    return np.sum(np.log(pdf_vals + 1e-12))

def safe_gaussian_kde(values, bw_method=None):

    """
    scipy.stats.gaussian_kde を安全に実行するためのラッパー関数です。
    KDEの失敗時には、微小なノイズを加えたり、簡易的な正規分布近似にフォールバックします。

    Parameters:
    values (array-like): KDEを行う対象のデータ
    bw_method (str or float, optional): バンド幅の指定方法（scipy標準に準拠）

    Returns:
    object: KDEオブジェクト（通常は gaussian_kde、失敗時は簡易KDEクラス）

    Notes:
        - データの分散がゼロの場合や、KDEが失敗する場合に備えて、複数のフォールバック処理を行います。
        - 最終的には平均と分散に基づく正規分布近似を返すことがあります。
        - 返されるオブジェクトは `.pdf(x)` メソッドを持ち、任意のxに対する密度を評価できます。
    """

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

# ---------------------------
# 事前・事後・MCMC 関連（FFT-KDE版）
# ---------------------------

def log_prior_informative(params, data_scale=None):
    """
    傾き a と切片 b に対する情報量のある事前分布（正規分布）を対数形式で返します。
    Parameters:
        - params (tuple): 回帰パラメータ (a, b)
        - data_scale (dict, optional): スケーリング情報（x_range, y_std）に基づいて事前分布の分散を調整
    Returns:
        - float: 対数事前確率（log prior）。パラメータが範囲外なら -inf を返します。
    Notes:
        - 分布はゼロ平均の正規分布で、分散は固定またはデータスケールに応じて変化します。
        - sigma_a=0.5 は、年間の log(conc)の変化率が pm 0.5 程度であることを許容する、比較的タイトな事前分布で、非情報的な比較を行うには狭すぎます。
        - $y_std / x_range の値が非常に小さい場合、sigma_a は 0.01 に設定されます。sigma_a=0.01 は極めて狭い事前分布であり、データが実質的に傾き a approx 0 を示している今回のケースで p_M1 を過大評価する最大の原因と考えられます
    課題:
        - 事前分布の幅が周辺尤度に与える影響（事前感度）線形モデル M1 の傾き a に対して、事前分布として𝑎∼𝑁(0,𝜎𝑎2)を用いている。
        - 周辺尤度（モデル証拠）𝑝(data∣𝑀)は 事前の幅 σₐ に強く依存する。
            - 狭い事前（σₐ が極小）：事前がデータの尤度ピークに重なる場合、事前の体積が小さいため周辺尤度が高くなる（Occam's razor 効果）。結果として p(M1|data) を押し上げることがある。
            - 広い事前（σₐ が大きい）： 事前体積が大きく “説明範囲が広すぎる” ため、周辺尤度が減少する傾向がある。
            - このため、事前分布の設定はモデル比較に大きく影響するため、事前感度解析（σₐ = 0.01, 0.1, 0.5, 1.0 の比較）を行うことを推奨する。   
    """

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

    """
    FFT-KDE による対数尤度と事前分布を組み合わせた事後分布（log posterior）を返します。

    Parameters:
        - params (tuple): 回帰パラメータ (a, b)
        - x_centered (array-like): 中心化された説明変数
        - y (array-like): 応答変数
        - lookup (dict): FFT-KDE による密度補間関数などを含む辞書
        - data_scale (dict, optional): 事前分布のスケーリング情報

    Returns:
        - float: 対数事後確率（log posterior）。事前が無限小なら -inf を返します。
    """

    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + ll

def compute_Rhat(chains, axis=0):
    """
    Gelman-Rubin 診断統計量（R̂）を計算し、MCMCの収束性を評価します。

    Parameters:
        - chains (ndarray): MCMCチェーン（形状: [n_chains, n_samples, n_params]）
        - axis (int): 計算軸（通常は0）

    Returns:
        - ndarray: 各パラメータに対する R̂ 値（1に近いほど収束している）

    Notes:
        - 収束していない場合やエラー時は NaN を返します。
    """

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

def kde_log_likelihood_from_residuals_plain(a, b, x_centered, y):

    """
    scipy の Gaussian KDE を用いて、回帰残差の対数尤度を計算します。

    Parameters:
        - a (float): 傾き
        - b (float): 切片
        - x_centered (array-like): 中心化された説明変数
        - y (array-like): 応答変数

    Returns:
        - float: 残差に対する対数尤度の合計。KDE失敗時は極端な低値を返します。
    """
    res = y - (a * x_centered + b)
    try:
        kde = gaussian_kde(res)
        pdf_vals = kde.pdf(res)
        return np.sum(np.log(pdf_vals + 1e-12))
    except Exception:
        return -1e20

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

    """
    FFT-KDE を用いたベイズ回帰推定を MCMC により実行し、傾向の事後確率と収束判定を行います。

    自動的に収束判定を行い、安定した傾向（CI幅が狭く、符号が一貫）で早期停止します。

    Parameters:
        - df (DataFrame): 'year' と 'log_conc' を含むデータフレーム
        - fixed_lookup (dict, optional): FFT-KDE の事前計算済みルックアップ（省略時はMLEから生成）
        - grid_n (int): KDEグリッド数（デフォルト4096）
        - bw_factor (float): KDEバンド幅の調整係数
        - n_walkers (int): MCMCのチェーン数（最低2×パラメータ数）
        - n_steps (int): 最大ステップ数
        - chunk (int): MCMCの1回の実行ステップ数
        - burn_frac (float): バーンイン比率（初期ステップの破棄割合）
        - verbose (bool): ログ出力の有無
        - ci_width_threshold (float): CI幅の収束判定閾値
        - stable_rounds (int): 安定判定の連続回数（収束条件）

    Returns:
        - dict: 推定結果を含む辞書
        - 'samples': MCMCサンプル（burn-in後）
        - 'a_mean', 'b_mean': 傾き・切片の平均
        - 'a_ci', 'b_ci': 95%信用区間
        - 'p_neg_slope': 傾きが負である事後確率
        - 'steps_completed': 実行ステップ数
        - 'stopped_early': 早期収束したかどうか
        - 'Rhat': Gelman-Rubin統計量
        - 'lookup': 使用したFFT-KDEルックアップ

    Notes:
        - MCMCは emcee の EnsembleSampler を使用。
        - KDEは FFT による高速化済み。
        - 自動収束判定により、計算時間を短縮可能。
    """    

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

# ---------------------------
# 既存の解析関数群（元ファイルのロジック）
# ---------------------------

def determine_split_year(df):

    """
    データフレーム内の 'year' 列から中央値の年を求め、分析の分割年として返します。

    Parameters:
        - df (DataFrame): 'year' 列を含むデータフレーム

    Returns:
        - int or None: 中央年（int）。年が4つ未満の場合は None を返します。

    Notes:
        - 欠損値（NaN）は自動的に除外されます。
        - 分割年は、時系列を前後に分けるための基準として使われます。
    """

    years = df['year'].dropna().unique()
    if len(years) < 4:
        return None
    return int(np.median(years))

def compare_periods(df, split_year=2015, R=10000):

    """
    指定された分割年でデータを2期間に分け、ブートストラップにより平均差の95%信頼区間を推定します。

    Parameters:
        - df (DataFrame): 'year' および 'log_conc' 列を含むデータフレーム
        - split_year (int): 分割年（この年を境に前後で比較）
        - R (int): ブートストラップの繰り返し回数（デフォルトは10,000）

    Returns:
        - ndarray: 2.5%点と97.5%点からなる平均差の95%信頼区間（配列）
               データが不足している場合は [NaN, NaN] を返します。
    """
    A = df[df['year'] <= split_year]['log_conc'].values
    B = df[df['year'] > split_year]['log_conc'].values
    if len(A) < 2 or len(B) < 2:
        return np.array([np.nan, np.nan])
    diffs = []
    for _ in range(R):
        A_sample = resample(A, replace=True)
        B_sample = resample(B, replace=True)
        diffs.append(np.mean(A_sample) - np.mean(B_sample))
    return np.percentile(diffs, [2.5, 97.5])

def neg_log_likelihood(params, x, y):
    """
    線形モデル（y = ax + b）に対する負の対数尤度をKDEで近似して返します。
    Parameters:
        - params (tuple): 回帰パラメータ (a, b)
        - x (array-like): 説明変数（中心化済み）
        - y (array-like): 応答変数
    Returns:
        - float: 負の対数尤度（最小化対象）。KDEが失敗した場合は大きな値（1e10）を返します。
    Notes:
        - 線形モデルの残差分布を Gaussian KDE により推定し、対数尤度を非パラメトリックに近似します。
        - 正規分布などの仮定を置かず、データに基づいて柔軟に分布を構築します。
    """
    a, b = params
    residuals = y - (a * x + b)
    # === バンド幅を0.3に固定する例 ===
    #h_fixed = 1
    try:
        kde = gaussian_kde(residuals)
        #kde = gaussian_kde(residuals, bw_method=h_fixed) # ここで設定
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        #print(kde.factor) hの値0.33程度であった。
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10
    
def neg_log_likelihood_1(params, x, y, initial_kde):
    """
    反復計算用
    線形モデル（y = ax + b）に対する負の対数尤度をKDEで近似して返します。
    Parameters:
        - params (tuple): 回帰パラメータ (a, b)
        - x (array-like): 説明変数（中心化済み）
        - y (array-like): 応答変数
    Returns:
        - float: 負の対数尤度（最小化対象）。KDEが失敗した場合は大きな値（1e10）を返します。
    Notes:
        - 線形モデルの残差分布を Gaussian KDE により推定し、対数尤度を非パラメトリックに近似します。
        - 正規分布などの仮定を置かず、データに基づいて柔軟に分布を構築します。
    """
    a, b = params
    residuals = y - (a * x + b)
    # === バンド幅を0.3に固定する例 ===
    #h_fixed = 1
    try:
        kde = gaussian_kde(residuals)
        #kde = gaussian_kde(residuals, bw_method=h_fixed) # ここで設定
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        #print(kde.factor) hの値0.33程度であった。
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10

def neg_log_likelihood_0(params, x, y):
    """
    定数モデル（y = b）の負の対数尤度をKDEで近似して返します。
    Parameters:
        - params (tuple): 切片のみを含むパラメータ (b,)
        - x (array-like): 説明変数（使用しませんが引数として必要）
        - y (array-like): 応答変数
    Returns:
        - float: 負の対数尤度（最小化対象）。KDEが失敗した場合は大きな値（1e10）を返します。
    Note:
        - シルバーマンの規則を使用する バンド幅 h = 1.06 x σ x n^{-1/5}
            kde_silverman = gaussian_kde(residuals, bw_method='silverman')
        - スコットの規則を使用する　バンド幅 h = 1.00 x σ x n^{-1/5}
            kde_scott = gaussian_kde(residuals, bw_method='scott')
    """

    b0 = params[0]
    residuals = y - b0
    # === バンド幅を0.3に固定する例 ===
    #h_fixed = 1
    try:
        kde = gaussian_kde(residuals)
        #kde = gaussian_kde(residuals, bw_method=h_fixed) # ここで設定
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10

def run_mle_and_aic_non_iterative(df):
    """
    線形モデルと定数モデルの最尤推定（MLE）を行い、AICによるモデル比較を実施します。

    Parameters:
        - df (DataFrame): 'year' および 'log_conc' 列を含むデータフレーム
    Returns:
        - tuple:
        - a1 (float): 線形モデルの傾き（MLE）
        - b1 (float): 線形モデルの切片（MLE）
        - aic0 (float): 定数モデルのAIC
        - aic1 (float): 線形モデルのAIC
        - p1 (float): 線形モデルが定数モデルより良い確率（AIC差に基づく）
        - x_base (float): 年の最小値（中心化の基準）

    Notes:
        - KDEを用いた非パラメトリック尤度に基づくMLEを使用しています。
        - AICはモデルの複雑さを考慮した情報量基準です。
        - `p1` は AIC差から導かれるモデル選択確率の近似です。
        - KDEは1度のみ実行し、残渣分布は固定です。

    課題：
    KDE を含む尤度の AIC への利用についての注意

        - AIC は　AIC=2k−2logLの形で計算されるが、その導出は パラメトリックモデルにおける有効自由度 k を前提としている。
        - KDE（非パラメトリック成分）を尤度に含めると、KDE の帯域幅が実質的な自由度に影響 smoothing に伴う自由度が AIC の “k” に含まれないといった理由により、AIC の厳密な意味は失われる。
        - 本ドキュメントでは AIC を“KDE 連続モデルの比較に対する経験的（ヒューリスティックな）尺度”として用いているが、理論的に完全な AIC ではないことを注意として記す。
        - KDE を含むモデル比較のより厳密な指標としては、

            - ベイズ周辺尤度（log evidence）
            - 汎化誤差のクロスバリデーション（LOO-CV）
            - Widely Applicable Information Criterion（WAIC）

        などを併用することが推奨される。 
    """

    np.set_printoptions(precision=10, suppress=True)
    x_raw = df['year'].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df['log_conc'].values
    result = minimize(neg_log_likelihood, [-0.1, np.mean(y)], args=(x_centered, y), method='L-BFGS-B')
    a1, b1 = result.x
    max_log_likelihood_1 = -result.fun
    result_0 = minimize(neg_log_likelihood_0, [np.mean(y)], args=(x_centered, y), method='L-BFGS-B')
    max_log_likelihood_0 = -result_0.fun
    aic1 = -2 * max_log_likelihood_1 + 2 * 2
    aic0 = -2 * max_log_likelihood_0 + 2 * 1
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator
    p1 = numerator / denominator if denominator != 0 else (0.0 if delta_aic > 0 else 1.0)
    #linear_validity = np.exp(-(aic1 - aic0) / 2)
    #return a1, b1, aic0, aic1, p1, x_base, linear_validity
    return a1, b1, aic0, aic1, p1, x_base

def run_mle_and_aic(df, max_iter=20, tolerance=1e-6):
    """
    非パラメトリック尤度推定とモデル選択（AIC）を行う関数。
        線形モデル（M1）と定数モデル（M0）を比較し、尤度最大化とAIC計算を通じてモデル適合度を評価します。
        線形モデルでは、残差分布をカーネル密度推定（KDE）で逐次更新しながら、最尤推定（MLE）を行います。

        - 初期モデルの推定: 最初の残差分布を構築するために、np.polyfit で初期の a と b を決めます。
        - ループの追加: max_iter と tolerance を設定した for ループを run_mle_and_aic_non_iterative() に追加しました。
        - 尤度関数の再構築: ループ内で、現在のパラメータの残差に基づき gaussian_kde(residuals) を再実行し、残差分布（尤度関数）を更新しています。
        - 最適化: minimize 関数は、更新されたKDEオブジェクトを引数として受け取るように再定義した固定尤度関数に対して実行されます。
        - 収束判定: 新しいパラメータ (a_new, b_new) と古いパラメータ (a_current, b_current) の変化が tolerance 以下になったら、ループを抜けます。 

    Parameters
        df : pandas.DataFrame
            入力データフレーム。以下の2列が必要です:
            - 'year': 説明変数（時系列など）
            - 'log_conc': 目的変数（対数変換済み濃度など）

        max_iter : int, optional
            線形モデルの反復最適化の最大回数（デフォルトは 20）。

        tolerance : float, optional
            パラメータ変化の収束判定閾値（デフォルトは 1e-6）。

    Returns
        a1 : float
            線形モデル M1 の最尤推定された傾きパラメータ。
        b1 : float
            線形モデル M1 の最尤推定された切片パラメータ。
        max_log_likelihood_1 : float
            線形モデル M1 の最大対数尤度。
        max_log_likelihood_0 : float
            定数モデル M0 の最大対数尤度。
        aic_diff : float
            線形モデルと定数モデルの AIC 差（M0 - M1）。正なら M1 が優れている。
        x_base : float
            年度の基準値（最小値）。センタリングに使用。
    Notes
        - 線形モデル M1 は `y = ax + b` の形で、残差分布を KDE により非パラメトリックに推定。
        - 定数モデル M0 は `y = c` の形で、単一パラメータによる最尤推定。
        - AIC は `2k - 2LL` に基づき、モデルの複雑さと適合度のバランスを評価。
        - `neg_log_likelihood_1` と `neg_log_likelihood_0` は外部定義関数として必要。
    課題：
    KDE を含む尤度の AIC への利用についての注意

        - AIC は　AIC=2k−2logLの形で計算されるが、その導出は パラメトリックモデルにおける有効自由度 k を前提としている。
        - KDE（非パラメトリック成分）を尤度に含めると、KDE の帯域幅が実質的な自由度に影響 smoothing に伴う自由度が AIC の “k” に含まれないといった理由により、AIC の厳密な意味は失われる。
        - 本ドキュメントでは AIC を“KDE 連続モデルの比較に対する経験的（ヒューリスティックな）尺度”として用いているが、理論的に完全な AIC ではないことを注意として記す。
        - KDE を含むモデル比較のより厳密な指標としては、

            - ベイズ周辺尤度（log evidence）
            - 汎化誤差のクロスバリデーション（LOO-CV）
            - Widely Applicable Information Criterion（WAIC）

        などを併用することが推奨される。 
    Examples
        >>> df = pd.DataFrame({'year': [2000, 2001, 2002], 'log_conc': [1.2, 1.5, 1.7]})
        >>> run_mle_and_aic(df)
        (0.25, 1.0, -1.23, -2.45, 2.44, 2000.0)
    
    """
    
    np.set_printoptions(precision=10, suppress=True)
    x_raw = df['year'].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df['log_conc'].values
    n_data = len(y)
    
    # 1. 初期値設定と初期残差分布の構築
    # 最初の残差分布構築のための初期モデルとして、最小二乗法の結果を流用
    a_current, b_current = np.polyfit(x_centered, y, 1) # 初期モデルの推定
    
    # 尤度関数オブジェクト (KDE) を保持する変数
    kde_current = None
    
    # 2. 反復的な最適化 (線形モデル M1)
    for i in range(max_iter):
        
        # a. 現在のパラメータでの残差計算
        residuals = y - (a_current * x_centered + b_current)
        
        # b. 残差分布の更新 (FFT-KDE/KDEの再構築)
        try:
            # 新しい残差でKDEオブジェクトを再構築 (バンド幅は自動選択)
            kde_new = gaussian_kde(residuals)
        except np.linalg.LinAlgError:
            print(f"警告: 反復 {i} でKDE構築エラー。スキップします。")
            break
            
        # c. 固定された新しいKDEでパラメータを最適化 (MLE)
        # 目的関数: neg_log_likelihood は残し、新しい kde_new を引数として受け取るようにneg_log_likelihood_1()を追加
        # ただし、scipy.minimize の引数構造に合わせるため、ここでは引数initial_kdeにkde_newを渡す
        
        # NOTE: 目的関数の呼び出し方を変える必要があるため、lambda関数でinitial_kdeを固定してminimizeに渡します
        def neg_log_likelihood_fixed(params, x, y):
             return neg_log_likelihood_1(params, x, y, kde_new)

        result = minimize(neg_log_likelihood_fixed, [a_current, b_current], 
                          args=(x_centered, y), method='L-BFGS-B')
        
        a_new, b_new = result.x
        max_log_likelihood_1 = -result.fun
        
        # d. 収束判定
        # パラメータがほとんど変化しなくなった場合、または尤度が収束した場合
        param_change = np.linalg.norm(np.array([a_new, b_new]) - np.array([a_current, b_current]))
        
        if param_change < tolerance:
            print(f"反復 {i+1} で収束しました。")
            a1, b1 = a_new, b_new
            break
        
        # e. 更新
        a_current, b_current = a_new, b_new
        kde_current = kde_new
        
    else: # forループが max_iter に達した場合
        print(f"警告: 最大反復回数 {max_iter} に達しましたが収束しませんでした。")
        a1, b1 = a_current, b_current
        
    # 3. 定数モデル M0 の推定 (反復は不要)
    # ※ neg_log_likelihood_0 関数が存在すると仮定
    try:
        result_0 = minimize(neg_log_likelihood_0, [np.mean(y)], args=(x_centered, y), method='L-BFGS-B')
        max_log_likelihood_0 = -result_0.fun
    except Exception:
        # エラー処理
        print("定数モデルの最適化に失敗しました。")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_base

    # 4. AICとP1の計算 (ここからは元のコードと同じ)
    aic1 = -2 * max_log_likelihood_1 + 2 * 2
    aic0 = -2 * max_log_likelihood_0 + 2 * 1
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator
    p1 = numerator / denominator if denominator != 0 else (0.0 if delta_aic > 0 else 1.0)
    
    return a1, b1, aic0, aic1, p1, x_base


def binomial_trend_analyze(df):
    """
    検出限界以下の割合に対して、年次傾向を二項ロジスティック回帰で評価し、AICによるモデル比較を行います。

    Parameters:
        - df (DataFrame): 'year', 'concentration', 'detection_limit' を含むデータフレーム

    Returns:
        - tuple:
        - aic0 (float): 定数モデル（傾きなし）のAIC
        - aic1 (float): 線形モデル（傾きあり）のAIC
        - p1 (float): 線形モデルが定数モデルより優れている確率（AIC差に基づく）
        - slope (float): 線形モデルの傾き（年次変化の方向性）

    Notes:
        - 各年ごとに検出限界以下の件数と総件数を集計し、割合をロジスティック回帰でモデル化します。
        - `PerfectSeparationWarning` を抑制し、安全にモデルをフィットします。
        - AICに基づくモデル選択確率 `p1` は、傾向の有無を判断する指標として使えます。
    """

    unified_dl = df['detection_limit'].max()
    df['low_flag'] = (df['concentration'] < unified_dl).astype(int)
    yearly_data = df.groupby('year').agg(
        low_count=('low_flag', 'sum'),
        total_count=('low_flag', 'count')
    ).reset_index()
    x_year = yearly_data['year'].astype(float).to_numpy()
    X1 = sm.add_constant(x_year).astype(float)
    low_count = yearly_data['low_count'].astype(float).to_numpy()
    total_count = yearly_data['total_count'].astype(float).to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        y_ratio = np.where(total_count > 0, low_count / total_count, np.nan)
    n_weights = total_count
    valid_mask = (~np.isnan(y_ratio)) & (~np.isnan(X1).any(axis=1)) & (~np.isnan(n_weights)) & (n_weights > 0)
    X1 = X1[valid_mask]
    y_ratio = y_ratio[valid_mask]
    n_weights = n_weights[valid_mask]
    X0 = np.ones((len(y_ratio), 1), dtype=float)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=PerfectSeparationWarning)
        model1 = sm.GLM(y_ratio, X1, family=sm.families.Binomial())
        model1_fit = model1.fit(freq_weights=n_weights)
        aic1 = model1_fit.aic
        model0 = sm.GLM(y_ratio, X0, family=sm.families.Binomial())
        model0_fit = model0.fit(freq_weights=n_weights)
        aic0 = model0_fit.aic
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    p1 = numerator / (1 + numerator)
    slope = model1_fit.params[1] if len(model1_fit.params) > 1 else np.nan
    return aic0, aic1, p1, slope

def clean_value_user(val):
    """
    ユーザー入力値（文字列）を数値に変換し、欠損値や特殊表記を処理します。

    Parameters:
        - val (str or any): 入力値（文字列または数値）

    Returns:
        - float or np.nan: 数値に変換された値。変換できない場合は NaN を返します。

    Notes:
        - 'tr(...)' や '---', 'nd' などの特殊表記を除去・変換します。
        - 空文字列や変換不能な値は NaN に変換されます。
        - 主にデータクリーニング前処理に使用します。
    """
    if isinstance(val, str):
        val = val.replace('tr(', '').replace(')', '')
        val = val.replace('---', '')
        val = val.replace('nd','0')
    try:
        if val == '': return np.nan
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def is_data_acceptable(a, p1, ci):

    """
    傾向の有意性と方向性に基づいて、データが受け入れ可能かどうかを判定します。

    Parameters:
        - a (float): 傾き（回帰係数）
        - p1 (float): 線形モデルの選択確率（AICに基づく）
        - ci (array-like): 傾きの95%信用区間（[下限, 上限]）

    Returns:
        - bool: 条件をすべて満たす場合は True（受け入れ可能）、それ以外は False

    判定条件:
        - 傾きが負（a < 0）
        - モデル選択確率 p1 ≥ 0.95（強い傾向の証拠）
        - 信用区間がゼロをまたがない（傾向の符号が一貫）

    Notes:
        - 主にベイズ推定や回帰分析の結果をフィルタリングするために使用します。
    """
    if a >= 0:
        return False
    if p1 < 0.95:
        return False
    if ci[0] < 0 and ci[1] > 0:
        return False
    return True
# -------------------------------------------------------------------
# 核となる FFT-KDE（スムージングされた密度グリッドを返す）
"""fast_kde, fast_kde_log_likelihood_from_residuals, fast_kde_log_likelihood を
あなたの HTML にそのまま置き換え可能です。"""
# -------------------------------------------------------------------
def fast_kde(samples, gridsize=512, norm=True):
    """
    FFT による Gaussian KDE。
    samples : 1D residual array
    """
    samples = np.asarray(samples)
    n = samples.size

    # グリッド（FFT 用に均等幅）
    smin, smax = samples.min(), samples.max()
    padding = (smax - smin) * 0.02 + 1e-9
    xmin, xmax = smin - padding, smax + padding

    grid = np.linspace(xmin, xmax, gridsize)
    dx = grid[1] - grid[0]

    # ヒストグラム（FFT の入力）
    hist, _ = np.histogram(samples, bins=gridsize, range=(xmin, xmax))
    hist = hist.astype(float)

    # Silverman bandwidth
    h = 1.06 * samples.std() * n ** (-1/5)
    if h <= 0:
        h = (samples.max() - samples.min())/20.0 + 1e-6

    # Gaussian kernel（FFT 用にグリッド上で作る）
    x = (np.arange(gridsize) - gridsize//2) * dx
    kernel = np.exp(-(x**2)/(2*h*h))
    kernel = np.roll(kernel, gridsize//2)

    # FFT convolution
    kde_fft = irfft(rfft(hist) * rfft(kernel), n=gridsize)

    # 正規化
    if norm:
        kde_fft = kde_fft / (kde_fft.sum() * dx)

    return grid, kde_fft, h


# -------------------------------------------------------------------
# FFT-KDE の結果から任意点の密度を補間評価
# -------------------------------------------------------------------
def evaluate_kde_at_points(grid, density, points):
    return np.interp(points, grid, density, left=1e-12, right=1e-12)


# -------------------------------------------------------------------
# ★★ FFT-KDE + LOO 補正を適用した “残差の対数尤度” 関数 ★★
# -------------------------------------------------------------------
def fast_kde_log_likelihood_from_residuals_3input(residuals, gridsize=512, floor=1e-12):
    """
    LOO-KDE を利用した対数尤度を返す。
    残差 → fast_kde() → 各点の密度 → LOO補正 → log-likelihood
    """
    residuals = np.asarray(residuals)
    n = residuals.size

    # FFT-KDE 構築（1回だけ）
    grid, dens, h = fast_kde(residuals, gridsize=gridsize)

    # 通常の（自己評価バイアスのある）密度
    f_raw = evaluate_kde_at_points(grid, dens, residuals)

    # ---- LOO 補正項（Gaussian Kernel の K(0)）----
    K0 = 1 / np.sqrt(2*np.pi)
    loo_correction = (1 / (n * h)) * K0

    # LOO 密度
    f_loo = f_raw - loo_correction

    # 数値安定化
    f_loo = np.clip(f_loo, floor, None)

    return np.sum(np.log(f_loo))

def fast_kde_log_likelihood_from_residuals_5input(a, b, x_centered, y, lookup):
    """
    5引数版（互換性修正版）
    FFT-KDE補間テーブル lookup を使って、残差の log-likelihood を返す。
    古い3引数版と同じ動作を維持しつつ、新しい引数形式に完全対応。
    """
    # a,b,x_centered,y から残差を生成
    residuals = y - (a * x_centered + b)

    # lookup には 'interp' が必ず存在する前提（build_fft_kde_lookup の出力）
    pdf_vals = lookup['interp'](residuals)

    # log の underflow を避ける
    pdf_vals = np.clip(pdf_vals, 1e-300, None)

    return np.sum(np.log(pdf_vals))

# -------------------------------------------------------------------
# MCMC / TI などで呼び出す “データ全体の log-likelihood”
# -------------------------------------------------------------------
def fast_kde_log_likelihood(x, y, a, b):
    """
    y ~ f(x; a,b) + residual
    残差 = y - (a x + b)
    その残差を FFT-KDE + LOO-KDE で評価して log-likelihood を返す。
    """
    residuals = y - (a * x + b)
    return fast_kde_log_likelihood_from_residuals(residuals)
# ---------------------------
# メイン処理（FFT-KDE ベイズを統合）
# ---------------------------
"""
このスクリプトは、環境モニタリングデータに基づいて、物質ごとの年次傾向を評価するための解析パイプラインを実行します。

主な処理内容:
1. データの読み込みと前処理
   - CSVファイルから 'concentration' と 'detection_limit' を読み込み
   - 欠損値や検出限界未満の値を処理（DL/√2 に置換）
   - 対数変換（log10）を適用し、前処理済みデータを保存

2. 対象物質ごとの解析（データ数が24件以上のもの）
   (1) 最尤推定（MLE）とAICによるモデル比較
       - KDEによる非パラメトリック尤度で線形モデルと定数モデルを評価
       - AIC差から線形傾向の有無を判定

   (2) ベイズ推定（FFT-KDE + MCMC）
       - MLE残差からFFT-KDEを構築し、MCMCで事後分布を推定
       - 傾きの事後分布から減少傾向の確率を評価
       - 熱力学積分による周辺尤度 logZ を用いたベイズ因子と事後確率の計算

   (3) ブートストラップ検定
       - 年次中央値で前後に分割し、平均値差の95%信頼区間を推定
       - 有意な濃度差（増加・減少）を判定

   (4) 二項ロジスティック回帰
       - 検出限界未満の割合に対して、年次傾向をモデル化
       - AICによるモデル比較とオッズ比の算出

   (5) 回帰直線のプロット
       - 実測データ、検出限界、MLE回帰直線、分割年を可視化
       - PNG画像として保存（plots_regression ディレクトリ）

3. 結果の格納と整合性チェック
   - MLEとブートストラップの傾向が一致しているかを判定
   - 各物質ごとの解析結果を辞書形式で `results` に保存

前提条件:
    - 入力ファイルは 'data/' ディレクトリ内にあり、'substance', 'year', 'site_id', 'concentration', 'detection_limit' の列を含む
    - 必要なライブラリ（pandas, numpy, scipy, statsmodels, emcee, matplotlib など）がインストールされていること

備考:
    - ベイズ推定は自動収束判定付きの高速MCMCを使用
    - 結果は標準出力に表示され、プロットは画像として保存されます


"""
if __name__ == "__main__":
    filenames = ["13HCBD_gemini"]
    #filenames = ["大気温暖期","大気寒冷期","水質","底質","生物"]
    #filenames = ["生物"]
    for filename in filenames:
        print(f"\n\n==================== ファイル処理開始: {filename} ====================")
        filepath = f"data/{filename}.csv"
        if not os.path.exists(filepath):
            print(f"ファイルが見つかりません: {filepath}")
            continue

        df_all = pd.read_csv(filepath, usecols=['substance','year', 'site_id', 'concentration', 'detection_limit'])
        df_all['concentration'] = df_all['concentration'].apply(clean_value_user)
        df_all['detection_limit'] = df_all['detection_limit'].apply(clean_value_user)
        is_nondetect = df_all['concentration'] <= df_all['detection_limit']
        substitute_value = df_all['detection_limit'] / np.sqrt(2)
        df_all['concentration_sub'] = df_all['concentration'].copy()
        df_all.loc[is_nondetect, 'concentration_sub'] = substitute_value[is_nondetect]
        df_all['log_conc'] = np.log10(df_all['concentration_sub'])
        df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_conc', 'detection_limit', 'concentration'])
        output_path = f"data/{filename}_conv.csv"
        df_all.to_csv(output_path, index=False, encoding="utf-8")
        print(f"→ 変換後のデータを保存しました: {output_path}")

        substances = df_all['substance'].unique()
        print(f"対象物質: {substances}")

        results = []
        PLOT_DIR = "plots_regression"
        os.makedirs(PLOT_DIR, exist_ok=True)
        print(f"プロット画像は '{PLOT_DIR}' ディレクトリに保存されます。")

        for substance in substances:
            print(f"\n==================== {substance} ====================")
            df = df_all[df_all['substance'] == substance].copy()
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df = df.dropna(subset=['year'])
            if len(df) < 24:
                print("データ不足のためスキップ（地点数 < 24）")
                continue

            # 初期値を明確にしておく（例外発生時に上書きされないように） linear_validity
            #a, b, aic0, aic1, p1, x_base,  linear_validity = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            a, b, aic0, aic1, p1, x_base = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ci_bs = np.array([np.nan, np.nan])
            a_bin, p1_bin = np.nan, np.nan

            # ベイズ結果用の一時変数（ここで必ず初期化）
            a_bayes = np.nan
            b_bayes = np.nan
            ci_a = np.array([np.nan, np.nan])
            ci_b = np.array([np.nan, np.nan])
            p_neg_slope = np.nan
            bayes_trend = "未実行"
            """
            # ① MLE + AIC  not iterative　
            try:
                #a, b, aic0, aic1, p1, x_base, linear_validity = run_mle_and_aic(df)
                a, b, aic0, aic1, p1, x_base = run_mle_and_aic_non_iterative(df)
                print("--- 1. 対数線形回帰モデル (MLE) non iterative ---")
                print(f"傾き a = {a:.4f}, 切片 b = {b:.4f}, AIC事後確率 p1 = {p1:.4f}")
                print(f"aic1: {aic1} - aic0: {aic0}")
                trend = "p1 >= 0.95 傾向あり" if p1 >= 0.95 else "p1 < 0.95 傾向なし"
                print(f"→ {trend}")
            except Exception as e:
                print(f"最尤推定エラー: {e}")
                trend = "エラー"
                continue
            """
            # ① MLE + AIC
            try:
                #a, b, aic0, aic1, p1, x_base, linear_validity = run_mle_and_aic(df)
                a, b, aic0, aic1, p1, x_base = run_mle_and_aic(df)
                print("--- 1. 対数線形回帰モデル (MLE) ---")
                print(f"傾き a = {a:.4f}, 切片 b = {b:.4f}, AIC事後確率 p1 = {p1:.4f}")
                print(f"aic1: {aic1} - aic0: {aic0}")
                trend = "p1 >= 0.95 傾向あり" if p1 >= 0.95 else "p1 < 0.95 傾向なし"
                print(f"→ {trend}")
            except Exception as e:
                print(f"最尤推定エラー: {e}")
                trend = "エラー"
                continue

            # OLS など（プロット用）
            x_centered = df['year'].values - x_base
            y_true = df['log_conc'].values
            y_pred = a * x_centered + b
            r2 = r2_score(y_true, y_pred)
            X = sm.add_constant(x_centered)
            model = sm.OLS(y_true, X).fit()
            x_fit_centered = np.linspace(x_centered.min(), x_centered.max(), 100)
            X_fit = sm.add_constant(x_fit_centered)
            y_fit = model.predict(X_fit)
            predictions = model.get_prediction(X_fit)
            ci_ols = predictions.conf_int(alpha=0.05)

            # ② FFT-KDE ベイズ MCMC（自動早期打ち切り）
            print("\n--- 2. FFT-KDE ベイズ事後（高速 MCMC、自動早期停止） ---")
            try:
                residuals_mle = y_true - (a * x_centered + b)
                lookup = build_fft_kde_lookup(residuals_mle, grid_n=4096, bw_factor=1.0)

                bayes_res = run_bayes_kde_auto_stop_fast(
                    df,
                    fixed_lookup=lookup,
                    grid_n=4096,
                    bw_factor=1.0,
                    n_walkers=16,
                    n_steps=2000,
                    chunk=100,
                    burn_frac=0.5,
                    verbose=True,
                    ci_width_threshold=0.03,
                    stable_rounds=1
                )

                # 結果を安全に一度に代入（途中例外で上書きされるのを避ける）
                a_bayes = bayes_res.get("a_mean", a_bayes)
                b_bayes = bayes_res.get("b_mean", b_bayes)
                ci_a = np.array(bayes_res.get("a_ci", ci_a))
                ci_b = np.array(bayes_res.get("b_ci", ci_b))
                p_neg_slope = bayes_res.get("p_neg_slope", p_neg_slope)
                bayes_trend = f"a: {a_bayes} < 0 and P: {p_neg_slope} >= 0.95 減少傾向あり" if (not np.isnan(a_bayes) and a_bayes < 0 and (not np.isnan(p_neg_slope) and p_neg_slope >= 0.95)) else "減少傾向なし"
                #p_neg_slope = bayes_res["p_neg_slope"]
                p_pos_slope = 1.0 - p_neg_slope
                #==========================================================
                res_lin = compute_log_marginal_TL(df, model='linear', verbose=True)
                res_const = compute_log_marginal_TL(df, model='constant', verbose=True)

                cmp = compare_models_logZ(res_lin, res_const)

                print("logZ(linear): ", cmp['logZ1'])
                print("logZ(const):  ", cmp['logZ0'])
                print("対数ベイズ因子logBF:", cmp['logBF'])
                print("BF:", cmp['BF'])
                print("モデル事後確率 p(M1|y):", cmp['p_M1'])


                #===========================================================
                print(f"ベイズ推定 a = {a_bayes:.4f} (95% CI {ci_a})")
                print(f"ベイズ推定 b = {b_bayes:.4f} (95% CI {ci_b})")
                print(f"減少傾向の事後確率 P(a < 0) = {p_neg_slope:.4f}")
                print(f"→ {bayes_trend} (ベイズ)")

            except Exception as e:
                # ここでは a_bayes 等を上書きしない（既に成功している場合に消さないため）
                print("ベイズ推定失敗:", e)
                bayes_trend = "エラー"

            # ③ ブートストラップ
            try:
                split_year = determine_split_year(df)
                if split_year is None:
                    ci_bs = np.array([np.nan, np.nan])
                    bs_trend = "有意差なし"
                else:
                    ci_bs = compare_periods(df, split_year=split_year)
                    print("\n--- 3. ブートストラップ検定 ---")
                    print(f"分割年: {split_year}")
                    print(f"平均値差の95%信頼区間: {ci_bs}")
                    if ci_bs[0] > 0 and not np.isnan(ci_bs[0]):
                        bs_trend = "後期は有意に低濃度"
                    elif ci_bs[1] < 0 and not np.isnan(ci_bs[1]):
                        bs_trend = "後期は有意に高濃度"
                    else:
                        bs_trend = "有意差なし"
                    print(f"→ {bs_trend}")
            except Exception as e:
                print(f"ブートストラップエラー: {e}")
                bs_trend = "エラー"
                ci_bs = np.array([np.nan, np.nan])

            # ④ 二項回帰
            try:
                print("\n--- 4. 二項ロジスティック回帰モデル ---")
                df_binomial = df[['year', 'concentration', 'detection_limit']].copy()
                aic0_bin, aic1_bin, p1_bin, a_bin = binomial_trend_analyze(df_binomial)
                print(f"傾き a = {a_bin:.4f}, オッズ比 ≈ {math.exp(a_bin):.4f}, AIC事後確率 p1 = {p1_bin:.3f}")
                trend_bin = "傾向あり" if p1_bin >= 0.95 else "傾向なし"
                print(f"→ {trend_bin}")
            except Exception as e:
                print(f"二項回帰エラー: {e}")
                trend_bin = "エラー"
                a_bin, p1_bin = np.nan, np.nan

            # ⑤ プロット
            try:
                print("\n--- 5. 回帰直線のプロット ---")
                x_data = df['year'].values
                y_data = df['log_conc'].values
                dl_log = np.log10(df['detection_limit'])
                x_fit_plot = np.array([x_data.min(), x_data.max()])
                x_fit_centered = x_fit_plot - x_base
                y_fit = a * x_fit_centered + b
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hlines(dl_log, xmin=x_data - 0.2, xmax=x_data + 0.2,
                          color='tab:orange', alpha=0.5, linewidth=1.5, label='検出限界 (log10)')
                ax.scatter(x_data, y_data, label='実測データ (代替値込)', color='tab:blue', alpha=0.6)
                ax.plot(x_fit_plot, y_fit, label=f'MLE回帰直線: y = {a:.4f}(x - {x_base}) + {b:.4f}', color='tab:red')
                if 'split_year' in locals() and split_year is not None:
                    ax.axvline(x=split_year, color='gray', linestyle='--', linewidth=1.5, label=f'分割年: {split_year}')
                ax.set_title(f'{substance} の対数線形回帰\nMLE p1 = {p1:.4f} / ベイズ P(a<0) = {p_neg_slope if not np.isnan(p_neg_slope) else 0:.4f} ({filename})')
                ax.set_xlabel('年')
                ax.set_ylabel('log10濃度')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                y_all = np.concatenate([y_data, y_fit, dl_log])
                ax.set_ylim(y_all.min() - 0.1, y_all.max() + 0.1)
                fig.tight_layout()
                filename_safe = f"{substance}_{filename}".replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_")
                plot_path = os.path.join(PLOT_DIR, f'{filename_safe}_regression.png')
                fig.savefig(plot_path)
                plt.close(fig)
                print(f"→ {plot_path} に保存しました。")
            except Exception as e:
                print(f"プロットエラー: {e}")

            # 結果格納
            reliability = is_data_acceptable(a, p1, ci_bs)
            print(f"減少傾向のモデル間整合性チェック (MLE + Boot): {reliability}")

            results.append({
                'substance': substance,
                'year':df['year'].unique().tolist(),
                'slope_MLE': a,
                'ci_a': ci_a,
                'ci_b': ci_b,
                'intercept_MLE': b,
                #'linear_validity': linear_validity,
                'p1_MLE': p1,
                'trend_MLE': trend,
                'bootstrap_CI': ci_bs,
                'bs_trend': bs_trend,
                'binomial_slope': a_bin,
                'binomial_p1': p1_bin,
                'binomial_trend': trend_bin,
                #base
                'bayes_slope': a_bayes,
                'bayes_intercept': b_bayes,
                'bayes_CI_a': ci_a.tolist() if isinstance(ci_a, np.ndarray) else ci_a,
                'bayes_CI_b': ci_a.tolist() if isinstance(ci_b, np.ndarray) else ci_b,
                'bayes_p_m1': cmp['p_M1'],
                'bayes_trend': bayes_trend,
                'bayes_p_neg': p_neg_slope,   # ← ★追加
                'bayes_p_pos': p_pos_slope,
                'reliability': reliability
            })

        # 整形・保存（安全に丸め）
        import numpy as np

        def format_ci(ci, digits=4):
            """信用区間を安全に丸めてリスト形式に統一"""
            try:
                arr = np.asarray(ci, dtype=float)
                if arr.size == 2 and not np.isnan(arr).any():
                    return [round(float(arr[0]), digits), round(float(arr[1]), digits)]
            except Exception:
                pass
            return [np.nan, np.nan]

        def format_float(val, digits=4):
            """数値を安全に丸める"""
            try:
                return round(float(val), digits)
            except Exception:
                return np.nan
            
        for r in results:
            # 信用区間の整形（すべて format_ci に統一）
            r['bootstrap_CI'] = format_ci(r.get('bootstrap_CI'))
            r['bayes_CI_a'] = format_ci(r.get('bayes_CI_a'))
            r['bayes_CI_b'] = format_ci(r.get('bayes_CI_b'))
            r['ci_a'] = format_ci(r.get('ci_a'))
            r['ci_b'] = format_ci(r.get('ci_b'))

            # 数値項目の整形
            for key in ['slope_MLE', 'intercept_MLE', 'binomial_slope', 'p1_MLE', 'binomial_p1',
                        'bayes_slope', 'bayes_intercept', 'bayes_p_m1', 'bayes_p_neg', 'bayes_p_pos']:
                r[key] = format_float(r.get(key))
        """
        後半処理：解析結果の整形・保存・HTMLレポート生成

        6. 結果の整形と丸め処理
        - 各物質の解析結果（辞書形式）を走査し、数値項目（傾き、切片、事後確率など）を小数点4桁に丸める
        - 信頼区間（CI）も安全に処理し、NaNや例外に対応
        - 整形済みの結果を pandas の DataFrame に変換し、CSVファイルとして保存

        7. HTMLレポートの生成
        - 各物質ごとの解析結果を HTML 形式で整形
        - 回帰プロット画像（PNG）を埋め込み、視覚的に傾向を確認可能
        - MLE、ベイズ推定、ブートストラップ、二項回帰の結果をセクションごとに表示
        - モデル選択確率や傾向判定を明示し、解釈しやすいレポートを構築

        出力ファイル:
        - CSV形式: `<filename>_2025_reliable.csv`
        - HTML形式: `<filename>_regression_bayes_pM1.html`
        - プロット画像: `plots_regression/` ディレクトリ内に保存

        備考:
        - HTMLレポートはローカルで閲覧可能な静的ファイルとして生成されます
        - グラフ画像が存在しない場合はその旨を表示
        - 結果の信頼性（reliability）は MLEとブートストラップの整合性に基づいて判定されます
        
        for r in results:
            # bootstrap_CI
            try:
                if isinstance(r.get('bootstrap_CI'), (list, tuple, np.ndarray)):
                    r['bootstrap_CI'] = [round(float(r['bootstrap_CI'][0]), 4), round(float(r['bootstrap_CI'][1]), 4)]
            except Exception:
                r['bootstrap_CI'] = [np.nan, np.nan]

            # bayes_CI_a: 安全に丸める
            try:
                ci_val = r.get('bayes_CI_a', [np.nan, np.nan])
                ci_arr = np.asarray(ci_val, dtype=float)
                if ci_arr.size == 2 and not (np.isnan(ci_arr[0]) or np.isnan(ci_arr[1])):
                    r['bayes_CI_a'] = [round(float(ci_arr[0]), 4), round(float(ci_arr[1]), 4)]
                else:
                    r['bayes_CI_a'] = [np.nan, np.nan]
            except Exception:
                r['bayes_CI_a'] = [np.nan, np.nan]

            # numeric rounding for other fields
            try:
                r['slope_MLE'] = round(float(r.get('slope_MLE', np.nan)), 4)
            except Exception:
                r['slope_MLE'] = np.nan
            try:
                r['intercept_MLE'] = round(float(r.get('intercept_MLE', np.nan)), 4)
            except Exception:
                r['intercept_MLE'] = np.nan
            try:
                r['binomial_slope'] = round(float(r.get('binomial_slope', np.nan)), 4)
            except Exception:
                r['binomial_slope'] = np.nan
            try:
                r['p1_MLE'] = round(float(r.get('p1_MLE', np.nan)), 4)
            except Exception:
                r['p1_MLE'] = np.nan
            try:
                r['binomial_p1'] = round(float(r.get('binomial_p1', np.nan)), 4)
            except Exception:
                r['binomial_p1'] = np.nan
        """
        df_results = pd.DataFrame(results)
        print("\n--- 最終結果（テーブル形式） ---")
        print(df_results.to_string(index=False))
        df_results.to_csv(f"{filename}_2025_reliable.csv", index=False, encoding='utf-8-sig')

        # HTMLレポート作成
        html_path = f"{filename}_regression_bayes_pM1.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<html><head><meta charset='utf-8'><title>{filename} 回帰分析レポート</title></head><body>")
            f.write("<h1>回帰分析レポート</h1>")

            for r in results:
                if max(r['year']) > 0:
                    filename_safe = f"{r['substance']}_{filename}".replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_")
                    image_path = os.path.join(PLOT_DIR, f"{filename_safe}_regression.png")


                    f.write("<hr>")
                    # ---- Graph ----
                    if os.path.exists(image_path):
                        f.write(f"<img src='{image_path}' width='600'><br>")
                    else:
                        f.write("<p><i>グラフ画像が見つかりませんでした。</i></p>")                    
                    f.write(f"<h2>{r['substance']}</h2>")
                    f.write(f"<p><b>対象:</b> {filename}<br>")
                    f.write(f"<b>年データ:</b> {r['year']}<br><br>")

                    # ---- MLE ----
                    f.write("<h3>最尤推定（MLE / AIC）</h3>")
                    f.write(f"<b>傾き a (MLE):</b> {r['slope_MLE']}  ")
                    f.write(f"<b>ci_a (MLE):</b> {r['ci_a']}<br>")                   
                    f.write(f"<b>切片 b (MLE):</b> {r['intercept_MLE']}  ")
                    f.write(f"<b>ci_b (MLE):</b> {r['ci_b']}<br>")  
                    #f.write(f"<b>AIC事後確率 p1:</b> {r['p1_MLE']}<br>")
                    f.write(f"<b>AICによるモデル選択確率 p1:</b> {r['p1_MLE']}<br>")
                    #f.write(f"<b>線形妥当性 (ΔAIC)) P:</b> {r['aic1']}-{r['aic0']}<br>")
                    f.write(f"<b>傾向判定 (MLE):</b> {r['trend_MLE']}<br><br>")

                    # ---- Bayes (FFT-KDE MCMC) ----
                    f.write("<h3>ベイズ推定（MCMC / FFT-KDE尤度）</h3>")
                    f.write(f"<b>傾き a (ベイズ平均):</b> {r['bayes_slope']}<br>")
                    f.write(f"<b>傾き95%信用区間:</b> {r['bayes_CI_a']}<br>")
                    f.write(f"<b>切片 b (ベイズ平均):</b> {r['bayes_intercept']}<br>")
                    f.write(f"<b>切片95%信用区間:</b> {r['bayes_CI_b']}<br>")
                    f.write(f"<b>減少傾向の事後確率 P(a &lt; 0):</b> {r.get('bayes_p_neg', '---')}<br>")
                    f.write(f"<b>増加傾向の事後確率 P(a &gt; 0):</b> {r['bayes_p_pos']}<br>")
                    f.write(f"<b>傾向判定 (ベイズMCMC):</b> {r['bayes_trend']}<br>")
                    f.write(f"<b>モデル比較の事後確率 pM1:</b> {r['bayes_p_m1']}<br><b>モデル妥当性判定 (ベイズ):</b> {'1次式が妥当' if r['bayes_p_m1'] >= 0.95 else '1次式とする強いエビデンスなし'}<br>")
                    # p_m1 を使用して、妥当性判定のメッセージを修正
                    #f.write(f"<b>モデル比較の事後確率 pM1:</b> {r['bayes_p_m1']}<br>")
                    # ---- Bootstrap ----
                    f.write("<h3>ブートストラップ検定</h3>")
                    f.write(f"<b>平均差 95% CI:</b> {r['bootstrap_CI']}<br>")
                    f.write(f"<b>傾向判定 (Boot):</b> {r['bs_trend']}<br><br>")

                    # ---- Binomial ----
                    f.write("<h3>二項ロジスティック回帰</h3>")
                    f.write(f"<b>傾き:</b> {r['binomial_slope']}<br>")
                    f.write(f"<b>AIC事後確率 p1:</b> {r['binomial_p1']}<br>")
                    f.write(f"<b>傾向判定:</b> {r['binomial_trend']}<br><br>")


            f.write("</body></html>")

        print(f"\n→ HTMLレポートを作成しました: {html_path}")
