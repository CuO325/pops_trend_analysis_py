# pops_fft_kde_analysis_clean.py
# FFT-KDE + MLE + MCMC + TI（熱力学積分）を統合したクリーン版
version = "script: pops_fft_analysis_clean.py, author:H.S."

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import json
import emcee
import time
import os
from sklearn.utils import resample
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.integrate import trapezoid
from sklearn.metrics import r2_score

from analysis.fft_kde_common_engine import prepare_fft_kde_lookup, fast_kde_log_likelihood

# JSON encoder
def log_posterior(params, x, y, lookup):
    lp = log_prior_standard(params)
    if not np.isfinite(lp): return -np.inf
    return lp + fast_kde_log_likelihood(params, x, y, lookup)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def binomial_trend_analyze(df, flag_col=None):
    """
    二項ロジスティック回帰分析を行う関数
    - flag_col が指定されていればその列(0/1)を使用
    - 指定がなければ detection_limit と concentration から判定
    """
    df_work = df.copy()
    
    # 1. 0/1 フラグ列の確定
    if flag_col is not None and flag_col in df_work.columns:
        df_work['low_flag'] = df_work[flag_col].astype(int)
    elif 'low_flag' in df_work.columns:
        pass # すでに low_flag があればそのまま使用
    elif 'binary' in df_work.columns:
        df_work['low_flag'] = df_work['binary'].astype(int)
    elif 'detection_limit' in df_work.columns and 'concentration' in df_work.columns:
        unified_dl = df_work['detection_limit'].max()
        df_work['low_flag'] = (df_work['concentration'] < unified_dl).astype(int)
    else:
        raise KeyError("二項化に必要な列 ('low_flag', 'binary', または 'detection_limit'/'concentration') が見つかりません。")

    # 2. 年ごとに「フラグ(1)の数 (low_count)」と「フラグ(0)の数 (not_low_count)」を集計
    yearly_data = df_work.groupby('year').agg(
        low_count=('low_flag', 'sum'),
        total_count=('low_flag', 'count')
    ).reset_index()
    
    yearly_data['not_low_count'] = yearly_data['total_count'] - yearly_data['low_count']
    
    x_year = yearly_data['year'].astype(float).to_numpy()
    X1 = sm.add_constant(x_year).astype(float)
    
    # 応答変数を [成功数(1), 失敗数(0)] の2列行列として作成
    y_binomial = yearly_data[['low_count', 'not_low_count']].to_numpy().astype(float)
    
    # 有効データのフィルタリング
    valid_mask = (yearly_data['total_count'] > 0) & (~np.isnan(X1).any(axis=1))
    X1 = X1[valid_mask]
    y_binomial = y_binomial[valid_mask]
    
    if len(y_binomial) == 0:
        return np.nan, np.nan, np.nan, np.nan

    X0 = np.ones((len(y_binomial), 1), dtype=float)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=PerfectSeparationWarning)
        
        model1 = sm.GLM(y_binomial, X1, family=sm.families.Binomial())
        model1_fit = model1.fit()
        aic1 = model1_fit.aic
        
        model0 = sm.GLM(y_binomial, X0, family=sm.families.Binomial())
        model0_fit = model0.fit()
        aic0 = model0_fit.aic
        
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    p1 = numerator / (1 + numerator)
    slope = model1_fit.params[1] if len(model1_fit.params) > 1 else np.nan
    
    return aic0, aic1, p1, slope


# ---------------------------
# ブートストラップ用関数
# ---------------------------

def determine_split_year(df):

    years = df['year'].dropna().unique()
    if len(years) < 4:
        return None
    return int(np.median(years))

def compare_periods(df, split_year=2015, R=10000):

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

# ---------------------------
# MLE MCMC 使用関数
# ---------------------------

def neg_log_likelihood(params, x, y):

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

def run_mle_and_aic(df, max_iter=20, tolerance=1e-6):
    
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
    

    #-------------------------------------
    # OLS (正規性仮定の線形解析)
    X = sm.add_constant(x_centered)
    ols_model = sm.OLS(y, X).fit()

    stderr_b, stderr_a = ols_model.bse   # 標準誤差
    pvals = ols_model.pvalues.tolist()  # p値
    r2 = ols_model.rsquared             # 決定係数

    # 95%信頼区間
    ci = ols_model.conf_int(alpha=0.05)  # デフォルトは95%
    #ols_ci_b, ols_ci_a = ci[0], ci[1]  # 順番に注意 (切片, 傾き)
    ols_ci_b = ci[0]
    ols_ci_a = ci[1]  # 順番に注意 (切片, 傾き)
    print(f'傾き a (ols): {a1:.4f}')
    print(f'切片 b (ols): {b1:.4f}')
    print(f"aの95%CI (ols): {ols_ci_a[0]:.4f} ～ {ols_ci_a[1]:.4f}")
    print(f"bの95%CI (ols): {ols_ci_b[0]:.4f} ～ {ols_ci_b[1]:.4f}")

    contains_zero_mle = (ols_ci_a[0] <= 0 <= ols_ci_a[1])

    if p1 < 0.95:
        trend_mle = "MLE: 1次式モデルは選択されない。傾向は不明"
    elif p1 >= 0.95 and a1 < 0:
        trend_mle = "MLE: 1次式モデルが選択される。減少傾向あり"
    elif p1 >= 0.95 and a1 > 0:
        trend_mle = "MLE: 1次式モデルが選択される。増加傾向あり"

    # OLSの補足を追記
    if contains_zero_mle:
        trend_mle += "　OLSによる補足: 傾きの信頼区間に0を含む"

    print(trend_mle)
    #return a1, b1, aic0, aic1, p1, x_base
    # 辞書形式で返す
    return {
        'a1': a1,
        'b1': b1,
        'aic0': aic0,
        'aic1': aic1,
        'loglik0': max_log_likelihood_0,
        'loglik1': max_log_likelihood_1,
        'p1': p1,
        'x_base': x_base,
        'n_data': n_data,

        # OLS 解析結果を別項目として追加
        'ols_stderr_a': stderr_a,
        'ols_stderr_b': stderr_b,
        'ols_pvalues': pvals,
        'ols_r2': r2,
        'ols_ci_a': ols_ci_a,
        'ols_ci_b': ols_ci_b,
        'trend_mle':trend_mle

    }

# ---------------------------
# FFT-KDE 尤度
# ---------------------------
def fast_kde_log_likelihood_from_residuals(a, b, x, y, lookup):
    return fast_kde_log_likelihood([a, b], x, y, lookup)

# ---------------------------
# 事前分布
# ---------------------------
def log_prior_informative(params, data_scale):
    a, b = params
    if not (-100 < a < 100 and -100 < b < 100):
        return -np.inf
    x_range = data_scale['x_range']
    y_std = data_scale['y_std']
    a_sd = max(0.1, y_std / max(1e-6, x_range) * 2.0)
    b_sd = max(0.1, y_std * 2.0)
    return - (a**2)/(2*a_sd**2) - (b**2)/(2*b_sd**2)

# ---------------------------
# MCMC（FFT-KDE）
# ---------------------------
def log_posterior_fast(params, x, y, lookup, data_scale):
    lp = log_prior_informative(params, data_scale)
    if not np.isfinite(lp): return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x, y, lookup)
    return lp + ll


# ---------------------------
# 熱力学積分（TI）
# ---------------------------
def estimate_E_loglike_for_beta(beta, x, y, lookup, init_center, data_scale):
    ndim = 2
    rng = np.random.default_rng()
    initial_pos = init_center + 1e-3 * rng.normal(size=(20, ndim))

    def _logprob(theta):
        lp = log_prior_informative(theta, data_scale)
        if not np.isfinite(lp): return -np.inf
        ll = fast_kde_log_likelihood_from_residuals(theta[0], theta[1], x, y, lookup)
        return lp + beta * ll

    sampler = emcee.EnsembleSampler(20, ndim, _logprob)
    sampler.run_mcmc(initial_pos, 2000)
    chain = sampler.get_chain(discard=800, flat=True)

    ll_vals = np.array([
        fast_kde_log_likelihood_from_residuals(th[0], th[1], x, y, lookup)
        for th in chain
    ])
    return np.mean(ll_vals)

def compute_log_marginal_TL(df, model='linear', bw_factor=1.0, verbose=True):
    years = df['year'].values.astype(float)
    x0 = years.min()
    x = years - x0
    y = df['log_conc'].values.astype(float)

    # MLE
    if model == 'linear':
        res = minimize(neg_log_likelihood_linear, [-0.01, np.mean(y)], args=(x, y))
        a_mle, b_mle = res.x
        residuals = y - (a_mle * x + b_mle)
    else:
        res = minimize(neg_log_likelihood_const, [np.mean(y)], args=(x, y))
        b0 = res.x[0]
        a_mle, b_mle = 0.0, b0
        residuals = y - b0

    lookup = prepare_fft_kde_lookup(residuals, bw_factor=bw_factor)

    betas = np.concatenate([
        np.linspace(0, 0.9, 30, endpoint=False),
        np.linspace(0.9, 1.0, 21)
    ])

    data_scale = {'x_range': x.max() - x.min(), 'y_std': np.nanstd(y)}
    init_center = np.array([a_mle, b_mle])

    E_logL = [estimate_E_loglike_for_beta(beta, x, y, lookup, init_center, data_scale)
              for beta in betas]

    logZ = trapezoid(E_logL, betas)

    return {
        'model': model,
        'logZ': logZ,
        'betas': betas,
        'E_logL': np.array(E_logL),
        'a_mle': a_mle,
        'b_mle': b_mle
    }

# ---------------------------
# モデル比較（Bayes 因子）
# ---------------------------
def compare_models_logZ(res1, res0, prior_ratio=1.0):
    l1 = max(res1['logZ'], -1e10)
    l0 = max(res0['logZ'], -1e10)
    logBF = l1 - l0 + np.log(prior_ratio)

    if logBF > 700:
        return {'logBF': logBF, 'BF': np.inf, 'p_M1': 1.0}
    if logBF < -700:
        return {'logBF': logBF, 'BF': 0.0, 'p_M1': 0.0}

    BF = np.exp(logBF)
    p_M1 = 1 / (1 + np.exp(-logBF))
    return {'logBF': logBF, 'BF': BF, 'p_M1': p_M1}
# ---------------------------
# 高速 FFT-KDE ユーティリティ
#ルックアップ作成の統一: build_fft_kde_lookup の定義を削除し、共通モジュールの prepare_fft_kde_lookup に置き換えます。

# --- 置き換え用：fast_kde_log_likelihood_from_residuals（改良版） ---
#def fast_kde_log_likelihood_from_residuals_loo_corrected(a, b, x_centered, y, lookup):
# --- 修正前: build_fft_kde_lookup (削除) ---
# --- 修正前: fast_kde_log_likelihood_from_residuals (整理) ---
#def fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup, use_loo_correction=True):
    #"""共通エンジンの尤度計算関数を呼び出す"""
    #return fast_kde_log_likelihood([a, b], x_centered, y, lookup, use_loo_correction=use_loo_correction)
def fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup, use_loo=True):

    # fast_kde_log_likelihood は [params, x, y, lookup_func] の4引数のみを受け取ります
    return fast_kde_log_likelihood([a, b], x_centered, y, lookup)

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

def kde_log_likelihood_from_residuals_plain(a, b, x_centered, y):

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
        #lookup = build_fft_kde_lookup(residuals_mle, grid_n=grid_n, bandwidth=None, bw_factor=bw_factor)
        lookup = prepare_fft_kde_lookup(residuals_mle, bw_factor=bw_factor, use_loo=False)
        print("L686 DEBUG-----lookup@ run_bayes_kde_auto_stop_fast", type(lookup))
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
#---------------general use----------------------------
def clean_value_user(val):

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

    if a >= 0:
        return False
    if p1 < 0.95:
        return False
    if ci[0] < 0 and ci[1] > 0:
        return False
    return True

def bayes_trend_judgement(contains_zero, cmp, p_neg_slope, p_pos_slope):
    trend=""
    if cmp['p_M1'] >= 0.95:
        trend = "1次式モデルが選択される。"
    else:
        trend = "1次式モデルは選択されない。"
    # 減少傾向の判定
    if p_neg_slope >= 0.95:
        trend +=   " 減少傾向あり。"

    elif p_pos_slope >= 0.95:
        trend +=   " 増加傾向あり。"
        # どちらでもない場合
    else:
        trend +=   " 傾向は不明。"
    # 1. CIに0を含む
    if contains_zero:
        trend +=  " 傾きの信頼区間に 0 を含む。"   

    return trend
def interpret_bayes_factor(logBF=None, BF=None):
    """
    ベイズ因子 (Bayes Factor) の判定を返す関数    Jeffreys (1961) の基準に基づく
    Parameters
        logBF : float, optional        対数ベイズ因子 logBF
        BF : float, optional        ベイズ因子 BF (exp(logBF))
    Returns
      dict        {           "BF": BF値,            "logBF": logBF値,            "interpretation": 判定文字列        }
    """
    # BF が与えられていない場合は logBF から計算
    if BF is None and logBF is not None:
        BF = math.exp(logBF)
    elif BF is None and logBF is None:
        raise ValueError("logBF または BF のいずれかを指定してください")   
    if logBF is None and BF is not None:
        logBF = math.log(BF)
        # 判定基準 (Jeffreys scale)
    if BF < 1:
        interpretation = "BF < 1　定数モデル支持"
    elif 1 <= BF < 3:
        interpretation = " 1 <= BF < 3　線形モデル支持の弱い証拠"
    elif 3 <= BF < 10:
        interpretation = "3 <= BF < 10　線形モデル支持の中程度の証拠"
    elif 10 <= BF < 30:
        interpretation = "10 <= BF < 30　線形モデル支持の強い証拠"
    elif 30 <= BF < 100:
        interpretation = "30 <= BF < 100　線形モデル支持の非常に強い証拠"
    else:
        interpretation = "100 <= BF　線形モデル支持の決定的な証拠"
    
    return {
        "BF": BF,
        "logBF": logBF,
        "interpretation": interpretation
    }
#---------------------------------------------------------------------
if __name__ == "__main__":
    #filenames = ["PFOS大気温暖期"]
    #filenames = ["大気温暖期","大気寒冷期","水質","底質","生物"]
    #filenames = ["水質","底質","生物"]
    #filenames = ["13HCBD"] 
    filenames = ["HCB大気温暖期2013"]

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
        
        reports = [] 
        results = []
        PLOT_DIR = "outputs/plots_fft_kde"
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


            # ① MLE + AIC
            try:
                #a, b, aic0, aic1, p1, x_base, linear_validity = run_mle_and_aic(df)

                print("--- 1. 対数線形回帰モデル (MLE) ---")                
                res = run_mle_and_aic(df)
                a = res['a1']
                b = res['b1']
                aic0 = res['aic0']
                aic1 = res['aic1']
                loglik0 = res['loglik0']
                loglik1 = res['loglik1']
                p1 = res['p1']
                x_base = res['x_base']
                n_data = res['n_data']
                trend_mle = res['trend_mle']
                a = float(a)
                b = float(b)
                aic0 = float(aic0)
                aic1 = float(aic1)
                p1 = float(p1)


                print(f"傾き a = {res['a1']:.4f}, 切片 b = {res['b1']:.4f}, AIC事後確率 p1 = {res['p1']:.4f}")
                print(f"aic1: {res['aic1']:.4f} - aic0: {res['aic0']:.4f}")
                trend = "p1 >= 0.95 傾向あり" if res['p1'] >= 0.95 else "p1 < 0.95 傾向なし"
                print(f"→ {trend}")
            except Exception as e:
                print(f"最尤推定エラー: {e}")
                trend = "エラー"
                continue

            # OLS など（プロット用）
            x_centered = df['year'].values - x_base
            y_true = df['log_conc'].values
            a = res['a1']
            b = res['b1']
            y_true = df['log_conc'].values
            y_pred = a * (df['year'].values - res['x_base']) + b

            r2 = r2_score(y_true, y_pred)
            print(f"R² = {r2:.4f}")

            #y_pred = a * x_centered + b
            #r2 = r2_score(y_true, y_pred)
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
                lookup = prepare_fft_kde_lookup(residuals_mle, bw_factor=1.0, use_loo=False)
                #bw_factor:シルバーマンの法則などで計算された標準的な帯域幅（Bandwidth）を「何倍にするか」を指定する係数
                print("L965 DEBUG ----lookup@ FFT-KDE ベイズ事後（高速 MCMC、自動早期停止",type(lookup))
                #lookup = build_fft_kde_lookup(residuals_mle, grid_n=4096, bw_factor=1.0)

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

                # JSONに入れるときは list に変換
                bayes_CI_a = ci_a.tolist() if isinstance(ci_a, np.ndarray) else ci_a
                # 0を含むかどうか判定
                contains_zero = (bayes_CI_a[0] <= 0 <= bayes_CI_a[1])
                #p_neg_slope = bayes_res["p_neg_slope"]
                p_pos_slope = 1.0 - p_neg_slope
                #==========================================================
                res_lin = compute_log_marginal_TL(df, model='linear', verbose=True)
                res_const = compute_log_marginal_TL(df, model='constant', verbose=True)

                cmp = compare_models_logZ(res_lin, res_const)
                print("モデル事後確率 p(M1|y):", cmp['p_M1'])                
                #bayes_trend = f"a: {a_bayes} < 0 and P: {p_neg_slope} >= 0.95 減少傾向あり" if (not np.isnan(a_bayes) and a_bayes < 0 and (not np.isnan(p_neg_slope) and p_neg_slope >= 0.95)) else "減少傾向なし"
                bayes_trend  = bayes_trend_judgement(contains_zero, cmp, p_neg_slope, p_pos_slope)              
                bf_interpretation = interpret_bayes_factor(BF=cmp['BF'])
                print("logZ(linear): ", cmp['logZ1'])
                print("logZ(const):  ", cmp['logZ0'])
                print("対数ベイズ因子logBF:", cmp['logBF'])
                print("BF:", cmp['BF'])
                print("BF評価:", bf_interpretation['interpretation'])
                            
                #===========================================================
                 
                print(f"ベイズ推定 a = {a_bayes:.4f} (95% CI {ci_a})")
                print(f"ベイズ推定 b = {b_bayes:.4f} (95% CI {ci_b})")
                print(f"減少傾向の事後確率 P(a < 0) = {p_neg_slope:.4f}")
                print(f"→ {bayes_trend}")

            except Exception as e:
                # ここでは a_bayes 等を上書きしない（既に成功している場合に消さないため）
                print("ベイズ推定失敗:", e)
                bayes_trend = "エラー"

            # ③ ブートストラップ
            try:
                split_year = determine_split_year(df)
                if split_year is None:
                    ci_bs = np.array([np.nan, np.nan])
                    bs_trend = "BSでは評価できない"
                else:
                    ci_bs = compare_periods(df, split_year=split_year)
                    print("\n--- 3. ブートストラップ検定 ---")
                    print(f"分割年: {split_year}")
                    print(f"平均値差の95%信頼区間: {ci_bs}")
                    if ci_bs[0] > 0 and not np.isnan(ci_bs[0]):
                        bs_trend = f"分割年: {split_year}　後期は有意に低濃度"
                    elif ci_bs[1] < 0 and not np.isnan(ci_bs[1]):
                        bs_trend = f"分割年: {split_year}　後期は有意に高濃度"
                    else:
                        bs_trend = f"分割年: {split_year}　有意差なし"
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
                trend_bin = f"統一下限値 {max(df_binomial[ 'detection_limit'])} 以下の増加傾向あり" if p1_bin >= 0.95 else  f"統一下限値 {max(df_binomial[ 'detection_limit'])} 以下で傾向なし"
                print(f"→ {trend_bin}")
            except Exception as e:
                print(f"二項回帰エラー: {e}")
                trend_bin = "エラー"
                a_bin, p1_bin = np.nan, np.nan

             # ------------ 変数の事前初期化 ------------
            a_dl, p1_dl = np.nan, np.nan
            a_q25, p1_q25 = np.nan, np.nan
            a_q75, p1_q75 = np.nan, np.nan

            # 1. DL閾値での二項回帰実行
            if 'detection_limit' in df.columns and not df['detection_limit'].dropna().empty:
                try:
                    threshold_dl = df['detection_limit'].max()
                    df_dl = df[['year', 'concentration']].copy()
                    df_dl['binary'] = (df['concentration'] < threshold_dl).astype(int)

                    aic0_dl, aic1_dl, p1_dl, a_dl = binomial_trend_analyze(df_dl, flag_col='binary')
                    print(f"DL閾値 ({threshold_dl:.4f}): 傾き={a_dl:.4f}, p1={p1_dl:.3f}")
                except Exception as e:
                    print(f"DL閾値での二項回帰エラー: {e}")

            # 2. 25%タイルでの二項回帰実行
            try:
                threshold_q25 = df['concentration'].quantile(0.25)
                df_q25 = df[['year', 'concentration']].copy()
                df_q25['binary'] = (df['concentration'] <= threshold_q25).astype(int)

                aic0_q25, aic1_q25, p1_q25, a_q25 = binomial_trend_analyze(df_q25, flag_col='binary')
                print(f"25%タイル閾値 ({threshold_q25:.4f}): 傾き={a_q25:.4f}, p1={p1_q25:.3f}")
            except Exception as e:
                print(f"25%タイルでの二項回帰エラー: {e}")

            # 3. 75%タイルでの二項回帰実行
            try:
                threshold_q75 = df['concentration'].quantile(0.75)
                df_q75 = df[['year', 'concentration']].copy()
                df_q75['binary'] = (df['concentration'] >= threshold_q75).astype(int)

                aic0_q75, aic1_q75, p1_q75, a_q75 = binomial_trend_analyze(df_q75, flag_col='binary')
                print(f"75%タイル閾値 ({threshold_q75:.4f}): 傾き={a_q75:.4f}, p1={p1_q75:.3f}")
            except Exception as e:
                print(f"75%タイルでの二項回帰エラー: {e}")

            #PLOT
            try:
                print("\n--- 5. 回帰直線のプロット ---")
                x_data = df['year'].values
                y_data = df['log_conc'].values
                dl_log = np.log10(df['detection_limit'])

                # プロット用の x 範囲
                x_fit_plot = np.array([x_data.min(), x_data.max()])
                x_fit_centered = x_fit_plot - x_base

                # MLE 回帰線
                y_fit_mle = a * x_fit_centered + b

                # ベイズ回帰線
                a_bayes = bayes_res.get("a_mean", a_bayes)
                b_bayes = bayes_res.get("b_mean", b_bayes)
                y_fit_bayes = a_bayes * x_fit_centered + b_bayes

                fig, ax = plt.subplots(figsize=(8, 5))

                # 検出限界
                ax.hlines(dl_log, xmin=x_data - 0.2, xmax=x_data + 0.2,
                        color='tab:orange', alpha=0.5, linewidth=1.5, label='検出限界 (log10)')

                # 実測データ
                ax.scatter(x_data, y_data, label='実測データ (代替値込)', color='tab:blue', alpha=0.6)

                # MLE 回帰直線
                ax.plot(x_fit_plot, y_fit_mle,
                        label=f'MLE回帰直線: y = {a:.4f}(x - {x_base}) + {b:.4f}',
                        color='tab:red')

                # ベイズ回帰直線
                ax.plot(x_fit_plot, y_fit_bayes,
                        label=f'ベイズ回帰直線: y = {a_bayes:.4f}(x - {x_base}) + {b_bayes:.4f}',
                        color='tab:green', linestyle='--')

                # 分割年がある場合
                if 'split_year' in locals() and split_year is not None:
                    ax.axvline(x=split_year, color='gray', linestyle='--', linewidth=1.5, label=f'分割年: {split_year}')

                # タイトルなど
                ax.set_title(f'{substance} の対数線形回帰\nMLE p1 = {p1:.4f} / ベイズ P(a<0) = {p_neg_slope if not np.isnan(p_neg_slope) else 0:.4f} ({filename})')
                ax.set_xlabel('年')
                ax.set_ylabel('log10濃度')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)

                # y 範囲調整
                y_all = np.concatenate([y_data, y_fit_mle, y_fit_bayes, dl_log])
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
            analysis_report = {
                'substance': substance,
                'filename': filename,
                'year': df['year'].unique().tolist(),
                'split_year': split_year,
                'n_obs': len(df),
                'n_censored': (df['log_conc'] < np.log10(df['detection_limit'])).sum(),

                'methods': {
                    'MLE': {
                        'slope': res['a1'],
                        'intercept': res['b1'],
                        'aic0': res['aic0'],
                        'aic1': res['aic1'],
                        'loglik0': res['loglik0'],
                        'loglik1': res['loglik1'],
                        'p1': res['p1'],
                        'trend': trend_mle
                    },
                    'OLS': {
                        'slope': res['a1'],
                        'intercept': res['b1'],
                        'stderr_a': res['ols_stderr_a'],
                        'stderr_b': res['ols_stderr_b'],
                        'ci_a': res['ols_ci_a'],
                        'ci_b': res['ols_ci_b'],
                        'pvalues': res['ols_pvalues'],
                        'r2': res['ols_r2']
                    },
                    'Bayes': {
                        'slope': a_bayes,
                        'intercept': b_bayes,
                        'p_M1':cmp['p_M1'],
                        'bayes_CI_a': ci_a.tolist() if isinstance(ci_a, np.ndarray) else ci_a,
                        'bayes_CI_b': ci_b.tolist() if isinstance(ci_b, np.ndarray) else ci_b,
                        'p_neg': p_neg_slope,
                        'p_pos': p_pos_slope,
                        'logZ_linear': cmp['logZ1'],
                        'logZ_const':  cmp['logZ0'],
                        'logBF':cmp['logBF'],
                        'BF':cmp['BF'],
                        'bayes_trend': bayes_trend,
                        "BF評価": bf_interpretation['interpretation']
                            
                    },
                    'Bootstrap': {
                        'bootstrap_CI': ci_bs,
                        'bootstrap_summary': {'mean': np.mean(ci_bs), 'std': np.std(ci_bs)},
                        'bs_trend': bs_trend
                    },
                    'Binominal':{   
                        'binomial_slope': a_bin,
                        'binomial_p1': p1_bin,
                        'binomial_trend': trend_bin,

                        'DL閾値_slope': a_dl,
                        'DL閾値_p1': p1_dl,
  
                        '25%タイル閾値_slope': a_q25,
                        '25%タイル閾値_p1': p1_q25,  

                        '75%タイル閾値_slope':a_q75,
                        '75%タイル閾値_p1': p1_q75     
                    } 
                
                }
            }
            reports.append(analysis_report)


        # 保存
        with open(f"{filename}_analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2, cls=NpEncoder)
        # 整形・保存（安全に丸め）
        """ 

        def format_ci(ci, digits=4):
            #信用区間を安全に丸めてリスト形式に統一
            try:
                arr = np.asarray(ci, dtype=float)
                if arr.size == 2 and not np.isnan(arr).any():
                    return [round(float(arr[0]), digits), round(float(arr[1]), digits)]
            except Exception:
                pass
            return [np.nan, np.nan]

        def format_float(val, digits=4):
            #数値を安全に丸める
            try:
                return round(float(val), digits)
            except Exception:
                return np.nan
        """


    html_path = f"{filename}_regression_report.html"
    #author = f"{version}"  # レポートの著者名を指定
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>{filename} 回帰分析レポート</title></head><body>")
        f.write(f"<h1>{filename} 分析レポート </h1>")
        f.write(f" {version}   <br><br>")
        for analysis_report in reports:
            filename_safe = f"{analysis_report['substance']}_{filename}".replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_")
            image_path = os.path.join(PLOT_DIR, f"{filename_safe}_regression.png")


            f.write("<hr>")
            f.write(f"<h2>{analysis_report['substance']}</h2>")  
            # ---- Graph ----
            if os.path.exists(image_path):
                f.write(f"<img src='{image_path}' width='600'><br>")
            else:
                f.write(f"<p><i>グラフ画像が見つかりませんでした。</i></p>{image_path}")     
            # 基本情報
            #f.write("<hr>")
   
            f.write(f"<p><b>対象ファイル:</b> {analysis_report['filename']}<br>")
            f.write(f"<b>年データ:</b> {analysis_report['year']}<br>")
            f.write(f"<b>分割年:</b> {analysis_report['split_year']}<br>")
            f.write(f"<b>観測数:</b> {analysis_report['n_obs']}<br>")
            f.write(f"<b>検出限界未満:</b> {analysis_report['n_censored']}<br></p>")
            f.write("<hr>")    
            # ---- OLS ----
            ols = analysis_report['methods']['OLS']
            f.write("<h3>最小二乗法（OLS）非推奨</h3>")
            f.write(f"<b>傾き a:</b> {ols['slope']:.6f}<br>")
            f.write(f"<b>切片 b:</b> {ols['intercept']:.6f}<br>")
            f.write(f"<b>標準誤差 a:</b> {ols['stderr_a']:.6f}<br>")
            f.write(f"<b>標準誤差 b:</b> {ols['stderr_b']:.6f}<br>")
            f.write(f"<b>95%CI a:</b> [{ols['ci_a'][0]:.6f}, {ols['ci_a'][1]:.6f}]<br>")
            f.write(f"<b>95%CI b:</b> [{ols['ci_b'][0]:.6f}, {ols['ci_b'][1]:.6f}]<br>")
            f.write(f"<b>p値:</b> [{ols['pvalues'][0]:.3e}, {ols['pvalues'][1]:.6f}]<br>")
            f.write(f"<b>決定係数 R²:</b> {ols['r2']:.6f}<br>")
            f.write("<hr>")                   
            # ---- MLE ----
            mle = analysis_report['methods']['MLE']
            f.write("<h3>最尤推定（KDE-MLE / AIC）環境省法</h3>")  #最尤推定（KDE‑based MLE / AIC）
            f.write(f"<b>傾き a:</b> {mle['slope']:.6}<br>")
            f.write(f"<b>切片 b:</b> {mle['intercept']:.6}<br>")

            f.write("<br><b>---AICによるモデル比較---</b><br>")
            f.write(f"<b>AIC0:</b> {mle['aic0']:.6}<br>")
            f.write(f"<b>AIC1:</b> {mle['aic1']:.6}<br>")
            f.write(f"<b>モデル選択確率 p1:</b> {mle['p1']:.6}<br>")
            f.write(f"<b>傾向判定:</b> {mle['trend']}<br>")
            f.write("<hr>")

            # ---- Bayes ----
            bayes = analysis_report['methods']['Bayes']
            f.write("<h3>ベイズ推定（FFT-KDE-Bayes / BF）</h3>")    #ベイズ推定（FFT‑KDE Bayesian inference / Bayes Factor）
            f.write(f"<b>傾き a:</b> {bayes['slope']:.6f}<br>")
            f.write(f"<b>切片 b:</b> {bayes['intercept']:.6f}<br>")
            f.write(f"<b>事後確率:</b> {bayes['p_M1']:.6f}<br>")
            
            f.write(f"<b>傾き95%CI:</b> [{bayes['bayes_CI_a'][0]:.6f}, {bayes['bayes_CI_a'][1]:.6f}]<br>")
            f.write(f"<b>切片95%CI:</b> [{bayes['bayes_CI_b'][0]:.6f}, {bayes['bayes_CI_b'][1]:.6f}]<br>")
            f.write(f"<b>減少傾向の事後確率 P(a<0):</b> {bayes['p_neg']:.6f}<br>")
            f.write(f"<b>増加傾向の事後確率 P(a>0):</b> {bayes['p_pos']:.6f}<br>")

            f.write("<br><b>---Bayes Factorによるモデル比較---</b><br>")
            f.write(f"<b>線形モデルの対数周辺尤度(モデルエビデンス) logZ(linear):</b> {bayes['logZ_linear']:.6f}<br>")
            f.write(f"<b>定数モデルの対数周辺尤度(モデルエビデンス) logZ(const):</b> {bayes['logZ_const']:.6f}<br>")
            f.write(f"<b>対数ベイズ因子 logBF:</b> {bayes['logBF']:.6f}<br>")
            f.write(f"<b>ベイズ因子 BF:</b> {bayes['BF']:.6f}<br>")
            f.write(f"<b>傾向判定:</b> {bayes['bayes_trend']}<br>")
            f.write(f"<b>BF評価:</b> {bayes['BF評価']}<br>")

            f.write("<hr>")
            # ---- Bootstrap ----
            boot = analysis_report['methods']['Bootstrap']
            f.write("<h3>ブートストラップ検定</h3>")
            #f.write(f"<b>95%CI:</b> {boot['bootstrap_CI']:.6}<br>")
            
            ci = boot['bootstrap_CI']
            f.write(f"<b>95%CI:</b> [{ci[0]:.6f}, {ci[1]:.6f}]<br>")
            f.write(f"<b>傾きの平均推定値:</b> {boot['bootstrap_summary']['mean']:.6}<br>")
            f.write(f"<b>標準偏差:</b> {boot['bootstrap_summary']['std']:.6}<br>")
            f.write(f"<b>傾向判定:</b> {boot['bs_trend']}<br>")
            f.write("<hr>")

            # ---- Binomial ----
            bino = analysis_report['methods']['Binominal']
            f.write("<h3>二項ロジスティック回帰</h3>")
            f.write(f"<b>傾き:</b> {bino['binomial_slope']:.6}<br>")
            f.write(f"<b>モデル選択確率 p1:</b> {bino['binomial_p1']:.6}<br>")
            f.write(f"<b>傾向判定:</b> {bino['binomial_trend']}<br>")
            f.write(f"<b>DL閾値: 傾き:</b> {bino['DL閾値_slope']:.4f}, p1={ bino['DL閾値_p1']:.3f}<br>")
            f.write(f"<b>25%タイル閾値:</b> 傾き={bino['25%タイル閾値_slope']:.4f}, p1={bino['25%タイル閾値_p1']:.3f}<br>")
            f.write(f"<b>75%タイル閾値:</b> 傾き={bino['75%タイル閾値_slope']:.4f}, p1={bino['75%タイル閾値_p1']:.3f}<br>")
            f.write("<hr>")

            f.write(f"<br>　→グラフは {image_path} に保存されている。")


        f.write("</body></html>")
        # カレントディレクトリを取得
        current_dir = os.getcwd()
        # フルパスを作成
        full_path = os.path.join(current_dir, html_path)


