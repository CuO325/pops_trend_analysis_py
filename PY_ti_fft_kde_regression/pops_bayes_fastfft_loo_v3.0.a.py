# pops_bayes_fastfft_loo_v3.0.py
# FFT-KDE 高速ベイズ（MCMC + TI）BayesConfig 対応版


version = f"pops_bayes_fastfft_loo_v3.0   (by H. Shiraishi)"

from datetime import datetime
import os
import time
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
import emcee
#custum modules
from analysis.config import BayesConfig, build_beta_grid
from analysis.fft_kde_common_engine3 import (
    prepare_fft_kde_lookup,
    fast_kde_log_likelihood_from_residuals,
)

def binomial_trend_analyze(df, flag_col='binary'):
    """
    二項ロジスティック回帰による傾向解析
    df: DataFrame (必須列: 'year' と flag_col)
    flag_col: 二項化された列名 (default='binary')
    """

    # 年ごとの集計
    yearly_data = df.groupby('year').agg(
        low_count=(flag_col, 'sum'),
        total_count=(flag_col, 'count')
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

def binomial_trend_analyze_old(df):

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
# ---------------------------
# 既存の解析関数群（元ファイルのロジック）
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

def neg_log_likelihood_0(params, x, y):

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
    
def numerical_hessian(func, params, eps=1e-5):
    params = np.asarray(params)
    n = len(params)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps

            fpp = func(params + ei + ej)
            fpm = func(params + ei - ej)
            fmp = func(params - ei + ej)
            fmm = func(params - ei - ej)

            H[i,j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)

    return H

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
    
 



    return a1, b1, aic0, aic1, p1, x_base

    #return a1, b1, aic0, aic1, p1, x_base
#-------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import jarque_bera, skew, kurtosis


def evaluate_ols_model(model, X, y):
    #results = model.fit()
    results = model
    resid = results.resid

    # --- 正規性 ---
    jb_stat, jb_p = jarque_bera(resid)
    skew_val = skew(resid)
    kurt_val = kurtosis(resid, fisher=False)

    # --- 自己相関 ---
    dw = durbin_watson(resid)
    lb_stat, lb_p = acorr_ljungbox(resid, lags=[10], return_df=False)

    # --- 分散不均一性 ---
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, results.model.exog)

    # --- VIF ---
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        var_names = X.columns
    else:
        X_values = X
        var_names = [f"x{i}" for i in range(X.shape[1])]

    vif_df = pd.DataFrame({
        "variable": var_names,
        "VIF": [variance_inflation_factor(X_values, i) for i in range(X_values.shape[1])]
    })

    summary = {
        "R_squared": results.rsquared,
        "Adj_R_squared": results.rsquared_adj,
        "coef": results.params.to_dict(),
        "p_values": results.pvalues.to_dict(),

        "Normality_JB_stat": jb_stat,
        "Normality_JB_p": jb_p,
        "Skewness": skew_val,
        "Kurtosis": kurt_val,

        "Durbin_Watson": dw,
        "LjungBox_stat(10)": lb_stat[0],
        "LjungBox_p(10)": lb_p[0],

        "BreuschPagan_stat": bp_stat,
        "BreuschPagan_p": bp_p,

        "VIF": vif_df
    }

    return summary




import numpy as np
import pandas as pd
import statsmodels.api as sm
def run_ols_analysis(df, x_col='year', y_col='log_conc'):
    # --- データのクリーニングと型変換 ---
    temp_df = df[[x_col, y_col]].copy()
    temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='coerce')
    temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='coerce')
    temp_df = temp_df.dropna()

    if len(temp_df) < 2:
        raise ValueError("有効な数値データが少なすぎます。")

    # --- 変数の準備 ---
    x_raw = temp_df[x_col].values
    y = temp_df[y_col].values

    # 中心化
    x_centered = x_raw - x_raw.mean()
    n_data = len(y)

    # --- X を必ず DataFrame にする（ここが最重要） ---
    X = pd.DataFrame({
        "const": 1.0,
        "x_centered": x_centered.astype(float)
    })

    try:
        ols_model = sm.OLS(y, X).fit()
        result = evaluate_ols_model(ols_model, X, y)

    except Exception as e:
        print(f"OLSの実行中にエラーが発生しました: {e}")
        return None

    # 結果表示
    for k, v in result.items():
        print(k, ":", v)

    print(ols_model.summary().as_text())

    # パラメータ抽出
    b1 = ols_model.params["const"]
    a1 = ols_model.params["x_centered"]

    stderr_b = ols_model.bse["const"]
    stderr_a = ols_model.bse["x_centered"]

    pvals = ols_model.pvalues.tolist()
    r2_ols = ols_model.rsquared

    ci = ols_model.conf_int(alpha=0.05)
    ols_ci_b = ci.loc["const"].values
    ols_ci_a = ci.loc["x_centered"].values

    p_slope = ols_model.pvalues["x_centered"]

    print(f"傾き a: {a1:.4f}, p値: {p_slope:.4f}")

    return {
        'ols_a1': a1,
        'ols_b1': b1,
        'ols_p1': p_slope,
        'n_data': n_data,
        'r2_ols': r2_ols,
        'ols_ci_a': ols_ci_a,
        'ols_ci_b': ols_ci_b,
        'aic': ols_model.aic,
        'loglik': ols_model.llf
    }

def run_ols_analysis_old(df, x_col='year', y_col='log_conc'):
    # --- データのクリーニングと型変換 ---
    # 数値に変換できない値（文字列など）を強制的にNaNにし、その後削除する
    temp_df = df[[x_col, y_col]].copy()
    temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='coerce')
    temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='coerce')
    
    # 欠損値(NaN)がある行を除外しないとOLSは実行できません
    temp_df = temp_df.dropna()

    if len(temp_df) < 2:
        raise ValueError("有効な数値データが少なすぎます。")

    # --- 変数の準備 ---
    x_raw = temp_df[x_col].values
    y = temp_df[y_col].values  # ここで確実に float64 になります
    
    # 中心化の計算（どちらの定義を使うか統一してください）
    # ここでは元のコードの後半の定義「平均引き」を採用します
    x_centered = x_raw - x_raw.mean()
    n_data = len(y)

    # --- OLS (線形回帰) ---
    # X を明示的に float 型で作成
    X = sm.add_constant(x_centered).astype(float)
    
    try:

        ols_model = sm.OLS(y, X)
        #model = sm.OLS(y, X)
        result = evaluate_ols_model(ols_model, X, y)        
    except Exception as e:
        print(f"OLSの実行中にエラーが発生しました: {e}")
        return None


    for k, v in result.items():
        print(k, ":", v)
        # --- 以下、解析結果の抽出 ---
        ols_summary = ols_model.summary().as_text()

    print(ols_summary)
    b1 = ols_model.params[0]  # 切片
    a1 = ols_model.params[1]  # 傾き
    
    stderr_b, stderr_a = ols_model.bse
    pvals = ols_model.pvalues.tolist()
    r2_ols = ols_model.rsquared

    ci = ols_model.conf_int(alpha=0.05)
    # statsmodelsの戻り値がDataFrameかndarrayかによってアクセスを変える
    ols_ci_b = ci.iloc[0].values if hasattr(ci, 'iloc') else ci[0]
    ols_ci_a = ci.iloc[1].values if hasattr(ci, 'iloc') else ci[1]

    # 傾向判定用の仮のp値（傾きのp値を使用）
    p_slope = ols_model.pvalues[1]

    # ... (以下、判定ロジックとreturn処理)
    print(f"傾き a: {a1:.4f}, p値: {p_slope:.4f}")
    
    return {
        'ols_a1': a1,
        'ols_b1': b1,
        'ols_p1': p_slope,
        'n_data': n_data,
        'ols_ci_a': ols_ci_a,
        'ols_ci_b': ols_ci_b,
        'aic': ols_model.aic,
        'loglik': ols_model.llf
    }


# ---------------------------
# 事前分布
# ---------------------------

def log_prior_informative(params, data_scale=None):
    a, b = params
    if not (-100 < a < 100 and -100 < b < 100):
        return -np.inf

    if data_scale is None:
        return - (a**2) / (2 * 2.0**2) - (b**2) / (2 * 2.0**2)
    else:
        x_range = data_scale.get("x_range", 1.0)
        y_std = data_scale.get("y_std", 1.0)
        a_sd = max(0.1, y_std / max(1e-6, x_range) * 2.0)
        b_sd = max(0.1, y_std * 2.0)
        return - (a**2) / (2 * a_sd**2) - (b**2) / (2 * b_sd**2)

# ---------------------------
# 事後分布（通常 / tempered）
# ---------------------------

def log_posterior_fast(params, x_centered, y, lookup, data_scale=None):
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + ll


def tempered_log_posterior(params, beta, x_centered, y, lookup, data_scale=None):
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + beta * ll

# ---------------------------
# β ごとの E_beta[logL] 推定
# ---------------------------

def estimate_E_loglike_for_beta(
    beta,
    x_centered,
    y,
    lookup,
    cfg: BayesConfig,
    init_center=None,
    data_scale=None,
    show_progress=False,
):
    ndim = 2

    if init_center is None:
        init_center = np.array([0.0, np.mean(y)])

    rng = np.random.default_rng()
    initial_pos = init_center + 1e-5 * rng.normal(size=(cfg.n_walkers, ndim))

    def _logprob(theta):
        return tempered_log_posterior(
            theta, beta, x_centered, y, lookup, data_scale=data_scale
        )

    sampler = emcee.EnsembleSampler(cfg.n_walkers, ndim, _logprob)
    pos, _, _ = sampler.run_mcmc(initial_pos, cfg.n_steps, progress=show_progress)
    chain = sampler.get_chain(discard=int(cfg.n_steps * cfg.burn_frac), flat=True)

    ll_vals = np.array([
        fast_kde_log_likelihood_from_residuals(th[0], th[1], x_centered, y, lookup)
        for th in chain
    ])
    return np.mean(ll_vals), chain, sampler

# ---------------------------
# 熱力学積分による logZ
# ---------------------------

def compute_log_marginal_TL(
    df,
    model="linear",
    cfg: BayesConfig = None,
    verbose=True,
):
    if cfg is None:
        cfg = BayesConfig()

    years = df["year"].values.astype(float)
    x_base = years.min()
    x_centered = years - x_base
    y = df["log_conc"].values.astype(float)

    # MLE による残差 → KDE lookup 作成
    if model == "linear":
        res = minimize(
            neg_log_likelihood,
            [-0.01, np.mean(y)],
            args=(x_centered, y),
            method="L-BFGS-B",
        )
        a_mle, b_mle = res.x
        residuals = y - (a_mle * x_centered + b_mle)
    else:
        res0 = minimize(
            neg_log_likelihood_0,
            [np.mean(y)],
            args=(x_centered, y),
            method="L-BFGS-B",
        )
        b0 = res0.x[0]
        a_mle, b_mle = 0.0, b0
        residuals = y - b0

    lookup = prepare_fft_kde_lookup(
        residuals,
        bw_factor=cfg.bw_factor,
        grid_size=cfg.grid_size_lookup,
        use_loo=cfg.mode_loo,
    )
    if verbose:
        print("DEBUG lookup @bayes:", type(lookup),)

    betas = build_beta_grid(cfg)

    if verbose:
        print(f"\n[TI] model={model}")
        print(" betas =", np.round(betas, 3))

    E_logL = []
    last_sampler = None  # <--- ここで初期化しておく
    data_scale = {
        "x_range": float(x_centered.max() - x_centered.min()),
        "y_std": float(np.nanstd(y)),
    }

    init_center = np.array([a_mle, b_mle])

    for beta in betas:
        if verbose:
            print(f" β={beta:.3f} ... ", end="")
        E, chain, sampler = estimate_E_loglike_for_beta(
            beta,
            x_centered,
            y,
            lookup,
            cfg=cfg,
            init_center=init_center,
            data_scale=data_scale,
            show_progress=False,
        )
        if verbose:
            print(f"E[logL]={E:.3f}")
        E_logL.append(E)
        last_sampler = sampler  # ここで代入される
# 台形公式で対数周辺尤度を計算
    logZ = trapezoid(E_logL, betas)

# --- Rhatの計算 ---
    # ここでは、最後のβ=1.0のサンプルを使ってRhatを計算することにする。
    # burn_fracを有効にするために、最後のサンプルからチェーンを取得する際に discard を指定する必要がある。
    # burn_frac と n_steps を定義しておく必要がある
    n_steps = cfg.n_steps
    burn_frac = cfg.burn_frac
    discard = int(burn_frac * n_steps)
    chain = last_sampler.get_chain(discard=discard)
    # last_sampler から 3Dチェーン (n_steps, n_walkers, n_params) を取得
    # compute_Rhat の期待する軸 (n_walkers, n_steps, n_params) に変換    
    chain3d = np.transpose(chain, (1, 0, 2))
    n_samples = chain3d.shape[1] # ← ここ重要　これが burn-in 後のサンプル数になる。これをもとに Rhat を計算する際のチェーンの形状を決定する。

    #chain3d = np.transpose(last_sampler.get_chain(), axes=(1, 0, 2))
    #n_samples = chain3d.shape[1] 

    # 安全ガード（超重要）
    if n_samples < 20:
        Rhat = np.array([np.nan, np.nan])
    else:

        split_chains = np.concatenate(
            [
                chain3d[:, :n_samples//2, :], # 各ウォーカーの前半分
                chain3d[:, n_samples//2:, :]  # 各ウォーカーの後半分
            ],
            axis=0 # 縦に繋ぐ（32本 + 32本 = 64本とみなす）
        )
        # この「擬似的に増やした64本」で収束を判定する
        # Rhatの計算結果を、モデルの次元に合わせて取得
        Rhat_raw = compute_Rhat(split_chains)
        Rhat = np.atleast_1d(Rhat_raw)
        #Rhat = compute_Rhat(chain3d)
    if verbose:
        if model == "linear":
            print(f"Rhat (TI-Linear) [a, b]: {Rhat}")
        else:
            # 0次モデルの場合、aは固定なのでb(インデックス0)のみ表示
            print(f"Rhat (TI-Const) [b]: {Rhat}")
    return {
        "model": model,
        "betas": betas,
        "E_logL": np.array(E_logL),
        "logZ": logZ,
        "a_mle": a_mle,
        "b_mle": b_mle,
        "Rhat": Rhat,
        "lookup": lookup,
    }

# ---------------------------
# モデル比較
# ---------------------------
def compare_models_logZ(res1, res0, prior_ratio=1.0):
    """
    周辺尤度 logZ に基づいてベイズ因子と事後確率を計算する (数値安定版)
    """
    # 1. 極端な logZ 値による math domain error を防ぐためのクリッピング
    # TIで失敗している場合、-1e18のような値が入るため、現実的な下限を設定
    l1 = max(res1['logZ'], -1e10)
    l0 = max(res0['logZ'], -1e10)
    
    # 2. 対数ベイズ因子の計算
    logBF = l1 - l0 + np.log(prior_ratio)
    
    # 3. 事後確率 p_M1 の安定計算
    # p = BF / (1 + BF) は p = 1 / (1 + exp(-logBF)) と等価 (シグモイド関数)
    # logBF が大きすぎると exp(-logBF) は 0 に、小さすぎると inf になるのを防ぐ
    if logBF > 700:
        p_M1 = 1.0
        BF = np.inf
    elif logBF < -700:
        p_M1 = 0.0
        BF = 0.0
    else:
        BF = np.exp(logBF)
        p_M1 = 1.0 / (1.0 + np.exp(-logBF))

    return {
        'logZ1': l1,
        'logZ0': l0,
        'logBF': logBF,
        'BF': BF,
        'p_M1': p_M1,
    }
def compare_models_logZ_ok(res1, res0, prior_ratio=1.0):
    logZ1 = res1["logZ"]
    logZ0 = res0["logZ"]
    log_prior_ratio = np.log(prior_ratio)
    
    # 直接 BF を計算せず、対数スケールで p_M1 を計算する
    # p_M1 = 1 / (1 + exp(logZ0 - logZ1 - log_prior_ratio))
    diff = logZ0 - logZ1 - log_prior_ratio
    
    # diff が非常に小さい（logZ1が圧倒的に大きい）場合は p_M1 は 1.0
    if diff < -700:
        p_M1 = 1.0
        BF = np.inf
    else:
        BF = np.exp(-diff) # ここで overflow する可能性はあるが BF 表示用
        p_M1 = 1.0 / (1.0 + np.exp(diff))
        
    return {
        "logZ1": logZ1,
        "logZ0": logZ0,
        "logBF": -diff,
        "BF": BF,
        "p_M1": p_M1,
    }


# ---------------------------
# Rhat （Gelman–Rubin 診断）1.00 に近いほど収束している. 1.05 を超えると怪しい. 1.10 を超えると収束していない可能性が高い.
# ---------------------------

def compute_Rhat(chains, axis=0):
    try:
        n_chains, n_samples, n_params = chains.shape
        chain_means = np.mean(chains, axis=1)
        chain_vars = np.var(chains, axis=1, ddof=1)
        W = np.mean(chain_vars, axis=0)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)
        var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        Rhat = np.sqrt(var_hat / W)
        print(f"DEBUG Rhat values: {Rhat}")
        return Rhat
    except Exception:
        return np.array([np.nan] * chains.shape[-1])


# ---------------------------
# MLE 用 KDE 尤度
# ---------------------------

def kde_log_likelihood_from_residuals_plain(a, b, x_centered, y):
    res = y - (a * x_centered + b)
    try:
        kde = gaussian_kde(res)
        pdf_vals = kde.pdf(res)
        return np.sum(np.log(pdf_vals + 1e-12))
    except Exception:
        return -1e20


def neg_log_likelihood(params, x, y):
    a, b = params
    residuals = y - (a * x + b)
    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10


def neg_log_likelihood_0(params, x, y):
    b0 = params[0]
    residuals = y - b0
    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10


# ---------------------------
# 高速 MCMC（auto-stop）
# ---------------------------
def run_bayes_kde_auto_stop_fast(
    df,
    cfg: BayesConfig,
    fixed_lookup=None,
    verbose=True,
    ci_width_threshold=0.02,
    stable_rounds=2,
    chunk=100,   # v2.1 で追加：チャンク長も引数に
):
    years = df["year"].values.astype(float)
    x_base = years.min()
    x_centered = years - x_base
    y = df["log_conc"].values.astype(float)
    ndim = 2

    # MLE で初期値
    try:
        mle_res = minimize(
            lambda p: -kde_log_likelihood_from_residuals_plain(
                p[0], p[1], x_centered, y
            ),
            x0=[0.0, np.mean(y)],
            method="L-BFGS-B",
        )
        a_mle, b_mle = mle_res.x
    except Exception:
        a_mle, b_mle = 0.0, np.mean(y)

    # lookup 準備
    if fixed_lookup is None:
        residuals_mle = y - (a_mle * x_centered + b_mle)
        lookup = prepare_fft_kde_lookup(
            residuals_mle,
            bw_factor=cfg.bw_factor,
            grid_size=cfg.grid_size_lookup,
            use_loo=cfg.mode_loo,
        )
        if verbose:
            print("DEBUG lookup@run_bayes_kde_auto_stop_fast", type(lookup), "Rhat (MLE):", mle_res.get("rhat_a") if mle_res is not None else "N/A")
    else:
        lookup = fixed_lookup

    data_scale = {
        "x_range": x_centered.max() - x_centered.min(),
        "y_std": np.nanstd(y),
    }

    rng = np.random.default_rng(int(time.time() % 2**31))
    n_walkers = max(2 * ndim, cfg.n_walkers)
    initial_pos = np.vstack([
        np.array([a_mle, b_mle]) + rng.normal(scale=[0.02, 0.1], size=2)
        for _ in range(n_walkers)
    ])

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_posterior_fast,
        args=(x_centered, y, lookup, data_scale),
    )

    pos = initial_pos
    n_chunks = max(1, cfg.n_steps // chunk)
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
            print(
                f"[FastMCMC {min(total_steps_done, cfg.n_steps)}/{cfg.n_steps}] "
                f"a_mean={a_mean if not np.isnan(a_mean) else 'nan'} "
                f"CI={ci} P(a<0)={p_neg if not np.isnan(p_neg) else 'nan'}"
            )

        ci_width = ci[1] - ci[0] if not np.any(np.isnan(ci)) else np.inf
        ci_is_stable = ci_width < ci_width_threshold
        direction_stable = (
            (not np.any(np.isnan(ci))) and ((ci[1] < 0) or (ci[0] > 0))
        )

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

        if total_steps_done >= cfg.n_steps:
            break

    total_steps = sampler.iteration
    discard = int(cfg.burn_frac * total_steps)
    samples = sampler.get_chain(discard=discard, flat=True)

    if samples.size == 0:
        samples = sampler.get_chain(flat=True)

    a_samps = samples[:, 0] if samples.size != 0 else np.array([np.nan])
    b_samps = samples[:, 1] if samples.size != 0 else np.array([np.nan])

    chain3d = np.transpose(sampler.get_chain()[:, :, :], axes=(1, 0, 2))
    Rhat = compute_Rhat(chain3d)

    result = {
        "samples": samples,
        "a_mean": np.mean(a_samps) if a_samps.size > 0 else np.nan,
        "b_mean": np.mean(b_samps) if b_samps.size > 0 else np.nan,
        "a_ci": np.percentile(a_samps, [2.5, 97.5])
        if a_samps.size > 0
        else np.array([np.nan, np.nan]),
        "b_ci": np.percentile(b_samps, [2.5, 97.5])
        if b_samps.size > 0
        else np.array([np.nan, np.nan]),
        "p_neg_slope": np.mean(a_samps < 0) if a_samps.size > 0 else np.nan,
        "steps_completed": total_steps,
        "stopped_early": stable_count >= stable_rounds,
        "mcmc_summary": f"Steps: {total_steps}/{cfg.n_steps}, a_mean: {np.mean(a_samps):.4f}, CI: [{ci[0]:.4f}, {ci[1]:.4f}], P(a<0): {p_neg:.4f}", # ★追加
        "Rhat": Rhat,
        "lookup": lookup,
    }
    return result

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
    # 1. まずBFの強度を取得
    bf_info = interpret_bayes_factor(BF=cmp['BF'])
    bf_label = bf_info['interpretation']
    
    # 2. 基本的なモデル支持の判定
    if cmp['BF'] >= 3:
        model_decision = "線形モデルを支持。"
    elif cmp['BF'] >= 1:
        model_decision = "線形モデルの兆候があるが、決定不足。"
    else:
        model_decision = "定数モデル（傾向なし）を支持。"

    # 3. 方向性の判定（確信度が高い場合のみ断定）
    if p_neg_slope >= 0.95:
        direction = "減少傾向。"
    elif p_pos_slope >= 0.95:
        direction = "増加傾向。"
    else:
        direction = "明確な方向性は認められない。"

    # 4. 信頼区間の情報を付与
    ci_status = "（信頼区間が0を跨ぐ）" if contains_zero else "（信頼区間が0を跨がない）"

    # 全てを統合したメッセージ
    return f"【判定】BF: {model_decision} P_slope: {direction}  ci:{ci_status} <br> [証拠の強度 (Jeffreys scale): {bf_label}]"


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
        # ベイズ因子 BF の解釈には、統計学で広く用いられる Jeffreys scale（Jeffreys, 1961）を採用した。
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

# -------------------------------------------------------------------
# FFT-KDE の結果から任意点の密度を補間評価
# -------------------------------------------------------------------
def evaluate_kde_at_points(grid, density, points):
    return np.interp(points, grid, density, left=1e-12, right=1e-12)

# -------------------------------------------------------------------
# MCMC / TI などで呼び出す “データ全体の log-likelihood”
# -------------------------------------------------------------------

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
#===================================================================================================    

if __name__ == "__main__":

    cfg = BayesConfig()
    #betas = build_beta_grid(cfg)
    print("BayesConfig:", cfg)
    print(f"beta_schedule: {cfg.beta_schedule}")
    print(f"Leave One Out" if cfg.mode_loo else "通常のTI", f" - beta_grid: {build_beta_grid(cfg)}")

    #filenames = ["HCB大気温暖期2013test"]
    filenames = ["PCB_water"] 
    #filenames = ["PFOS大気温暖期"]
    for filename in filenames:
        print(f"\n\n==================== ファイル処理開始: {filename} ====================")
        filepath = f"data/{filename}.csv"
        if not os.path.exists(filepath):
            print(f"ファイルが見つかりません: {filepath}")
            continue

        df_all = pd.read_csv(
            filepath,
            usecols=["substance", "year", "site_id", "concentration", "detection_limit"],
        )
     

        df_all["concentration"] = df_all["concentration"].apply(clean_value_user)
        df_all["detection_limit"] = df_all["detection_limit"].apply(clean_value_user)

        is_nondetect = df_all["concentration"] <= df_all["detection_limit"]
        substitute_value = df_all["detection_limit"] / np.sqrt(2)
        df_all["concentration_sub"] = df_all["concentration"].copy()
        df_all.loc[is_nondetect, "concentration_sub"] = substitute_value[is_nondetect]
        df_all["log_conc"] = np.log10(df_all["concentration_sub"])
        df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["log_conc", "detection_limit", "concentration"]
        )


        # ここで TI / MCMC / モデル比較などを呼び出す
        output_path = f"data/{filename}_conv.csv"
        df_all.to_csv(output_path, index=False, encoding="utf-8")
        print(f"→ 変換後のデータを保存しました: {output_path}")

        substances = df_all['substance'].unique()
        print(f"対象物質: {substances}")

        results = []
        PLOT_DIR = "outputs/plots_loo_corrected"
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
            # OLS分析の結果を格納する変数もここで初期化しておく（例外発生時に上書きされないように）

            #ols_trend = "未実行"
            ols = run_ols_analysis(df) # ここでOLS分析を実行して、結果を表示するように変更予定（現状はプロット用のOLSのみ）

            # ① MLE + AIC
            try:
                #a, b, aic0, aic1, p1, x_base, linear_validity = run_mle_and_aic(df)
                a, b, aic0, aic1, p1, x_base = run_mle_and_aic(df) #hessianからの信頼区間も受け取るように変更予定
                print("--- 1. 対数線形回帰モデル (MLE) ---")
                print(f"傾き a = {a:.4f}, 切片 b = {b:.4f}, AIC事後確率 p1 = {p1:.4f}")
                print(f"aic1: {aic1} - aic0: {aic0}")
                trend_mle = "p1 >= 0.95 対数線形回帰モデルを選択する" if p1 >= 0.95 else "p1 < 0.95 対数線形回帰モデルを選択しない"
                print(f"→ {trend_mle}")
            except Exception as e:
                print(f"最尤推定エラー: {e}")
                trend_mle  = "エラー"
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
                # 共通の準備処理
                residuals_mle = y_true - (a * x_centered + b)
                # 1回だけ実行
                lookup = prepare_fft_kde_lookup(residuals_mle, bw_factor=1.0, use_loo=True)
                print(f"DEBUG: lookup created, type={type(lookup)}, Rhat (MLE): N/A (まだMCMC未実行)")

                # MCMC実行（1回目と2回目を統合）
                # chunk や stable_rounds などのパラメータを適切に設定
                bayes_res = run_bayes_kde_auto_stop_fast(
                    df,
                    cfg=cfg,
                    fixed_lookup=lookup,
                    verbose=True,
                    ci_width_threshold=0.03,
                    stable_rounds=1,
                    chunk=100,  # 1回目にあった引数を採用
                )

                # 結果の抽出
                a_bayes = bayes_res.get("a_mean", np.nan)
                b_bayes = bayes_res.get("b_mean", np.nan)
                ci_a = np.array(bayes_res.get("a_ci", [np.nan, np.nan]))
                ci_b = np.array(bayes_res.get("b_ci", [np.nan, np.nan]))
                p_neg_slope = bayes_res.get("p_neg_slope", np.nan)
                p_pos_slope = 1.0 - p_neg_slope if not np.isnan(p_neg_slope) else np.nan
                contains_zero = (ci_a[0] <= 0 <= ci_a[1]) if not np.any(np.isnan(ci_a)) else True
                rhat_linear1 = bayes_res.get("Rhat") # ここでも計算し後で参照する[Rhat_a, Rhat_b]
                print(f"DEBUG: MCMC計算完了, Rhat (線形モデル): {rhat_linear1}, a_mean: {a_bayes}, CI_a: {ci_a}, P(a<0): {p_neg_slope:.4f}, P(a>0): {p_pos_slope:.4f}")

                # --- モデル比較 (TI) ---
                # モデル比較（線形 vs 定数）には
                # TIが必須となり、非常に重い処理なので、必要に応じconfig.pyで調整し実行する. 
                # TIの計算は、MCMCの結果を利用して行うため、MCMCが成功した場合にのみ実行するようにする。
                res1 = compute_log_marginal_TL(df_all, model="linear", cfg=cfg, verbose=True)
                rhat_linear = res1.get("Rhat") # [Rhat_a, Rhat_b]
                print(f"DEBUG: TI計算完了, Rhat (線形モデル): {rhat_linear}")
                res0 = compute_log_marginal_TL(df_all, model="constant", cfg=cfg, verbose=True)
                rhat_const = res0.get("Rhat")  # [Rhat_b]
                print(f"DEBUG: TI計算完了, Rhat (定数モデル): {rhat_const}")
                cmp = compare_models_logZ(res1, res0)

                # 判定
                bayes_trend = bayes_trend_judgement(contains_zero, cmp, p_neg_slope, p_pos_slope)
                bf_interpretation = interpret_bayes_factor(BF=cmp['BF'])

                # 表示
                print(f"logZ(linear): {cmp['logZ1']:.4f}, logZ(const): {cmp['logZ0']:.4f}")
                print(f"BF: {cmp['BF']:.4f}, 評価: {bf_interpretation['interpretation']}")
                print(f"→ {bayes_trend} (ベイズ)")

            except Exception as e:
                print(f"ベイズ推定工程でエラーが発生しました: {e}")
                bayes_trend = "エラー"
                # 必要に応じて a_bayes 等に NaN を代入

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
            """try:
                print("\n--- 4. 二項ロジスティック回帰モデル ---")
                df_binomial = df[['year', 'concentration', 'detection_limit']].copy()
                aic0_bin, aic1_bin, p1_bin, a_bin = binomial_trend_analyze(df_binomial)
                print(f"傾き a = {a_bin:.4f}, オッズ比 ≈ {math.exp(a_bin):.4f}, AIC事後確率 p1 = {p1_bin:.3f}")
                trend_bin = "傾向あり" if p1_bin >= 0.95 else "傾向なし"
                print(f"→ {trend_bin}")
            except Exception as e:
                print(f"二項回帰エラー: {e}")
                trend_bin = "エラー"
                a_bin, p1_bin = np.nan, np.nan"""

            try:
                print("\n--- 4. 二項ロジスティック回帰モデル ---")
                df_binomial = df[['year', 'concentration', 'detection_limit']].copy()
                aic0_bin, aic1_bin, p1_bin, a_bin = binomial_trend_analyze_old(df_binomial)
                print(f"傾き a = {a_bin:.4f}, オッズ比 ≈ {math.exp(a_bin):.4f}, AIC事後確率 p1 = {p1_bin:.3f}")
                trend_bin = f"統一下限値 {max(df_binomial[ 'detection_limit'])} 以下の増加傾向あり" if p1_bin >= 0.95 else  f"統一下限値 {max(df_binomial[ 'detection_limit'])} 以下で傾向なし"
                print(f"→ {trend_bin}")
            except Exception as e:
                print(f"二項回帰エラー: {e}")
                trend_bin = "エラー"
                a_bin, p1_bin = np.nan, np.nan


            # 閾値の定義
            threshold_dl = df['detection_limit'].max()       # 最高検出下限値
            threshold_q25 = df['concentration'].quantile(0.25)  # 25%タイル
            threshold_q75 = df['concentration'].quantile(0.75)  # 75%タイル
            # 全データの75%タイル
            #threshold_q75_global = df['concentration'].quantile(0.75)
            # 年ごとの75%タイルを計算し、その最大値を閾値にする
            threshold_q75_yearly_max = df.groupby('year')['concentration'].quantile(0.75).max()
            
            # 二項化（低濃度・高濃度判定）
            # 25%タイル
            threshold_q25 = df['concentration'].quantile(0.25)
            df_binomial_q25 = df[['year', 'concentration']].copy()
            df_binomial_q25['binary'] = (df['concentration'] <= threshold_q25).astype(int)

            # 75%タイル
            threshold_q75 = df['concentration'].quantile(0.75)
            df_binomial_q75 = df[['year', 'concentration']].copy()
            df_binomial_q75['binary'] = (df['concentration'] >= threshold_q75).astype(int)

            # DL閾値（従来通り detection_limit がある場合のみ）
            if 'detection_limit' in df.columns:
                threshold_dl = df['detection_limit'].max()
                df_binomial_dl = df[['year', 'concentration']].copy()
                df_binomial_dl['binary'] = (df['concentration'] <= threshold_dl).astype(int)

            # 解析関数に渡す
            try:
                aic0_dl, aic1_dl, p1_dl, a_dl = binomial_trend_analyze(df_binomial_dl)
                print(f"DL閾値: 傾き={a_dl:.4f}, p1={p1_dl:.3f}")
            except Exception as e:
                print(f"DL閾値での二項回帰エラー: {e}")

            try:
                aic0_q25, aic1_q25, p1_q25, a_q25 = binomial_trend_analyze(df_binomial_q25)
                print(f"25%タイル閾値: 傾き={a_q25:.4f}, p1={p1_q25:.3f}")
            except Exception as e:
                print(f"25%タイルでの二項回帰エラー: {e}")

            try:
                aic0_q75, aic1_q75, p1_q75, a_q75 = binomial_trend_analyze(df_binomial_q75)
                print(f"75%タイル閾値: 傾き={a_q75:.4f}, p1={p1_q75:.3f}")
            except Exception as e:
                print(f"75%タイルでの二項回帰エラー: {e}")
            # ⑤ プロット
            try:
                print("\n--- 5. 回帰直線のプロット ---")
                x_data = df['year'].values
                y_data = df['log_conc'].values
                dl_log = np.log10(df['detection_limit'])
                x_fit_plot = np.array([x_data.min(), x_data.max()])
                x_fit_centered = x_fit_plot - x_base
                y_fit = a * x_fit_centered + b
                # ベイズ回帰線
                a_bayes = bayes_res.get("a_mean", a_bayes)
                b_bayes = bayes_res.get("b_mean", b_bayes)
                y_fit_bayes = a_bayes * x_fit_centered + b_bayes
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hlines(dl_log, xmin=x_data - 0.2, xmax=x_data + 0.2,
                            color='tab:orange', alpha=0.5, linewidth=1.5, label='検出限界 (log10)')
                ax.scatter(x_data, y_data, label='実測データ (代替値込)', color='tab:blue', alpha=0.6)
                ax.plot(x_fit_plot, y_fit, label=f'MLE回帰直線: y = {a:.4f}(x - {x_base}) + {b:.4f}', color='tab:red')
                # ベイズ回帰直線
                ax.plot(x_fit_plot, y_fit_bayes,
                        label=f'ベイズ回帰直線: y = {a_bayes:.4f}(x - {x_base}) + {b_bayes:.4f}',
                        color='tab:green', linestyle='--')
                if 'split_year' in locals() and split_year is not None:
                    ax.axvline(x=split_year, color='gray', linestyle='--', linewidth=1.5, label=f'分割年: {split_year}')
                ax.set_title(f'{substance} の対数線形回帰\nMLE p1 = {p1:.4f} / ベイズ P(a<0) = {p_neg_slope if not np.isnan(p_neg_slope) else 0:.4f} ({filename})')
                ax.set_xlabel('年')
                ax.set_ylabel('log10濃度')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                y_all = np.concatenate([y_data, y_fit, y_fit_bayes, dl_log])
                ax.set_ylim(y_all.min() - 0.1, y_all.max() + 0.1)
                fig.tight_layout()
                filename_safe = f"{substance}_{filename}".replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_")
                plot_path = os.path.join(PLOT_DIR, f'{filename_safe}_regression.png')
                fig.savefig(plot_path)
                plt.close(fig)
                print(f"→ {plot_path} に保存しました。")
            except Exception as e:
                print(f"プロットエラー: {e}")

            reliability = is_data_acceptable(a, p1, ci_bs)
            print(f"減少傾向のモデル間整合性チェック (MLE + Boot): {reliability}")

            Rhat = bayes_res.get("Rhat", np.array([np.nan, np.nan]))

            # 結果格納
            results.append({
                "substance": substance,
                'filename': filename,                
                'year':df['year'].unique().tolist(),
                'split_year': split_year,
                "n": len(df),
                'n_censored': (df['log_conc'] < np.log10(df['detection_limit'])).sum(),
            #OLS
                'r2': r2,
                'r2_ols': ols['r2_ols'],
                'ols_a1': ols['ols_a1'],
                'ols_b1': ols['ols_b1'],
                'n_data': ols['n_data'],
                #'ols_ci_b1': ols['ols_ci_b1'],
                #'ci_ols': ci_ols.tolist() if isinstance(ci_ols, np.ndarray) else ci_ols,
                "model_summary_ols": model.summary().as_html(),
            #MLE/AIC
                "a_mle": a,
                "b_mle": b,
                "aic0": aic0,
                "aic1": aic1,
                "delta_aic": aic1 - aic0,
                "p1_aic": p1,
                'trend_MLE': trend_mle,                

            #Bayes
                "mcmc_summary": bayes_res.get("mcmc_summary", "N/A"), # ★追加
                "steps_done": bayes_res.get("steps_completed", 0),    # ★追加
                "stopped_early": bayes_res.get("stopped_early", False), # ★追加
                'Rhat_a': bayes_res['Rhat'][0],  # 傾きaのRhat
                'Rhat_b': bayes_res['Rhat'][1],  # 切片bのRhat
                'rhat_a': rhat_linear[0],
                'rhat_b_lin': rhat_linear[1],
                'rhat_b_const': rhat_const[0], # これを追加
                            #'Rhat': Rhat,
                #'ci_b': ci_b,
            #'Bayes'
                'bayes_slope': a_bayes,
                'bayes_intercept': b_bayes,
                'bayes_p_m1': cmp['p_M1'],         
                'bayes_CI_a': ci_a.tolist() if isinstance(ci_a, np.ndarray) else ci_a,
                'bayes_CI_b': ci_b.tolist() if isinstance(ci_b, np.ndarray) else ci_b,
                'bayes_p_neg': p_neg_slope,   
                'bayes_p_pos': p_pos_slope,
                'logZ_linear': cmp['logZ1'],
                'logZ_const':  cmp['logZ0'],
                'logBF':cmp['logBF'],
                'BF':cmp['BF'],
                'bayes_trend': bayes_trend,
                "BF評価": bf_interpretation['interpretation'],               
 

                'reliability': reliability,       
            #'Bootstrap'

                'bootstrap_CI': ci_bs,
                'bootstrap_summary': {'mean': np.mean(ci_bs), 'std': np.std(ci_bs)},
                'bootstrap_mean': np.mean(ci_bs),
                'bootstrap_std': np.std(ci_bs),

                'bs_trend': bs_trend,
            #'Binominal'
                'binomial_slope': a_bin,
                'binomial_p1': p1_bin,
                'binomial_trend': trend_bin,

                'DL閾値_slope': a_dl,
                'DL閾値_p1': p1_dl,

                '25%タイル閾値_slope': a_q25,
                '25%タイル閾値_p1': p1_q25,  

                '75%タイル閾値_slope':a_q75,
                '75%タイル閾値_p1': p1_q75                     

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
            #r['ci_a'] = format_ci(r.get('ci_a'))
            #r['ci_b'] = format_ci(r.get('ci_b'))

            # 数値項目の整形
            for key in ['a_mle', 'b_mle', 'binomial_slope', 'p1_MLE', 'binomial_p1',
                        'bayes_slope', 'bayes_intercept', 'bayes_p_m1', 'bayes_p_neg', 'bayes_p_pos']:
                r[key] = format_float(r.get(key))

        df_results = pd.DataFrame(results)
        print("\n--- 最終結果（テーブル形式） ---")
        print(df_results.to_string(index=False))
        df_results.to_csv(f"{filename}_2025_reliable.csv", index=False, encoding='utf-8-sig')

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  
        # -----------------------------HTMLレポート作成------------------------------------

        html_path = f"outputs/{filename}_LOO2_FFT-KDE-Bayes_{timestamp}.html"
        df_results.to_csv(f"outputs/{filename}_2025_reliable_{timestamp}.csv", index=False, encoding='utf-8-sig')

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<html><head><meta charset='utf-8'><title>{filename} 回帰分析レポート</title></head><body>")
            f.write("<h1>回帰分析レポート</h1>")
            f.write(f" {version}   <br><br>")
            f.write(f" analysis date:{timestamp}<br><br>")    

            for r in results:
                if max(r['year']) > 0:
                    filename_safe = f"{r['substance']}_{filename}".replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_")
                    image_path = os.path.join(PLOT_DIR, f"{filename_safe}_regression.png")


                    f.write("<hr>")
                    # ---- Graph ----
                    if os.path.exists(image_path):
                        #f.write(f"<img src='{image_path}' width='600'><br>")
                        f.write(f"<img src='plots_loo_corrected/{filename_safe}_regression.png' width='600'><br>")                        
                    else:
                        f.write("<p><i>グラフ画像が見つかりませんでした。</i></p>")                    
                    f.write(f"<h2>{r['substance']}</h2>")
                    f.write(f"<p><b>対象:</b> {filename}<br>")

                    f.write(f"<b>年データ:</b> {r['year']}<br><br>")
                    f.write(f"<b>データ数:</b> {r['n']}<br>")
                    f.write(f"<b>検出限界以下のデータ数:</b> {r['n_censored']}<br>")                    
                    f.write("<hr>")    
                    # ---- OLS ---- 
                    f.write("<h3>最小二乗法（OLS）</h3>")
                    f.write(f"<pre><b>summary:</b> {r['model_summary_ols']}<br></pre>")
                   
                    f.write(f"<b>傾き a:</b> {r['ols_a1']:.4f}<br>")
                    f.write(f"<b>切片 b:</b> {r['ols_b1']:.4f}<br>")
                    #f.write(f"<b>標準誤差 a:</b> {r['stderr_a']:.6f}<br>")
                    #f.write(f"<b>標準誤差 b:</b> {r['stderr_b']:.6f}<br>")
                    #f.write(f"<b>95%CI a:</b> [{r['ci_a'][0]:.6f}, {r['ci_a'][1]:.6f}]<br>")
                    #f.write(f"<b>95%CI b:</b> [{r['ci_b'][0]:.6f}, {r['ci_b'][1]:.6f}]<br>")
                    #f.write(f"<b>p値:</b> [{r['pvalues'][0]:.3e}, {r['pvalues'][1]:.6f}]<br>")
                    f.write(f"<b>決定係数 R²:</b> {r['r2_ols']:.4f}<br>")
                    f.write(f"<b>補正決定係数 R²:</b> {r['r2']:.4f}<br>")                    
                    f.write("<hr>")  
                    # ---- MLE ----
                    f.write("<h3>最尤推定（MLE / AIC）</h3>")
                    f.write(f"<b>傾き a (MLE):</b> {r['a_mle']:.4f}  <br>")
               
                    f.write(f"<b>切片 b (MLE):</b> {r['b_mle']:.4f} <br> ")
 
                    f.write(f"<b>AICによるモデル選択　事後確率 p1:</b> {r['p1_aic']:.4f}<br>")
                    #f.write(f"<b>線形妥当性 (ΔAIC)) P:</b> {r['aic1']}-{r['aic0']}<br>")
                    f.write(f"<b>傾向判定 (MLE):</b> {r['trend_MLE']}<br>")
                    f.write("<hr>")      

                    # ---- Bayes (FFT-KDE MCMC) ----
                    f.write(f"<h3>ベイズ推定（MCMC / FFT-KDE尤度）</h3>")
                    f.write(f"--条件設定と収束結果-- <br> beta_schedule: {cfg.beta_schedule}<br>")
                    f.write(f"mode_loo: {cfg.mode_loo}<br>    {str(cfg)} <br>")
                    #Rhatの表示を追加
                    f.write(f" <pre> Rhat_a (mcmc): {r.get('Rhat_a'):.4f}<br>")
                    f.write(f" Rhat_b (mcmc):</b> {r.get('Rhat_b'):.4f}<br>")
                    f.write(f" rhat_a(TI): {r.get('rhat_a'):.4f}<br>")
                    f.write(f" rhat_b_lin(TI)': {r.get('rhat_b_lin'):.4f}<br>") 
                    f.write(f" rhat_b_const(TI)':{r.get('rhat_b_const'):.4f}</pre> <br>") # これを追加
                    # MCMCの実行ログと自動早期停止の情報を追加
                    f.write(f"<p style='color: #666;'><b>MCMC実行ログ:</b> {r.get('mcmc_summary')}")
                    if r.get('stopped_early'):
                        f.write(" <span style='color: green;'>(自動早期停止が適用されました)</span>")
                    f.write("</p>")
                    # ベイズ推定の結果を表示
                    f.write(f"<b>傾き a (ベイズ平均):</b> {r['bayes_slope']}<br>")
                    f.write(f"<b>切片 b (ベイズ平均):</b> {r['bayes_intercept']}<br>")                    
                    f.write(f"<b>傾き95%信用区間:</b> {r['bayes_CI_a']}<br>")

                    f.write(f"<b>切片95%信用区間:</b> {r['bayes_CI_b']}<br>")
                    f.write(f"<b>減少傾向の事後確率 P(a &lt; 0):</b> {r.get('bayes_p_neg', '---')}<br>")
                    f.write(f"<b>増加傾向の事後確率 P(a &gt; 0):</b> {r.get('bayes_p_pos', '---')}<br>")           
                    f.write(f"<b>モデル比較の事後確率 pM1:</b> {r['bayes_p_m1']}<br><pre>+ モデル比較判定 (ベイズ):{'1次式が妥当 bayes_p_m1 >= 0.95 ' if r['bayes_p_m1'] >= 0.95 else '1次式とする強いエビデンスなし bayes_p_m1 < 0.95'}</pre>")

                    f.write("<b>---Bayes Factorによるモデル比較---</b><br>")
                    f.write(f"<b>線形モデルの対数周辺尤度(モデルエビデンス) logZ(linear):</b> {r['logZ_linear']:.3f}<br>")
                    f.write(f"<b>定数モデルの対数周辺尤度(モデルエビデンス) logZ(const):</b> {r['logZ_const']:.3f}<br>")
                    f.write(f"<b>対数ベイズ因子 logBF:</b> {r['logBF']:.3f}<br>")
                    f.write(f"<b>ベイズ因子 BF:</b> {r['BF']:.3e}<br>")

                    f.write(f"<b>BF評価:</b>   {r['BF評価']} <br>")
                    f.write(
            """<pre>
            参考    Jeffreys scale（Jeffreys, 1961）
            -------------------------------------------
            BF < 1         定数モデル支持
            1 <= BF < 3    線形モデル支持の弱い証拠
            3 <= BF < 10   線形モデル支持の中程度の証拠
            10 <= BF < 30  線形モデル支持の強い証拠
            30 <= BF < 100 線形モデル支持の非常に強い証拠
            100 <= BF      線形モデル支持の決定的な証拠
            </pre>"""
                        )
                    f.write(f"<b>ベイズ判定:</b> {r['bayes_trend']}<br>")  
                    f.write("<hr>")        
                    # ---- Bootstrap ----
                    f.write("<h3>ブートストラップ検定</h3>")
                    ci = r['bootstrap_CI']
                    f.write(f"<b>対数平均濃度の差(前期―後期):</b> {r['bootstrap_mean']:.6}<br>")
                    f.write(f"<b>95% CI:</b> [{ci[0]:.4f}, {ci[1]:.4f}]<br>")

                    f.write(f"<b>標準偏差:</b> {r['bootstrap_std']:.6}<br>")    
                    #f.write(f"<b>summary:</b> {r['bootstrap_summary']:.6}<br>")                   

                    f.write(f"<b>傾向判定 (Boot):</b> {r['bs_trend']}<br><br>")
                    f.write("<hr>")    
                    # ---- Binomial ----
                    f.write("<h3>二項ロジスティック回帰</h3>")
                    f.write(f"<b>傾き:</b> {r['binomial_slope']}<br>")
                    f.write(f"<b>AIC事後確率 p1:</b> {r['binomial_p1']}<br>")
                    f.write(f"<b>傾向判定:</b> {r['binomial_trend']}<br><br>")
                    f.write(f"<b>DL閾値: 傾き:</b> {r['DL閾値_slope']:.4f}, p1={ r['DL閾値_p1']:.3f}<br>")
                    f.write(f"<b>25%タイル閾値:</b> 傾き={r['25%タイル閾値_slope']:.4f}, p1={r['25%タイル閾値_p1']:.3f}<br>")
                    f.write(f"<b>75%タイル閾値:</b> 傾き={r['75%タイル閾値_slope']:.4f}, p1={r['75%タイル閾値_p1']:.3f}<br>")

                    f.write(f"<br>　→グラフは {image_path} に保存。")


            f.write("</body></html>")
        # カレントディレクトリを取得
        current_dir = os.getcwd()
        # フルパスを作成
        full_path = os.path.join(current_dir, html_path)
        print(f"\n→ HTMLレポートを作成しました: {full_path}")

