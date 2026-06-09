#run_analysis.py


#from turtle import st
import os
from unittest import result

import streamlit as st
#st.write("DEBUG type of st:", type(st))

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm

from experimental.pops_bayes_fastfft_docscript import kde_log_likelihood_from_residuals_plain
#custum modules
from analysis.config import BayesConfig, build_beta_grid
from analysis.fft_kde_common_engine3 import (
    prepare_fft_kde_lookup,
    fast_kde_log_likelihood_from_residuals,
)

#-OLS------元の内容を忠実に移植-------------------------------
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
def run_ols_analysis(df, x_col='year', y_col='log_conc'):
    safe_f_substance =df["safe_f_substance"].iloc[0]
    substance =df["substance"].iloc[0]
    # --- データのクリーニングと型変換 ---
    temp_df = df[[x_col, y_col]].copy()
    temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='coerce')
    temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='coerce')
    temp_df = temp_df.dropna()

    if len(temp_df) < 2:
        raise ValueError("There is not enough valid numerical data.")

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

    print(f"Slope a: {a1:.4f}, p-value: {p_slope:.4f}")

    return {

        'safe_f_substance':safe_f_substance,
        "substance": substance,
        "analysis_type": "OLS",
        #"file_name": file_name,
        #"config": cfg,

        'ols_a1': a1,
        'ols_b1': b1,
        'ols_p1': p_slope,
        'n_data': n_data,
        'r2_ols': r2_ols,
        'ols_ci_a': ols_ci_a,
        'ols_ci_b': ols_ci_b,
        'aic': ols_model.aic,
        'loglik': ols_model.llf,
        'summary': ols_model.summary().as_text(),
        'summary_html': ols_model.summary().as_html()
    }



#-MLE + AIC------内容を出来るだけ忠実に移植-------------------------------
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.optimize import minimize

def run_mle_and_aic(df, x_col='year', y_col='log_conc'):
    """MLE + AIC の実装は、以下の関数 run_mle_and_aic_simple 内で行われています。
    この関数は、与えられたデータフレームから年と対数濃度の列を抽出し、線形モデルのパラメータを最大化するための最適化ルーチンを実行します。
    最終的に、傾き、切片、AIC値、および線形モデルの有意性を示す指標を返します。"""   
    safe_f_substance =df["safe_f_substance"].iloc[0]
    substance =df["substance"].iloc[0]
    try:
        a, b, aic0, aic1, p1, x_base, linear_validity, b1_corrected = run_mle_and_aic_simple(df)
    except Exception as e:
        return {"error": f"MLE 実行中にエラー: {e}"}

    return {

        "safe_f_substance":safe_f_substance,
        "substance":substance,
        "analysis_type": "mle_aic",
        "slope_mle": a,
        "intercept_mle": b,

        "aic_const": aic0,
        "aic_linear": aic1,
        "p_linear": p1,
        "linear_validity": linear_validity,
        "intercept_b1_corrected": b1_corrected,
        #"Rhat":rhat
    }



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

"""
重複している関数は、run_mle_and_aic_simple 内で定義されている neg_log_likelihood と neg_log_likelihood_0 です。
これらは MLE の最適化に使用される目的関数であり、kde_log_likelihood_from_residuals_plain を内部で呼び出す形になっています。
この構造により、MLE の最適化ルーチンは、残差の KDE に基づく対数尤度を最大化するための目的関数を使用して、線形モデルのパラメータを推定します。"""


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



# ---------------------------
# 事前分布
# ---------------------------
# ここでは、a と b に対して、平均0、標準偏差2の正規分布を事前分布として使用する例を示します。
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
    
def log_prior_const(params, data_scale=None):

    b = params[0]

    # 範囲制限
    if not (-100 < b < 100):
        return -np.inf

    # 弱情報事前分布
    if data_scale is None:
        b_sd = 2.0
    else:
        y_std = data_scale.get("y_std", 1.0)
        b_sd = max(0.1, y_std * 2.0)

    return -(b**2) / (2 * b_sd**2)

# ---------------------------
# 事後分布（通常 / tempered）
# ---------------------------
# ここでは、通常の事後分布と、温度付き事後分布の両方を定義します。温度付き事後分布は、ベータ値を使用して尤度の影響を調整します。
def log_posterior_fast(params, x_centered, y, lookup, data_scale=None):
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + ll

def tempered_log_posterior(
    params,
    beta,
    x_centered,
    y,
    lookup,
    data_scale=None,
    model_type="linear",
):

    # --- 線形モデル ---
    if model_type == "linear":

        lp = log_prior_informative(
            params,
            data_scale=data_scale
        )

        if not np.isfinite(lp):
            return -np.inf

        a, b = params

        ll = fast_kde_log_likelihood_from_residuals(
            a, b, x_centered, y, lookup
        )

        return lp + beta * ll

    # --- 定数モデル ---
    elif model_type == "constant":

        lp = log_prior_const(
            params,
            data_scale=data_scale
        )

        if not np.isfinite(lp):
            return -np.inf

        b = params[0]

        ll = fast_kde_log_likelihood_from_residuals(
            0.0, b, x_centered, y, lookup
        )

        return lp + beta * ll

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def tempered_log_posterior_old(
    params,
    beta,
    x_centered,
    y,
    lookup,
    data_scale=None,
    model_type="linear",
):
    # 事前
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf

    # --- 線形モデル ---
    if model_type == "linear":
        a, b = params
        ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
        return lp + beta * ll

    # --- 定数モデル ---
    elif model_type == "constant":
        # params = [b] の 1 パラメータ
        b = params[0]
        # 線形項 a=0 として扱う
        ll = fast_kde_log_likelihood_from_residuals(0.0, b, x_centered, y, lookup)
        return lp + beta * ll

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    
def run_mle_and_aic_simple(df, max_iter=20, tolerance=1e-6):
    #st向けに改変
    np.set_printoptions(precision=10, suppress=True)
    x_raw = df['year'].values
    #x_base = x_raw.min()
    x_base = x_raw.mean()
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
            print(f"Warning: KDE build error in iteration {i}. Skipping.")
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
            print(f"The iteration converged at {i+1}.")
            a1, b1 = a_new, b_new
            break
        
        # e. 更新
        a_current, b_current = a_new, b_new
        kde_current = kde_new
        
    else: # forループが max_iter に達した場合
        print(f"Warning: The maximum number of iterations {max_iter} has been reached, but convergence has not been achieved.")
        a1, b1 = a_current, b_current
        
    # 3. 定数モデル M0 の推定 (反復は不要)
    # ※ neg_log_likelihood_0 関数が存在すると仮定
    try:
        result_0 = minimize(neg_log_likelihood_0, [np.mean(y)], args=(x_centered, y), method='L-BFGS-B')
        max_log_likelihood_0 = -result_0.fun

    except Exception:
        # エラー処理
        print("Failed to optimize the constant model.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_base

    # 4. AICとP1の計算 (ここからは元のコードと同じ)
    aic1 = -2 * max_log_likelihood_1 + 2 * 2
    aic0 = -2 * max_log_likelihood_0 + 2 * 1
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator
    p1 = numerator / denominator if denominator != 0 else (0.0 if delta_aic > 0 else 1.0)
    
    #linear_validity = (p1 > 0.95) and (a1 < 0)
    linear_validity = bool((p1 > 0.95) and (a1 < 0))
    x_mean = x_raw.mean()
    b1_corrected = b1 - a1 * (x_base - x_mean) #baseのx_baseをこちらに合わせたので、もはや不要。b1と同じ値になるので、削除しべし

    #return a1, b1, aic0, aic1, p1, x_base, linear_validity
    return a1, b1, aic0, aic1, p1, x_base, linear_validity, b1_corrected


import datetime
#timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#-FFT-KDE ベイズ------内容を忠実に移植-------------------------------
# FFT-KDE ベイズの実装は、以下の関数 run_bayes_kde_auto_stop_fast 内で行われています。
# この関数は、与えられたデータフレームから年と対数濃度の列を抽出し、FFT-KDE を使用して残差の分布を推定しながら、MCMC を実行して線形モデルのパラメータをサンプリングします。さらに、MCMC の収束を自動的に判定するロジックも含まれています。

import time
import emcee

def compute_Rhat(chains, axis=0):

    try:
        n_chains, n_samples, n_params = chains.shape

        chain_means = np.mean(chains, axis=1)
        chain_vars  = np.var(chains, axis=1, ddof=1)

        W = np.mean(chain_vars, axis=0)

        B = n_samples * np.var(
            chain_means,
            axis=0,
            ddof=1
        )

        var_hat = (
            ((n_samples - 1) / n_samples) * W
            + (1 / n_samples) * B
        )

        # ← ここに移動
        W = np.maximum(W, 1e-12)
        var_hat = np.maximum(var_hat, 1e-12)

        Rhat = np.sqrt(var_hat / W)

        print(f"DEBUG Rhat values: {Rhat}")
        print("chain_vars.shape =", chain_vars.shape)
        print("W =", W)
        print("B =", B)
        print("var_hat =", var_hat)

        return Rhat

    except Exception as e:

        print("Rhat ERROR:", e)

        return np.array([np.nan] * chains.shape[-1])
    
def compute_Rhat_old(chains, axis=0):
    W = max(W, 1e-12)
    var_hat = max(var_hat, 1e-12)
    try:
        n_chains, n_samples, n_params = chains.shape
        chain_means = np.mean(chains, axis=1)
        chain_vars = np.var(chains, axis=1, ddof=1)
        W = np.mean(chain_vars, axis=0)
        B = n_samples * np.var(chain_means, axis=0, ddof=1)
        var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
        Rhat = np.sqrt(var_hat / W)
        print(f"DEBUG Rhat values: {Rhat}")
        print(chain_vars.shape)
        print(W)
        print(B)
        print(var_hat)
        return Rhat
    except Exception:
        return np.array([np.nan] * chains.shape[-1])
    
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
    #x_base = years.min()
    x_base = years.mean()
    x_centered = years - x_base
    y = df["log_conc"].values.astype(float)
    ndim = 2

    # MLE で初期値------
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

    # lookup 準備-------
    if fixed_lookup is None:
        residuals_mle = y - (a_mle * x_centered + b_mle)
        lookup = prepare_fft_kde_lookup(
            residuals_mle,
            bw_factor=cfg.bw_factor,
            grid_size=cfg.grid_size_lookup,
            use_loo=cfg.mode_loo,
        )
        if verbose:
            print("DEBUG lookup@run_bayes_kde_auto_stop_fast", type(lookup), "Rhat (base MLE):", mle_res.get("rhat_a") if mle_res is not None else "N/A")
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
                print(f"  → Candidates for convergence ({stable_count}/{stable_rounds})")
        else:
            stable_count = 0

        if stable_count >= stable_rounds:
            if verbose:
                print("######## MCMC terminated due to automatic early convergence detection ########")
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
        "config": cfg, 
    }
    return result

def run_fft_bayes(df, cfg, x_col='year', y_col='log_conc'):
    safe_f_substance =df["safe_f_substance"].iloc[0]
    substance =df["substance"].iloc[0]
    try:
        result = run_bayes_kde_auto_stop_fast(
            df,
            cfg=cfg,
            fixed_lookup=None,
            verbose=False
        )
    except Exception as e:
        return {"error": f"FFT-KDE ベイズ実行中にエラー: {e}"}

    return {

        "safe_f_substance":safe_f_substance,
        "substance": substance,
        "analysis_type": "fft_bayes",

        "samples": result.get("samples"),   # ★ 追加
        "a_mean": round(float(result.get("a_mean")),4),
        "b_mean": round(float(result.get("b_mean")),4),
        #"b_mean": result.get("b_mean"),
        "ci_a": format_ci(result.get("a_ci")),
        "ci_b": format_ci(result.get("b_ci")),
        "Rhat": format_ci(result.get("Rhat")),       
        #"ci_a": result.get("a_ci").tolist() if result.get("a_ci") is not None else None,
        #"ci_b": result.get("b_ci").tolist() if result.get("b_ci") is not None else None,
        #"Rhat": result.get("Rhat").tolist() if result.get("Rhat") is not None else None,

        "p_neg_slope": result.get("p_neg_slope"),
        "p_pos_slope": 1 - result.get("p_neg_slope", np.nan),

        #"n_iter": result.get("n_iter"),
        #"converged": result.get("converged")
        "steps_completed": int(result.get("steps_completed", 0)),
        "stopped_early": bool(result.get("stopped_early", False)),
        "mcmc_summary": result.get("mcmc_summary"),
        "config": cfg, 
    }


# ---------------------------
# β ごとの E_beta[logL] 推定
    #MCMC はここで 1 回だけ実行
    #burn-in 後の chain を flat=False で取得
    #Rhat をここで計算
    #E[logL] もここで計算
    #compute_logL_at_beta() では何も追加計算しない
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
    model_type="linear",
):
    
    if model_type == "linear":
        ndim = 2
    elif model_type == "constant":
        ndim = 1  # 定数モデルも a と b の2パラメータで扱う（aは線形項として0に固定される）


    if init_center is None:
        #init_center = np.array([0.0, np.mean(y)])
        if model_type == "linear":
            init_center = np.array([0.0, np.mean(y)])

        elif model_type == "constant":
            init_center = np.array([np.mean(y)])

    rng = np.random.default_rng()
    initial_pos = init_center + 1e-5 * rng.normal(size=(cfg.n_walkers, ndim))

    def _logprob(theta):
        return tempered_log_posterior(
            theta, beta, x_centered, y, lookup, data_scale=data_scale, model_type=model_type
        )

    sampler = emcee.EnsembleSampler(cfg.n_walkers, ndim, _logprob)
    sampler.run_mcmc(initial_pos, cfg.n_steps, progress=show_progress)

    # --- burn-in 後のチェーン（flat=False が重要） ---
    chain = sampler.get_chain(discard=int(cfg.n_steps * cfg.burn_frac), flat=False)
    chain3d = np.transpose(chain, (1, 0, 2))  # (walkers, steps, params)

    # --- Rhat ---
    rhat_val = compute_Rhat(chain3d)

    # --- E[logL] ---

    chain_flat = chain.reshape(-1, chain.shape[-1])
    """    
    ll_vals = np.array([
        fast_kde_log_likelihood_from_residuals(th[0], th[1], x_centered, y, lookup)
        for th in chain_flat
    ])
    """
    if model_type == "linear":
        ll_vals = np.array([
            fast_kde_log_likelihood_from_residuals(
                th[0], th[1], x_centered, y, lookup
            )
            for th in chain_flat
        ])

    elif model_type == "constant":
        ll_vals = np.array([
            fast_kde_log_likelihood_from_residuals(
                0.0, th[0], x_centered, y, lookup
            )
            for th in chain_flat
        ])

    E = np.mean(ll_vals)
    var_logL = np.var(ll_vals)

    return E, var_logL, rhat_val     #format_ci(rhat_val)



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
def interpret_bayes_factor(logBF=None, BF=None):

    if logBF is None and BF is not None:

        BF = max(BF, 1e-300)
        logBF = np.log(BF)

    elif BF is None and logBF is not None:

        if logBF > 700:
            BF = np.inf
        elif logBF < -700:
            BF = 0.0
        else:
            BF = np.exp(logBF)

    else:
        raise ValueError(
            "Please specify either logBF or BF"
        )

    # interpretation
    if logBF < 0:
        interp = "Supports constant model"
    elif logBF < 1:
        interp = "Weak evidence"
    elif logBF < 2.5:
        interp = "Moderate evidence"
    elif logBF < 5:
        interp = "Strong evidence"
    else:
        interp = "Decisive evidence"

    return {
        "BF": BF,
        "logBF": logBF,
        "interpretation": interp
    }
def interpret_bayes_factor_old(logBF=None, BF=None):
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
        raise ValueError("Please specify either logBF or BF")   
    #if logBF is None and BF is not None:
        #logBF = math.log(BF)
        # ベイズ因子 BF の解釈には、統計学で広く用いられる Jeffreys scale（Jeffreys, 1961）の1以下を２区分に改変した。

    # Interpretation of Bayes Factor (BF) based on a modified Jeffreys scale.
    if BF < 0.5:
        interpretation = "BF < 0.5: Evidence in favor of the constant model"
    elif 0.5 <= BF < 1:
        interpretation = "0.5 ≤ BF < 1: No meaningful difference between models (indistinguishable given TI precision)"
    elif 1 <= BF < 3:
        interpretation = "1 ≤ BF < 3: Weak evidence supporting the linear model"
    elif 3 <= BF < 10:
        interpretation = "3 ≤ BF < 10: Moderate evidence supporting the linear model"
    elif 10 <= BF < 30:
        interpretation = "10 ≤ BF < 30: Strong evidence supporting the linear model"
    elif 30 <= BF < 100:
        interpretation = "30 ≤ BF < 100: Very strong evidence supporting the linear model"
    else:
        interpretation = "BF ≥ 100: Decisive evidence supporting the linear model"

    """
    if BF < 0.5:
        interpretation = "BF < 0.5　定数モデル支持"
    elif 0.5 <= BF < 1:
        interpretation = "0.5 <= BF < 1　両モデルに有意な差はない (TI の精度では区別困難）"
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
    """
    return {
        "BF": BF,
        "logBF": logBF,
        "interpretation": interpretation
    }


def bayes_trend_judgement(contains_zero, cmp, p_neg_slope, p_pos_slope):
    # 1. まずBFの強度を取得
    bf_info = interpret_bayes_factor(BF=cmp['BF'])
    bf_label = bf_info['interpretation']
    
    # 2. 基本的なモデル支持の判定
    if cmp['BF'] >= 3:
        model_decision = "Supports linear models."
    elif cmp['BF'] >= 1:
        model_decision = "There are signs of a linear model, but the evidence is inconclusive."
    else:
        model_decision = "Supports the constant model (no trend)."

    # 3. 方向性の判定（確信度が高い場合のみ断定）
    if p_neg_slope >= 0.95:
        direction = "On the decreasing trend."
    elif p_pos_slope >= 0.95:
        direction = "On an increasing trend."
    else:
        direction = "No clear direction is apparent."

    # 4. 信頼区間の情報を付与
    ci_status = "(The confidence interval crosses zero)" if contains_zero else "(The confidence interval does not cross zero)"

    # 全てを統合したメッセージ
    return f"Judgment: BF: {model_decision} P_slope: {direction}  ci:{ci_status} <br> [Strength of evidence (Jeffreys scale): {bf_label}]"


# ---------------------------
# 熱力学積分による logZ
# ---------------------------

def compute_logL_at_beta(df, beta, cfg, model_type='linear'):
    x = df["year"].values.astype(float)
    y = df["log_conc"].values.astype(float)

    lookup = prepare_fft_kde_lookup(
        y,
        bw_factor=cfg.bw_factor,
        grid_size=cfg.grid_size_lookup,
        use_loo=cfg.mode_loo,
    )

    rng = np.random.default_rng()
    # ← これを追加
    init_center = None

    if model_type == "linear":

        ndim = 2

        if init_center is None:
            init_center = np.array([0.0, np.mean(y)])

    elif model_type == "constant":

        ndim = 1

        if init_center is None:
            init_center = np.array([np.mean(y)])

    else:
        raise ValueError(model_type)

    initial_pos = (
        init_center
        + 1e-5 * rng.normal(size=(cfg.n_walkers, ndim))
    )


    E, var_logL, rhat_val = estimate_E_loglike_for_beta(
        beta,
        x - x.min(),
        y,
        lookup,
        cfg=cfg,
        init_center=init_center,       
        data_scale={"x_range": float(x.max() - x.min()), "y_std": float(np.nanstd(y))},
        show_progress=False,
        model_type=model_type,
    )

    return {"E_logL": E, "var_logL": var_logL, "Rhat": rhat_val}



def compute_logZ_from_records(ti_records):
    #st用に新規作成
    betas = np.array([rec["beta"] for rec in ti_records])
    E_logL = np.array([rec["E_logL"] for rec in ti_records])

    # 台形公式で積分
    logZ = np.trapezoid(E_logL, betas)
    return logZ

import matplotlib.pyplot as plt

def plot_beta_rhat(ti_records_linear, ti_records_const):
    beta = [r["beta"] for r in ti_records_linear]
    rhat_lin_a = [r["Rhat"][0] for r in ti_records_linear]
    rhat_lin_b = [r["Rhat"][1] for r in ti_records_linear]

    rhat_const = [r["Rhat"][0] for r in ti_records_const]

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(beta, rhat_lin_a, marker="o", label="Linear model: Rhat(a)")
    ax.plot(beta, rhat_lin_b, marker="o", label="Linear model: Rhat(b)")
    ax.plot(beta, rhat_const, marker="o", label="Constant model: Rhat")

    ax.axhline(1.1, color="red", linestyle="--", label="Rhat=1.1 threshold")

    ax.set_xlabel("β")
    ax.set_ylabel("Rhat")
    ax.set_title("β–Rhat diagnostic")
    ax.legend()
    ax.grid(True)

    return fig

def plot_beta_ElogL(ti_records_linear, ti_records_const):
    beta = [r["beta"] for r in ti_records_linear]
    E_lin = [r["E_logL"] for r in ti_records_linear]
    E_const = [r["E_logL"] for r in ti_records_const]

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(beta, E_lin, marker="o", label="Linear model")
    ax.plot(beta, E_const, marker="o", label="Constant model")

    ax.set_xlabel("β")
    ax.set_ylabel("E[log L]")
    ax.set_title("β–E[logL] curve (TI)")
    ax.legend()
    ax.grid(True)

    return fig


def run_ti(df, cfg, safe_f_substance, x_col='year', y_col='log_conc',
    progress_box=None, table_box=None, progress_bar=None):
    #st用に新規作成
    substance =df["substance"].iloc[0]
    try:
        if progress_box:
            progress_box.write("Step 1/3: Calculating the FFT-KDE Bayesian posterior distribution....")

        bayes_res = run_fft_bayes(df, cfg, x_col=x_col, y_col=y_col)

        a_bayes = bayes_res.get("a_mean", np.nan)
        b_bayes = bayes_res.get("b_mean", np.nan)
        ci_a = np.array(bayes_res.get("ci_a", [np.nan, np.nan]))
        ci_b = np.array(bayes_res.get("ci_b", [np.nan, np.nan]))
        Rhat_linear_prior = bayes_res.get("Rhat", np.array([np.nan, np.nan]))

        p_neg_slope = bayes_res.get("p_neg_slope", np.nan)
        p_pos_slope = 1.0 - p_neg_slope if not np.isnan(p_neg_slope) else np.nan
        contains_zero = (ci_a[0] <= 0 <= ci_a[1]) if not np.any(np.isnan(ci_a)) else True

        beta_grid = build_beta_grid(cfg)
        n_beta = len(beta_grid)

        ti_records_linear = []
        ti_records_const = []

        table_box_linear = st.empty()
        table_box_const = st.empty()

        if progress_box:
            progress_box.write("Step 2 of 3: Performing TI (thermodynamic integration)...")

        # --- 線形モデル TI ---
        for i, beta in enumerate(beta_grid):
            res = compute_logL_at_beta(df, beta, cfg, model_type='linear')  # 線形モデル版
            ti_records_linear.append({
                "beta": beta,
                "E_logL": res["E_logL"],
                "var_logL": res["var_logL"],
                "Rhat": res["Rhat"],
            })
            table_box_linear.dataframe(pd.DataFrame(ti_records_linear))
            if progress_bar:
                progress_bar.progress((i+1)/n_beta)

        logZ_linear = compute_logZ_from_records(ti_records_linear)
        Rhat_linear = ti_records_linear[-1]["Rhat"]

        # --- 定数モデル TI ---
        for i, beta in enumerate(beta_grid):
            res = compute_logL_at_beta(df, beta, cfg, model_type='constant')  # 定数モデル版に分けても良い
            ti_records_const.append({
                "beta": beta,
                "E_logL": res["E_logL"],
                "var_logL": res["var_logL"],
                "Rhat": res["Rhat"],
            })
            table_box_const.dataframe(pd.DataFrame(ti_records_const))
            if progress_bar:
                progress_bar.progress((i+1)/n_beta)

        logZ_const = compute_logZ_from_records(ti_records_const)
        Rhat_const = ti_records_const[-1]["Rhat"]

        cmp = compare_models_logZ({"logZ": logZ_linear}, {"logZ": logZ_const})
        logBF = cmp["logBF"]
        BF = cmp["BF"]
        BF評価 = interpret_bayes_factor(BF=BF)['interpretation']

        if progress_box:
            progress_box.write("Step 3/3: TI Complete! Displaying graphs...")

        fig_elogl = plot_beta_ElogL(  ti_records_linear, ti_records_const)
        st.pyplot(fig_elogl)
        fig_rhat = plot_beta_rhat(ti_records_linear, ti_records_const)       
        st.pyplot(fig_rhat)

        #グラフの保存
        os.makedirs("outputs/ti", exist_ok=True)
        #import re
        #cfg_str = str(cfg)
        #safe_cfg = re.sub(r'[\\/:"*?<>|,\'.]', '_', cfg_str)
        #safe_cfg = re.sub(r'[^A-Za-z0-9_.-]+', '_', cfg_str)

        cfg_name = (
        f"loo{int(cfg.mode_loo)}"
        f"_gs{cfg.grid_size_lookup}"
        f"_bw{cfg.bw_factor}"
        f"_nw{cfg.n_walkers}"
        f"_ns{cfg.n_steps}"
        f"_sched_{cfg.beta_schedule}"
        )
        print("DEBUG safe_substance at gaph:",safe_f_substance)
        #print("DEBUG safe_substance:",df['safe_f_substance'].iloc[0])
        #fig_elogl.savefig(f"outputs/ti/{cfg_name}_{df['safe_substance'].iloc[0]}_elogl_plot.png", dpi=300, bbox_inches="tight")
        #fig_rhat.savefig(f"outputs/ti/{cfg_name}_{df['safe_substance'].iloc[0]}_rhat_plot.png", dpi=300, bbox_inches="tight")
        fig_elogl.savefig(f"outputs/ti/{cfg_name}_{safe_f_substance}_elogl_plot.png", dpi=300, bbox_inches="tight")
        fig_rhat.savefig(f"outputs/ti/{cfg_name}_{safe_f_substance}_rhat_plot.png", dpi=300, bbox_inches="tight")

        return {

            "safe_f_substance": safe_f_substance,
            "substance": substance,
            "analysis_type": "ti",
            "a_mean": a_bayes,
            "b_mean": b_bayes,
            "ci_a": ci_a,
            "ci_b": ci_b,
            "p_neg": p_neg_slope,
            "p_pos": p_pos_slope,

            "logZ_linear": logZ_linear,
            "logZ_const": logZ_const,
            "logBF": logBF,
            "BF": BF,
            "BF評価": BF評価,
            "contains_zero": contains_zero,

            "Rhat_linear": Rhat_linear,
            "Rhat_const": Rhat_const,

            "ti_records_linear": ti_records_linear,
            "ti_records_const": ti_records_const,
            "cfg": cfg,
        }

    except Exception as e:
        if progress_box:
            progress_box.write(f"エラー発生: {e}")
        return {"error": str(e)}
    
def extract_rhat_max_list(rhat_value):
    """If `Rhat` is a list, it returns the maximum value. If it is a float, it returns the value as is."""
    if isinstance(rhat_value, (list, tuple)):
        return max(rhat_value)
    return float(rhat_value)
#import numpy as np

def extract_rhat_max(rhat_value):
    """If `Rhat` is a list, tuple, or ndarray, the maximum value is returned. If it is a float, the value is returned as-is."""
    if isinstance(rhat_value, (list, tuple, np.ndarray)):
        return float(np.max(rhat_value))
    return float(rhat_value)

def evaluate_ti_rhat(df_lin, df_const, verbose=True):
    # Rhat を float に変換
    df_lin["Rhat_max"] = df_lin["Rhat"].apply(extract_rhat_max)
    df_const["Rhat_max"] = df_const["Rhat"].apply(extract_rhat_max)

    # β=1 の Rhat
    rhat_lin_final = df_lin[df_lin["beta"] == 1]["Rhat_max"].iloc[0]
    rhat_const_final = df_const[df_const["beta"] == 1]["Rhat_max"].iloc[0]

    # 全体の最大 Rhat
    rhat_max_overall = max(df_lin["Rhat_max"].max(), df_const["Rhat_max"].max())

    # 判定
    def judge(r):
        if r <= 1.01:
            return " <= 1.01 OK (converged well)"
        elif r <= 1.05:
            return "<= 1.05 Note (nearly converged)）"
        elif r <= 1.1:
            return "<= 1.1 Warning (Insufficient convergence)"
        else:
            return ">1.1 Danger (Convergence Failed)"

    result = {
        "Rhat_linear_final": rhat_lin_final,
        "Rhat_const_final": rhat_const_final,
        "Rhat_max_overall": rhat_max_overall,
        "judge_linear": judge(rhat_lin_final),
        "judge_const": judge(rhat_const_final),
        "judge_overall": judge(rhat_max_overall),
    }

    if verbose:
        print("=== TI Rhat 評価 ===")
        print(f"線形モデル β=1 Rhat: {rhat_lin_final:.4f} → {judge(rhat_lin_final)}")
        print(f"定数モデル β=1 Rhat: {rhat_const_final:.4f} → {judge(rhat_const_final)}")
        print(f"全βの最大 Rhat: {rhat_max_overall:.4f} → {judge(rhat_max_overall)}")

    return result
def rhat_dict_to_html(d):
    html = []
    html.append("<ul>")
    for k, v in d.items():
        html.append(f"<li><b>{k}</b>: {v}</li>")
    html.append("</ul>")
    return "\n".join(html)

def format_ci(ci, digits=4):
    """信用区間を安全に丸めてリスト形式に統一"""
    try:
        arr = np.asarray(ci, dtype=float)
        if arr.size == 2 and not np.isnan(arr).any():
            return [round(float(arr[0]), digits), round(float(arr[1]), digits)]
    except Exception:
        pass
    return [np.nan, np.nan]

import os
from pathlib import Path

def generate_html_report(df, ols_res, mle_res, bayes_res, ti_res, plot_path, out_path, safe_f_substance):

    html = []
    filename = out_path.replace("outputs/", "").replace(".html", "").split("_")[0]
    html.append("<html><head><meta charset='utf-8'>")
    html.append(f"<title>{safe_f_substance} Regression Analysis Report</title></head><body>")
    html.append(f"<h1>{safe_f_substance}<br> Regression Analysis Report</h1>")

    #データへの警告
    years = df["year"].values
    if len(np.unique(years)) < 2:
        # 年次変化を評価できない
        html.append("<h2>Since there is only one year of data, we cannot assess the trend.</h2>")

    else:
        html.append("<h2>Linear Trend Analysis</h2>")
    # --- グラフ画像 ---
    html.append(f"<img src='{plot_path}' width='600'><br><br>")

    """
    if os.path.exists(f"outputs/{plot_path}"):
        html.append(f"<img src='{plot_path}' width='600'><br><br>")
        #print(f"DEBUG: グラフ画像のパス: {plot_path}, 存在: {str(os.path.exists(plot_path))}")
    else:
        #html.append(f"<p>グラフ画像が見つかりません。{plot_path}, 存在: {os.path.exists(plot_path)}</p>")  
        html.append(f"<img src='{plot_path}' width='600'><br><br>")"""
    

    # --- Data Overview ---
    html.append("<h2>Data Overview</h2>")
    html.append(f"Number of data points: {len(df)}<br>")
    html.append(f"year: {df['year'].min()} ～ {df['year'].max()}<br>")
    html.append(f"filename: {filename}.csv <br>")

    #html.append(f"substance: {df['substance'].iloc[0]}<br>")
    html.append(f"safe_f_substance: {safe_f_substance}<br>")    
    # --- OLS ---
    if ols_res and ols_res.get("safe_f_substance") == safe_f_substance:
    #if ols_res and safe_f_substance in st.session_state["ols_results"]:
        html.append("<h2>OLS Regression</h2>")
        html.append(f"Slope a: {ols_res['ols_a1']:.4f}<br>")
        html.append(f"Intercept b: {ols_res['ols_b1']:.4f}<br>")
        html.append(f"p-value: {ols_res['ols_p1']:.4g}<br>")
        html.append(f"R²: {ols_res['r2_ols']:.4f}<br>") 
       
        html.append(f"summary {safe_f_substance}: {ols_res['summary_html']}<br>")     

    # --- MLE ---
    if mle_res and ("slope_mle" in mle_res)  and mle_res.get("safe_f_substance") == safe_f_substance:
    #if mle_res and safe_f_substance in st.session_state["mle_results"]:
        html.append("<h2>MLE + AIC (MOE-style)</h2>")
        html.append(f"Slope a: {mle_res['slope_mle']:.4f}<br>")
        html.append(f"Intercept b (centered-x): {mle_res['intercept_mle']:.4f}<br>")
        html.append(f"AIC(Constant model): {mle_res['aic_const']:.2f}<br>")
        html.append(f"AIC(Linear model): {mle_res['aic_linear']:.2f}<br>")
        html.append(f"Akaike weight for linear model: {mle_res['p_linear']:.3f}<br>")
     
    # --- Bayes ---
    #if bayes_res:
    if bayes_res and  bayes_res.get("safe_f_substance") == safe_f_substance:
        html.append("<h2>FFT-KDE Bayesian</h2>")
       
        html.append(f"Posterior mean of slope a: {bayes_res['a_mean']:.4f}<br>")
        html.append(f"95% credible interval (a): {bayes_res['ci_a']}<br>")     
        html.append(f"P(a&lt;0): {bayes_res['p_neg_slope']:.4f}<br><br>")
        #html.append(f"P(a&gt;0): {bayes_res['p_pos_slope']:.4f}<br>") 

        html.append(f"Intercept b (centered-x mean): {bayes_res['b_mean']:.4f}<br>")
        html.append(f"95% credible interval (b): {bayes_res['ci_b']}<br><br>")
 

        html.append(f"Rhat (a, b): {bayes_res['Rhat']}<br> ")
        #html.append(f"{bayes_res.get('mcmc_summary','')}<br> ") 


    # --- TIによるモデル比較 ---
    if ti_res is not None:
        
        cfg=ti_res['cfg']
        cfg_name = (
        f"loo{int(cfg.mode_loo)}"
        f"_gs{cfg.grid_size_lookup}"
        f"_bw{cfg.bw_factor}"
        f"_nw{cfg.n_walkers}"
        f"_ns{cfg.n_steps}"
        f"_sched_{cfg.beta_schedule}")
        

        html.append(f"The confidence interval crosses zero: {ti_res['contains_zero']}<br>")  

        html.append("<h2>Model Comparison Using TI (Thermodynamic Integration)</h2>")

        html.append(f"TI Settings: {ti_res['cfg']}<br>")        
        html.append(f"logZ(linear): {ti_res['logZ_linear']:.4f}<br>")
        html.append(f"logZ(const): {ti_res['logZ_const']:.4f}<br>")
        html.append(f"logBF: {ti_res['logBF']:.4f}<br>")
        html.append(f"BF: {ti_res['BF']:.3e}<br>")
        html.append(f"Rating: {ti_res['BF評価']}<br>")
        #TIの詳細
        df_lin = pd.DataFrame(ti_res["ti_records_linear"])
        df_const = pd.DataFrame(ti_res["ti_records_const"])
        eval_rhat = evaluate_ti_rhat(df_lin, df_const, verbose=True)
        eval_rhat = evaluate_ti_rhat(df_lin, df_const, verbose=True)
        html.append(rhat_dict_to_html(eval_rhat))
        #html.append(f"Rhat評価: {eval_rhat}<br>")
        #plotを追加
        html.append(f"<img src='ti/{cfg_name}_{df['safe_f_substance'].iloc[0]}_elogl_plot.png' width='600'><br><br>")
        html.append(f"<img src='ti/{cfg_name}_{df['safe_f_substance'].iloc[0]}_rhat_plot.png' width='600'><br><br>")
        html.append("<h3>TI β–E[logL] (linear model)</h3>")
        html.append(df_lin.to_html(index=False,formatters={
            "E_logL": "{:.2f}".format,
            "var_logL": "{:.2e}".format,
        }))
        html.append("<h3>TI β–E[logL] (Constant Model)</h3>")
        html.append(df_const.to_html(index=False, formatters={
            "E_logL": "{:.2f}".format,
            "var_logL": "{:.2e}".format,
        }))


    html.append("</body></html>")
    #Rhatでcausion

    Path(out_path).write_text("\n".join(html), encoding="utf-8")
    file_name =safe_f_substance.split("_",1)[0]
    substance =safe_f_substance.split("_",1)[1]
    fig, plot_path = plot_regression_results(
        df,
        substance,
        file_name,
        safe_f_substance,
        ols_res,
        mle_res,
        bayes_res,
    )

    st.pyplot(fig)

    plt.close(fig)    

    return out_path

def plot_regression_results(
    df,
    substance,
    file_name,
    safe_f_substance,
    ols_result,
    mle_result,
    bayes_result,
):

    x = df["year"].values
    y = df["log_conc"].values

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(x, y, s=10, alpha=0.4, label="data")

    # OLS
    if ols_result:
        a = ols_result["ols_a1"]
        b = ols_result["ols_b1"]

        ax.plot(
            x,
            a * (x - x.mean()) + b,
            color="green",
            label="OLS",
        )

    # MLE
    if (
        mle_result
        and "slope_mle" in mle_result
        and "intercept_mle" in mle_result
    ):

        a = mle_result["slope_mle"]
        b = mle_result["intercept_mle"]

        ax.plot(
            x,
            a * (x - x.mean()) + b,
            color="orange",
            label="MLE",
        )
    """  
    else:
        ax.text(
            0.05,
            0.9,
            "MLEなし",
            transform=ax.transAxes,
            color="orange"
        )"""  

    # Bayes
    if bayes_result:

        a = bayes_result["a_mean"]
        b = bayes_result["b_mean"]

        ax.plot(
            x,
            a * (x - x.mean()) + b,
            color="red",
            label="Bayes",
        )

    ax.legend()

    ax.set_title(f"{file_name}: {substance}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("outputs/plots_regression", exist_ok=True)

    plot_path = (
        f"plots_regression/"
        f"{safe_f_substance}_regression_{timestamp}.png"
    )

    fig.savefig(
        f"outputs/{plot_path}",
        dpi=150,
        bbox_inches="tight"
    )

    return fig, plot_path
"""#
fig, plot_path = plot_regression_results(
    df,
    substance,
    file_name,
    safe_f_substance,
    ols_result,
    mle_result,
    bayes_result,
)

st.pyplot(fig)

plt.close(fig)
"""