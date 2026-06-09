#run_analysis.py


from turtle import st

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

# st.button("OLS 実行"):
def run_ols_analysis_ok(df, x_col='year', y_col='log_conc'):

    temp_df = df[[x_col, y_col]].copy()
    temp_df[x_col] = pd.to_numeric(temp_df[x_col], errors='coerce')
    temp_df[y_col] = pd.to_numeric(temp_df[y_col], errors='coerce')
    temp_df = temp_df.dropna()

    if len(temp_df) < 2:
        return {"error": "有効なデータが不足しています"}

    x_raw = temp_df[x_col].values
    y = temp_df[y_col].values

    x_centered = x_raw - x_raw.mean()

    X = pd.DataFrame({
        "const": 1.0,
        "x_centered": x_centered.astype(float)
    })

    model = sm.OLS(y, X).fit()

    ci = model.conf_int(alpha=0.05)
    ci_b = ci.loc["const"].values
    ci_a = ci.loc["x_centered"].values

    return {
        "slope": model.params["x_centered"],
        "intercept": model.params["const"],
        "p_slope": model.pvalues["x_centered"],
        "r2": model.rsquared,
        "ci_slope": ci_a,
        "ci_intercept": ci_b,
        "aic": model.aic,
        "loglik": model.llf,
        "summary": model.summary().as_text()
    }
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
        'loglik': ols_model.llf,
        'summary': ols_model.summary().as_text()
    }


"""MLE + AIC の実装は、以下の関数 run_mle_and_aic_simple 内で行われています。この関数は、与えられたデータフレームから年と対数濃度の列を抽出し、線形モデルのパラメータを最大化するための最適化ルーチンを実行します。最終的に、傾き、切片、AIC値、および線形モデルの有意性を示す指標を返します。"""
#-MLE + AIC------内容を忠実に移植-------------------------------
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.optimize import minimize

def run_mle_and_aic(df, x_col='year', y_col='log_conc'):
    try:
        a, b, aic0, aic1, p1, x_base, linear_validity, b1_corrected = run_mle_and_aic_simple(df)
    except Exception as e:
        return {"error": f"MLE 実行中にエラー: {e}"}

    return {
        "slope_mle": a,
        "intercept_mle": b,

        "aic_const": aic0,
        "aic_linear": aic1,
        "p_linear": p1,
        "linear_validity": linear_validity,
        "intercept_b1_corrected": b1_corrected
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
    try:
        kde = gaussian_kde(residuals)
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


def tempered_log_posterior(params, beta, x_centered, y, lookup, data_scale=None):
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + beta * ll
    
def run_mle_and_aic_simple(df, max_iter=20, tolerance=1e-6):
    
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
    
    #linear_validity = (p1 > 0.95) and (a1 < 0)
    linear_validity = bool((p1 > 0.95) and (a1 < 0))
    x_mean = x_raw.mean()
    b1_corrected = b1 - a1 * (x_base - x_mean)

    #return a1, b1, aic0, aic1, p1, x_base, linear_validity
    return a1, b1, aic0, aic1, p1, x_base, linear_validity, b1_corrected


import datetime
#timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#-FFT-KDE ベイズ------内容を忠実に移植-------------------------------
# FFT-KDE ベイズの実装は、以下の関数 run_bayes_kde_auto_stop_fast 内で行われています。この関数は、与えられたデータフレームから年と対数濃度の列を抽出し、FFT-KDE を使用して残差の分布を推定しながら、MCMC を実行して線形モデルのパラメータをサンプリングします。さらに、MCMC の収束を自動的に判定するロジックも含まれています。

import time
import emcee

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

def run_fft_bayes(df, cfg, x_col='year', y_col='log_conc'):
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
        "samples": result.get("samples"),   # ★ 追加
        "a_mean": result.get("a_mean"),
        "b_mean": result.get("b_mean"),
        #"ci_a": result.get("a_ci"),
        #"ci_b": result.get("b_ci"),
        #"rhat": result.get("Rhat"),       
        "ci_a": result.get("a_ci").tolist() if result.get("a_ci") is not None else None,
        "ci_b": result.get("b_ci").tolist() if result.get("b_ci") is not None else None,
        "rhat": result.get("Rhat").tolist() if result.get("Rhat") is not None else None,

        "p_neg_slope": result.get("p_neg_slope"),
        "p_pos_slope": 1 - result.get("p_neg_slope", np.nan),

        #"n_iter": result.get("n_iter"),
        #"converged": result.get("converged")
        "steps_completed": int(result.get("steps_completed", 0)),
        "stopped_early": bool(result.get("stopped_early", False)),
        "mcmc_summary": result.get("mcmc_summary"),
    }
import os
from pathlib import Path

def generate_html_report(df, ols_res, mle_res, bayes_res, plot_path, out_path):
    html = []
    html.append("<html><head><meta charset='utf-8'>")
    html.append("<title>回帰分析レポート</title></head><body>")
    html.append("<h1>回帰分析レポート</h1>")

    # --- グラフ画像 ---
    """
    if os.path.exists(plot_path):
        html.append(f"<img src='{plot_path}' width='600'><br><br>")
        #html.append(f"<img src='plots_regression/regression_plot.png' width='600'><br><br>")
    else:
        html.append("<p><i>グラフ画像が見つかりませんでした。</i></p>")"""
    
    html.append(f"<img src='{plot_path}' width='600'><br><br>")
    # --- データ概要 ---
    html.append("<h2>データ概要</h2>")
    html.append(f"データ数: {len(df)}<br>")
    html.append(f"year: {df['year'].min()} ～ {df['year'].max()}")

    # --- OLS ---
    if ols_res:
        html.append("<h2>OLS 回帰</h2>")
        html.append(f"傾き a: {ols_res['ols_a1']:.4f}<br>")
        html.append(f"切片 b: {ols_res['ols_b1']:.4f}<br>")
        html.append(f"p値: {ols_res['ols_p1']:.4g}<br>")
        html.append(f"R²: {ols_res['r2_ols']:.4f}<br>")

    # --- MLE ---
    if mle_res:
        html.append("<h2>MLE + AIC</h2>")
        html.append(f"傾き a: {mle_res['slope_mle']:.4f}<br>")
        html.append(f"切片 b (raw): {mle_res['intercept_mle']:.4f}<br>")
        html.append(f"切片 b (補正後): {mle_res['intercept_b1_corrected']:.4f}<br>")
        html.append(f"AIC(定数): {mle_res['aic_const']:.2f}<br>")
        html.append(f"AIC(線形): {mle_res['aic_linear']:.2f}<br>")
        html.append(f"線形モデル重み p1: {mle_res['p_linear']:.3f}<br>")

    # --- Bayes ---
    if bayes_res:
        html.append("<h2>FFT-KDE ベイズ</h2>")
        html.append(f"傾き a (平均): {bayes_res['a_mean']:.4f}<br>")
        html.append(f"切片 b (平均): {bayes_res['b_mean']:.4f}<br>")
        html.append(f"95% CI (a): {bayes_res['ci_a']}<br>")
        html.append(f"95% CI (b): {bayes_res['ci_b']}<br>")
        html.append(f"P(a&lt;0): {bayes_res['p_neg_slope']:.3f}<br>")
        html.append(f"Rhat: {bayes_res['rhat']}<br> ")
        html.append(f"<pre>{bayes_res.get('mcmc_summary','')}</pre>")

    html.append("</body></html>")

    Path(out_path).write_text("\n".join(html), encoding="utf-8")
    return out_path
