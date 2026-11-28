# pops_fft_kde_analysis_v1.2.py
# v1.1b のリファクタリング版（解析ロジック・出力形式は維持）
#
# 構成:
#   0. 設定
#   1. 共通ユーティリティ
#   2. KDE 尤度 / MLE / OLS / AIC
#   3. ベイズ (FFT-KDE + emcee) / 熱力学積分 (TI)
#   4. ブートストラップ / 二項ロジスティック回帰
#   5. データ読込 / プロット / レポート
#   6. 実行部 (main)
from __future__ import annotations

import json
import math
import os
import time
import warnings

import emcee
import japanize_matplotlib  # noqa: F401  (日本語フォント設定のため import のみ)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.integrate import trapezoid  # scipy.stats.trapezoid(台形分布)とは別物
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from PY_MOE_KDE_AIC.PY_pops_moe_analysis_plus.analysis.fft_kde_common_engine import (
    fast_kde_log_likelihood,
    prepare_fft_kde_lookup,
)

# =====================================================================
# 0. 設定
# =====================================================================
VERSION = "script: pops_fft_kde_analysis_v1.2.py (refactored from v1.1b), 2026 (H.S.)"

#DEFAULT_FILENAMES = ["大気温暖期", "大気寒冷期", "水質", "底質", "生物"]
DEFAULT_FILENAMES = ["pfos大気温暖期"]
PLOT_DIR = "outputs/plots_fft_kde"
RUN_TI = False          # True で熱力学積分(TI)によるベイズ因子計算を実行
MIN_SAMPLES = 24        # これ未満のデータ数の物質はスキップ
PROB_THRESHOLD = 0.95   # 傾向あり判定の確率しきい値

NO_TI_MESSAGE = "<b> run_ti = True</b>　とすれば、TIによる解析が可能です。"
EMPTY_CMP = {"p_M1": None, "logZ1": None, "logZ0": None, "logBF": None, "BF": None}


# =====================================================================
# 1. 共通ユーティリティ
# =====================================================================
class NpEncoder(json.JSONEncoder):
    """numpy 型を JSON に変換する"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def clean_value_user(val):
    """'tr(0.1)' '---' 'nd' などの表記を数値(または NaN)に変換する"""
    if isinstance(val, str):
        val = val.replace("tr(", "").replace(")", "")
        val = val.replace("---", "")
        val = val.replace("nd", "0")
    try:
        if val == "":
            return np.nan
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def fmt(value, spec=".6f", default="N/A"):
    """None / NaN を安全に整形する（レポート出力用）"""
    try:
        if value is None or np.isnan(value):
            return default
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def ci_str(ci, spec=".6f"):
    return f"[{fmt(ci[0], spec)}, {fmt(ci[1], spec)}]"


def safe_name(substance, filename):
    return (f"{substance}_{filename}"
            .replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_"))


def extract_xy(df):
    """df から (x_base, x_centered, y) を取り出す"""
    years = df["year"].to_numpy(dtype="int64")
    x_base = years.min()
    return x_base, years - x_base, df["log_conc"].to_numpy(dtype=float)


def model_probability_m1(aic1, aic0):
    """AIC から 1次式モデルの選択確率 p1 を計算する（数値安定なシグモイド形式）"""
    return float(expit(-0.5 * (aic1 - aic0)))


# =====================================================================
# 2. KDE 尤度 / MLE / OLS
# =====================================================================
def kde_neg_loglik(residuals):
    """残差の gaussian_kde による負の対数尤度"""
    try:
        kde = gaussian_kde(residuals)
        return -np.sum(np.log(kde.pdf(residuals) + 1e-10))
    except np.linalg.LinAlgError:
        return 1e10


def neg_log_likelihood_linear(params, x, y):
    a, b = params
    return kde_neg_loglik(y - (a * x + b))


def neg_log_likelihood_fixed_kde(params, x, y, kde):
    """固定した KDE（直前の残差から構築）で評価する負の対数尤度"""
    a, b = params
    residuals = y - (a * x + b)
    return -np.sum(np.log(kde.pdf(residuals) + 1e-10))


def neg_log_likelihood_const(params, x, y):
    return kde_neg_loglik(y - params[0])


def kde_log_likelihood_plain(a, b, x_centered, y):
    """gaussian_kde による対数尤度（MCMC 初期値用、1e-12 のフロア付き）"""
    res = y - (a * x_centered + b)
    try:
        return np.sum(np.log(gaussian_kde(res).pdf(res) + 1e-12))
    except Exception:
        return -1e20


def _fit_linear_mle(x, y, max_iter, tolerance, center="median"):
    """1次式モデルの MLE（固定 KDE の反復法）。
    現在の残差から KDE を再構築 → その KDE を固定して (a, b) を最適化 →
    残差の中央値（または平均）を b に繰り込む、を収束まで繰り返す。

    KDE 残差モデルでは残差分布の位置が自由なため、b は尤度上ほぼ不定になる。
    毎反復で残差の位置（center）を 0 に揃えることで b を一意に固定する。
    戻り値: (a, b, 最大対数尤度)
    """
    center_fn = {"median": np.median, "mean": np.mean}[center]
    a, b = np.polyfit(x, y, 1)  # 初期モデル（最小二乗）
    b += center_fn(y - (a * x + b))
    result = None
    for i in range(max_iter):
        # a. 現在のパラメータでの残差から KDE を再構築（バンド幅は自動選択）
        try:
            kde_new = gaussian_kde(y - (a * x + b))
        except np.linalg.LinAlgError:
            print(f"警告: 反復 {i} でKDE構築エラー。スキップします。")
            break

        # b. 固定した KDE のもとで (a, b) を最適化
        result = minimize(neg_log_likelihood_fixed_kde, [a, b],
                          args=(x, y, kde_new), method="L-BFGS-B")
        a_new, b_new = result.x

        # c. 残差の位置を b に繰り込み、b を一意に固定
        b_new += center_fn(y - (a_new * x + b_new))

        # d. 収束判定
        if np.linalg.norm([a_new - a, b_new - b]) < tolerance:
            print(f"反復 {i + 1} で収束しました。")
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    else:
        print(f"警告: 最大反復回数 {max_iter} に達しましたが収束しませんでした。")

    if result is None:
        raise RuntimeError("KDE構築に失敗したため最尤推定できませんでした。")
    # 最終パラメータの対数尤度は、M0 と同じ「自己構築KDE」で評価（位置不変で AIC 比較が整合）
    return a, b, -kde_neg_loglik(y - (a * x + b))


def run_ols(x_centered, y):
    """OLS（正規性仮定の線形解析）"""
    ols = sm.OLS(y, sm.add_constant(x_centered)).fit()
    stderr_b, stderr_a = ols.bse
    ci = ols.conf_int(alpha=0.05)  # 行順: (切片, 傾き)
    return {
        "ols_slope": ols.params[1],
        "ols_intercept": ols.params[0],
        "ols_stderr_a": stderr_a,
        "ols_stderr_b": stderr_b,
        "ols_pvalues": ols.pvalues.tolist(),
        "ols_r2": ols.rsquared,
        "ols_ci_a": ci[1],
        "ols_ci_b": ci[0],
    }


def judge_mle_trend(p1, a):
    if p1 < PROB_THRESHOLD:
        return "MLE: 1次式モデルは選択されない。傾向は不明"
    if a < 0:
        return "MLE: 1次式モデルが選択される。減少傾向あり"
    if a > 0:
        return "MLE: 1次式モデルが選択される。増加傾向あり"
    return "MLE: 1次式モデルが選択される。傾きは 0"


def run_mle_and_aic(df, max_iter=20, tolerance=1e-6):
    """KDE-MLE による 1次式(M1) vs 定数(M0) の AIC 比較 + OLS 補足。dict を返す"""
    np.set_printoptions(precision=10, suppress=True)
    x_base, x_centered, y = extract_xy(df)

    # M1: 1次式
    a1, b1, loglik1 = _fit_linear_mle(x_centered, y, max_iter, tolerance)

    # M0: 定数
    try:
        result_0 = minimize(neg_log_likelihood_const, [np.mean(y)],
                            args=(x_centered, y), method="L-BFGS-B")
        loglik0 = -result_0.fun
    except Exception as e:
        raise RuntimeError("定数モデルの最適化に失敗しました。") from e

    aic1 = -2 * loglik1 + 2 * 2
    aic0 = -2 * loglik0 + 2 * 1
    p1 = model_probability_m1(aic1, aic0)

    ols = run_ols(x_centered, y)
    ci_a, ci_b = ols["ols_ci_a"], ols["ols_ci_b"]
    print(f"傾き a (MLE): {a1:.4f}")
    print(f"切片 b (MLE): {b1:.4f}")
    print(f"aの95%CI (ols): {ci_a[0]:.4f} ～ {ci_a[1]:.4f}")
    print(f"bの95%CI (ols): {ci_b[0]:.4f} ～ {ci_b[1]:.4f}")

    trend_mle = judge_mle_trend(p1, a1)
    if ci_a[0] <= 0 <= ci_a[1]:
        trend_mle += "　OLSによる補足: 傾きの信頼区間に0を含む"
    print(trend_mle)

    return {
        "a1": a1, "b1": b1,
        "aic0": aic0, "aic1": aic1,
        "loglik0": loglik0, "loglik1": loglik1,
        "p1": p1, "x_base": x_base, "n_data": len(y),
        "trend_mle": trend_mle,
        **ols,
    }


# =====================================================================
# 3. ベイズ (FFT-KDE + emcee) / 熱力学積分
# =====================================================================
def log_prior_informative(params, data_scale=None):
    a, b = params
    if not (-100 < a < 100 and -100 < b < 100):
        return -np.inf
    if data_scale is None:
        a_sd = b_sd = 2.0
    else:
        x_range = data_scale.get("x_range", 1.0)
        y_std = data_scale.get("y_std", 1.0)
        a_sd = max(0.1, y_std / max(1e-6, x_range) * 2.0)
        b_sd = max(0.1, y_std * 2.0)
    return -(a ** 2) / (2 * a_sd ** 2) - (b ** 2) / (2 * b_sd ** 2)


def fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup):
    """共通エンジンの FFT-KDE 対数尤度"""
    return fast_kde_log_likelihood([a, b], x_centered, y, lookup)


def tempered_log_posterior(params, beta, x_centered, y, lookup, data_scale=None):
    """log prior + β * logL（β=1 が通常の事後分布）"""
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    return lp + beta * fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)


def log_posterior_fast(params, x_centered, y, lookup, data_scale=None):
    return tempered_log_posterior(params, 1.0, x_centered, y, lookup, data_scale)


def compute_Rhat(chains):
    """Gelman-Rubin R-hat。chains: (n_chains, n_samples, n_params)"""
    try:
        _, n_samples, _ = chains.shape
        W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
        B = n_samples * np.var(np.mean(chains, axis=1), axis=0, ddof=1)
        var_hat = (n_samples - 1) / n_samples * W + B / n_samples
        return np.sqrt(var_hat / W)
    except Exception:
        return np.full(chains.shape[-1], np.nan)


def _initial_positions(a_mle, b_mle, n_walkers, seed=None):
    rng = np.random.default_rng(int(time.time() % 2 ** 31) if seed is None else seed)
    return np.array([a_mle, b_mle]) + rng.normal(scale=[0.02, 0.1], size=(n_walkers, 2))


def _estimate_mle_start(x_centered, y):
    """MCMC 初期値用の MLE（失敗したら a=0, b=mean(y)）"""
    try:
        res = minimize(lambda p: -kde_log_likelihood_plain(p[0], p[1], x_centered, y),
                       x0=[0.0, np.mean(y)], method="L-BFGS-B")
        return res.x
    except Exception:
        return 0.0, np.mean(y)


def run_bayes_kde_auto_stop_fast(
    df, fixed_lookup=None, bw_factor=1.0, n_walkers=16, n_steps=2000, chunk=100,
    burn_frac=0.5, verbose=True, ci_width_threshold=0.02, stable_rounds=2, seed=None,
):
    """FFT-KDE 尤度の ベイズ MCMC。傾き a の 95%CI が安定したら早期終了する"""
    _, x_centered, y = extract_xy(df)
    ndim = 2
    a_mle, b_mle = _estimate_mle_start(x_centered, y)

    if fixed_lookup is None:
        lookup = prepare_fft_kde_lookup(y - (a_mle * x_centered + b_mle),
                                        bw_factor=bw_factor, use_loo=False)
    else:
        lookup = fixed_lookup

    data_scale = {"x_range": x_centered.max() - x_centered.min(), "y_std": np.nanstd(y)}
    n_walkers = max(2 * ndim, n_walkers)
    pos = _initial_positions(a_mle, b_mle, n_walkers, seed)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_fast,
                                    args=(x_centered, y, lookup, data_scale))

    stable_count = 0
    n_chunks = max(1, n_steps // chunk)
    for i in range(n_chunks):
        pos, _, _ = sampler.run_mcmc(pos, chunk, progress=False)

        # 最終結果と同じ burn-in 除去を現在のステップ数に適用して判定する
        a_partial = sampler.get_chain(discard=int(burn_frac * sampler.iteration), flat=True)[:, 0]
        ci = np.percentile(a_partial, [2.5, 97.5])
        if verbose:
            print(f"[FastMCMC {min((i + 1) * chunk, n_steps)}/{n_steps}] "
                  f"a_mean={np.mean(a_partial)} CI={ci} P(a<0)={np.mean(a_partial < 0)}")

        ci_is_stable = (ci[1] - ci[0]) < ci_width_threshold
        direction_stable = (ci[1] < 0) or (ci[0] > 0)
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

    total_steps = sampler.iteration
    samples = sampler.get_chain(discard=int(burn_frac * total_steps), flat=True)
    if samples.size == 0:
        samples = sampler.get_chain(flat=True)
    a_samps, b_samps = samples[:, 0], samples[:, 1]

    return {
        "samples": samples,
        "a_mean": np.mean(a_samps),
        "b_mean": np.mean(b_samps),
        "a_ci": np.percentile(a_samps, [2.5, 97.5]),
        "b_ci": np.percentile(b_samps, [2.5, 97.5]),
        "p_neg_slope": np.mean(a_samps < 0),
        "steps_completed": total_steps,
        "stopped_early": stable_count >= stable_rounds,
        "Rhat": compute_Rhat(np.transpose(sampler.get_chain(), axes=(1, 0, 2))),
        "lookup": lookup,
    }


def bayes_trend_judgement(contains_zero, p_m1, p_neg_slope, p_pos_slope):
    """傾向判定文を返す。p_m1 は TI によるモデル事後確率（未計算なら None）。
    None のときはモデル選択に関する文を含めない（傾きの事後確率のみで判定）。
    """
    trend = ""
    if p_m1 is not None:
        trend = ("1次式モデルが選択される。" if p_m1 >= PROB_THRESHOLD
                 else "1次式モデルは選択されない。")
    if p_neg_slope >= PROB_THRESHOLD:
        trend += " 減少傾向あり。"
    elif p_pos_slope >= PROB_THRESHOLD:
        trend += " 増加傾向あり。"
    else:
        trend += " 傾向は不明。"
    if contains_zero:
        trend += " 傾きの信頼区間に 0 を含む。"
    return trend.strip()


# ---- 熱力学積分 (TI) ----
def default_betas():
    """β グリッド（1.0 付近を密に：計51点）"""
    return np.concatenate([
        np.linspace(0, 0.9, 30, endpoint=False),
        np.linspace(0.9, 1.0, 21),
    ])


def estimate_E_loglike_for_beta(
    beta, x_centered, y, lookup, n_walkers=20, n_steps=2000, burn_frac=0.4,
    init_center=None, data_scale=None, show_progress=False,
):
    """温度 β での MCMC を回し E_β[logL] を推定する"""
    ndim = 2
    if init_center is None:
        init_center = np.array([0.0, np.mean(y)])
    initial_pos = init_center + 1e-3 * np.random.default_rng().normal(size=(n_walkers, ndim))

    def _logprob(theta):
        return tempered_log_posterior(theta, beta, x_centered, y, lookup, data_scale=data_scale)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, _logprob)
    sampler.run_mcmc(initial_pos, n_steps, progress=show_progress)
    chain = sampler.get_chain(discard=int(n_steps * burn_frac), flat=True)

    # TI のため tempered ではない logL で評価
    ll_vals = [fast_kde_log_likelihood_from_residuals(th[0], th[1], x_centered, y, lookup)
               for th in chain]
    return np.mean(ll_vals), chain, sampler


def _mle_for_model(model, x, y):
    if model == "linear":
        res = minimize(neg_log_likelihood_linear, [-0.01, np.mean(y)],
                       args=(x, y), method="L-BFGS-B")
        return res.x
    res0 = minimize(neg_log_likelihood_const, [np.mean(y)], args=(x, y), method="L-BFGS-B")
    return 0.0, res0.x[0]


def compute_log_marginal_TL(
    df, model="linear", betas=None, n_walkers=20, n_steps=600, burn_frac=0.4,
    bw_factor=1.0, verbose=True,
):
    """熱力学積分で周辺尤度 logZ を推定する（model: 'linear' | 'constant'）"""
    _, x_centered, y = extract_xy(df)

    a_mle, b_mle = _mle_for_model(model, x_centered, y)
    residuals = y - (a_mle * x_centered + b_mle)
    lookup = prepare_fft_kde_lookup(residuals, bw_factor=bw_factor, use_loo=False)

    betas = np.unique(np.sort(default_betas() if betas is None else betas))
    if verbose:
        print(f"\n[TI] model={model}")
        print(" betas =", np.round(betas, 3))

    data_scale = {"x_range": float(x_centered.max() - x_centered.min()),
                  "y_std": float(np.nanstd(y))}
    init_center = np.array([a_mle, b_mle])

    E_logL = []
    for beta in betas:
        if verbose:
            print(f" β={beta:.3f} ... ", end="")
        E, _, _ = estimate_E_loglike_for_beta(
            beta, x_centered, y, lookup, n_walkers=n_walkers, n_steps=n_steps,
            burn_frac=burn_frac, init_center=init_center, data_scale=data_scale)
        if verbose:
            print(f"E[logL]={E:.3f}")
        E_logL.append(E)

    return {
        "model": model, "betas": betas, "E_logL": np.array(E_logL),
        "logZ": trapezoid(E_logL, betas),
        "a_mle": a_mle, "b_mle": b_mle, "lookup": lookup,
    }


def compare_models_logZ(res1, res0, prior_ratio=1.0):
    """logZ からベイズ因子とモデル事後確率を計算する（数値安定版）"""
    l1 = max(res1["logZ"], -1e10)
    l0 = max(res0["logZ"], -1e10)
    logBF = l1 - l0 + np.log(prior_ratio)

    if logBF > 700:
        BF, p_M1 = np.inf, 1.0
    elif logBF < -700:
        BF, p_M1 = 0.0, 0.0
    else:
        BF, p_M1 = np.exp(logBF), expit(logBF)

    return {"logZ1": l1, "logZ0": l0, "logBF": logBF, "BF": BF, "p_M1": p_M1}


BF_SCALE = [  # (上限, 判定文)  Jeffreys (1961)
    (1, "BF < 1　定数モデル支持"),
    (3, " 1 <= BF < 3　線形モデル支持の弱い証拠"),
    (10, "3 <= BF < 10　線形モデル支持の中程度の証拠"),
    (30, "10 <= BF < 30　線形モデル支持の強い証拠"),
    (100, "30 <= BF < 100　線形モデル支持の非常に強い証拠"),
]


def interpret_bayes_factor(logBF=None, BF=None):
    """BF の Jeffreys 基準による判定。BF か logBF のどちらかを指定する"""
    if BF is None and logBF is None:
        raise ValueError("logBF または BF のいずれかを指定してください")
    if BF is None:
        BF = math.exp(logBF)
    if logBF is None:
        logBF = math.log(BF)

    interpretation = "100 <= BF　線形モデル支持の決定的な証拠"
    for upper, text in BF_SCALE:
        if BF < upper:
            interpretation = text
            break
    return {"BF": BF, "logBF": logBF, "interpretation": interpretation}


# =====================================================================
# 4. ブートストラップ / 二項ロジスティック回帰
# =====================================================================
def determine_split_year(df):
    years = df["year"].dropna().unique()
    if len(years) < 4:
        return None
    return int(np.median(years))


def _bootstrap_means(values, R, rng, batch=500):
    """values のブートストラップ平均を R 個返す（メモリ節約のためバッチ処理）"""
    n = len(values)
    out = np.empty(R)
    for s in range(0, R, batch):
        e = min(s + batch, R)
        out[s:e] = values[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    return out


def compare_periods(df, split_year, R=10000, seed=None):
    """前期 - 後期 の log 濃度平均差のブートストラップ分布。
    戻り値: (95%信頼区間, 平均差の推定値(ブートストラップ平均), 標準偏差)
    """
    A = df[df["year"] <= split_year]["log_conc"].to_numpy(dtype=float)
    B = df[df["year"] > split_year]["log_conc"].to_numpy(dtype=float)
    if len(A) < 2 or len(B) < 2:
        return np.array([np.nan, np.nan]), np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = _bootstrap_means(A, R, rng) - _bootstrap_means(B, R, rng)
    return np.percentile(diffs, [2.5, 97.5]), np.mean(diffs), np.std(diffs, ddof=1)


def binomial_trend_analyze(df, flag_col=None):
    """年を説明変数とする二項ロジスティック回帰 (GLM) の AIC 比較。
    戻り値: (aic0, aic1, p1, slope)
    """
    df_work = df.copy()

    if flag_col is not None and flag_col in df_work.columns:
        df_work["low_flag"] = df_work[flag_col].astype(int)
    elif "low_flag" in df_work.columns:
        pass
    elif "binary" in df_work.columns:
        df_work["low_flag"] = df_work["binary"].astype(int)
    elif "detection_limit" in df_work.columns and "concentration" in df_work.columns:
        unified_dl = df_work["detection_limit"].max()
        df_work["low_flag"] = (df_work["concentration"] < unified_dl).astype(int)
    else:
        raise KeyError("二項化に必要な列 ('low_flag', 'binary', または "
                       "'detection_limit'/'concentration') が見つかりません。")

    yearly = df_work.groupby("year").agg(
        low_count=("low_flag", "sum"), total_count=("low_flag", "count")
    ).reset_index()
    yearly["not_low_count"] = yearly["total_count"] - yearly["low_count"]

    X1 = sm.add_constant(yearly["year"].astype(float).to_numpy()).astype(float)
    y_bin = yearly[["low_count", "not_low_count"]].to_numpy().astype(float)

    valid = (yearly["total_count"] > 0).to_numpy() & ~np.isnan(X1).any(axis=1)
    X1, y_bin = X1[valid], y_bin[valid]
    if len(y_bin) == 0:
        return np.nan, np.nan, np.nan, np.nan

    X0 = np.ones((len(y_bin), 1), dtype=float)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", category=PerfectSeparationWarning)
        fit1 = sm.GLM(y_bin, X1, family=sm.families.Binomial()).fit()
        fit0 = sm.GLM(y_bin, X0, family=sm.families.Binomial()).fit()

    p1 = model_probability_m1(fit1.aic, fit0.aic)
    slope = fit1.params[1] if len(fit1.params) > 1 else np.nan
    return fit0.aic, fit1.aic, p1, slope


# =====================================================================
# 5. 解析ステップ（物質ごと）
# =====================================================================
def analyze_bayes(df, x_centered, y, a_mle, b_mle, run_ti):
    """FFT-KDE ベイズ解析（+ オプションで TI）。失敗しても必ず全キーを持つ dict を返す"""
    out = {
        "a": np.nan, "b": np.nan,
        "ci_a": np.array([np.nan, np.nan]), "ci_b": np.array([np.nan, np.nan]),
        "p_neg": np.nan, "p_pos": np.nan,
        "trend": "未評価",
        "cmp": dict(EMPTY_CMP),
        "bf_text": "N/A",
    }
    try:
        lookup = prepare_fft_kde_lookup(y - (a_mle * x_centered + b_mle),
                                        bw_factor=1.0, use_loo=False)
        bayes_res = run_bayes_kde_auto_stop_fast(
            df, fixed_lookup=lookup, bw_factor=1.0, n_walkers=16, n_steps=2000,
            chunk=100, burn_frac=0.5, verbose=True, ci_width_threshold=0.03, stable_rounds=1)

        out["a"], out["b"] = bayes_res["a_mean"], bayes_res["b_mean"]
        out["ci_a"], out["ci_b"] = np.asarray(bayes_res["a_ci"]), np.asarray(bayes_res["b_ci"])
        out["p_neg"] = bayes_res["p_neg_slope"]
        out["p_pos"] = 1.0 - out["p_neg"]

        print(f"ベイズ推定 a = {out['a']:.4f} (95% CI {out['ci_a']})")
        print(f"ベイズ推定 b = {out['b']:.4f} (95% CI {out['ci_b']})")
        print(f"減少傾向の事後確率 P(a < 0) = {out['p_neg']:.4f}")

        if run_ti:
            try:
                res_lin = compute_log_marginal_TL(df, model="linear", verbose=True)
                res_const = compute_log_marginal_TL(df, model="constant", verbose=True)
                cmp = compare_models_logZ(res_lin, res_const)
                out["cmp"] = cmp
                out["bf_text"] = interpret_bayes_factor(BF=cmp["BF"])["interpretation"]
                print("p_M1 モデル事後確率 p(M1|y):", cmp["p_M1"])
                print("logZ_linear: ", cmp["logZ1"])
                print("logZ_const:  ", cmp["logZ0"])
                print("logBF 対数ベイズ因子:", cmp["logBF"])
                print("BF:", cmp["BF"])
                print("BF評価:", out["bf_text"])
            except Exception as e_ti:
                print("TI（logZ計算）は失敗しましたが、ベイズ事後の傾向判定は有効です:", e_ti)
        else:
            out["bf_text"] = NO_TI_MESSAGE

        # 傾向判定: モデル選択の文は TI で p_M1 が得られた場合のみ含める
        contains_zero = out["ci_a"][0] <= 0 <= out["ci_a"][1]
        out["trend"] = bayes_trend_judgement(
            contains_zero, out["cmp"]["p_M1"], out["p_neg"], out["p_pos"])
        print(f"→ {out['trend']}")

    except Exception as e:
        print("ベイズ推定失敗:", e)
        out["trend"] = "エラー"
    return out


def analyze_bootstrap(df):
    """期間分割ブートストラップ検定"""
    res = {"split_year": None, "ci": np.array([np.nan, np.nan]),
           "mean": np.nan, "std": np.nan, "trend": "エラー"}
    try:
        split_year = determine_split_year(df)
        res["split_year"] = split_year
        if split_year is None:
            res["trend"] = "BSでは評価できない"
            return res

        ci, mean_diff, std_diff = compare_periods(df, split_year=split_year)
        res["ci"], res["mean"], res["std"] = ci, mean_diff, std_diff
        print("\n--- 3. ブートストラップ検定 ---")
        print(f"分割年: {split_year}")
        print(f"平均値差(前期-後期)の推定値: {mean_diff:.4f} (SD {std_diff:.4f})")
        print(f"平均値差の95%信頼区間: {ci}")
        if ci[0] > 0:
            res["trend"] = f"分割年: {split_year}　後期は有意に低濃度"
        elif ci[1] < 0:
            res["trend"] = f"分割年: {split_year}　後期は有意に高濃度"
        else:
            res["trend"] = f"分割年: {split_year}　有意差なし"
        print(f"→ {res['trend']}")
    except Exception as e:
        print(f"ブートストラップエラー: {e}")
    return res


def analyze_binomial(df):
    """統一下限値での二項ロジスティック回帰"""
    print("\n--- 4. 二項ロジスティック回帰モデル ---")
    try:
        df_bin = df[["year", "concentration", "detection_limit"]].copy()
        _, _, p1, slope = binomial_trend_analyze(df_bin)
        print(f"傾き a = {slope:.4f}, オッズ比 ≈ {math.exp(slope):.4f}, AIC事後確率 p1 = {p1:.3f}")
        dl = df_bin["detection_limit"].max()
        trend = (f"統一下限値 {dl} 以下の増加傾向あり" if p1 >= PROB_THRESHOLD
                 else f"統一下限値 {dl} 以下で傾向なし")
        print(f"→ {trend}")
        return {"slope": slope, "p1": p1, "trend": trend}
    except Exception as e:
        print(f"二項回帰エラー: {e}")
        return {"slope": np.nan, "p1": np.nan, "trend": "エラー"}


def analyze_threshold_binomials(df):
    """DL / 25%タイル / 75%タイル を閾値とした二項ロジスティック回帰"""
    conc = df["concentration"]
    thresholds = []  # (ラベル, 閾値, 0/1 フラグ)
    if not df["detection_limit"].dropna().empty:
        t = df["detection_limit"].max()
        thresholds.append(("DL閾値", t, conc < t))
    t25 = conc.quantile(0.25)
    thresholds.append(("25%タイル閾値", t25, conc <= t25))
    t75 = conc.quantile(0.75)
    thresholds.append(("75%タイル閾値", t75, conc >= t75))

    out = {label: {"slope": np.nan, "p1": np.nan}
           for label in ("DL閾値", "25%タイル閾値", "75%タイル閾値")}
    for label, thr, flag in thresholds:
        try:
            d = df[["year"]].copy()
            d["binary"] = flag.astype(int)
            _, _, p1, slope = binomial_trend_analyze(d, flag_col="binary")
            out[label] = {"slope": slope, "p1": p1}
            print(f"{label} ({thr:.4f}): 傾き={slope:.4f}, p1={p1:.3f}")
        except Exception as e:
            print(f"{label}での二項回帰エラー: {e}")
    return out


# =====================================================================
# 6. データ読込 / プロット / レポート
# =====================================================================
def load_dataset(filename):
    """CSV を読み込み、不検出値を DL/√2 で代入して log10 濃度列を追加する"""
    filepath = f"{DATA_DIR}/{filename}.csv"
    if not os.path.exists(filepath):
        print(f"ファイルが見つかりません: {filepath}")
        return None

    df = pd.read_csv(filepath, usecols=["substance", "year", "site_id",
                                        "concentration", "detection_limit"])
    for col in ("concentration", "detection_limit"):
        df[col] = df[col].apply(clean_value_user)

    is_nd = df["concentration"] <= df["detection_limit"]
    df["concentration_sub"] = df["concentration"].where(~is_nd, df["detection_limit"] / np.sqrt(2))
    df["log_conc"] = np.log10(df["concentration_sub"])
    df = (df.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["log_conc", "detection_limit", "concentration"]))

    output_path = f"{DATA_DIR}/{filename}_conv.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"→ 変換後のデータを保存しました: {output_path}")
    return df


def plot_regression(df, substance, filename, mle, bayes, split_year, p1):
    """MLE / ベイズ回帰直線のプロットを保存し、パスを返す"""
    x_data = df["year"].to_numpy(dtype="int64")
    y_data = df["log_conc"].to_numpy(dtype=float)
    dl_log = np.log10(df["detection_limit"].to_numpy(dtype=float))
    x_base, a, b = mle["x_base"], mle["a1"], mle["b1"]

    x_fit = np.array([x_data.min(), x_data.max()])
    x_fit_c = x_fit - x_base
    y_fit_mle = a * x_fit_c + b
    y_fit_bayes = bayes["a"] * x_fit_c + bayes["b"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hlines(dl_log, xmin=x_data - 0.2, xmax=x_data + 0.2,
              color="tab:orange", alpha=0.5, linewidth=1.5, label="検出限界 (log10)")
    ax.scatter(x_data, y_data, label="実測データ (代替値込)", color="tab:blue", alpha=0.6)
    ax.plot(x_fit, y_fit_mle, color="tab:red",
            label=f"MLE回帰直線: y = {a:.4f}(x - {x_base}) + {b:.4f}")
    if not np.isnan(bayes["a"]):
        ax.plot(x_fit, y_fit_bayes, color="tab:green", linestyle="--",
                label=f"ベイズ回帰直線: y = {bayes['a']:.4f}(x - {x_base}) + {bayes['b']:.4f}")
    if split_year is not None:
        ax.axvline(x=split_year, color="gray", linestyle="--", linewidth=1.5,
                   label=f"分割年: {split_year}")

    p_neg = 0 if np.isnan(bayes["p_neg"]) else bayes["p_neg"]
    ax.set_title(f"{substance} の対数線形回帰\n"
                 f"MLE p1 = {p1:.4f} / ベイズ P(a<0) = {p_neg:.4f} ({filename})")
    ax.set_xlabel("年")
    ax.set_ylabel("log10濃度")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    y_all = np.concatenate([y_data, y_fit_mle, y_fit_bayes, dl_log])
    ax.set_ylim(np.nanmin(y_all) - 0.1, np.nanmax(y_all) + 0.1)

    fig.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"{safe_name(substance, filename)}_regression.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def build_analysis_report(substance, filename, df, split_year, mle, bayes, boot, binom, thr):
    """JSON / HTML 出力用のレポート dict（v1.1b とキー互換）"""
    cmp = bayes["cmp"]
    ci_bs = boot["ci"]
    return {
        "substance": substance,
        "filename": filename,
        "year": df["year"].unique().tolist(),
        "split_year": split_year,
        "n_obs": len(df),
        "n_censored": (df["log_conc"] < np.log10(df["detection_limit"])).sum(),
        "methods": {
            "MLE": {
                "slope": mle["a1"], "intercept": mle["b1"],
                "aic0": mle["aic0"], "aic1": mle["aic1"],
                "loglik0": mle["loglik0"], "loglik1": mle["loglik1"],
                "p1": mle["p1"], "trend": mle["trend_mle"],
            },
            "OLS": {
                "slope": mle["ols_slope"], "intercept": mle["ols_intercept"],
                "stderr_a": mle["ols_stderr_a"], "stderr_b": mle["ols_stderr_b"],
                "ci_a": mle["ols_ci_a"], "ci_b": mle["ols_ci_b"],
                "pvalues": mle["ols_pvalues"], "r2": mle["ols_r2"],
            },
            "Bayes": {
                "slope": bayes["a"], "intercept": bayes["b"],
                "p_M1": cmp["p_M1"],
                "bayes_CI_a": bayes["ci_a"].tolist(), "bayes_CI_b": bayes["ci_b"].tolist(),
                "p_neg": bayes["p_neg"], "p_pos": bayes["p_pos"],
                "logZ_linear": cmp["logZ1"], "logZ_const": cmp["logZ0"],
                "logBF": cmp["logBF"], "BF": cmp["BF"],
                "bayes_trend": bayes["trend"],
                "BF評価": bayes["bf_text"],
            },
            "Bootstrap": {
                "bootstrap_CI": ci_bs,
                "bootstrap_summary": {"mean": boot["mean"], "std": boot["std"]},
                "bs_trend": boot["trend"],
            },
            "Binominal": {  # (綴りは JSON 互換のため v1.1b のまま)
                "binomial_slope": binom["slope"],
                "binomial_p1": binom["p1"],
                "binomial_trend": binom["trend"],
                "DL閾値_slope": thr["DL閾値"]["slope"],
                "DL閾値_p1": thr["DL閾値"]["p1"],
                "25%タイル閾値_slope": thr["25%タイル閾値"]["slope"],
                "25%タイル閾値_p1": thr["25%タイル閾値"]["p1"],
                "75%タイル閾値_slope": thr["75%タイル閾値"]["slope"],
                "75%タイル閾値_p1": thr["75%タイル閾値"]["p1"],
            },
        },
    }


# ---- HTML レポート ----
def _row(label, value):
    return f"<b>{label}:</b> {value}<br>"


def _html_substance_section(r, image_path, run_ti):
    m = r["methods"]
    ols, mle, bayes, boot, bino = m["OLS"], m["MLE"], m["Bayes"], m["Bootstrap"], m["Binominal"]

    h = ["<hr>", f"<h2>{r['substance']}</h2>"]
    if os.path.exists(image_path):
        h.append(f"<img src='{image_path}' width='600'><br>")
    else:
        h.append(f"<p><i>グラフ画像が見つかりませんでした。</i></p>{image_path}")

    h.append("<p>" + _row("対象ファイル", r["filename"]) + _row("年データ", r["year"])
             + _row("分割年", r["split_year"]) + _row("観測数", r["n_obs"])
             + _row("検出限界未満", r["n_censored"]) + "</p>")
    h.append("<hr>")

    # OLS
    h += ["<h3>最小二乗法（OLS）非推奨</h3>",
          _row("傾き a", fmt(ols["slope"])), _row("切片 b", fmt(ols["intercept"])),
          _row("標準誤差 a", fmt(ols["stderr_a"])), _row("標準誤差 b", fmt(ols["stderr_b"])),
          _row("95%CI a", ci_str(ols["ci_a"])), _row("95%CI b", ci_str(ols["ci_b"])),
          _row("p値", f"[{fmt(ols['pvalues'][0], '.3e')}, {fmt(ols['pvalues'][1])}]"),
          _row("決定係数 R²", fmt(ols["r2"])), "<hr>"]

    # MLE
    h += ["<h3>最尤推定（KDE-MLE / AIC）環境省法</h3>",
          _row("傾き a", fmt(mle["slope"], ".6")), _row("切片 b", fmt(mle["intercept"], ".6")),
          "<br><b>---AICによるモデル比較---</b><br>",
          _row("AIC0", fmt(mle["aic0"], ".6")), _row("AIC1", fmt(mle["aic1"], ".6")),
          _row("モデル選択確率 p1", fmt(mle["p1"], ".6")), _row("傾向判定", mle["trend"]),
          "<hr>"]

    # Bayes
    h += ["<h3>ベイズ推定（FFT-KDE-Bayes / BF）</h3>",
          _row("傾き a", fmt(bayes["slope"])), _row("切片 b", fmt(bayes["intercept"])),
          _row("傾き95%CI", ci_str(bayes["bayes_CI_a"])),
          _row("切片95%CI", ci_str(bayes["bayes_CI_b"])),
          _row("減少傾向の事後確率 P(a<0)", fmt(bayes["p_neg"])),
          _row("増加傾向の事後確率 P(a>0)", fmt(bayes["p_pos"])),
          _row("傾向判定", bayes["bayes_trend"])]
    if run_ti:
        h += ["<br><b>---Bayes Factorによるモデル比較-TI--</b><br>",
              _row("線形モデルの対数周辺尤度(モデルエビデンス) logZ(linear)", fmt(bayes["logZ_linear"])),
              _row("定数モデルの対数周辺尤度(モデルエビデンス) logZ(const)", fmt(bayes["logZ_const"])),
              _row("対数ベイズ因子 logBF", fmt(bayes["logBF"])),
              _row("ベイズ因子 BF", fmt(bayes["BF"])),
              _row("傾向判定", bayes["bayes_trend"]),
              _row("BF評価", bayes["BF評価"])]
    else:
        h += ["<b><br>BF（TI計算）によるモデル比較は実行されませんでした。</b><br>",
              _row("BF評価", bayes["BF評価"])]
    h.append("<hr>")

    # Bootstrap
    summary = boot["bootstrap_summary"]
    h += ["<h3>ブートストラップ検定</h3>",
          _row("95%CI", ci_str(boot["bootstrap_CI"])),
          _row("平均値差(前期-後期)の推定値", fmt(summary["mean"], ".6")),
          _row("標準偏差(ブートストラップ標準誤差)", fmt(summary["std"], ".6")),
          _row("傾向判定", boot["bs_trend"]), "<hr>"]

    # Binomial
    def _slope_p1(prefix):
        return (f"傾き={fmt(bino[f'{prefix}_slope'], '.4f')}, "
                f"p1={fmt(bino[f'{prefix}_p1'], '.3f')}")

    h += ["<h3>二項ロジスティック回帰</h3>",
          _row("傾き", fmt(bino["binomial_slope"], ".6")),
          _row("モデル選択確率 p1", fmt(bino["binomial_p1"], ".6")),
          _row("傾向判定", bino["binomial_trend"]),
          _row("DL閾値: 傾き", f"{fmt(bino['DL閾値_slope'], '.4f')}, "
                               f"p1={fmt(bino['DL閾値_p1'], '.3f')}"),
          _row("25%タイル閾値", _slope_p1("25%タイル閾値")),
          _row("75%タイル閾値", _slope_p1("75%タイル閾値")),
          "<hr>", f"<br>　→グラフは {image_path} に保存されている。"]
    return "\n".join(h)


def write_html_report(reports, filename, run_ti):
    html_path = f"{filename}_regression_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
    parts = [f"<html><head><meta charset='utf-8'><title>{filename} 回帰分析レポート</title></head><body>",
             f"<h1>{filename} 分析レポート </h1>",
             f" {VERSION}   <br><br>"]
    for r in reports:
        image_path = os.path.join(PLOT_DIR, f"{safe_name(r['substance'], filename)}_regression.png")
        parts.append(_html_substance_section(r, image_path, run_ti))
    parts.append("</body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"\n→ HTMLレポートを作成しました: {os.path.join(os.getcwd(), html_path)}")


# =====================================================================
# 7. 実行部
# =====================================================================
def analyze_substance(df_all, substance, filename, run_ti):
    """1物質の全解析を行い、レポート dict を返す（スキップ時は None）"""
    print(f"\n==================== {substance} ====================")
    df = df_all[df_all["substance"] == substance].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])
    if len(df) < MIN_SAMPLES:
        print(f"データ不足のためスキップ（地点数 < {MIN_SAMPLES}）")
        return None

    # ① MLE + AIC
    print("--- 1. 対数線形回帰モデル (MLE) ---")
    try:
        mle = run_mle_and_aic(df)
    except Exception as e:
        print(f"最尤推定エラー: {e}")
        return None
    a, b, p1 = float(mle["a1"]), float(mle["b1"]), float(mle["p1"])
    print(f"傾き a = {a:.4f}, 切片 b = {b:.4f}, AIC事後確率 p1 = {p1:.4f}")
    print(f"aic1: {mle['aic1']:.4f} - aic0: {mle['aic0']:.4f}")
    print("→ " + ("p1 >= 0.95 傾向あり" if p1 >= PROB_THRESHOLD else "p1 < 0.95 傾向なし"))

    _, x_centered, y = extract_xy(df)
    print(f"R² = {r2_score(y, a * x_centered + b):.4f}")

    # ② FFT-KDE ベイズ MCMC
    print("\n--- 2. FFT-KDE ベイズ事後（高速 MCMC、自動早期停止） ---")
    bayes = analyze_bayes(df, x_centered, y, a, b, run_ti)

    # ③ ブートストラップ / ④ 二項回帰 / ⑤ 閾値別二項回帰
    boot = analyze_bootstrap(df)
    binom = analyze_binomial(df)
    thr = analyze_threshold_binomials(df)

    # プロット
    try:
        print("\n--- 5. 回帰直線のプロット ---")
        plot_path = plot_regression(df, substance, filename, mle, bayes, boot["split_year"], p1)
        print(f"→ {plot_path} に保存しました。")
    except Exception as e:
        print(f"プロットエラー: {e}")

    return build_analysis_report(substance, filename, df, boot["split_year"],
                                 mle, bayes, boot, binom, thr)


def process_file(filename, run_ti):
    print(f"\n\n==================== ファイル処理開始: {filename} ====================")
    df_all = load_dataset(filename)
    if df_all is None:
        return

    substances = df_all["substance"].unique()
    print(f"対象物質: {substances}")
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"プロット画像は '{PLOT_DIR}' ディレクトリに保存されます。")

    reports = []
    for substance in substances:
        report = analyze_substance(df_all, substance, filename, run_ti)
        if report is not None:
            reports.append(report)

    with open(f"{filename}_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    write_html_report(reports, filename, run_ti)


def main(filenames=None, run_ti=RUN_TI):
    for filename in (filenames or DEFAULT_FILENAMES):
        process_file(filename, run_ti)


if __name__ == "__main__":
    main()