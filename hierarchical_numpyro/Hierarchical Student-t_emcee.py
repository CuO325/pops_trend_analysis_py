# Hierarchical Student-t regression (per-site slopes and intercepts) with emcee. site_t_emcee.py 
# -*- coding: utf-8 -*-
"""
Hierarchical Student-t regression (per-site slopes and intercepts) with emcee.
高速化・安定化版 v4.2: site_t_std_update_fixed_v4_2a.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from scipy.special import gammaln
import japanize_matplotlib  # 日本語フォント
from emcee.autocorr import integrated_time 


# ------------------------------
# フォント設定（起動時に一度）
# ------------------------------
plt.rcParams['font.size'] = 10
# japanize_matplotlib を import 済みなので、追加設定は不要

#　入力設定
filenames = ["水質"]  # data/<name>.csv
substance_data = ["α-HCH"] #「リストに中身があれば」選択、[]で「すべての物質」の意味

# 出力設定
PLOT_DIR = ""     #　傾きのバーグラフのみ分離して1か所に集めたいときに指定する　空 ""の設定で　{filemane_substance}　
SAVE_NPZ = False
CSV_PATH ="_site_summary.csv"

"""--------
Rhat値の範囲,解釈,実務上の対応
Rhat≥1.1:       収束の失敗。このパラメータの結果は信頼できない。サンプリングの続行、モデル/初期値の変更が必須。
1.05≤Rhat<1.1:  弱い収束。結果は疑わしい。サンプル数を増やすか、モデルの再確認を推奨。
1.01≤Rhat<1.05: 許容範囲。実用的な結果が得られることが多い。余裕があればさらにサンプリングを増やす。
Rhat<1.01:      理想的。十分に収束している。終了。
---------"""
RHAT_CHK = 1.05

# データ数による除外 ---
MIN_OBS_FOR_SITE = 5   # Total 観測数（観測年）# サイトごとの最小データ列（例:　11以上を残す）
# 最終年度による除外
MIN_END_YEAR = 2010  # 調査最終年度 # 最終観測年がこの年以下の場合に除外 (例:　2016年以上のサイトを残す)
#　解析期間の範囲指定
MIN_ANALYSIS_YEAR = 1999  # この年より大きい（指定した年は含まない）データを採用
MAX_ANALYSIS_YEAR = 2030  # この年以下（指定年まで含む）データを採用

#PLOTの出力
MAKE_DIAG_PLOTS = True  # Falseで図は描かない
MAKE_SITE_PLOTS = False  #['1', '30']   診断図を出力したいサイトIDを指定できる Falseで全指定
MAKE_TRACE_PLOT = False  #['1', '30']   診断図を出力したいサイトIDを指定できる Falseで全指定

DPI_PLOT = 150          # 図の解像度（200→150で保存高速化）


# ==========emcee 重要=========

N_WALKERS = 360         # 400 → 360（下限は 2×ndim=276 を厳守）
STEPS     = 10000    # 軽量6000　最終　30000    デフォルトで　burn_in = 5000　thin    = 100　であるので注意。
P0_SPREAD = 0.25        # 0.20 → 0.25（初期雲を広げる）
USE_POOL  = False       # 直列が速かったため、False維持
MOVE_CONF = [
    (emcee.moves.StretchMove(a=2.8), 0.85),  # あるウォーカーが別のウォーカーの座標を参考にして移動。デフォルトは2.0。 aが大きいほど、ウォーカーはより大きなジャンプをする。この移動が選択される確率。デフォルト設定では 0.85 に設定
    (emcee.moves.DEMove(),           0.15), #異なるウォーカー間の差分ベクトルを利用して移動。アンサンブルが細長く引き延ばされた形になっている場合に、よりランダムで大域的な探索を助けます.この移動が選択される確率。
    (emcee.moves.DESnookerMove(),    0.2),  #DEMove のバリエーションで、特に複雑な分布やマルチモーダルな分布（複数のピークを持つ分布）に対して有効です。
]
BURN_IN = 5000
THIN    = 100

# 半減の定数
LOG10_2 = np.log10(2.0)  # ≈0.30102999566
HALF_EPS = 1e-6          # 傾きが極小のサンプルを除外する閾値

# ------------------------------
# Student-t logpdf（自前・数値安定化）
# ------------------------------

def student_t_logpdf(x: np.ndarray, nu: float, loc: float, scale: float) -> np.ndarray:
    z = (x - loc) / scale
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
        - (nu + 1.0) / 2.0 * np.log1p((z * z) / nu)
    )

# ------------------------------
# Posterior log-probability（階層 t 回帰）
# ------------------------------

def log_prob(params: np.ndarray, x_data: np.ndarray, y_data: np.ndarray, site_idx: np.ndarray, n_sites: int) -> float:
    # params layout: [a_1..a_n, b_1..b_n, mu_a, sigma_a, mu_b, sigma_b, nu, sigma]
    a = params[:n_sites]
    b = params[n_sites:2 * n_sites]
    mu_a = params[2 * n_sites]
    sigma_a = params[2 * n_sites + 1]
    mu_b = params[2 * n_sites + 2]
    sigma_b = params[2 * n_sites + 3]
    nu = params[2 * n_sites + 4]
    sigma = params[2 * n_sites + 5]

    # 制約
    if sigma_a <= 0 or sigma_b <= 0 or sigma <= 0 or nu <= 1:
        return -np.inf

    # 事前：階層正規 + 弱情報事前
    lp = 0.0
    lp += -0.5 * np.sum(((a - mu_a) ** 2) / (sigma_a ** 2) + np.log(2 * np.pi * sigma_a ** 2))
    lp += -0.5 * np.sum(((b - mu_b) ** 2) / (sigma_b ** 2) + np.log(2 * np.pi * sigma_b ** 2))
    lp += -np.log(sigma_a) - np.log(sigma_b) - np.log(sigma)
    lp += -0.5 * (mu_a ** 2 + mu_b ** 2)
    if not np.isfinite(lp):
        return -np.inf

    # 尤度
    y_pred = a[site_idx] * x_data + b[site_idx]
    residual = y_data - y_pred
    ll = np.sum(student_t_logpdf(residual, nu, 0.0, sigma))
    return lp + ll

import numpy as np
# student_t_logpdf は、t分布の対数PDFを計算する外部関数と仮定します

def log_prob_non_centered(params: np.ndarray, x_data: np.ndarray, y_data: np.ndarray, site_idx: np.ndarray, n_sites: int) -> float:
    # 1. パラメータのレイアウト変更 (非中心化)
    # params layout: [theta_a_1..n, theta_b_1..n, mu_a, sigma_a, mu_b, sigma_b, nu, sigma]
    theta_a = params[:n_sites]          # 標準化された傾きパラメータ
    theta_b = params[n_sites:2 * n_sites] # 標準化された切片パラメータ
    
    # グローバルパラメータは同じインデックス
    mu_a = params[2 * n_sites]
    sigma_a = params[2 * n_sites + 1]
    mu_b = params[2 * n_sites + 2]
    sigma_b = params[2 * n_sites + 3]
    nu = params[2 * n_sites + 4]
    sigma = params[2 * n_sites + 5]

    # 制約 (変更なし)
    if sigma_a <= 0 or sigma_b <= 0 or sigma <= 0 or nu <= 1:
        return -np.inf

    # 2. 物理パラメータ a, b の計算 (非中心化の核となる部分)
    # a_i = mu_a + sigma_a * theta_a_i
    # b_i = mu_b + sigma_b * theta_b_i
    a = mu_a + sigma_a * theta_a
    b = mu_b + sigma_b * theta_b

    # 3. 事前分布 (Prior Probability: lp)
    lp = 0.0
    
    # 階層事前分布: 標準化されたパラメータ theta_a と theta_b が 標準正規分布 N(0, 1) に従う
    # lp(theta_a) = -0.5 * sum(theta_a^2) - N*0.5*log(2*pi) (定数項は無視してもMCMCは機能するが、ここでは残す)
    # 標準正規分布の対数PDF: log(N(theta | 0, 1)) = -0.5*theta^2 - 0.5*log(2*pi)
    lp += -0.5 * np.sum(theta_a ** 2) - 0.5 * n_sites * np.log(2 * np.pi)
    lp += -0.5 * np.sum(theta_b ** 2) - 0.5 * n_sites * np.log(2 * np.pi)

    # 弱情報事前分布 (グローバルパラメータの事前分布はそのまま維持)
    lp += -np.log(sigma_a) - np.log(sigma_b) - np.log(sigma) # J(sigma) ∝ 1/sigma
    lp += -0.5 * (mu_a ** 2 + mu_b ** 2) # mu_a, mu_b ~ N(0, 1)

    if not np.isfinite(lp):
        return -np.inf

    # 4. 尤度 (Likelihood: ll) (変更なし)
    # 物理パラメータ a, b を使用して予測値を計算
    y_pred = a[site_idx] * x_data + b[site_idx]
    residual = y_data - y_pred
    ll = np.sum(student_t_logpdf(residual, nu, 0.0, sigma))
    
    return lp + ll

def plot_walker_traces(sampler, param_index, burn_in=300, max_walkers=None, title=None):
    """
    sampler: emcee.EnsembleSampler
    param_index: 可視化したいパラメータのインデックス (0-based)
    burn_in: バーンインとして捨てるステップ数
    max_walkers: 表示するウォーカー数の上限（多すぎる場合の可視化負担低減）
    title: 図タイトル（任意）
    """
    # 形状: (n_steps, n_walkers, n_dim)
    chain = sampler.get_chain(discard=burn_in, flat=False)
    n_steps, n_walkers, n_dim = chain.shape
    if param_index >= n_dim:
        raise IndexError(f"param_index={param_index} は範囲外です（n_dim={n_dim}）。")

    # 可視化ウォーカーを制限
    walkers_idx = np.arange(n_walkers)
    if max_walkers is not None and max_walkers < n_walkers:
        # 均等サンプリングで間引く
        walkers_idx = np.linspace(0, n_walkers - 1, max_walkers).astype(int)

    # 指定パラメータのみ抽出: (n_steps, n_walkers_sub)
    traces = chain[:, walkers_idx, param_index]

    plt.figure(figsize=(10, 6))
    # ウォーカー別にトレースを薄い線で重ねる
    for w in range(traces.shape[1]):
        plt.plot(traces[:, w], alpha=0.5, linewidth=0.8)

    plt.xlabel("ステップ")
    plt.ylabel(f"パラメータ[{param_index}] の値")
    ttl = title or f"ウォーカー別トレース（param[{param_index}]、burn_in={burn_in}）"
    plt.title(ttl)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

def split_walkers_into_groups(chain, n_groups=4):
    """
    chain: shape = (n_steps, n_walkers, n_dim)（burn-in除外済みを渡す）
    各グループ内のウォーカーを平均して、(n_steps, n_dim) の擬似チェーンにする。
    """
    n_steps, n_walkers, n_dim = chain.shape
    group_size = n_walkers // n_groups
    groups = []
    for g in range(n_groups):
        w_start = g * group_size
        w_end   = (g + 1) * group_size if g < n_groups - 1 else n_walkers
        group_chain = chain[:, w_start:w_end, :].mean(axis=1)  # (n_steps, n_dim)
        groups.append(group_chain)
    return groups  # list of arrays

def gelman_rhat(groups):
    """
    groups: list of arrays each shape (n_steps, n_dim)
    Returns: Rhat vector (n_dim,)
    """
    import numpy as np
    m = len(groups)
    n, d = groups[0].shape
    means = np.array([g.mean(axis=0) for g in groups])         # (m, d)
    vars_ = np.array([g.var(axis=0, ddof=1) for g in groups])  # (m, d)

    B = n * means.var(axis=0, ddof=1)      # between-chain variance
    W = vars_.mean(axis=0)                 # within-chain variance
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    Rhat = np.sqrt(var_hat / W)
    return Rhat

def compute_rhat_from_sampler(sampler, burn_in=300, n_groups=4):
    chain = sampler.get_chain(discard=burn_in, flat=False)
    groups = split_walkers_into_groups(chain, n_groups=n_groups)
    Rhat = gelman_rhat(groups)
    return Rhat  # shape=(n_dim,)

def get_param_indices_for_site(sid: str, site_map: dict, n_sites: int):
    """
    sid: 可視化したいサイトID（文字列 or 数値）。内部で文字列化。
    site_map: run_mcmc_t が返した {site_id(str): index(int)}。※同じものを使うこと
    n_sites: サイト数（len(site_map)）
    returns: (pidx_slope, pidx_intercept) のタプル
    """
    if site_map is None or not isinstance(site_map, dict) or len(site_map) == 0:
        raise ValueError("site_map が None/空/辞書でないため、インデックス解決ができません。run_mcmc_t(df) の返り値をそのまま渡してください。")

    sid_str = str(sid)

    # キー存在チェック（デバッグ情報付き）
    if sid_str not in site_map:
        # キーのプレビュー（最大10件）
        keys_preview = list(site_map.keys())[:10]
        raise KeyError(
            f"指定した sid='{sid_str}' は site_map に存在しません。\n"
            f"site_map のキー例（先頭10件）: {keys_preview} ...（全 {len(site_map)} 件）\n"
            f"※ run_mcmc_t 内で site_id は文字列化＋ソートされています。\n"
            f"  例: 数値ID 1 は文字列 '1'。'01' や 'S001' は別キーです。"
        )

    i = site_map[sid_str]
    if not (0 <= i < n_sites):
        raise IndexError(f"site_map['{sid_str}'] = {i} が範囲外です（n_sites={n_sites}）。site_map の構成を確認してください。")

    # 前半 n_sites が 傾き、後半 n_sites が 切片
    pidx_slope = i
    pidx_intercept = n_sites + i

    return pidx_slope, pidx_intercept

# ------------------------------

def label_rhat(Rhat, n_sites):
    labels = []
    for i in range(len(Rhat)):
        if i < n_sites:
            labels.append(f"a_site[{i}]")  # サイト別傾き
        elif i < 2*n_sites:
            labels.append(f"b_site[{i-n_sites}]")  # サイト別切片
        elif i == 2*n_sites:
            labels.append("mu_a")  # 傾きの階層平均
        elif i == 2*n_sites + 1:
            labels.append("sigma_a")  # 傾きの階層スケール
        elif i == 2*n_sites + 2:
            labels.append("mu_b")  # 切片の階層平均
        elif i == 2*n_sites + 3:
            labels.append("sigma_b")  # 切片の階層スケール
        elif i == 2*n_sites + 4:
            labels.append("nu_df")  # t分布自由度
        elif i == 2*n_sites + 5:
            labels.append("sigma")  # 観測誤差スケール
        else:
            labels.append(f"param[{i}]")  # 万一範囲外なら汎用ラベル
    return list(zip(labels, Rhat))
# ------------------------------

def run_mcmc_t(df: pd.DataFrame, nwalkers: int = N_WALKERS, steps: int = STEPS, seed: int | None = 42):
    rng = np.random.default_rng(seed)

    # site indexing（文字列統一）
    site_list = sorted(df["site_id"].astype(str).unique())
    site_map = {sid: i for i, sid in enumerate(site_list)}
    site_idx = df["site_id"].astype(str).map(site_map).values
    n_sites = len(site_list)

    # 標準化
    x_raw = df["year"].values.astype(float)
    y = df["log_conc"].values.astype(float)
    x_mean = float(x_raw.mean())
    x_scale = float(x_raw.std()) if x_raw.std() > 0 else 1.0
    x = (x_raw - x_mean) / x_scale

    # OLS で初期値（サイト別）
    a0 = np.zeros(n_sites)
    b0 = np.zeros(n_sites)
    for i, sid in enumerate(site_list):
        sel = df["site_id"].astype(str) == sid
        xi = (df.loc[sel, "year"].values.astype(float) - x_mean) / x_scale
        yi = df.loc[sel, "log_conc"].values.astype(float)
        if len(xi) >= 2 and np.std(xi) > 1e-6:
            aa, bb = np.polyfit(xi, yi, 1)
            a0[i] = aa
            b0[i] = bb
        else:
            a0[i] = 0.0
            b0[i] = yi.mean() if len(yi) else 0.0

    mu_a0 = float(np.mean(a0))
    mu_b0 = float(np.mean(b0))
    sigma_a0 = float(np.std(a0)) if np.std(a0) > 0 else 0.3
    sigma_b0 = float(np.std(b0)) if np.std(b0) > 0 else 0.5
    nu0 = 5.0
    sigma0 = 0.2

    #=================================center===========================
    center = np.concatenate([a0, b0, [mu_a0, sigma_a0, mu_b0, sigma_b0, nu0, sigma0]])
    ndim = len(center)

    # バリデーション：nwalkers は 2×ndim 以上
    if nwalkers < 2 * ndim:
        raise ValueError(f"nwalkers={nwalkers} は不正です（少なくとも 2×ndim={2*ndim} が必要）。")

    # 初期ウォーカー（スプレッド）
    p0 = center + P0_SPREAD * rng.standard_normal((nwalkers, ndim))
    p0[:, 2 * n_sites + 1] = np.abs(p0[:, 2 * n_sites + 1]) + 0.05  # sigma_a
    p0[:, 2 * n_sites + 3] = np.abs(p0[:, 2 * n_sites + 3]) + 0.05  # sigma_b
    p0[:, 2 * n_sites + 4] = np.abs(p0[:, 2 * n_sites + 4]) + 2.0   # nu
    p0[:, 2 * n_sites + 5] = np.abs(p0[:, 2 * n_sites + 5]) + 0.05  # sigma

    #========emcee sampler===================================
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y, site_idx, n_sites), moves=MOVE_CONF)
    print(f"MCMC RUN: nwalkers={nwalkers}, ndim={ndim}, steps={steps}")
    t0 = time.perf_counter()
    sampler.run_mcmc(p0, steps, progress=True)

    # ===============R-hat を計算==============
    Rhat = compute_rhat_from_sampler(sampler, burn_in=300, n_groups=4)
    labeled = label_rhat(Rhat, n_sites)
    print("R-hat values:", labeled)  

    # 収束していない（例: R-hat > 1.1）のパラメータだけ抽出
    bad_idx = np.where(Rhat > RHAT_CHK)[0] #1.1
    print("収束が不十分なパラメータ:", bad_idx)
    print("対応するR-hat値:", Rhat[bad_idx])
    bad_rhat = Rhat[bad_idx]
    #label_rhat(Rhat[bad_idx], n_sites)

    # 収束不十分なものだけ表示
    for name, val in labeled:
        if val > 1.1:
            print(f"{name}: R-hat={val:.3f}")

    t1 = time.perf_counter()
    print(f"[timer] MCMC run: {t1 - t0:.2f}s")
    #return sampler, site_map, x_mean, x_scale
    return sampler, site_map, x_mean, x_scale, labeled, bad_idx, bad_rhat

# ------------------------------
# 標準化→元スケールの逆変換（ベクトル）
# ------------------------------

def original_scale_from_samples_all(a_samps_all: np.ndarray, b_samps_all: np.ndarray, x_mean: float, x_scale: float):
    a_o_all = a_samps_all / x_scale
    b_o_all = b_samps_all - a_o_all * x_mean
    return a_o_all, b_o_all

# ------------------------------
# データクリーニング
# ------------------------------

def clean_value_user(val):
    if isinstance(val, str):
        val = val.replace('tr(', '').replace(')', '')
        val = val.replace('---', '')
        val = val.replace('nd', '0')
    try:
        if val == '':
            return np.nan
        return float(val)
    except (ValueError, TypeError):
        return np.nan

# ------------------------------
# サマリー/可視化（ベクトル化 + 半減時間）
# ------------------------------

def summarize_and_save(
    results: dict,
    df_source: pd.DataFrame,  # ★ MCMCに使った df（site_id, year, log_conc を含む）
    out_prefix: str = "results",
    plot_dir: str = "plots_regression_t",
):
    os.makedirs(plot_dir, exist_ok=True)

    samples = results["samples"]      # shape = (N_samples, 2*n_sites+6)
    site_ids = results["site_ids"]    # 並びはサンプル列順と一致
    x_mean = results.get("x_mean", 0.0)
    x_scale = results.get("x_scale", 1.0)

    n_sites = len(site_ids)
    a_samps_all = samples[:, :n_sites]             # (N, S) 傾き（標準化スケール）
    b_samps_all = samples[:, n_sites:2 * n_sites]  # (N, S) 切片（標準化スケール）
    # グローバルパラメータ（インデックスは log_prob と一致）
    mu_a_samps    = samples[:, 2 * n_sites]
    sigma_a_samps = samples[:, 2 * n_sites + 1]
    mu_b_samps    = samples[:, 2 * n_sites + 2]
    sigma_b_samps = samples[:, 2 * n_sites + 3]
    #nu_samps      = samples[:, 2 * n_sites + 4]   HS resultsに移動
    #sigma_samps   = samples[:, 2 * n_sites + 5]   HS resultsに移動
    # 予測区間計算のために results から取得
    nu_samps = np.array(results["nu_samps"]) # リストからNumPyに戻す
    sigma_samps = np.array(results["sigma_samps"]) # リストからNumPyに戻す

  
    # summarize_and_save 関数内（mu_a_samps, mu_b_samps などが定義された後）

    # --- mu_a の事後確率の計算 ---
    total_samples = len(mu_a_samps)

    # mu_a が負である確率
    prob_mu_a_negative = np.sum(mu_a_samps < 0) / total_samples
    # mu_a が正である確率 (1 から引くか、直接計算)
    prob_mu_a_positive = np.sum(mu_a_samps > 0) / total_samples

    # ※ 厳密にはゼロに等しい確率は含めないか、別途処理します。


    # 逆変換（元スケール）

    a_o_all, b_o_all = original_scale_from_samples_all(a_samps_all, b_samps_all, x_mean, x_scale)

    # 事後統計（標準化スケール）
    a_mean_std_all = a_samps_all.mean(axis=0)
    a_ci_std_all   = np.percentile(a_samps_all, [2.5, 97.5], axis=0)
    b_mean_std_all = b_samps_all.mean(axis=0)
    b_ci_std_all   = np.percentile(b_samps_all, [2.5, 97.5], axis=0)

    # 事後統計（元スケール）
    a_mean_orig_all = a_o_all.mean(axis=0)
    a_ci_all        = np.percentile(a_o_all, [2.5, 97.5], axis=0)
    b_mean_orig_all = b_o_all.mean(axis=0)
    b_ci_all        = np.percentile(b_o_all, [2.5, 97.5], axis=0)

    # 半減時間（年）
    abs_a_all   = np.abs(a_o_all)
    t_half_all  = np.where(abs_a_all > HALF_EPS, LOG10_2 / abs_a_all, np.nan)
    t_half_mean_all = np.nanmean(t_half_all, axis=0)
    t_half_ci_all   = np.nanpercentile(t_half_all, [2.5, 97.5], axis=0)

    # 傾きの事後確率（符号）
    total = a_samps_all.shape[0]
    prob_a_negative = np.sum(a_samps_all < 0, axis=0) / total
    prob_a_positive = 1.0 - prob_a_negative

    # === 観測年の集計（サイト別）
    df_years = df_source.copy()
    df_years["year"] = pd.to_numeric(df_years["year"], errors="coerce")
    df_years = df_years.dropna(subset=["year"])
    # 年が整数のみなら int に、そうでないなら float のままで
    try:
        df_years["year"] = df_years["year"].astype(int)
    except Exception:
        pass

    g_year = df_years.groupby("site_id")["year"]
    df_obs_info = pd.DataFrame({
        "site_id": g_year.apply(lambda s: s.name),
        "n_obs_years": g_year.nunique(),
        "year_start": g_year.min(),
        "year_end": g_year.max(),
        # 総レコード数も欲しければ:
        "n_records": df_years.groupby("site_id").size().values,
    }).reset_index(drop=True)

    # === OLS 推定（サイト別）
    # 標準化 x を復元してから、サイト別に np.polyfit で傾き・切片を計算
    df_ols = []
    # ここで site_ids の順に合わせて計算（site_ids はソート済み文字列の並び）
    site_set = set(df_years["site_id"].astype(str).unique())
    for sid in site_ids:
        sel = (df_years["site_id"].astype(str) == str(sid))
        sub = df_years.loc[sel, ["year", "log_conc"]].dropna()

        #-----------------
        import statsmodels.api as sm # ★ これを追加

        if len(sub) >= 2 and np.nanstd(sub["year"].values) > 1e-12:
            x_std = (sub["year"].values.astype(float) - x_mean) / (x_scale if x_scale != 0 else 1.0)
            y = sub["log_conc"].values.astype(float)
            
            # ------------------------------------------------------------------
            # ★ ここから statsmodels による OLS 回帰を実行
            # 1. 切片項（定数）を x データに追加
            X = sm.add_constant(x_std)
            
            # 2. OLS モデルを定義し、適合させる
            model = sm.OLS(y, X, missing='drop') # missing='drop' で NaN を自動的に除外
            ols_results = model.fit()
            
            # 3. 結果の取得 (statsmodels は [intercept, slope] の順で返す)
            b_std = ols_results.params[0] # 切片 (const)
            a_std = ols_results.params[1] # 傾き (x_std)
            
            # 4. 傾きの p 値を取得 (1番目のパラメータ、つまり傾きに対応)
            ols_a_pvalue = ols_results.pvalues[1] # pvalues[0] は切片のp値
            # ------------------------------------------------------------------

            # 元スケールへ変換
            a_orig = a_std / (x_scale if x_scale != 0 else 1.0)
            b_orig = b_std - a_std * (x_mean / (x_scale if x_scale != 0 else 1.0))
            
            # 辞書に追加
            df_ols.append({
                "site_id": sid,
                "ols_a_std": a_std,
                "ols_b_std": b_std,
                "ols_a_orig": a_orig,
                "ols_b_orig": b_orig,
                "ols_n_points": len(sub),
                "ols_a_pvalue": ols_a_pvalue, # ★ p値を追加
            })
        else:
            df_ols.append({
                "site_id": sid,
                "ols_a_std": np.nan,
                "ols_b_std": np.nan,
                "ols_a_orig": np.nan,
                "ols_b_orig": np.nan,
                "ols_n_points": len(sub),
                "ols_a_pvalue": np.nan, # ★ p値を追加
            })

# ----------------------------

    df_ols = pd.DataFrame(df_ols)

    # === サマリー DataFrame（サイト順は site_ids に合わせる）
    df_summary = pd.DataFrame({
        "site_id": site_ids,
        # 標準化スケール（事後）
        "a_mean_std":  a_mean_std_all,
        "a_ci_l_std":  a_ci_std_all[0],
        "a_ci_u_std":  a_ci_std_all[1],
        "b_mean_std":  b_mean_std_all,
        "b_ci_l_std":  b_ci_std_all[0],
        "b_ci_u_std":  b_ci_std_all[1],
        # 元スケール（事後）
        "a_mean_orig": a_mean_orig_all,
        "a_ci_l_orig": a_ci_all[0],
        "a_ci_u_orig": a_ci_all[1],
        "b_mean_orig": b_mean_orig_all,
        "b_ci_l_orig": b_ci_all[0],
        "b_ci_u_orig": b_ci_all[1],
        # 半減時間（年）
        "t_half_mean_year": t_half_mean_all,
        "t_half_ci_l_year": t_half_ci_all[0],
        "t_half_ci_u_year": t_half_ci_all[1],
        # 傾きの事後確率（符号）
        "Prob_a_Negative": prob_a_negative,
        "Prob_a_Positive": prob_a_positive,
        "Prob_a_Negative_%": np.array([f"{p*100.0:.4f}%" for p in prob_a_negative]),

    })

    # 観測年・OLS を結合（site_id で左結合）
    df_summary = df_summary.merge(df_obs_info, on="site_id", how="left")
    df_summary = df_summary.merge(df_ols, on="site_id", how="left")

    # 保存
    csv_path = f"{out_prefix}/{out_prefix}{CSV_PATH}"
    df_summary.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"→ site summary saved: {csv_path}")

    overall = {
        # === 既存のグローバルパラメータのサマリー ===
        "steps": results["steps"],
        "n_walkers": results["n_walkers"],
        "mu_a_mean": float(np.mean(mu_a_samps)),    #全サイトの平均的な傾きの推定値（点推定）
        # グローバル平均傾き mu_a の事後確率
        "Prob_mu_a_Negative": float(prob_mu_a_negative),    #全サイトの平均的な傾きが負である事後確率
        "Prob_mu_a_Positive": float(prob_mu_a_positive),
        "mu_a_ci": np.percentile(mu_a_samps, [2.5, 97.5]).tolist(), # 全サイトの平均的な傾きの95%信用区間（CI）
        "mu_b_mean": float(np.mean(mu_b_samps)),
        "mu_b_ci": np.percentile(mu_b_samps, [2.5, 97.5]).tolist(), # リストに変換
        "sigma_a_mean": float(np.mean(sigma_a_samps)),  #サイト間のばらつき（階層性）大きなばらつきがある場合、階層モデルを採用したことは適切
        "sigma_b_mean": float(np.mean(sigma_b_samps)),  #サイト間のばらつき（階層性）
        "nu_mean": float(np.mean(nu_samps)),    #t分布の自由度。ロバスト性: nu_meanが小さい値は、t-分布として機能している。正規分布に近いモデル（nu >>30）と評価
        "sigma_mean": float(np.mean(sigma_samps)),  #残差の標準偏差
        "acceptance_fraction": float(results.get("acceptance_fraction", np.nan)),

        # === results 辞書から追加したい情報 (NumPy型を変換) ===
        "substance": results["substance"],
        "mean_conc_input": float(results["mean"]), # キー名を明確化
        "std_conc_input": float(results["std_conc"]), # キー名を明確化

        # 'n': counts（pandas Series）をサイトごとの観測数辞書として保存
        # counts は `df.groupby("site_id").size()` の結果なので、辞書に変換
        "n_records_by_site": results["n"].to_dict(), 

        # 'valid_sites': Indexオブジェクトを文字列のリストとして保存
        #"valid_sites_used": results["valid_sites"].astype(str).tolist(), 

        #"site_ids_sorted": results["site_ids"].tolist(), # NumPy arrayをリストに変換
        "site_map": results["site_map"],
        
        "x_mean": float(results["x_mean"]),
        "x_scale": float(results["x_scale"]),

        "autocorr_tau": results["autocorr"]["tau"].tolist(), # NumPy arrayをリストに変換
        "autocorr_burn_in": int(results["autocorr"]["burn_in"]),
        "autocorr_thin": int(results["autocorr"]["thin"]),
        
        # 'labeled' (Rhatのリスト) を保存
        "rhat_labeled": results["labeled"], # NumPy arrayをリストに変換
        "bad_idx": results["bad_idx"],  
        "bad_rhat": results["bad_rhat"],        
        # ※ 注意: results["samples"] (flat_samples) は巨大なので通常は保存しません
    }
    os.makedirs(out_prefix, exist_ok=True)

    # 1. bad_idx のリストが空かどうか（=問題のあるインデックスが存在しないかどうか）を判定
    has_bad_indices = len(results["bad_rhat"]) > 0

    # 2. 条件に基づいてファイル名を決定
    if has_bad_indices:
        # 問題がある場合（bad_idxが空でない場合）は、ファイル名にそれを明記
        file_name = f"badidx_{has_bad_indices}_{out_prefix}_overall_summary.json"
    else:
        # 問題がない場合（bad_idxが空の場合）は、通常のファイル名
        file_name = f"{out_prefix}_overall_summary.json"


    overall_path = f"{out_prefix}/{file_name}"
    with open(overall_path, "w", encoding="utf-8") as f:
        # NumPyのデータ型を標準Python型に変換済みなので、そのままダンプ
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"→ overall summary saved: {overall_path}")


    overall_path = f"{out_prefix}_overall_summary.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"→ overall summary saved: {overall_path}")

    # ===== バープロット（元スケールの傾き） with 95%CI =====
    plt.figure(figsize=(10, 5))
    order = df_summary.sort_values("a_mean_orig", ascending=False)
    x_labels = order["site_id"].astype(str).values
    y_means  = order["a_mean_orig"].values
    yerr_lower = y_means - order["a_ci_l_orig"].values
    yerr_upper = order["a_ci_u_orig"].values - y_means
    yerr = np.vstack([yerr_lower, yerr_upper])

    plt.bar(x_labels, y_means, color="#1f77b4", alpha=0.85)
    plt.errorbar(x_labels, y_means, yerr=yerr, fmt='none',
                 ecolor='black', elinewidth=1.2, capsize=3, capthick=1.2, zorder=10)
    plt.axhline(0, linestyle='--', color='gray', linewidth=1)
    plt.ylabel("傾きの事後平均 (元スケール)")
    plt.xlabel("Site ID")
    plt.title(f"{out_prefix} 各サイトの傾き (t回帰)")
    plt.xticks(rotation=90)
    plot_file = os.path.join(plot_dir, f"{out_prefix}_slopes_t.png")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=DPI_PLOT)
    plt.close()
    print(f"→ slope plot saved: {plot_file}")

    # （任意）半減時間ヒストグラム
    try:
        plt.figure(figsize=(8, 4))
        df_th = df_summary.dropna(subset=["t_half_mean_year"])
        plt.hist(df_th["t_half_mean_year"], bins=20, color="#2ca02c", alpha=0.8)
        plt.xlabel("半減時間（年）")
        plt.ylabel("サイト数")
        plt.title(f"{out_prefix} 半減時間（サイト別平均）分布")
        hist_file = os.path.join(plot_dir, f"{out_prefix}_t_half_hist.png")
        plt.tight_layout()
        plt.savefig(hist_file, dpi=DPI_PLOT)
        plt.close()
        print(f"→ half-life histogram saved: {hist_file}")
    except Exception as e:
        print(f"[warn] 半減ヒストグラム出力に失敗: {type(e).__name__}: {e}")

    return df_summary


# ------------------------------
# MCMC結果の保存（軽量）
# ------------------------------

def save_mcmc_samples(results: dict, out_prefix: str = "results"):
    print(f"MCMCサンプルデータを保存中.......")
    save_path = f"{out_prefix}/{out_prefix}_mcmc_samples.npz"
    np.savez_compressed(
        save_path,
        samples=results["samples"],
        site_ids=np.array(results["site_ids"], dtype=str),
        x_mean=results.get("x_mean", 0.0),
        x_scale=results.get("x_scale", 1.0),
    )
    print(f"→ MCMCサンプルデータを保存しました: {save_path}")
    return save_path


# サイト診断プロット
# ------------------------------

def diagnose_site_fit(df: pd.DataFrame, results: dict, site_original_id: str | int, out_dir: str = "diagnostics"):
    os.makedirs(out_dir, exist_ok=True)
    site_ids = list(map(str, results.get("site_ids", [])))
    try:
        i = site_ids.index(str(site_original_id))
    except ValueError:
        print(f"[diagnose] site_id {site_original_id} が results['site_ids'] に見つかりません。")
        return None

    site_mask = df['site_id'].astype(str) == str(site_original_id)
    df_site = df[site_mask].copy()
    if df_site.shape[0] == 0:
        print(f"[diagnose] 元データに site {site_original_id} が見つかりません。")
        return None

    years = df_site['year'].astype(float).values
    conc = df_site['concentration_sub'].astype(float).values
    logc = np.log10(conc)

    x_plot = np.arange(int(years.min()), int(years.max()) + 1)
    X_ols = years - years.min()

    # OLS
    ols_slope = np.nan
    ols_intercept = np.nan
    ols_y = np.full_like(x_plot, np.nan, dtype=float)
    try:
        if len(X_ols) >= 2 and X_ols.std() > 1e-6:
            ols_coef = np.polyfit(X_ols, logc, 1)
            ols_slope = ols_coef[0]
            ols_intercept = ols_coef[1]
            ols_y = ols_slope * (x_plot - years.min()) + ols_intercept
    except Exception:
        pass

    x_mean = results.get('x_mean', years.mean())
    x_scale = results.get('x_scale', 1.0)
    samples = results['samples']
    n_sites = len(site_ids)
    a_samps = samples[:, :n_sites]
    b_samps = samples[:, n_sites:2 * n_sites]
    a_site = a_samps[:, i]
    b_site = b_samps[:, i]

    a_o = a_site / x_scale
    b_o = b_site - a_o * x_mean
    post_a = float(np.mean(a_o))
    post_b = float(np.mean(b_o))

    # 各サンプルから予測値を生成（予測平均の不確実性のみ）
    post_y_samps = np.array([a_o[j] * x_plot + b_o[j] for j in range(len(a_o))])
    post_y = np.mean(post_y_samps, axis=0)
    # 95% 信用区間 (CI) の計算 回帰平均の95% CI
    y_low = np.percentile(post_y_samps, 2.5, axis=0)
    y_high = np.percentile(post_y_samps, 97.5, axis=0)
    #-----------------

    #予測区間
    import scipy.stats as st
    # ※ results 辞書内に sigma_samps と nu_samps があると仮定
    nu_samps = np.array(results["nu_samps"]) # リストからNumPyに戻す
    sigma_samps = np.array(results["sigma_samps"]) # リストからNumPyに戻す
    #sigma_samps = results['sigma_samps'] 
    #nu_samps = results['nu_samps']
    # post_y_samps を予測区間用のサンプリング結果として再定義
    post_y_pred_samps = []

    for j in range(len(a_o)):
        # 1. 平均値 mu を計算
        mu = a_o[j] * x_plot + b_o[j]
        
        # 2. そのMCMCステップでの sigma と nu を取得
        sigma_j = sigma_samps[j]
        nu_j = nu_samps[j]
        
        # 3. t分布から予測値（ノイズ込み）をサンプリング
        # 各x_plotの点に対して、平均 mu、スケール sigma、自由度 nu の t分布からランダムにサンプリング
        # ※ t分布のスケールパラメータは標準偏差に対応します
        
        # t分布の予測分布は解析的に難しいことが多いので、ここでは単純にノイズとしてt分布を使う例
        t_noise = st.t.rvs(df=nu_j, loc=0, scale=sigma_j, size=len(x_plot))
        y_pred_j = mu + t_noise # ノイズを加えて予測値を生成
        
        post_y_pred_samps.append(y_pred_j)

    # 95% 予測区間 (PI) の計算
    post_y_pred_samps = np.array(post_y_pred_samps)
    y_low_pi = np.percentile(post_y_pred_samps, 2.5, axis=0)
    y_high_pi = np.percentile(post_y_pred_samps, 97.5, axis=0)
        

    #   
    # post_y_pred_samps の計算が完了していると仮定
    # post_y_pred_samps = ... (前の手順で計算済み)
    post_y_pred_samps = np.array(post_y_pred_samps)
    y_low_pi = np.percentile(post_y_pred_samps, 2.5, axis=0)
    y_high_pi = np.percentile(post_y_pred_samps, 97.5, axis=0)


    # --- 回帰直線（平均値）の95% 信用区間（Credible Interval: CI）事後信頼区間のプロット ---
    plt.figure(figsize=(8, 5))
    plt.scatter(years, logc, label='observed log10(conc)')

    if not np.isnan(ols_slope):
        plt.plot(x_plot, ols_y, linestyle='--', label=f'OLS slope={ols_slope:.3f}')
        
    plt.plot(x_plot, post_y, linestyle='-', label=f'Posterior mean slope={post_a:.3f}')

    # 1. 95% 予測区間 (PI) を描画 (より広い範囲、濃いグレー/低い透明度)
    # y_low_pi と y_high_pi を使用
    plt.fill_between(x_plot, y_low_pi, y_high_pi, 
                    color='darkred', alpha=0.1, 
                    label='95% PI (Prediction Interval)') # <-- PIを追加

    # 2. 95% 信用区間 (CI) を描画 (より狭い範囲、元の色)
    # y_low と y_high を使用
    plt.fill_between(x_plot, y_low, y_high, 
                    color='gray', alpha=0.4, 
                    label='95% CI (Credible Interval)') # <-- CIを維持

    plt.xlabel("Year"); plt.ylabel("log10(conc)")
    plt.title(f"{out_dir} site {site_original_id}: OLS vs Posterior")
    plt.legend()
    png = os.path.join(out_dir, f"{out_dir}_{site_original_id}.png")
    plt.tight_layout(); plt.savefig(png, dpi=150); plt.close()

    print(f"[diagnose] プロット保存: {site_original_id}.png")
        

    if MAKE_TRACE_PLOT:
        # --- トレースプロット（復活） ---
        plt.figure(figsize=(8, 3))
        plt.plot(a_site, alpha=0.6)
        plt.title(f"Trace of a_{site_original_id} (Std Scale)")
        plt.ylabel("a value (Std)"); plt.xlabel("sample index")
        p2 = os.path.join(out_dir, f"trace_a_{site_original_id}.png")
        plt.tight_layout(); plt.savefig(p2, dpi=DPI_PLOT); plt.close()
        print(f"[diagnose] trace 保存: {site_original_id}.png")

    # 傾きaの事後分布ヒストグラム
    plt.figure(figsize=(8, 4))
    plt.hist(a_o, bins=50, density=True, alpha=0.6, color='steelblue')
    ci_low, ci_high = np.percentile(a_o, [2.5, 97.5])
    plt.axvline(post_a, color='red', linestyle='-', label=f"mean={post_a:.3f}")
    plt.axvline(ci_low, color='black', linestyle='--', label=f"2.5%={ci_low:.3f}")
    plt.axvline(ci_high, color='black', linestyle='--', label=f"97.5%={ci_high:.3f}")
    plt.title(f"{out_dir} Posterior distribution of slope a (site {site_original_id})")
    plt.xlabel("a value"); plt.ylabel("Density")
    plt.legend()
    p3 = os.path.join(out_dir, f"{out_dir} posterior_a_{site_original_id}.png")
    plt.tight_layout(); plt.savefig(p3, dpi=150); plt.close()
    print(f"[diagnose] posterior 保存: {site_original_id}.png")

    return {
        "site": site_original_id,
        "ols_slope": ols_slope,
        "posterior_mean_a": post_a,
        "posterior_ci_a": (ci_low, ci_high),
        #"plots": (png, p3, p2),
        "plots": (png, p3, p2) if MAKE_TRACE_PLOT else (png, p3),
    }

# ------------------------------
# マッピング確認
# ------------------------------

def check_site_mapping(df: pd.DataFrame, results: dict):
    df_order = list(map(str, sorted(df["site_id"].astype(str).unique())))
    res_order = list(map(str, results["site_ids"]))
    print("DF side site_id order:", df_order)
    print("RESULTS side site_id order:", res_order)
    print("Equal? →", df_order == res_order)

# ------------------------------
# 一括プロット（フラグで制御）
# ------------------------------

def export_site_plots(df: pd.DataFrame, results: dict, df_summary: pd.DataFrame,
                      out_dir: str = "diagnostics_all"):
    if not MAKE_DIAG_PLOTS:
        print("診断プロットはスキップ（MAKE_DIAG_PLOTS=False）")
        return None

    os.makedirs(out_dir, exist_ok=True)
    site_ids = results["site_ids"]

    # サイト指定がある場合はフィルタリング
    if MAKE_SITE_PLOTS:
        target_ids = [sid for sid in site_ids if str(sid) in MAKE_SITE_PLOTS]
    else:
        target_ids = site_ids  # 指定がなければ全サイト

    for sid in target_ids:
        diagnose_site_fit(df, results, site_original_id=sid, out_dir=out_dir)

    return None

# ------------------------------
# メイン
# ------------------------------

def main():


    for filename in filenames:
        print(f"\n\n==================== ファイル処理開始: {filename} ====================")
        filepath = f"data/{filename}.csv"
        if not os.path.exists(filepath):
            print(f"ファイルが見つかりません: {filepath}")
            continue

        # 読み込み＋クリーニング
        df_all = pd.read_csv(
            filepath,
            usecols=['substance', 'year', 'site_id', 'concentration', 'detection_limit'],
            dtype={'substance': str, 'site_id': str}
        )
        df_all['concentration'] = df_all['concentration'].apply(clean_value_user)
        df_all['detection_limit'] = df_all['detection_limit'].apply(clean_value_user)

        # ND 代入
        is_nondetect = (df_all['concentration'] <= df_all['detection_limit']) & (~df_all['detection_limit'].isna())
        substitute_value = df_all['detection_limit'] / np.sqrt(2)
        df_all['concentration_sub'] = df_all['concentration'].copy()
        df_all.loc[is_nondetect, 'concentration_sub'] = substitute_value[is_nondetect]

        df_all['log_conc'] = np.log10(df_all['concentration_sub'].replace(0, np.nan))
        df_all = df_all.replace([np.inf, -np.inf], np.nan)
        df_all = df_all.dropna(subset=['log_conc', 'detection_limit', 'concentration'])


        # === ここで「グローバル除外」 ===
        min_samples = 2  # 2以下を除外 → 少なくとも 2以上を残す
        counts_all = df_all.groupby('site_id').size()               # site ごとの総観測数
        valid_sites_all = counts_all[counts_all >= min_samples].index
        removed_sites = set(df_all['site_id'].unique()) - set(valid_sites_all)

        df_all = df_all[df_all['site_id'].isin(valid_sites_all)].copy()

        print(f"ファイル名: {filename} 内で観測数（全データ数） {min_samples-1} 以下で除外された site: {sorted(list(removed_sites))}")
        print(f"残存 site 数: {df_all['site_id'].nunique()} / 総レコード数: {len(df_all)}")


        # 「if substance_data:」だけで「リストに中身があれば」という意味になります
        if substance_data:
            substances = substance_data 
        else:
            substances = df_all['substance'].unique()
        print(f"対象物質: {substances}")

        for substance in substances:
            print(f"\n==================== {substance} ====================")
            df = df_all[df_all['substance'] == substance].copy()

            #"""-----------
            # --- データ準備　パラメータ定義 ---
            # サイトごとの最小観測数（11以上を残す）
            # --------------------

            # site 観測数チェック (countsは pandas.Series, インデックスが site_id)
            counts = df.groupby("site_id").size()

            # 観測数が MIN_OBS_FOR_SITE 以上のサイトIDのみを取得
            valid_sites = counts[counts > MIN_OBS_FOR_SITE].index 

            # 除外されたサイトIDを特定 (診断用)
            removed_sites_local = set(df['site_id'].unique()) - set(valid_sites)

            # 除外処理: データフレーム df をフィルタリング
            df = df[df["site_id"].isin(valid_sites)].copy()

            # 診断結果の出力
            print(f"  > {substance}のデータ数が {MIN_OBS_FOR_SITE}以下のサイトを除外しました。")
            print(f"    - 除外されたサイト: {removed_sites_local}")
            print(f"    - 除外されたサイト数: {len(removed_sites_local)}")
            print(f"    - 残存サイト数: {df['site_id'].nunique()} / 総レコード数: {len(df)}")
            
            #------------------"""
            # --- データ準備　 パラメータ定義 ---
            #MIN_END_YEAR = 2020   # 最終観測年がこの年以下の場合に除外 (2016年以上のサイトを残す)
            # --------------------

            # site ごとの年データの最大値 (最終観測年) を取得
            g_year = df.groupby("site_id")["year"]
            df_year_info = pd.DataFrame({
                "site_id": g_year.apply(lambda s: s.name),
                "year_end": g_year.max(),
            }).reset_index(drop=True)
            # 最終観測年が MIN_END_YEAR より大きいサイトIDのみを取得 (例: 2016年以上のサイト)
            valid_sites_by_year = df_year_info[df_year_info["year_end"] > MIN_END_YEAR]["site_id"]

            # 除外されたサイトIDを特定 (診断用)
            all_sites_current = set(df['site_id'].unique())
            removed_sites_year = all_sites_current - set(valid_sites_by_year)

            # データフレーム df をフィルタリング
            df = df[df["site_id"].isin(valid_sites_by_year)].copy()

            # 診断結果の出力
            print(f"  > 最終観測年 {MIN_END_YEAR}以下で除外されたサイト: {removed_sites_year}")
            print(f"    - 最終観測年 {MIN_END_YEAR}以下で除外されたサイト数: {len(removed_sites_year)}")
            print(f"    - 残存サイト数: {df['site_id'].nunique()} / 総レコード数: {len(df)}")

            # --------------------
            # 年度の型変換が確実に行われているか再確認
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year']).copy() 
            df['year'] = df['year'].astype(int) 

            
            # 条件1: year > 2015 (2016年以降)
            condition_min = (df['year'] > MIN_ANALYSIS_YEAR)       
            condition_max = (df['year'] <= MAX_ANALYSIS_YEAR)

            # 両方の条件を満たすレコードのみを抽出
            df_filtered = df[condition_min & condition_max].copy() 

            # 診断結果の出力
            original_records = len(df)
            filtered_records = len(df_filtered)

            print(f"  > 観測年 [{MIN_ANALYSIS_YEAR + 1}年〜{MAX_ANALYSIS_YEAR}年] のデータに限定しました。")
            print(f"    - 除外されたレコード数: {original_records - filtered_records}")
            print(f"    - 残存サイト数: {df_filtered['site_id'].nunique()} / 残存レコード数: {filtered_records}")

            # ★ この後の df_filtered を使って MCMC 解析にかけます
            df = df_filtered
            # ----------------------------------------------------
            # concentrationの平均と標準偏差を計算
            mean_conc = df['concentration'].mean()
            std_conc = df['concentration'].std()
            
            # 出力
            print(f"濃度 (concentration) の平均: {mean_conc:.4f}")
            print(f"濃度 (concentration) の標準偏差: {std_conc:.4f}")
            # ----------------------------------


            # 年度整形
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(float)

            # site 観測数チェック
            counts = df.groupby("site_id").size()
            valid_sites = counts[counts >= 2].index
            df = df[df["site_id"].isin(valid_sites)]
            if df['site_id'].nunique() < 10:                 #観測数を設定
                print("地点数が不足 → スキップ")
                continue
            if len(df) < 10:
                print("総観測数が不足 → スキップ")
                continue

            #===================="階層ベイズ解析（t分布モデル）========================
            print("階層ベイズ解析（t分布モデル）を開始します...")
            sampler, site_map, x_mean, x_scale, labeled, bad_idx, bad_rhat = run_mcmc_t(df)
            #========================================================================

            try:
                # 代表パラメータだけで autocorr を計算 概ね失敗するので、Exceptionが大事
                tau = sampler.get_autocorr_time(var_names=["mu_a"], quiet=True)
                tau_max = float(np.max(tau))
                burn_in = min(max(300, int(20 * tau_max)), 2000)
                thin    = 100  # flattenを軽くするため強制thin
            except Exception:
                tau     = np.array([])
                tau_max = np.nan
                burn_in = BURN_IN   #5000
                thin    = THIN      #100

            t2 = time.perf_counter()
            flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
            t3 = time.perf_counter()
            print(f"[timer] flatten: {t3 - t2:.2f}s (burn_in={burn_in}, thin={thin})")
            # === ウォーカー別トレースの表示 ===

            plot_walker_traces(sampler, param_index=0, burn_in=burn_in, max_walkers=30, title="walker trace: slope (param_index=0)")

            # 指定サイトIDでの傾きを表示　オプション
            sid = "10"  # 本当にこの文字列が site_map に存在するか要確認
            n_sites = len(site_map)
            try:
                pidx_slope, pidx_intercept = get_param_indices_for_site(sid, site_map, n_sites)
                print(f"site_id='{sid}' → index={site_map[str(sid)]}, pidx_slope={pidx_slope}, pidx_intercept={pidx_intercept}")
                plot_walker_traces(sampler, param_index=pidx_slope, burn_in=burn_in, max_walkers=30,
                                title=f"walker trace: slope (site_id={sid})")
            except KeyError as ke:
                print(ke)
                # デバッグ用に site_map を少し表示してあげる
                print("site_map の先頭5件:", list(site_map.items())[:5])

            # 結果を辞書　必要な出力情報はここで確保する。samplerが本体で重い。
            site_ids_sorted = np.array(sorted(df["site_id"].astype(str).unique(), key=lambda s: site_map[s]))
            results = {
                "min_data_count": MIN_OBS_FOR_SITE,  # サイトごとの最小データ列
                "min_end_year": MIN_END_YEAR,   # 調査最終年度
                "min_anal_year": MIN_ANALYSIS_YEAR,
                "max_anal_year": MAX_ANALYSIS_YEAR, 
                "steps": STEPS,
                "n_walkers": N_WALKERS,       # 400 → 360（下限は 2×ndim=276 を厳守）
                "p0_spred": P0_SPREAD,
                "autocorr": {
                    "tau": tau,
                    "burn_in": burn_in,
                    "thin": thin,
                },

                "substance":substance,
                "mean": mean_conc,
                "std_conc": std_conc,
                "n": counts,
                "valid_sites": counts[counts >= 2].index, #これは誤りか？？
                "sampler": sampler,
                "samples": flat_samples,
                "site_ids": site_ids_sorted,
                "site_map": site_map,
                "x_mean": x_mean,
                "x_scale": x_scale,

                "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
                "labeled": labeled,
                "bad_idx": bad_idx.tolist(),
                "bad_rhat": bad_rhat.tolist(),
                #"nu_samps":nu_samps,
                #"sigma_samps" :sigma_samps,

            }
            #　95% 予測区間計算のため　"nu_samps":nu_samps,  #"sigma_samps" :sigma_samps,を追加する
            # n_sites と n_params_per_site を利用してインデックスを計算
            # 2 * n_sites + 4 が nu_samps、2 * n_sites + 5 が sigma_samps のインデックス
            n_sites_in_samples = len(site_ids_sorted) # site_ids_sorted は results 辞書に入れる直前に定義済み

            # グローバルパラメータのインデックスを計算 (samples配列内での位置)
            # A: 2*n_sites
            idx_nu = 2 * n_sites_in_samples + 4
            idx_sigma = 2 * n_sites_in_samples + 5

            # flat_samples から該当のサンプル配列を抽出
            # 注意: flat_samples は既に burn_in と thin 処理が適用されています。
            nu_samps = flat_samples[:, idx_nu]
            sigma_samps = flat_samples[:, idx_sigma]

            # results 辞書にグローバルパラメータのサンプル配列を追加
            # JSONシリアライズエラーを避けるため、抽出後にリストに変換します。
            results["nu_samps"] = nu_samps.tolist()
            results["sigma_samps"] = sigma_samps.tolist()

            # --------------------------------------------------------------------------------------

            print("=== MCMC diagnostics ===")
            print("収束が不十分なパラメータ:", bad_idx)
            print("収束が不十分なパラメータ:", bad_rhat)
            print("autocorr tau:", tau)
            print("tau max:", tau_max)           
            print("flattened samples length:", len(flat_samples))  
            #print("effective samples:", len(flat_samples)/tau_max)  # 参考値（厳密な ESS ではない）
            print(f"acceptance fraction: {results['acceptance_fraction']:.4f}")
            out_prefix = f"{filename}_{substance}".replace(" ", "_")

            if PLOT_DIR:
                df_summary = summarize_and_save(results, df_source=df,  out_prefix=out_prefix, plot_dir=PLOT_DIR)
            else:
                df_summary = summarize_and_save(results, df_source=df,  out_prefix=out_prefix, plot_dir=out_prefix)
            # μ_a のまとめ
            n_sites = len(results['site_ids'])
            mu_a_samples = results['samples'][:, 2 * n_sites]
            print(f"全体傾き μ_a 平均: {np.mean(mu_a_samples):.4f}")
            print("全体傾き μ_a 95%CI:", np.percentile(mu_a_samples, [2.5, 97.5]))
            check_site_mapping(df, results)

            if SAVE_NPZ:
                save_mcmc_samples(results, out_prefix=out_prefix)

            #export_site_plots(df, results, df_summary, out_dir=f"{filename}_{substance}_diagnose")
            export_site_plots(df, results, df_summary, out_dir=f"{filename}_{substance}")
    print("\nすべて完了しました。")

if __name__ == "__main__":
    main()