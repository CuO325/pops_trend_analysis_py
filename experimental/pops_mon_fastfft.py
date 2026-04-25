# pops_idea_4_update_fastfft.py
# 完全統合版：FFT高速KDE を用いたベイズ（MCMC）と既存解析フローの統合
# 既存の pops_idea_4_update.py をこのファイルで置き換えてください。

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

# ---------------------------
# 高速 FFT-KDE ユーティリティ
# ---------------------------

def build_fft_kde_lookup(values, grid_min=None, grid_max=None, grid_n=4096, bandwidth=None, bw_factor=1.0):
    """
    FFT ベースで KDE を計算して lookup (grid->pdf) を作る。
    - values: 1D array のデータ（残差）
    - grid_min/grid_max: グリッドの範囲。None なら data 範囲をベースに余白を取る。
    - grid_n: 格子点数（2^n に近いほうが速いが任意）
    - bandwidth: ガウスカーネルの標準偏差。None なら scott を利用
    - bw_factor: bandwidth に乗じる係数（1.0 が標準）
    戻り値: dict { 'grid': grid, 'pdf': pdf_on_grid, 'interp': interp_func, 'bw': bw }
    """
    values = np.asarray(values).astype(float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("build_fft_kde_lookup: 入力 values が空です。")
    # グリッド範囲
    vmin = values.min() if grid_min is None else grid_min
    vmax = values.max() if grid_max is None else grid_max
    # 余白を少し取る（裾の評価のため）
    rng = vmax - vmin
    if rng == 0:
        rng = max(1.0, abs(vmin))
    pad = rng * 0.5 + 1e-6
    gmin = vmin - pad
    gmax = vmax + pad
    # grid
    grid = np.linspace(gmin, gmax, grid_n)
    dx = grid[1] - grid[0]

    # bandwidth: scott rule に基づく推定（Gaussianカーネルの標準偏差）
    if bandwidth is None:
        # scott: n^{-1/5} * std
        sigma = np.std(values)
        n = len(values)
        bw = sigma * (n ** (-1.0/5.0))
        if bw <= 0 or np.isnan(bw):
            bw = max(1e-3, sigma if sigma>0 else 1.0)
    else:
        bw = float(bandwidth)
    bw *= bw_factor

    # データを grid にヒストグラム化（密度ではなく頻度）
    # histogram counts
    counts, _ = np.histogram(values, bins=grid_n, range=(gmin, gmax))
    # normalize to density: counts / (n * dx)
    nvals = len(values)
    counts = counts.astype(float) / (nvals * dx)

    # ガウスカーネルを grid 上に discretize
    # kernel grid: center at 0
    kx = np.linspace(- (grid_n//2)*dx, (grid_n//2)*dx, grid_n)
    kernel = np.exp(-0.5 * (kx / bw)**2)
    kernel /= (np.sum(kernel) * dx)  # normalize to unit area

    # 畳み込み（FFT）
    conv = fftconvolve(counts, kernel, mode='same')  # same length as counts
    pdf_on_grid = conv
    # pdf_on_grid は頻度的に grid の bin 中心に対応。補間を用意する。
    # define centers of bins
    bin_centers = np.linspace(gmin + dx/2.0, gmax - dx/2.0, grid_n)
    # For safety, clip tiny negatives
    pdf_on_grid = np.clip(pdf_on_grid, 0.0, None)
    # interp function (linear), extrapolate with tiny value
    interp = interp1d(bin_centers, pdf_on_grid, bounds_error=False, fill_value=1e-300)
    return {'grid': bin_centers, 'pdf': pdf_on_grid, 'interp': interp, 'bw': bw, 'dx': dx, 'gmin': gmin, 'gmax': gmax}


def fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup):
    """
    FFT-KDE の lookup を使った高速な対数尤度計算。
    - lookup: build_fft_kde_lookup が返す dict
    """
    residuals = y - (a * x_centered + b)
    pdf_vals = lookup['interp'](residuals)
    # underflow回避
    return np.sum(np.log(pdf_vals + 1e-12))


# 安定な gaussian_kde のフォールバック（小データ等）
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

# ---------------------------
# 事前・事後・MCMC 関連（FFT-KDE版）
# ---------------------------

def log_prior_informative(params, data_scale=None):
    a, b = params
    if not (-100 < a < 100 and -100 < b < 100):
        return -np.inf
    if data_scale is None:
        return - (a**2) / (2 * 0.5**2) - (b**2) / (2 * 2.0**2)
    else:
        x_range = data_scale.get('x_range', 1.0)
        y_std = data_scale.get('y_std', 1.0)
        a_sd = max(0.01, y_std / max(1e-6, x_range) * 2.0)
        b_sd = max(0.1, y_std * 2.0)
        return - (a**2) / (2 * a_sd**2) - (b**2) / (2 * b_sd**2)

def log_posterior_fast(params, x_centered, y, lookup, data_scale=None):
    lp = log_prior_informative(params, data_scale=data_scale)
    if not np.isfinite(lp):
        return -np.inf
    a, b = params
    ll = fast_kde_log_likelihood_from_residuals(a, b, x_centered, y, lookup)
    return lp + ll

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
    FFT-KDE + emcee の自動早期停止版（高速）
    - fixed_lookup: 事前に build_fft_kde_lookup の結果を渡せる
    - grid_n, bw_factor: lookup の設定
    - n_walkers, n_steps, chunk: MCMC 制御
    - ci_width_threshold, stable_rounds: 早期停止条件
    """
    # データ準備
    years = df['year'].values.astype(float)
    x_base = years.min()
    x_centered = years - x_base
    y = df['log_conc'].values.astype(float)
    ndim = 2

    # MLE 初期推定（残差の代表を得るため）
    try:
        # we use gaussian_kde(y) as initial kernel to optimize a,b — but it's just for center
        mle_res = minimize(
            lambda p: -kde_log_likelihood_from_residuals_plain(p[0], p[1], x_centered, y),
            x0=[0.0, np.mean(y)],
            method='L-BFGS-B'
        )
        a_mle, b_mle = mle_res.x
    except Exception:
        a_mle, b_mle = 0.0, np.mean(y)

    # build lookup if not provided (residuals from MLE)
    if fixed_lookup is None:
        residuals_mle = y - (a_mle * x_centered + b_mle)
        lookup = build_fft_kde_lookup(residuals_mle, grid_n=grid_n, bandwidth=None, bw_factor=bw_factor)
    else:
        lookup = fixed_lookup

    data_scale = {'x_range': x_centered.max() - x_centered.min(), 'y_std': np.nanstd(y)}

    # initial positions
    rng = np.random.default_rng(int(time.time() % 2**31))
    if n_walkers < 2*ndim:
        n_walkers = max(2*ndim, n_walkers)
    # MLEを中心に適度に分散させた初期点
    initial_pos = np.vstack([np.array([a_mle, b_mle]) + rng.normal(scale=[0.02, 0.1], size=2) for _ in range(n_walkers)])

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_fast, args=(x_centered, y, lookup, data_scale))

    pos = initial_pos
    n_chunks = max(1, n_steps // chunk)
    stable_count = 0
    total_steps_done = 0

    for i in range(n_chunks):
        # run chunk steps
        pos, _, _ = sampler.run_mcmc(pos, chunk, progress=False)
        total_steps_done += chunk

        # partial chain values (flattened)
        chain_partial = sampler.get_chain(flat=True)
        a_partial = chain_partial[:, 0]
        ci = np.percentile(a_partial, [2.5, 97.5])
        a_mean = np.mean(a_partial)
        p_neg = np.mean(a_partial < 0)

        if verbose:
            print(f"[FastMCMC {min(total_steps_done, n_steps)}/{n_steps}] a_mean={a_mean:.4f} CI={ci} P(a<0)={p_neg:.4f}")

        ci_width = ci[1] - ci[0]
        ci_is_stable = (ci_width < ci_width_threshold)
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

        # safety: don't exceed requested steps
        if total_steps_done >= n_steps:
            break

    total_steps = sampler.iteration
    discard = int(burn_frac * total_steps)
    samples = sampler.get_chain(discard=discard, flat=True)

    if samples.size == 0:
        # fallback: take last chain flattened
        samples = sampler.get_chain(flat=True)

    a_samps = samples[:, 0]
    b_samps = samples[:, 1]

    # diagnostics Rhat
    chain3d = np.transpose(sampler.get_chain()[:,:,:], axes=(1,0,2))
    Rhat = compute_Rhat(chain3d)

    result = {
        "samples": samples,
        "a_mean": np.mean(a_samps),
        "b_mean": np.mean(b_samps),
        "a_ci": np.percentile(a_samps, [2.5, 97.5]),
        "b_ci": np.percentile(b_samps, [2.5, 97.5]),
        "p_neg_slope": np.mean(a_samps < 0),
        "steps_completed": total_steps,
        "stopped_early": stable_count >= stable_rounds,
        "Rhat": Rhat,
        "lookup": lookup
    }
    return result

# --- 補助：MLEで通常KDE尤度を用いる簡易関数（最小用）
def kde_log_likelihood_from_residuals_plain(a, b, x_centered, y):
    res = y - (a * x_centered + b)
    try:
        kde = gaussian_kde(res)
        pdf_vals = kde.pdf(res)
        return np.sum(np.log(pdf_vals + 1e-12))
    except Exception:
        return -1e20

# ---------------------------
# 既存の解析関数群（元ファイルのロジックをほぼそのまま）
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

def run_mle_and_aic(df):
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
    return a1, b1, aic0, aic1, p1, x_base

def binomial_trend_analyze(df):
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

# ---------------------------
# メイン処理（既存フローに FFT-KDE ベイズを統合）
# ---------------------------

if __name__ == "__main__":
    filenames = ["生物"]
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

            a, b, aic0, aic1, p1, x_base = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ci_bs = np.array([np.nan, np.nan])
            a_bin, p1_bin = np.nan, np.nan
            a_bayes, ci_a, bayes_trend = np.nan, [np.nan, np.nan], "未実行"

            # ① MLE + AIC
            try:
                a, b, aic0, aic1, p1, x_base = run_mle_and_aic(df)
                print("--- 1. 対数線形回帰モデル (MLE) ---")
                print(f"傾き a = {a:.4f}, 切片 b = {b:.4f}, AIC事後確率 p1 = {p1:.4f}")
                trend = "傾向あり" if p1 >= 0.95 else "傾向なし"
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
                # MLE 残差で lookup を作成（bw_factor を調整して滑らかさを制御）
                residuals_mle = y_true - (a * x_centered + b)
                # grid_n: 4096 は多め。必要なら減らしてください(2048等)
                lookup = build_fft_kde_lookup(residuals_mle, grid_n=4096, bw_factor=1.0)
                # 高速 MCMC 実行
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
                    ci_width_threshold=0.03,  # 少し緩めにして早期終了を取りやすく
                    stable_rounds=1
                )
                a_bayes = bayes_res["a_mean"]
                b_bayes = bayes_res["b_mean"]
                ci_a = bayes_res["a_ci"]
                ci_b = bayes_res["b_ci"]
                p_neg_slope = bayes_res["p_neg_slope"]
                print(f"ベイズ推定 a = {a_bayes:.4f} (95% CI {ci_a})")
                print(f"ベイズ推定 b = {b_bayes:.4f} (95% CI {ci_b})")
                print(f"減少傾向の事後確率 P(a < 0) = {p_neg_slope:.4f}")
                bayes_trend = "傾向あり" if (a_bayes < 0 and p_neg_slope >= 0.95) else "傾向なし"
                print(f"→ {bayes_trend} (ベイズ)")
            except Exception as e:
                print("ベイズ推定失敗:", e)
                a_bayes, ci_a, bayes_trend = np.nan, [np.nan, np.nan], "エラー"

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
                print("\n--- 4. 二項回帰モデル ---")
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
                ax.set_title(f'{substance} の対数線形回帰\nMLE p1 = {p1:.4f} / ベイズ P(a<0) = {p_neg_slope:.4f} ({filename})')
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
                'intercept_MLE': b,
                'p1_MLE': p1,
                'trend_MLE': trend,
                'bootstrap_CI': ci_bs,
                'bs_trend': bs_trend,
                'binomial_slope': a_bin,
                'binomial_p1': p1_bin,
                'binomial_trend': trend_bin,
                'bayes_slope': a_bayes,
                'bayes_CI_a': ci_a,
                'bayes_trend': bayes_trend,
                'reliability': reliability
            })

        # 整形・保存
        for r in results:
            r['bootstrap_CI'] = [round(r['bootstrap_CI'][0], 4), round(r['bootstrap_CI'][1], 4)]
            r['bayes_CI_a'] = [round(r['bayes_CI_a'][0], 4), round(r['bayes_CI_a'][1], 4)]
            r['slope_MLE'] = round(r['slope_MLE'], 4)
            r['intercept_MLE'] = round(r['intercept_MLE'], 4)
            r['binomial_slope'] = round(r['binomial_slope'], 4)
            r['p1_MLE'] = round(r['p1_MLE'], 4)
            r['binomial_p1'] = round(r['binomial_p1'], 4)
            r['bayes_slope'] = round(r['bayes_slope'], 4)

        df_results = pd.DataFrame(results)
        print("\n--- 最終結果（テーブル形式） ---")
        print(df_results.to_string(index=False))
        df_results.to_csv(f"{filename}_2025_reliable.csv", index=False, encoding='utf-8-sig')

        # HTML レポート
        html_path = f"{filename}_regression_report_2024_01.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<html><head><meta charset='utf-8'><title>{filename} 回帰分析レポート</title></head><body>")
            f.write("<h1>回帰分析レポート</h1>")
            for r in results:
                if max(r['year'])>0:
                    filename_safe = f"{r['substance']}_{filename}".replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_")
                    image_path = os.path.join(PLOT_DIR, f"{filename_safe}_regression.png")
                    f.write("<hr>")
                    f.write(f"<h2>{r['substance']}</h2>")
                    f.write(f"<p><b>対象:</b> {filename}<br>")
                    f.write(f"<b>年データ:</b> {r['year']}<br>")
                    f.write("<h3>傾向分析結果 (MLE/AIC)</h3>")
                    f.write(f"<b>傾き (MLE):</b> {r['slope_MLE']}<br>")
                    f.write(f"<b>AIC事後確率 p1:</b> {r['p1_MLE']}<br>")
                    f.write(f"<b>傾向判定 (MLE):</b> {r['trend_MLE']}<br>")
                    f.write("<h3>ブートストラップ検定</h3>")
                    f.write(f"<b>CI (平均差):</b> {r['bootstrap_CI']}<br>")
                    f.write(f"<b>傾向判定 (Boot):</b> {r['bs_trend']}<br>")
                    f.write("<h3>ベイズ FFT-KDE (MCMC)</h3>")
                    f.write(f"<b>傾き平均 (ベイズ):</b> {r['bayes_slope']}<br>")
                    f.write(f"<b>傾き95%信用区間:</b> {r['bayes_CI_a']}<br>")
                    f.write(f"<b>傾向判定 (ベイズ):</b> {r['bayes_trend']}</p>")
                    f.write("<h3>二項回帰</h3>")
                    f.write(f"<b>傾き:</b> {r['binomial_slope']}<br>")
                    f.write(f"<b>AIC事後確率 p1:</b> {r['binomial_p1']}<br>")
                    f.write(f"<b>傾向判定:</b> {r['binomial_trend']}</p>")
                    if os.path.exists(image_path):
                        f.write(f"<img src='{image_path}' width='600'><br>")
                    else:
                        f.write("<p><i>グラフ画像が見つかりませんでした。</i></p>")
                else:
                    continue
            f.write("</body></html>")
        print(f"\n→ HTMLレポートを作成しました: {html_path}")
