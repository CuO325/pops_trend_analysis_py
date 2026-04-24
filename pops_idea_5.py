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
import os # ファイル保存のためにosモジュールを追加
import emcee

#追加した関数
def kde_likelihood(params, x, y, kde):
    """
    KDEを誤差分布とする尤度（log L）
    params = [a, b]
    """
    a, b = params
    residuals = y - (a * x + b)
    pdf_vals = kde.pdf(residuals)
    return np.sum(np.log(pdf_vals + 1e-12))  # underflow対策

def log_prior(params):  #事前分布（ガウス事前）
    a, b = params
    # 広めの正規事前：N(0, 10^2)
    if -20 < a < 20 and -20 < b < 20:
        return - (a**2 + b**2) / (2 * 100)
    return -np.inf

def log_posterior(params, x, y, kde):   #事後分布（log posterior）
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + kde_likelihood(params, x, y, kde)

def run_bayes_kde(df, initial_kde, x_base, n_walkers=20, n_steps=5000):
    """
    dfから KDE尤度ベイズ線形回帰の事後分布をサンプリング
    initial_kde: MLE残差から推定されたKDEオブジェクト
    x_base: 年の中心化の基準値
    """
    x_raw = df['year'].values
    x_centered = x_raw - x_base # MLEと同じ中心化を使用
    y = df['log_conc'].values

    # 初期位置をMLE結果の周辺に設定
    ndim = 2  # a, b
    # MLEの推定値 a, b
    a_mle = np.mean(df['year'].values - x_base) # 仮の初期値
    b_mle = np.mean(y) # 仮の初期値

    # より良い初期位置: MLEの結果を初期位置の中心とする
    try:
        # MLE結果に近い初期位置から始める
        result = minimize(neg_log_likelihood, [-0.1, np.mean(y)], args=(x_centered, y), method='L-BFGS-B')
        a_mle, b_mle = result.x
    except Exception:
        pass # MLEが失敗した場合は初期値を使用

    initial_pos = np.array([a_mle, b_mle]) + 1e-4 * np.random.randn(n_walkers, ndim)
    
    # log_posterior の引数から kde を削除し、kde_likelihood に渡すように変更
    # MCMCサンプラーのargsとして kde を渡す
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior, args=(x_centered, y, initial_kde)
    )

    # ... (MCMC実行ロジックは変更なし)

    # Burn-in/Thinning後のサンプル返却
    sampler.run_mcmc(initial_pos, n_steps, progress=False)
    samples = sampler.get_chain(discard=int(n_steps/2), flat=True)

    a_post = samples[:, 0]
    b_post = samples[:, 1]
    
    # 傾きが減少傾向（a < 0）である確率を計算
    p_neg_slope = np.sum(a_post < 0) / len(a_post)
    
    return {
        "samples": samples,
        "a_mean": np.mean(a_post),
        "b_mean": np.mean(b_post),
        "a_ci": np.percentile(a_post, [2.5, 97.5]),
        "b_ci": np.percentile(b_post, [2.5, 97.5]),
        "p_neg_slope": p_neg_slope # 減少傾向の確率
    }

# log_posterior も修正が必要です (kdeを引数に追加)
def log_posterior(params, x, y, kde):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    # kde_likelihood に kde を渡す
    return lp + kde_likelihood(params, x, y, kde)

# ------------------------------------------------
## 統合された分析関数
# ------------------------------------------------

def determine_split_year(df):
    """データの中央値の年をsplit_yearとして返す"""
    years = df['year'].dropna().unique()
    if len(years) < 4:
        return None  # 分割には不十分
    return int(np.median(years))

def compare_periods(df, split_year=2015, R=10000):
    """ブートストラップ法による平均値の差の検定"""
    # A群: 前期、B群: 後期にデータを分割
    A = df[df['year'] <= split_year]['log_conc'].values
    B = df[df['year'] > split_year]['log_conc'].values
    
    if len(A) < 2 or len(B) < 2:
        # データが少なすぎる場合の処理
        return np.array([np.nan, np.nan])

    diffs = []
    for _ in range(R):
        # A, B群から重複を許してランダムサンプリング (μ*_A, μ*_Bの算出)
        A_sample = resample(A, replace=True)
        B_sample = resample(B, replace=True)
        # 平均値の差を記録 (μ*_A - μ*_Bの算出)
        diffs.append(np.mean(A_sample) - np.mean(B_sample))
        
    # 95%信頼区間の算出
    ci = np.percentile(diffs, [2.5, 97.5])
    return ci

# --- 最尤推定法 + AIC (mle_estimation.py の統合) ---

def neg_log_likelihood(params, x, y):
    """最尤推定法のための負の対数尤度関数 (1次式)"""
    a, b = params
    residuals = y - (a * x + b)
    
    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10)) # ゼロ回避
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10

def neg_log_likelihood_0(params, x, y):
    """最尤推定法のための負の対数尤度関数 (0次式)"""
    b0 = params[0]
    residuals = y - b0
    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return 1e10
def run_mle_and_aic(df):
    """中心化した年を使って最尤推定とAICを計算"""
    np.set_printoptions(precision=10, suppress=True)

    x_raw = df['year'].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base  # 中心化

    y = df['log_conc'].values

    # 1次式モデルの最尤推定
    # initial_guesses = [[-0.1, np.mean(y)], [0.0, np.mean(y)], [0.1, np.mean(y)]]
    # methods = ['L-BFGS-B', 'Nelder-Mead', 'TNC']
    result = minimize(neg_log_likelihood, [-0.1, np.mean(y)], args=(x_centered, y), method='L-BFGS-B')
    a1, b1 = result.x
    max_log_likelihood_1 = -result.fun

    # 0次式モデルの最尤推定
    result_0 = minimize(neg_log_likelihood_0, [np.mean(y)], args=(x_centered, y), method='L-BFGS-B')
    max_log_likelihood_0 = -result_0.fun

    # AIC計算
    aic1 = -2 * max_log_likelihood_1 + 2 * 2
    aic0 = -2 * max_log_likelihood_0 + 2 * 1

    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator
    p1 = numerator / denominator if denominator != 0 else (0.0 if delta_aic > 0 else 1.0)

    return a1, b1, aic0, aic1, p1, x_base  # x_base を返す

def run_mle_and_ai_year(df):
    """最尤推定を実行し、その結果からAICと事後確率を算出"""
    # 桁数を10桁に設定し、小さな値での科学表記を抑制する
    np.set_printoptions(precision=10, suppress=True)
    
    x = df['year'].values
    y = df['log_conc'].values

    # 1. 1次式モデルの最尤推定
    result = minimize(neg_log_likelihood, [-0.1, np.mean(y)], args=(x, y), method='L-BFGS-B')
    #methods = ['L-BFGS-B', 'Nelder-Mead', 'TNC']

    a1, b1 = result.x
    max_log_likelihood_1 = -result.fun # 最大対数尤度 (log L1)

    # 2. 0次式モデルの最尤推定
    result_0 = minimize(neg_log_likelihood_0, [np.mean(y)], args=(x, y), method='L-BFGS-B')
    max_log_likelihood_0 = -result_0.fun # 最大対数尤度 (log L0)

    # 3. AICの算出 (AIC = -2 * log L + 2 * パラメータ数)
    aic1 = -2 * max_log_likelihood_1 + 2 * 2 # 1次式: a, b の2パラメータ
    aic0 = -2 * max_log_likelihood_0 + 2 * 1 # 0次式: b の1パラメータ

    # 4. AIC事後確率 p1 の算出（安定性を高めるためAICの差分を使用）
    delta_aic = aic1 - aic0 
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator

    if denominator == 0:
        p1 = 0.0 if delta_aic > 0 else 1.0
    else:
        p1 = numerator / denominator

    return a1, b1, aic0, aic1, p1

# --- 二項回帰モデル (binomial_trend.py の統合) ---

def binomial_trend_analyze(df):
    """
    二項回帰モデルによる低濃度地点の傾向分析。
    統一検出下限値を用いて低濃度地点の割合を算出し、経年傾向を分析する。
    """

    # 1. 統一検出下限値の決定: 解析期間における最も高い検出下限値
    unified_dl = df['detection_limit'].max()
    
    # 2. 低濃度地点フラグの作成: 統一検出下限値 (unified_dl) を下回る地点を「低濃度地点」と定義
    df['low_flag'] = (df['concentration'] < unified_dl).astype(int)
    
    # 3. 年度の低濃度地点数と総地点数を集計
    yearly_data = df.groupby('year').agg(
        low_count=('low_flag', 'sum'),
        total_count=('low_flag', 'count')
    ).reset_index()

    # 年度（説明変数）
    x_year = yearly_data['year'].astype(float).to_numpy()
    X1 = sm.add_constant(x_year).astype(float)

    # 成功割合（目的変数）と試行回数（重み）
    low_count = yearly_data['low_count'].astype(float).to_numpy()
    total_count = yearly_data['total_count'].astype(float).to_numpy()

    # 安全な割合計算
    with np.errstate(divide='ignore', invalid='ignore'):
        y_ratio = np.where(total_count > 0, low_count / total_count, np.nan)

    n_weights = total_count

    # 欠損除去（GLMが受け付けない行を除外）
    valid_mask = (~np.isnan(y_ratio)) & (~np.isnan(X1).any(axis=1)) & (~np.isnan(n_weights)) & (n_weights > 0)
    X1 = X1[valid_mask]
    y_ratio = y_ratio[valid_mask]
    n_weights = n_weights[valid_mask]
    # X0は定数項のみのモデルのための説明変数
    X0 = np.ones((len(y_ratio), 1), dtype=float)


    # 1次式モデル
    model1 = sm.GLM(y_ratio, X1, family=sm.families.Binomial())
    model1_fit = model1.fit(freq_weights=n_weights)
    #----------------------

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=PerfectSeparationWarning)
        model1 = sm.GLM(y_ratio, X1, family=sm.families.Binomial())
        model1_fit = model1.fit(freq_weights=n_weights)

        for warn in w:
            print(f"[警告] {warn.message}")


    #----------------------

    aic1 = model1_fit.aic

    # 0次式モデル
    model0 = sm.GLM(y_ratio, X0, family=sm.families.Binomial())
    model0_fit = model0.fit(freq_weights=n_weights)
    aic0 = model0_fit.aic
    #-------------------------------

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=PerfectSeparationWarning)
        model1 = sm.GLM(y_ratio, X1, family=sm.families.Binomial())
        model1_fit = model1.fit(freq_weights=n_weights)

        for warn in w:
            print(f"[警告] {warn.message}")


    #-------------------------------

    # AIC事後確率と傾き
    # p1 = exp(-0.5 * aic1) / (exp(-0.5 * aic0) + exp(-0.5 * aic1))
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    p1 = numerator / (1 + numerator)

    slope = model1_fit.params[1] if len(model1_fit.params) > 1 else np.nan

    return aic0, aic1, p1, slope

def clean_value_user(val):
    if isinstance(val, str):
        val = val.replace('tr(', '').replace(')', '').replace('---', '').replace('nd','0')
    try:
        # 空文字になった場合はNaNを返す
        if val == '': return np.nan
        return float(val)
    except (ValueError, TypeError):
        return np.nan
    
def is_data_acceptable(a, p1, ci):

    if p1 < 0.95:
        return False
    if ci[0] < 0 and ci[1] > 0:
        return False


    return True
#from sklearn.metrics import r2_score
#import statsmodels.api as sm
# ------------------------------------------------
## メイン処理
# ------------------------------------------------    
# 処理対象ファイル名リスト（拡張子なし）
#filenames = ["大気寒冷期", "水質", "底質", "生物"]
filenames = ["生物"]
for filename in filenames:
    print(f"\n\n==================== ファイル処理開始: {filename} ====================")

    filepath = f"data/{filename}.csv"
    if not os.path.exists(filepath):
        print(f"ファイルが見つかりません: {filepath}")
        continue

    # データ読み込みと前処理（ここは元のコードと同じ）
    df_all = pd.read_csv(filepath, usecols=['substance','year', 'site_id', 'concentration', 'detection_limit'])
    ts_raw = df_all['concentration'].apply(clean_value_user)
    df_all['concentration'] = ts_raw
    #df_all['log_conc'] = np.log10(df_all['concentration'].where(df_all['concentration'] > 0, np.nan))
    #    df_all['log_conc'] = np.log(df_all['concentration'].where(df_all['concentration'] > 0, np.nan))
    is_nondetect = df_all['concentration'] <= df_all['detection_limit']
    substitute_value = df_all['detection_limit'] / np.sqrt(2)
    df_all['concentration_sub'] = df_all['concentration'].copy()
    df_all.loc[is_nondetect, 'concentration_sub'] = substitute_value[is_nondetect]
    df_all['log_conc'] = np.log10(df_all['concentration_sub'])
    #    df_all['log_conc'] = np.log(df_all['concentration_sub'])
    df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_conc', 'detection_limit', 'concentration'])

    # 変換後のデータを保存
    output_path = f"data/{filename}_conv.csv"
    df_all.to_csv(output_path, index=False, encoding="utf-8")
    print(f"→ 変換後のデータを保存しました: {output_path}")


    substances = df_all['substance'].unique()
    print(f"対象物質: {substances}")

    results = []
    #---------------------------------
    # プロット保存用ディレクトリを作成
    PLOT_DIR = "plots_regression"
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"プロット画像は '{PLOT_DIR}' ディレクトリに保存されます。")


    # 各物質ごとに処理
    for substance in substances:
        print(f"\n==================== {substance} ====================")
        df = df_all[df_all['substance'] == substance].copy()
        
        # データ型の確認と調整
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['year']) # yearがNaNになった行を除去
        #n_point= len(df)
        if len(df) < 24:
            print("データ不足のためスキップ（地点数 < 24）")
            continue

        a, b, aic0, aic1, p1 = np.nan, np.nan, np.nan, np.nan, np.nan
        ci = np.array([np.nan, np.nan])
        a_bin, p1_bin = np.nan, np.nan

        # ① 最尤推定法 + AIC
        try:
            #a, b, aic0, aic1, p1 = run_mle_and_aic(df)
            a, b, aic0, aic1, p1, x_base = run_mle_and_aic(df)
            print("--- 1. 対数線形回帰モデル ---")
            print(f"傾き a = {a:.4f}, 切片 b = {b:.4f}, AIC事後確率 p1 = {p1:.4f}")
            trend = "傾向あり" if p1 >= 0.95 else "傾向なし"
            print(f"→ {trend}")
        except Exception as e:
            print(f"最尤推定エラー: {e}")
            trend = "エラー"
            continue
        #------------------------------==================
        # 中心化されたXとY
        x_centered = df['year'].values - x_base
        y_true = df['log_conc'].values
        y_pred = a * x_centered + b

        r2 = r2_score(y_true, y_pred)

        # 回帰に使うX（中心化済み）とY
        X = sm.add_constant(x_centered)
        model = sm.OLS(y_true, X).fit()

        # 回帰直線の予測と信頼区間
        x_fit_centered = np.linspace(x_centered.min(), x_centered.max(), 100)
        X_fit = sm.add_constant(x_fit_centered)
        y_fit = model.predict(X_fit)
        predictions = model.get_prediction(X_fit)
        ci = predictions.conf_int(alpha=0.05)  # 95%信頼区間
        # ... (MLE後のR2計算、OLS実行のブロック)
        #------------------------------==================

        # KDEベイズ事後（MCMC）の実行
        print("\n--- KDEベイズ事後（MCMC） ---")
        try:
            # 1. MLEの結果 a, b を使って残差を計算
            x_centered = df['year'].values - x_base
            y = df['log_conc'].values
            residuals_mle = y - (a * x_centered + b)
            
            # 2. 残差からKDEを推定（これをMCMC全体で固定）
            fixed_kde = gaussian_kde(residuals_mle)

            # 3. MCMCを実行
            bayes_res = run_bayes_kde(df, fixed_kde, x_base) # x_baseを渡す
            
            a_bayes = bayes_res["a_mean"]
            b_bayes = bayes_res["b_mean"]
            ci_a = bayes_res["a_ci"]
            ci_b = bayes_res["b_ci"]
            p_neg_slope = bayes_res["p_neg_slope"] # 減少傾向の確率
            
            # 結果出力
            print(f"ベイズ推定 a = {a_bayes:.4f} (95% CI {ci_a})")
            print(f"ベイズ推定 b = {b_bayes:.4f} (95% CI {ci_b})")
            print(f"減少傾向の事後確率 P(a < 0) = {p_neg_slope:.4f}")
            
            bayes_trend = "傾向あり" if (a_bayes < 0 and p_neg_slope >= 0.95) else "傾向なし"
            print(f"→ {bayes_trend} (ベイズ)")

        except Exception as e:
            print("ベイズ推定失敗:", e)
            a_bayes, ci_a, bayes_trend = np.nan, [np.nan, np.nan], "エラー"


        #-----------------------------------------
        # ② ブートストラップ検定
        try:
            split_year = determine_split_year(df)
            if split_year is None:
                print("有効な分割年が見つかりません。")
                ci = np.array([np.nan, np.nan])
                bs_trend = "有意差なし"
            else:
                ci = compare_periods(df, split_year=split_year)
                print("\n--- 2. ブートストラップ検定 ---")
                print(f"分割年: {split_year}")
                print(f"平均値差の95%信頼区間: {ci}")
                if ci[0] > 0 and not np.isnan(ci[0]):
                    bs_trend = "後期は有意に低濃度"
                elif ci[1] < 0 and not np.isnan(ci[1]):
                    bs_trend = "後期は有意に高濃度"
                else:
                    bs_trend = "有意差なし"
                print(f"→ {bs_trend}")
        except Exception as e:
            print(f"ブートストラップエラー: {e}")
            bs_trend = "エラー"

        # ③ 二項回帰モデル
        try:
            print("\n--- 3. 二項回帰モデル ---")        
            # 二項回帰に渡すデータフレームの準備（必要な列のみ）
            df_binomial = df[['year', 'concentration', 'detection_limit']].copy()
            df_binomial['year'] = pd.to_numeric(df_binomial['year'], errors='coerce')
            df_binomial['concentration'] = pd.to_numeric(df_binomial['concentration'], errors='coerce')
            df_binomial['detection_limit'] = pd.to_numeric(df_binomial['detection_limit'], errors='coerce')
            
            aic0_bin, aic1_bin, p1_bin, a_bin = binomial_trend_analyze(df_binomial)
            

            print(f"傾き a = {a_bin:.4f}, オッズ比 ≈ {math.exp(a_bin):.4f}, AIC事後確率 p1 = {p1_bin:.3f}")
            trend_bin = "傾向あり" if p1_bin >= 0.95 else "傾向なし"
            print(f"→ {trend_bin}")
        except Exception as e:
            print(f"二項回帰エラー: {e}")
            trend_bin = "エラー"

        # ④ プロット
        try:
            print("\n--- 4. 回帰直線のプロット ---")
            x_data = df['year'].values
            y_data = df['log_conc'].values
            # detection_limit を log10 に変換
            dl_log = np.log10(df['detection_limit'])




            # x_base は run_mle_and_aic() から取得済み
            x_centered = x_data - x_base
            x_fit_plot = np.array([x_data.min(), x_data.max()])
            x_fit_centered = x_fit_plot - x_base
            y_fit = a * x_fit_centered + b

            fig, ax = plt.subplots(figsize=(8, 5))
            # 横棒（マーカー）として検出限界を描画
            ax.hlines(dl_log, xmin=x_data - 0.2, xmax=x_data + 0.2,
            color='tab:orange', alpha=0.5, linewidth=1.5, label='検出限界')

            ax.scatter(x_data, y_data, label='実測データ', color='tab:blue', alpha=0.6)
            ax.plot(x_fit_plot, y_fit, label=f'回帰直線: y = {a:.4f}(x - {x_base}) + {b:.4f}', color='tab:red')
            #plt.fill_between(x_fit_centered + x_base, ci[:, 0], ci[:, 1], color='red', alpha=0.2, label='95%信頼区間')


            if 'split_year' in locals() and split_year is not None:
                ax.axvline(x=split_year, color='gray', linestyle='--', linewidth=1.5, label=f'分割年: {split_year}')

            #plt.title(f'{substance} の対数線形回帰')
            #plt.title(f'{substance} の対数線形回帰\nAIC事後確率 p1 = {p1:.4f} (参考 決定係数 R² = {r2:.3f})')
            ax.set_title(f'{substance} の対数線形回帰\nAIC事後確率 p1 = {p1:.4f}  {filename}')
            ax.set_xlabel('年')
            ax.set_ylabel('log濃度')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            # y軸の範囲を決定するための全データを結合
            y_all = np.concatenate([y_data, y_fit, dl_log])

            # y軸の範囲を設定（少し余白を持たせる）
            ax.set_ylim(y_all.min() - 0.1, y_all.max() + 0.1)
            #ax.set_ylim(min(y_data.min(), y_fit.min()) - 0.1, max(y_data.max(), y_fit.max()) + 0.1)
            fig.tight_layout()
            plt.show()
            #-------------------------------
            # ファイル名の設定（日本語名でも動作するように）
            #filename_safe = f"{substance}_{filename}".replace(' ', '_').replace('/', '_')
            filename_safe = f"{substance}_{filename}".replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_")
            plot_path = os.path.join(PLOT_DIR, f'{filename_safe}_regression.png')
        
            # グラフをファイルに保存
            fig.tight_layout()
            fig.savefig(plot_path)
            #plt.show(block=False)     #保存後に表示
            plt.close(fig) # メモリ解放
            print(f"→ {plot_path} に保存しました。")


        except Exception as e:
            print(f"プロットエラー: {e}")

        # 結果を格納
        reliability = is_data_acceptable(a, p1, ci)
        print(f"減少傾向のモデル1，2間の整合性？ : {reliability}")       

        results.append({
            'substance': substance,
            'year':df['year'].unique().tolist(),
            'slope': a,
            'intercept': b,
            'p1': p1,
            'trend': trend,
            'bayes_slope': round(a_bayes, 4),
            'bayes_CI': [round(ci_a[0], 4), round(ci_a[1], 4)],
            'bayes_trend': bayes_trend,
            #'bootstrap_CI': ci.tolist(),
            'bootstrap_CI': ci,
            'bs_trend': bs_trend,
            'binomial_slope': a_bin,
            'binomial_p1': p1_bin,
            'binomial_trend': trend_bin,
            'reliability': reliability
        })





    # 桁数を整える
    for r in results:
        ci = r['bootstrap_CI']
        #r['bootstrap_CI'] = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        r['bootstrap_CI'] = [round(ci[0], 4), round(ci[1], 4)]
        r['slope'] = round(r['slope'], 4)
        r['intercept'] = round(r['intercept'], 4)
        r['binomial_slope'] = round(r['binomial_slope'], 4)
        r['p1'] = round(r['p1'], 4)
        r['binomial_p1'] = round(r['binomial_p1'], 4)
 
    df_results = pd.DataFrame(results)
    print("\n--- 最終結果（テーブル形式） ---")
    print(df_results.to_string(index=False))

    df_results.to_csv(f"{filename}_2025_reliable.csv", index=False, encoding='utf-8-sig')

    # CSVファイルの読み込み
    #filename ="大気温暖期"
    html_path = f"{filename}_regression_report_2024_01.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>{filename} 回帰分析レポート</title></head><body>")
        f.write("<h1>回帰分析レポート</h1>")

        for r in results:
            #filename_safe = f"{r['substance']}_{filename}".replace(' ', '_').replace('/', '_')
            filename_safe = f"{r['substance']}_{filename}".replace("'", "").replace(",", "_").replace(" ", "_").replace("/", "_")
            image_path = os.path.join(PLOT_DIR, f"{filename_safe}_regression.png")
            #-------------------------------------年による出力制限------------------------
            if max(r['year'])>0:
                #-------------------------------------------------
                f.write("<hr>")
                f.write(f"<h2>{r['substance']}</h2>")
                f.write(f"<p><b>対象:</b> {filename}<br>")
                f.write(f"<p><b>year:</b> {r['year']}<br>")
                f.write(f"<b>傾き:</b> {r['slope']}<br>")
                f.write(f"<b>AIC事後確率 p1:</b> {r['p1']}<br>")
                f.write(f"<b>傾向判定:</b> {r['trend']}<br>")
                f.write(f"<b>ブートストラップCI:</b> {r['bootstrap_CI']}<br>")
                f.write(f"<b>ブートストラップ傾向:</b> {r['bs_trend']}<br>")
                f.write(f"<b>二項回帰傾き:</b> {r['binomial_slope']}<br>")
                f.write(f"<b>二項回帰 p1:</b> {r['binomial_p1']}<br>")
                f.write(f"<b>二項回帰傾向:</b> {r['binomial_trend']}</p>")

                if os.path.exists(image_path):
                    f.write(f"<img src='{image_path}' width='600'><br>")
                else:
                    f.write("<p><i>グラフ画像が見つかりませんでした。</i></p>")
            else:
                continue

        f.write("</body></html>")

    print(f"\n→ HTMLレポートを作成しました: {html_path}")