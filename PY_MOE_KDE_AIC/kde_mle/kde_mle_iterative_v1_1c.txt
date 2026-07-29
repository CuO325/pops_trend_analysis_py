
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
from module.module_kde_mle_2a import run_mle_and_aic_simple, run_mle_and_aic_iterative

from tqdm import tqdm  # tqdm がない場合は進捗表示を自作してください

# =====================================================================
# 修正・整理された目的関数群
# =====================================================================

def neg_log_likelihood_fixed_kde(params, x, y, kde_fixed):
    """
    【線形モデル用】
    渡された『固定されたKDE (kde_fixed)』を用いて、現在のパラメータの負の対数尤度を計算する。
    """
    a, b = params
    # 今回の仮のパラメータでの残差を計算
    residuals = y - (a * x + b)
    
    try:
        # 新しくKDEを作らず、固定されたKDEから確率密度（PDF）を計算する
        pdf_values = kde_fixed.pdf(residuals)
        log_likelihood = np.sum(np.log(pdf_values + 1e-10))
        return -log_likelihood
    except Exception:
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

# 傾き $a$ の小数点以下3桁目や4桁目のブレは、実質的な結論（傾向の有無）に全く影響しません。
# #tolerance を 1e-2 (0.01) または 5e-3 (0.005) 程度まで緩める.1e-6
def run_mle_and_aic(df, max_iter=50, tolerance=1e-3):

    x_raw = df['year'].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df['log_conc'].values

    # 初期値（OLS）
    a_current, b_current = np.polyfit(x_centered, y, 1)

    # 収束判定用
    prev_a, prev_b = a_current, b_current

    converged_status = "未収束"
    final_iter = max_iter

    for i in range(max_iter):

        # --- KDE を更新 ---
        residuals = y - (a_current * x_centered + b_current)
        kde_new = gaussian_kde(residuals)

        # --- 固定 KDE を使った NLL ---
        def neg_log_likelihood_fixed(params, x, y):
            return neg_log_likelihood_fixed_kde(params, x, y, kde_new)

        # --- 最適化 ---
        result = minimize(
            neg_log_likelihood_fixed,
            [a_current, b_current],
            args=(x_centered, y),
            method='L-BFGS-B'
        )
        a_new, b_new = result.x
        max_log_likelihood_1 = -result.fun

        # --- ★ iteration 間の差で収束判定 ---
        param_change = np.linalg.norm([a_new - prev_a, b_new - prev_b])

        if param_change < tolerance:
            converged_status = "収束"
            final_iter = i + 1
            a1, b1 = a_new, b_new
            break

        # --- 次の iteration のために更新 ---
        prev_a, prev_b = a_current, b_current
        a_current, b_current = a_new, b_new

    else:
        # 最大回数到達
        a1, b1 = a_current, b_current
        converged_status = "最大回数到達"
        final_iter = max_iter

    # --- 定数モデル ---
    result_0 = minimize(
        neg_log_likelihood_0,
        [np.mean(y)],
        args=(x_centered, y),
        method='L-BFGS-B'
    )
    max_log_likelihood_0 = -result_0.fun

    # --- AIC ---
    aic1 = -2 * max_log_likelihood_1 + 2 * 2
    aic0 = -2 * max_log_likelihood_0 + 2 * 1
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator
    p1 = numerator / denominator

    return a1, b1, aic0, aic1, p1, x_base, converged_status, final_iter


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

def binomial_trend_analyze(df):

    unified_dl = df['detection_limit'].max()
    df['low_flag'] = (df['concentration'] < unified_dl).astype(int)
    
    # 年ごとに「ND数 (low_count)」と「NDでない数 (not_low_count)」を集計
    yearly_data = df.groupby('year').agg(
        low_count=('low_flag', 'sum'),
        total_count=('low_flag', 'count')
    ).reset_index()
    
    # NDでない数（非ND数）を算出
    yearly_data['not_low_count'] = yearly_data['total_count'] - yearly_data['low_count']
    
    x_year = yearly_data['year'].astype(float).to_numpy()
    X1 = sm.add_constant(x_year).astype(float)
    
    # 応答変数を [ND数, 非ND数] の2列行列を作成
    y_binomial = yearly_data[['low_count', 'not_low_count']].to_numpy().astype(float)
    
    # 有効なデータ（全試行数が0でない行）のフィルタリング
    valid_mask = (yearly_data['total_count'] > 0) & (~np.isnan(X1).any(axis=1))
    X1 = X1[valid_mask]
    y_binomial = y_binomial[valid_mask]
    
    X0 = np.ones((len(y_binomial), 1), dtype=float)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=PerfectSeparationWarning)
        
        # freq_weights を使わず、y に 2列の行列を渡す
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
# メイン処理（）
# ----------------
def clean_value_user(val):

    if isinstance(val, str):
        val = val.replace('tr(', '').replace(')', '')
        val = val.replace('---', '')
        val = val.replace('nd','0')
        val = val.replace('ND','0')
    try:
        if val == '': return np.nan
        return float(val)
    except (ValueError, TypeError):
        return np.nan
   
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
if __name__ == "__main__":
    filenames = ["PFOS大気温暖期"]
    #filenames = ["大気温暖期","大気寒冷期","底質","生物"]
    #filenames = ["水質"]
    for filename in filenames:
        print(f"\n\n==================== ファイル処理開始: {filename} ====================")
        filepath = f"data/{filename}.csv"
        if not os.path.exists(filepath):
            print(f"ファイルが見つかりません: {filepath}")
            continue
        
        os.makedirs("outputs", exist_ok=True)

        df_all = pd.read_csv(filepath, usecols=['substance','year', 'site_id', 'concentration', 'detection_limit'])
        df_all['concentration'] = df_all['concentration'].apply(clean_value_user)
        df_all['detection_limit'] = df_all['detection_limit'].apply(clean_value_user)
        is_nondetect = df_all['concentration'] <= df_all['detection_limit']
        substitute_value = df_all['detection_limit'] / np.sqrt(2)
        df_all['concentration_sub'] = df_all['concentration'].copy()
        df_all.loc[is_nondetect, 'concentration_sub'] = substitute_value[is_nondetect]
        df_all['log_conc'] = np.log10(df_all['concentration_sub'])
        df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_conc', 'detection_limit', 'concentration'])
        output_path = f"outputs/{filename}_conv.csv"
        df_all.to_csv(output_path, index=False, encoding="utf-8-sig")    #utf-8ではエクセルで文字化け
        print(f"→ 変換後のデータを保存しました: {output_path}")

        substances = df_all['substance'].unique()

        #########出力指定#########debug######################
        #substances = substances[2:4]
        #
        ####################################################

        print(f"対象物質: {substances}")        
        results = []


        for substance in substances:
            print(f"\n==================== {filename} : {substance} ====================")
            df = df_all[df_all['substance'] == substance].copy()
            #　NDのみのケースを念頭に置く。除外する
            if np.std(df['log_conc']) < 1e-6:
                continue


            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df = df.dropna(subset=['year'])
            if len(df) < 24:
                print("データ不足のためスキップ（地点数 < 24）")
                continue

            # 初期値を明確にしておく（例外発生時に上書きされないように） linear_validity
            #a, b, aic0, aic1, p1_s = np.nan, np.nan, np.nan, np.nan, np.nan
            a1, b1, aic0_1, aic1, p1_s, x_base = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
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

            # === Run KDE-MLE (simple version) ===iterativeの初期値に使うが、これだけでも判定には十分====
            print("\n=== Running KDE-MLE (simple) ===")
            a1, b1, aic0_1, aic1_1, p1_s, x_base = run_mle_and_aic_simple(df)

            print(f"Slope (a): {a1:.4f}")
            print(f"Intercept (b): {b1:.4f}")
            print(f"AIC0 (constant): {aic0_1:.3f}")
            print(f"AIC1 (linear):   {aic1_1:.3f}")
            print(f"AIC posterior weight p1: {p1_s:.4f}")
            trend_s = "p1 >= 0.95 傾向あり" if p1_s >= 0.95 else "p1 < 0.95 傾向なし"
            print(f"→ {trend_s}")

            # ① MLE + AIC(itertive)　   閾値を自動判定にする
            print("\n=== Running KDE-MLE (iterative) ===")            
            try:
                a, b, aic0, aic1, p1, x_base, mle_status, mle_iters, tolerance = run_mle_and_aic_iterative(df,a1,b1,30)
                
                print("--- 1. 对数線形回帰モデル (MLE) ---")
                
                print(f"状態: {mle_status} tolerance: {tolerance:.04f} (反復回数: {mle_iters})") # 確認用ログ
                print(f"傾き a = {a:.4f}, 切片 b = {b:.4f}, AIC事後確率 p1 = {p1:.4f}")
                print(f"aic1: {aic1} - aic0: {aic0}")
                trend = "p1 >= 0.95 傾向あり" if p1 >= 0.95 else "p1 < 0.95 傾向なし"
                print(f"→ {trend}")
                ########################## DEBUG #########増減確認であればSimple版で十分なのではないか。
                print("OK　判定は一致しています" if trend==trend_s else "NO! 判定が不一致です!")    

                if trend !=trend_s:
                    trend += f"　simple版とは異なる結果です ({p1_s:.4f})"                
                else:
                    trend += f"　simple版と同じ結果です ({p1_s:.4f})"

            except Exception as e:
                print(f"最尤推定エラー: {e}")
                trend = "エラー"
                mle_status = "エラー"
                mle_iters = np.nan
                continue

            
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

            results.append({
                'substance': substance,
                'year':df['year'].unique().tolist(),
                'slope_MLE': a,
                'intercept_MLE': b,
                #'linear_validity': linear_validity,
                'p1_MLE': p1,
                'trend_MLE': trend,
                'mle_status': mle_status,  # ★ CSVに追加される
                'mle_iters': mle_iters,    # ★ CSVに追加される
                'tolerance': tolerance,     #追加
                'bootstrap_CI': ci_bs,
                'bs_trend': bs_trend,
                'binomial_slope': a_bin,
                'binomial_p1': p1_bin,
                'binomial_trend': trend_bin,
            })

        for r in results:
        # 信用区間の整形（すべて format_ci に統一）
            r['bootstrap_CI'] = format_ci(r.get('bootstrap_CI'))


            # 数値項目の整形
            for key in ['slope_MLE', 'intercept_MLE', 'binomial_slope', 'p1_MLE', 'binomial_p1']:
                r[key] = format_float(r.get(key))
    
        df_results = pd.DataFrame(results)
        print(f"\n--- 最終結果（テーブル形式）  {filename} ---")
        print(df_results.to_string(index=False))




        df_results.to_csv(f"outputs/{filename}_2025_moe.csv", index=False, encoding='utf-8-sig')    
        



 
        
