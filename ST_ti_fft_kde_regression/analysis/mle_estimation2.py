# analysis/mle_estimation.py (AICによるモデル選択機能も統合)
# analysis/model_selection.py (削除し、mle_estimation.pyに統合)
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import numpy as np
import statsmodels.api as sm
import numpy as np

# 桁数を10桁に設定し、小さな値での科学表記を抑制する
np.set_printoptions(precision=10, suppress=True) 

# これで p1 の値が 1.0000000000 ではなく、
# 0.999999999951 のような形で表示されるようになります。

def neg_log_likelihood(params, x, y):
    """最尤推定法のための負の対数尤度関数"""
    a, b = params
    residuals = y - (a * x + b)
    
    # 残差分布から確率密度関数(PDF)を推定（添付資料の非パラメトリック法に準拠）
    # ただし、gaussian_kdeはデータ数が少ないと不安定になる可能性がある
    try:
        kde = gaussian_kde(residuals)
        # 尤度の積を最大化するため、対数尤度の和を計算（最尤推定法）
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10)) # ゼロ回避
        return -log_likelihood
    except np.linalg.LinAlgError:
        # kdeの計算が失敗した場合、大きな負の対数尤度を返す
        return 1e10

def run_mle_and_aic_old(df):
    """最尤推定を実行し、その結果からAICと事後確率を算出"""
    x = df['調査年度'].values
    y = df['log_conc'].values

    # 1. 最尤推定による最適な回帰直線のパラメータ(a1, b1)を決定
    # L-BFGS-Bは制約付き最適化手法だが、ここでは安定性のために使用
    result = minimize(neg_log_likelihood, [-0.1, np.mean(y)], args=(x, y), method='L-BFGS-B')
    a1, b1 = result.x
    max_log_likelihood_1 = -result.fun # 最大対数尤度 (log L1)

    # 2. 傾き0のモデル (0次式) の最大対数尤度 (log L0)を決定
    def neg_log_likelihood_0(params, x, y):
        b0 = params[0]
        residuals = y - b0
        try:
            kde = gaussian_kde(residuals)
            log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
            return -log_likelihood
        except np.linalg.LinAlgError:
            return 1e10

    result_0 = minimize(neg_log_likelihood_0, [np.mean(y)], args=(x, y), method='L-BFGS-B')
    max_log_likelihood_0 = -result_0.fun # 最大対数尤度 (log L0)

    # 3. AICの算出 (AIC = -2 * log L + 2 * パラメータ数)
    aic1 = -2 * max_log_likelihood_1 + 2 * 2 # 1次式: a, b の2パラメータ
    aic0 = -2 * max_log_likelihood_0 + 2 * 1 # 0次式: b の1パラメータ
    """
    # 4. AIC事後確率 p1 の算出
    # p1 = exp(-0.5 * AIC1) / (exp(-0.5 * AIC0) + exp(-0.5 * AIC1))
    p1 = np.exp(-0.5 * aic1) / (np.exp(-0.5 * aic0) + np.exp(-0.5 * aic1))
    """
    # 4. AIC事後確率 p1 の算出
    # p1 = np.exp(-0.5 * aic1) / (np.exp(-0.5 * aic0) + np.exp(-0.5 * aic1))

    # 安定性を高めるため、AICの差分を使って計算する
    delta_aic = aic1 - aic0 # AIC1 - AIC0 を計算

    # p1 = exp(-0.5 * delta_aic) / (1 + exp(-0.5 * delta_aic))
    # np.exp() の引数が大きい負の値になっても、アンダーフローの可能性が減る
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator

    # ランタイム警告の回避のために、分母がゼロでないことを確認
    if denominator == 0:
        p1 = 0.0 if delta_aic > 0 else 1.0 # delta_aicが負ならp1=1
    else:
        p1 = numerator / denominator


    return a1, b1, aic0, aic1, p1
#import numpy as np
#from scipy.optimize import minimize
#from scipy.stats import gaussian_kde

def run_mle_and_aic(df, n_bootstraps=1000, ci_level=0.95):
    """最尤推定を実行し、AIC、事後確率、ブートストラップ法による信頼区間を算出"""
    x = df['調査年度'].values
    y = df['log_conc'].values

    # --- 1. 元データでの1次式MLE ---
    result = minimize(neg_log_likelihood, [-0.1, np.mean(y)], args=(x, y), method='L-BFGS-B')
    a1, b1 = result.x
    max_log_likelihood_1 = -result.fun

    # --- 2. 元データでの0次式MLE ---
    def neg_log_likelihood_0(params, x, y):
        b0 = params[0]
        residuals = y - b0
        try:
            kde = gaussian_kde(residuals)
            log_likelihood = np.sum(np.log(kde.pdf(residuals) + 1e-10))
            return -log_likelihood
        except np.linalg.LinAlgError:
            return 1e10

    result_0 = minimize(neg_log_likelihood_0, [np.mean(y)], args=(x, y), method='L-BFGS-B')
    max_log_likelihood_0 = -result_0.fun

    # --- 3. AIC & 事後確率の算出 ---
    aic1 = -2 * max_log_likelihood_1 + 2 * 2
    aic0 = -2 * max_log_likelihood_0 + 2 * 1

    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    denominator = 1 + numerator

    if denominator == 0:
        p1 = 0.0 if delta_aic > 0 else 1.0
    else:
        p1 = numerator / denominator

    # --- 4. ノンパラメトリック・ブートストラップ法による a1 の 95% 信頼区間 ---
    a1_boot = []
    n_samples = len(df)
    
    # 乱数シードの設定（再現性確保のため必要に応じて有効化）
    # np.random.seed(42)

    for _ in range(n_bootstraps):
        # データのペア (x, y) を重複を許してリサンプリング
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        x_resample = x[indices]
        y_resample = y[indices]

        try:
            res_boot = minimize(
                neg_log_likelihood, 
                [a1, b1],  # 初期値は元の推定値を利用して収束を高速化
                args=(x_resample, y_resample), 
                method='L-BFGS-B'
            )
            if res_boot.success:
                a1_boot.append(res_boot.x[0])
        except Exception:
            continue

    # パーセンタイル法による信頼区間の算出
    alpha = 1.0 - ci_level
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1.0 - alpha / 2.0) * 100

    if len(a1_boot) > 0:
        a1_ci_lower = np.percentile(a1_boot, lower_percentile)
        a1_ci_upper = np.percentile(a1_boot, upper_percentile)
    else:
        a1_ci_lower, a1_ci_upper = np.nan, np.nan

    return a1, b1, aic0, aic1, p1, a1_ci_lower, a1_ci_upper