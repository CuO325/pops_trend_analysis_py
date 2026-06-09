# analysis/bootstrap_test.py
from sklearn.utils import resample
import numpy as np

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