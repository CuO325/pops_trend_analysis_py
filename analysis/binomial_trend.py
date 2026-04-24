import statsmodels.api as sm
import numpy as np
import pandas as pd

def analyze(df):
    """
    二項回帰モデルによる低濃度地点の傾向分析。
    統一検出下限値を用いて低濃度地点の割合を算出し、経年傾向を分析する。
    """

    # 1. 統一検出下限値の決定: 解析期間における最も高い検出下限値
    # 【前提】df['detection_limit'] 列が存在すること
    unified_dl = df['detection_limit'].max()
    
    # 2. 低濃度地点フラグの作成: 統一検出下限値 (unified_dl) を下回る地点を「低濃度地点」と定義
    df['low_flag'] = (df['concentration'] < unified_dl).astype(int)
    
    # 3. 年度の低濃度地点数と総地点数を集計
    yearly_data = df.groupby('year').agg(
        low_count=('low_flag', 'sum'),
        total_count=('low_flag', 'count')
    ).reset_index()
    
    # GLMに渡すためのデータ準備
    #x_year = yearly_data['year']
    
    # 成功の割合をendogとして使用（0 <= y_ratio <= 1）
    #y_ratio = yearly_data['low_count'] / yearly_data['total_count']
    # 試行回数（総地点数）をweightsとして使用
    #n_weights = yearly_data['total_count']
    
    #x_year = pd.to_numeric(yearly_data['year'], errors='coerce')
    #y_ratio = pd.to_numeric(yearly_data['low_count'] / yearly_data['total_count'], errors='coerce')
    #n_weights = pd.to_numeric(yearly_data['total_count'], errors='coerce')
    # 年度（説明変数）
    """
    x_year = np.array(yearly_data['year'], dtype=float)
    X1_df = sm.add_constant(x_year)  # ← DataFrameとして返る
    X1 = np.array(X1_df, dtype=float)  # ← NumPy配列に変換（これが決定打）

    # 成功割合（目的変数）
    y_ratio = np.array(yearly_data['low_count'] / yearly_data['total_count'], dtype=float)

    # 試行回数（重み）
    n_weights = np.array(yearly_data['total_count'], dtype=float)
    """
    # 年度（説明変数）
    x_year = yearly_data['year'].astype(float).to_numpy()
    X1 = sm.add_constant(x_year).astype(float)

    # 成功割合（目的変数）
    low_count = yearly_data['low_count'].astype(float).to_numpy()
    total_count = yearly_data['total_count'].astype(float).to_numpy()

    # 安全な割合計算（ゼロ除外）
    with np.errstate(divide='ignore', invalid='ignore'):
        y_ratio = np.where(total_count > 0, low_count / total_count, np.nan)

    # 試行回数（重み）
    n_weights = total_count

    # 欠損除去（GLMが受け付けない行を除外）
    valid_mask = (~np.isnan(y_ratio)) & (~np.isnan(X1).any(axis=1)) & (~np.isnan(n_weights)) & (n_weights > 0)
    X1 = X1[valid_mask]
    y_ratio = y_ratio[valid_mask]
    n_weights = n_weights[valid_mask]
    X0 = np.ones((len(y_ratio), 1), dtype=float)

    # 1次式モデル
    model1 = sm.GLM(y_ratio, X1, family=sm.families.Binomial())
    model1_fit = model1.fit(freq_weights=n_weights)
    aic1 = model1_fit.aic

    # 0次式モデル
    X0 = np.ones((len(x_year), 1), dtype=float)
    model0 = sm.GLM(y_ratio, X0, family=sm.families.Binomial())
    model0_fit = model0.fit(freq_weights=n_weights)
    aic0 = model0_fit.aic

    # AIC事後確率と傾き
    p1 = np.exp(-0.5 * aic1) / (np.exp(-0.5 * aic0) + np.exp(-0.5 * aic1))
    slope = model1_fit.params[1] if len(model1_fit.params) > 1 else np.nan

    return aic0, aic1, p1, slope