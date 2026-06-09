import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import warnings
import re

warnings.filterwarnings('ignore')

# データの読み込み
df_raw = pd.read_csv('c:\kate_rd\data\POPs_Hedo2.csv')

# 調査対象物質ごとに内容確認のため表示（例：ヘキサクロロブタジエン）
target = '総PCB'
print(df_raw[df_raw['調査対象物質'] == target])


# 1. 調査対象物質リストの作成
# '総PCB'、'Cl:1'～'Cl:10'を除くすべての物質を対象とする
all_substances = df_raw['調査対象物質'].unique()
#pcb_substances = ['総PCB'] + [f'Cl:{i}' for i in range(1, 11)]
# PCB以外の物質リストを抽出
#target_substances = [sub for sub in all_substances if sub not in pcb_substances]
#target_substances = [sub for sub in all_substances if sub == 'ヘキサクロロブタジエン']
target_substances = [sub for sub in all_substances]
# ここでは、分析を容易にするため、単純にユニークな物質名を全て対象とします。
print(df_raw.describe())

print(f"--- 解析対象物質リスト ({len(target_substances)} 種類) ---")
# 最初の100個を表示
print(target_substances[0:100])

# 2. データ抽出と前処理 (ユーザー提供のクリーニングを適用)
df_log_monthly = {}
for substance in target_substances:
    df_sub = df_raw[df_raw['調査対象物質'] == substance].copy()
    
    # 日付インデックスの作成
    df_sub['Date'] = pd.to_datetime(df_sub['年'].astype(str) + '-' + df_sub['月'].astype(str))
    df_sub.set_index('Date', inplace=True)
    
    # '月平均値'列のクリーニングと数値変換（ユーザーの回避策を適用）
    def clean_value_user(val):
        if isinstance(val, str):
            val = val.replace('tr(', '').replace(')', '').replace('---', '').replace('nd','')
            val = val.replace(',', '')
        try:
            # 空文字になった場合はNaNを返す
            if val == '': return np.nan
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    ts_raw = df_sub['月平均値'].apply(clean_value_user)
    ts_raw.name = substance
    ts_log = np.log(ts_raw)
    
    # 連続した時系列データとして格納
    df_log_monthly[substance] = ts_log.sort_index()

# トレンド成分を格納する辞書
trend_results = {}
for substance, ts_log in df_log_monthly.items():
    # 欠損値を含む行を除外
    ts_log = ts_log.dropna()
    
    if len(ts_log) >= 24: # 最低2年分のデータ（24ヶ月）を要求
        # 連続したデータが必要なため、時系列を再構築
        # (ただし、ここではデータが連続していることを前提として進めます)
        try:
            stl = STL(ts_log, period=12, robust=True)
            res = stl.fit()
            trend_results[substance] = res.trend.dropna()
        except Exception as e:
            # STL分解が収束しない場合やデータに問題がある場合
            print(f"Warning: STL failed for {substance}: {e}")
            pass

print("\n--- 他のPOPs物質の季節調整済みトレンド解析詳細結果 ---")
print("----------------------------------------------------------")

for substance, trend in trend_results.items():
    N = len(trend)
    X = np.arange(N)
    X = sm.add_constant(X)
    y = trend.values

    model = sm.OLS(y, X)
    results = model.fit()

    # R2と傾き(x1)の95%信頼区間を出力
    r_squared = results.rsquared
    # NumPy配列のインデックス1（傾きx1の行）を取得
    slope_conf_int = results.conf_int()[1]
    
    print(f"\n======== 物質: {substance} (N={N}ヶ月) ========")
    print(f"決定係数 (R^2): {r_squared:.4f}")
    print(f"傾き x1 の95%信頼区間: [{slope_conf_int[0]:.4f}, {slope_conf_int[1]:.4f}] (月間 Log 濃度変化)")
    
    # P値と傾き（月間変化率）
    slope = results.params[1]
    p_value_slope = results.pvalues[1]
    
    if p_value_slope < 0.05 and slope > 0:
        conclusion = '✅有意な増加傾向 (警鐘)'
    elif p_value_slope < 0.05 and slope < 0:
        conclusion = '有意な減少傾向'
    else:
        # P>0.05 または 信頼区間がゼロを跨ぐ
        conclusion = '横ばい/有意な変化なし'
        
    print(f"【結論】: {conclusion}")
    # 係数テーブル（Summary Tableの2番目の表）を出力
    print(results.summary().tables[1].as_text())
    print("--------------------------------------------------")


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# STL分解後のトレンド成分（例: trend_results['ヘキサクロロブタジエン']）
trend_series = trend_results['ヘキサクロロブタジエン']
jp_font = fm.FontProperties(fname='C:/Windows/Fonts/msgothic.ttc')  # MSゴシックなど

# グラフ描画
plt.figure(figsize=(12, 6))
plt.plot(trend_series.index, trend_series.values, marker='o', linestyle='-', color='blue', label='STLトレンド')
plt.title('ヘキサクロロブタジエンのSTLトレンド（月平均濃度の対数）', fontproperties=jp_font)
plt.xlabel('年月', fontproperties=jp_font)
plt.ylabel('log(月平均濃度)', fontproperties=jp_font)
plt.grid(True)
plt.legend(prop=jp_font)
plt.tight_layout()
plt.show()

# 近年のデータ（例：2021年以降）を抽出
ts_recent = trend_series[trend_series.index >= '2021-01-01']

# 線形回帰を再実行
X_recent = np.arange(len(ts_recent))
X_recent = sm.add_constant(X_recent)
model_recent = sm.OLS(ts_recent.values, X_recent).fit()

print(model_recent.summary())

import pwlf
import matplotlib.pyplot as plt
import numpy as np

# STLトレンドデータ（例: trend_results['ヘキサクロロブタジエン']）
ts_log = trend_results['ヘキサクロロブタジエン'].dropna()

# x軸を数値化（0, 1, 2, ..., N-1）
x = np.arange(len(ts_log))
y = ts_log.values

# Piecewise Linear Fit モデルの作成
model = pwlf.PiecewiseLinFit(x, y)

# 変曲点の数を指定（例: 3区間 → 2つの変曲点）
breaks = model.fit(3)

# 傾きの取得
slopes = model.slopes

# 変曲点の位置（インデックス → 日付に変換）
break_dates = [ts_log.index[int(b)] for b in breaks[1:-1]]  # 最初と最後は除く

# 結果の表示
print("変曲点の位置（年月）:")
for i, bd in enumerate(break_dates):
    print(f"変曲点{i+1}: {bd.strftime('%Y-%m')}")

print("\n各区間の傾き:")
for i, slope in enumerate(slopes):
    print(f"区間{i+1}: 傾き = {slope:.5f} log濃度/月")

# 可視化
x_hat = np.linspace(0, len(x)-1, 100)
y_hat = model.predict(x_hat)
"""
plt.figure(figsize=(12, 6))
plt.plot(ts_log.index, y, 'o', label='STLトレンド')
plt.plot(ts_log.index[0] + pd.to_timedelta(x_hat, unit='M'), y_hat, 'r-', label='分割線形回帰')
for bd in break_dates:
    plt.axvline(bd, color='green', linestyle='--', label='変曲点')
plt.title('ヘキサクロロブタジエンのトレンド変曲点検出')
plt.xlabel('年月')
plt.ylabel('log(月平均濃度)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
from dateutil.relativedelta import relativedelta

# x_hat を datetime に変換
start_date = ts_log.index[0]
date_hat = [start_date + relativedelta(months=int(i)) for i in x_hat]
"""
# グラフ描画
plt.figure(figsize=(12, 6))
plt.plot(ts_log.index, y, 'o', label='STLトレンド')
plt.plot(date_hat, y_hat, 'r-', label='分割線形回帰')
for bd in break_dates:
    plt.axvline(bd, color='green', linestyle='--', label='変曲点')
plt.title('ヘキサクロロブタジエンのトレンド変曲点検出')
plt.xlabel('年月')
plt.ylabel('log(月平均濃度)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
#---------------------------
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
#from dateutil.relativedelta import relativedelta

jp_font = fm.FontProperties(fname='C:/Windows/Fonts/msgothic.ttc')

# グラフ描画
plt.figure(figsize=(12, 6))
plt.plot(ts_log.index, y, 'o', label='STLトレンド')
plt.plot(date_hat, y_hat, 'r-', label='分割線形回帰')
for bd in break_dates:
    plt.axvline(bd, color='green', linestyle='--', label='変曲点')
plt.title('ヘキサクロロブタジエンのトレンド変曲点検出', fontproperties=jp_font)
plt.xlabel('年月', fontproperties=jp_font)
plt.ylabel('log(月平均濃度)', fontproperties=jp_font)
plt.legend(prop=jp_font)
plt.grid(True)
plt.tight_layout()
plt.show()

import statsmodels.api as sm

# 変曲点のインデックスを取得
break1 = ts_log.index.get_loc('2018-06-01')
break2 = ts_log.index.get_loc('2020-07-01')

# 区間ごとのデータ
segments = [
    ts_log.iloc[:break1],
    ts_log.iloc[break1:break2],
    ts_log.iloc[break2:]
]

# 回帰式の構築と表示
for i, segment in enumerate(segments, 1):
    X = np.arange(len(segment))
    X = sm.add_constant(X)
    y = segment.values
    model = sm.OLS(y, X).fit()
    
    slope = model.params[1]
    intercept = model.params[0]
    r2 = model.rsquared
    pval = model.pvalues[1]
    
    print(f"\n【区間{i}】")
    print(f"回帰式: log濃度 = {intercept:.4f} + {slope:.5f} × 月数")
    print(f"決定係数 R² = {r2:.4f}, 傾きのP値 = {pval:.4f}")

# 例：12ヶ月の移動回帰で傾きを監視
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# STLトレンドデータ（例: trend_results['ヘキサクロロブタジエン']）
ts_log = trend_results['ヘキサクロロブタジエン'].dropna()

window = 12  # 12ヶ月の移動ウィンドウ
slopes = []
dates = []

for i in range(len(ts_log) - window):
    segment = ts_log.iloc[i:i+window]
    X = np.arange(window)
    X = sm.add_constant(X)
    y = segment.values
    model = sm.OLS(y, X).fit()
    slopes.append(model.params[1])
    dates.append(segment.index[-1])  # ウィンドウの終了月を記録

# 傾きの時系列をプロット
plt.figure(figsize=(12, 6))
plt.plot(dates, slopes, label='Moving Regression Slope', color='blue')
plt.axhline(0.05, color='red', linestyle='--', label='Alert threshold（0.05）')

# アラート候補（傾き > 0.05）を赤点で表示
for d, s in zip(dates, slopes):
    if s > 0.05:
        plt.plot(d, s, 'ro')

plt.title('Hexachlorobutadiene Migration Regression Slope Monitoring')
plt.xlabel('year-month')
plt.ylabel('slop（log_concentration/month）')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

alert_threshold = 0.05
window = 12
alert_summary = {}

for substance, ts_log in trend_results.items():
    ts_log = ts_log.dropna()
    if len(ts_log) < window:
        continue

    slopes = []
    dates = []

    for i in range(len(ts_log) - window):
        segment = ts_log.iloc[i:i+window]
        X = np.arange(window)
        X = sm.add_constant(X)
        y = segment.values
        model = sm.OLS(y, X).fit()
        slope = model.params[1]
        slopes.append(slope)
        dates.append(segment.index[-1])

    # アラート候補の抽出
    alert_dates = [d for d, s in zip(dates, slopes) if s > alert_threshold]
    alert_dates = sorted(set(alert_dates))
    if alert_dates:
        alert_summary[substance] = alert_dates

# 結果の表示
print("\nアラート候補物質一覧（傾き > 0.05）")
for substance, dates in alert_summary.items():
    print(f"\n物質: {substance}")
    for d in dates:
        print(f"  アラート月: {d.strftime('%Y-%m')}")

# 全物質・全月の移動回帰傾きを集計
all_slopes = []
for ts_log in trend_results.values():
    ts_log = ts_log.dropna()
    if len(ts_log) < 12: continue
    for i in range(len(ts_log) - 12):
        segment = ts_log.iloc[i:i+12]
        X = sm.add_constant(np.arange(12))
        y = segment.values
        model = sm.OLS(y, X).fit()
        all_slopes.append(model.params[1])

# 閾値を上位5%で設定
threshold = np.percentile(all_slopes, 95)
print(f"\n経験的閾値（95パーセンタイル）: {threshold:.5f}")