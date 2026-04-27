import pandas as pd
import numpy as np
import japanize_matplotlib
# 1. データの読み込み

file_id = "PCBv1_hs.csv"
df = pd.read_csv(f"data/{file_id.split(':')[-1]}")
# 2. 濃度データのクリーンアップと数値型への変換
# 欠損値を示す '---' を NaN に置換
df = df.replace('---', np.nan)

# 濃度カラムを数値型に変換（エラーが出る値を NaN にする）
concentration_cols = ['1日目', '2日目', '3日目']
for col in concentration_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. 月次代表濃度 (Monthly Average Concentration) の算出
# 3日間の測定値の平均を計算
df['Concentration'] = df[concentration_cols].mean(axis=1)

# 4. 時系列インデックスの作成
# 英語の列名にリネームしてpd.to_datetime()が認識できるようにする
df.rename(columns={'年': 'year', '月': 'month'}, inplace=True)

# 'year'と'month'を結合し、月初のDatetimeIndexを作成
df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

# 5. 必要なカラムのみを抽出
ts_data = df[['Date', 'Concentration']].copy()
print(ts_data)
# 6. 時系列データとして設定
ts_data = ts_data.set_index('Date').asfreq('MS') # 'MS'は月初を意味する

# 7. 欠損値の確認
initial_missing = ts_data['Concentration'].isnull().sum()
start_year = ts_data.index.min().year
end_year = ts_data.index.max().year

print("--- データの基本情報 ---")
print(f"期間: {start_year}年{ts_data.index.min().month}月 から {end_year}年{ts_data.index.max().month}月 まで")
print(f"データポイント総数: {len(ts_data)} 点")
print(f"初期の欠損値 (NaN): {initial_missing} 点\n")

# 確認のため、データの一部を表示
print("--- 時系列データ (ts_data) の先頭 ---")
print(ts_data.head(15))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 前処理ステップ1で読み込んだ ts_data を利用
# (データが読み込まれていない場合は前ステップのコードを実行してください)

# 欠損期間（2009年）を除外し、2010年1月以降のデータを抽出
start_date = '2010-01-01'
ts_data_clean = ts_data[ts_data.index >= start_date].copy()

# 欠損値の再確認
missing_count = ts_data_clean['Concentration'].isnull().sum()
total_count = len(ts_data_clean)

# 2. データの欠損処理（残りの欠損は前方の値で補完）
# SARIMAは連続したデータが必要なため、残った少数のNaNはf illna()で補完します
if missing_count > 0:
    # 欠損値の少ないデータのため、直前の有効な値で補完 (Forward Fill)
    ts_data_clean['Concentration'].fillna(method='ffill', inplace=True)
    
    # 補完後にも先頭にNaNが残る可能性があるため、その場合は除去
    ts_data_clean.dropna(inplace=True)

# 3. 安定性向上のための対数変換
# SARIMAモデルの適用や、残差の正規性確保のために、対数変換を適用します
ts_data_clean['Log_Concentration'] = np.log(ts_data_clean['Concentration'])

print("--- データの最終確認 ---")
print(f"解析期間: {ts_data_clean.index.min().strftime('%Y年%m月')} から {ts_data_clean.index.max().strftime('%Y年%m月')} まで")
print(f"データポイント総数: {len(ts_data_clean)} 点")
print(f"補完後の欠損値: {ts_data_clean['Log_Concentration'].isnull().sum()} 点")

# 4. 時系列データのプロット（対数濃度）
plt.figure(figsize=(10, 5))
plt.plot(ts_data_clean.index, ts_data_clean['Log_Concentration'], label='対数濃度')
plt.title('総PCB濃度（対数変換後, 2010年1月〜）の経年変化')
plt.xlabel('年')
plt.ylabel('ln(濃度)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# SARIMAモデルに渡す最終データ
ts_final = ts_data_clean['Log_Concentration']

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# データの対数変換 (ts_final を利用)
ts_final = ts_data_clean['Log_Concentration'] 

print("--- 1. 分解（トレンド・季節性・残差への分解） ---")
# 季節変動を分解 (周期 s=12)
decomposition = seasonal_decompose(ts_final, model='additive', period=12) 
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

print("\n--- 2. ADF (Augmented Dickey-Fuller) 検定 ---")
# ADF検定: 帰無仮説 H0: 非定常である（単位根が存在する）
result = adfuller(ts_final.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

if result[1] > 0.05:
    print("→ P値が0.05より大きいため、データは非定常である可能性が高いです (d=1が必要)")
    d = 1
else:
    print("→ P値が0.05以下であるため、データは定常であると見なせます (d=0で試行)")
    d = 0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# ts_final を利用 (Log_Concentrationデータ)

# 1. 非季節差分 (d=1) の適用
ts_diff1 = ts_final.diff(periods=1).dropna()

# 2. 季節差分 (D=1, 周期=12) の適用
# 季節差分は、非季節差分が適用されたデータに適用します
ts_diff12 = ts_diff1.diff(periods=12).dropna()

print("--- 差分適用後の定常性確認 ---")
# 季節差分後のADF検定
result_diff = adfuller(ts_diff12.dropna())
print('ADF Statistic (差分後): %f' % result_diff[0])
print('p-value (差分後): %f' % result_diff[1])
if result_diff[1] < 0.05:
    print("→ P値が0.05以下: データは定常であると見なせます。")
else:
    print("→ P値が0.05より大きい: さらに差分が必要かもしれません。")

# 差分後のデータのACFとPACFをプロット
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 1. 自己相関関数 (ACF) のプロット: MA(q) と季節MA(Q) の次数決定に利用
sm.graphics.tsa.plot_acf(ts_diff12, lags=30, ax=axes[0])
axes[0].set_title('自己相関関数 (ACF) - q/Q決定')

# 2. 偏自己相関関数 (PACF) のプロット: AR(p) と季節AR(P) の次数決定に利用
sm.graphics.tsa.plot_pacf(ts_diff12, lags=30, ax=axes[1])
axes[1].set_title('偏自己相関関数 (PACF) - p/P決定')

plt.tight_layout()
plt.show()

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 差分済みの定常データではなく、元の対数データ (ts_final) を使用します。
# SARIMAX関数が自動で差分処理を行います。
# (ts_final は前々回のステップで作成した Log_Concentration の Series です)

# 試行モデル: SARIMA(0, 1, 1) x (1, 1, 1)_12
order = (0, 1, 1)
seasonal_order = (1, 1, 1, 12)

# モデルの構築とフィッティング
try:
    model = SARIMAX(ts_final, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    results = model.fit(disp=False)
    
    print(results.summary())
    print(f"\n✅ 試行モデル: SARIMA{order} x {seasonal_order[:-1]}_12 のAIC: {results.aic:.4f}")
    
    # モデル残差の診断プロット
    print("\n--- モデル残差の診断プロット (残差がホワイトノイズか確認) ---")
    results.plot_diagnostics(figsize=(10, 8))
    plt.show()

except Exception as e:
    print(f"モデルのフィッティング中にエラーが発生しました: {e}")

# 予測期間の設定
forecast_steps = 12  # 12ヶ月先まで予測

# モデルによる予測
forecast = results.get_forecast(steps=forecast_steps)

# 予測値（Log_Concentration）と95%信頼区間
mean_forecast_log = forecast.predicted_mean
confidence_intervals_log = forecast.conf_int(alpha=0.05) # 95%予測区間

# 予測値と区間を元の濃度スケールに戻す（指数関数 e^x を適用）
mean_forecast_conc = np.exp(mean_forecast_log)
lower_bound_conc = np.exp(confidence_intervals_log['lower Log_Concentration'])
upper_bound_conc = np.exp(confidence_intervals_log['upper Log_Concentration'])

# 過去データも元のスケールに戻す
ts_actual_conc = np.exp(ts_final)

print("\n--- 警鐘システム: 予測と基準の準備完了 ---")

# 警鐘システムとしてのプロット
plt.figure(figsize=(12, 6))
plt.plot(ts_actual_conc.tail(36), label='実測濃度 (pg/m3)') # 直近3年間の実績
plt.plot(mean_forecast_conc, label='予測平均濃度', color='red')
plt.fill_between(mean_forecast_conc.index, 
                 lower_bound_conc, 
                 upper_bound_conc, 
                 color='pink', alpha=0.5, label='95%予測区間')

# 【警鐘ラインの設定】: 95%予測区間の上限を警鐘の基準とする
plt.plot(upper_bound_conc, linestyle='--', color='red', label='警鐘基準 (95%上限)')

plt.title('PCB濃度予測と警鐘基準 (SARIMAモデル)')
plt.xlabel('年')
plt.ylabel('PCB濃度 (pg/m3)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("\n✅ 警鐘基準の定義:")
print("次回の実測値が、予測区間の上限（95%予測区間の上限）を上回った場合に、警鐘（異常増加）と判断します。")