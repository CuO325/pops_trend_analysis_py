# main.py (実行スクリプト)

import matplotlib.pyplot as plt
import numpy as np
from analysis import utils, mle_estimation, bootstrap_test, binomial_trend

# model_selectionはmle_estimationに統合
#df = utils.load_data("data/concentration_data.csv")
df_all = utils.load_data("data/concentration_idea.csv")
# 対象物質リストの取得
substances = df_all['substance'].unique()
print(substances)
df = df_all[df_all['substance'] == substances[0]].copy()
print(df)
# ① 最尤推定法 + ② AICによる傾向判定
# 統合された関数を呼び出す
a, b, aic0, aic1, p1 = mle_estimation.run_mle_and_aic(df)
print("--- 1. 対数線形回帰モデル (最尤推定法 + AIC) ---")
print(f"最尤回帰直線: 傾き a = {a:.4f}, 切片 b = {b:.4f}")
print(f"AIC事後確率 p1 = {p1:.8f}")

# 傾向判定
if p1 >= 0.950:
    print("→ 傾きを持つ1次式が適切と判断 (傾向あり)")
else:
    print("→ 傾き0の0次式が適切と判断 (傾向なし)")

# ---

# ③ ブートストラップ検定
print("\n--- 2. ブートストラップ法による平均値の差の検定 ---")
# split_yearは例として2015を使用 (前期6か年と後期6か年の分割を想定)
ci = bootstrap_test.compare_periods(df, split_year=2015) 
print(f"平均値差の95%信頼区間 (前期 - 後期): {ci}")

# 有意差の判定
if ci[0] > 0:
    print("→ 95%信頼区間が正の範囲: 前期と比較して後期は有意に低濃度")
elif ci[1] < 0:
    print("→ 95%信頼区間が負の範囲: 前期と比較して後期は有意に高濃度")
else:
    print("→ 有意差なし (0を含む)")

# ---

# ④ 二項回帰モデル
print("\n--- 3. 二項回帰モデルによる低濃度地点の傾向分析 ---")
aic0_bin, aic1_bin, p1_bin, a_bin = binomial_trend.analyze(df)
print(f"低濃度地点の回帰直線の傾き(ロジットリンク関数による係数): {a_bin:.4f}")
import math
print(f"オッズ比（e^{a_bin:.4f} ≈ {math.exp(a_bin):.4f}）")

print(f"低濃度地点のAIC事後確率 p1 = {p1_bin:.3f}")

# 傾向判定
if p1_bin >= 0.950:
    print("→ 傾きを持つ1次式が適切と判断 (低濃度地点に傾向あり)")
else:
    print("→ 傾向なし")
# ---

# 5. 結果のプロット (対数線形回帰モデル)
import japanize_matplotlib
print("\n--- 4. 対数線形回帰モデルのプロット ---")

# 1. プロット用データの準備
# 元のデータ (x=year, y=log_conc)
x_data = df['year'].values
y_data = df['log_conc'].values
# 【回帰分析のX軸の基準年】回帰分析でX軸が年-X_BASEで計算されたと仮定
# ここでは、最小の年を X_BASE と仮定します
X_BASE = x_data.min() 
x_centered_data = x_data - X_BASE # X軸を基準化したデータ

# 2. 回帰直線の計算
# プロットするxの範囲をデータ範囲に合わせる
x_fit_centered = np.array([x_centered_data.min(), x_centered_data.max()])
# 推定された回帰式 y = ax + b
y_fit = a * x_fit_centered + b 
# 傾き0のモデル (0次式) の切片（ここではb0として傾き0のモデルの結果は省略）
# y_fit0 = b0 # 傾き0のモデルをプロットしたい場合

# 3. グラフの描画
plt.figure(figsize=(8, 5))

# 散布図 (元のデータ)
plt.scatter(x_data, y_data, label='実測データ (対数濃度)', color='tab:blue', alpha=0.6)

# 回帰直線 (1次式: 傾向ありモデル)
"""
plt.plot(x_fit, y_fit, 
         label=f'最尤回帰直線: $y = {a:.4f}x + {b:.4f}$', 
         color='tab:red', linestyle='-', linewidth=2)
         """
# 回帰直線: X軸を元の 'year' に戻してプロット
plt.plot(x_fit_centered + X_BASE, y_fit, # X軸を X_fit_centered + X_BASE で元の年に戻す
         label=f'最尤回帰直線: $y = {a:.4f}x\' + {b:.4f}$ ($x\'=x-{X_BASE}$)', 
         color='tab:red', linestyle='-', linewidth=2)
# 傾き0の回帰直線 (0次式: 傾向なしモデル) をプロットしたい場合はコメント解除
# plt.axhline(y=np.mean(y_data), 
#             label=f'傾き0モデル ($y = {np.mean(y_data):.4f}$)', 
#             color='tab:green', linestyle='--', linewidth=1)


# 4. グラフの装飾
plt.title('対数線形回帰モデルによるPOPs濃度の経年変化')
plt.xlabel('年 (Year)')
plt.ylabel('対数濃度 (log Concentration)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 【Y軸範囲の修正】データの最小値と最大値に基づいてY軸の範囲を設定
# データ範囲より少し広めに設定 (例: 最小値-0.1 から 最大値+0.1)
y_min = y_data.min() - 0.1
y_max = y_data.max() + 0.1
plt.ylim(y_min, y_max) # Y軸の範囲を設定

# グラフを出力
plt.show()