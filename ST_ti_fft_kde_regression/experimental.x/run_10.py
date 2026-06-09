import subprocess
import pandas as pd

#filenames = ["HCB大気温暖期2013"]
filename = ["PCB_water"] 
n= 5 # 実行回数
# 実行結果を格納するリスト
all_results = []

for i in range(1, n + 1):
    print(f"--- 実行回数: {i}/{n} ---")
    # A.py を外部プロセスとして実行
    subprocess.run(["python", "pops_bayes_fastfft_loo_v2.2c.py"])
    
    # A.py が出力する CSV（HCB大気温暖期2013_2025_reliable.csv 等）を読み込む
    # ※ファイル名は A.py の出力設定に合わせて変更してください
    try:
        df = pd.read_csv(f"outputs/{filename}_2025_reliable.csv")
        df['run_count'] = i
        all_results.append(df)
    except FileNotFoundError:
        print(f"実行 {i} 回目の CSV が見つかりませんでした。")

# 10回分の結果を結合して一つのCSVに保存
if all_results:
    final_df = pd.concat(all_results)
    final_df.to_csv(f"mcmc_{n}times_comparison_{filename}.csv", index=False, encoding='utf-8-sig')
    print(f"{n}回分の比較用CSVを作成しました: mcmc_{n}times_comparison_{filename}.csv")