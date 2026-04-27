## バージョン履歴
- v7: KDEベイズ自動収束、Rhat、KDE安定化（2025/11/15）
- v5: MLE + ベイズ + ブートストラップ統合版


# Pops Trend Analysis Project

階層ベイズモデルを中心とした、高度なトレンド分析プロジェクトです。
分析のフェーズに応じて、以下の3つのスクリプトを使い分けます。

## 実行スクリプトの使い分け

| ファイル名 | 役割 | 特徴 |
|:---|:---|:---|
| `pops_trend_analysis.py` | **基礎分析・試作** | 単一の時系列に対するベイズ推論。モデルのプロトタイプ作成や基本設定の検証に使用。 |
| `pops_hierarchical_numpyro.py` | **多階層分析（本命）** | カテゴリ間や地域間の共通性を考慮した階層モデル。最も精緻な推定が可能ですが、計算負荷は最大です。 |
| `pops_fft_kde_analysis.py` | **高速運用・可視化** | FFT-KDEを用いた高速近似モデル。日々の定点観測や、大量のレポートを短時間で出す際に活用。 |

## ディレクトリ構造

- `data/`: 解析対象の生データ
- `outputs/`: 3つのスクリプトそれぞれの実行結果（グラフ・統計量）
- `hierarchical_numpyro/`: モデルの重みや中間ファイルの保存先

## 今後の展望
- 階層モデルの精度向上に向けた事前分布の微調整。
- FFT-KDEによる近似と階層モデルの推定結果の乖離チェック（バリデーション）。

## 更新履歴
- **階層モデルの統合**: 多階層分析用の `pops_hierarchical_numpyro.py` を追加。
- **ファイル名の整理**: 旧 `pops_final_test2.py` を、機能を表す `pops_fft_kde_analysis.py` へ変更。
- **pops_fft_kde_analysis.py の追加**: FFT-KDEを統合し、推論速度を大幅に向上。
- **ディレクトリ構造の整理**: 出力先を `outputs/` に統一。
- **Git管理の最適化**: `.gitignore` を更新し、不要なキャッシュや中間ファイルを追跡対象外に設定。
最尤推定法、AIC、ブートストラップ検定、二項回帰モデルなど複数の手法を組み合わせ、物質ごとの濃度変化を多角的に評価します。


## 使用ライブラリとライセンス

本プロジェクトは以下のオープンソースライブラリを利用しています。

- NumPy (BSD-3-Clause)
- SciPy (BSD-3-Clause)
- pandas (BSD-3-Clause)
- scikit-learn (BSD-3-Clause)
- statsmodels (BSD-3-Clause)
- matplotlib (PSF/BSD)
- japanize-matplotlib (MIT)
- emcee (MIT)

各ライブラリのライセンス条件に従い、本プロジェクトでは著作権表示を保持しています。

---

## 🚀 主な特徴

- **不検出値（ND）の統一処理**：検出下限値の 1/√2 による代替  
- **対数線形回帰（最尤推定 + AIC）**：AIC事後確率 p1 による傾向判定  
- **ブートストラップ検定**：前期・後期の平均値差の信頼区間を推定  
- **二項回帰モデル**：低濃度地点割合の経年変化を評価  
- **可視化**：回帰直線とデータ点をプロット  
- **結果の自動集計と CSV 出力**

---

## 📁 入力データ形式

CSV は以下の列を含む必要があります：

| 列名 | 説明 |
|------|------|
| `substance` | 物質名 |
| `year` | 年度（整数） |
| `site_id` | 測定地点 ID |
| `concentration` | 濃度（ND 表記含む） |
| `detection_limit` | 検出下限値 |

例：`data/大気寒冷期.csv`

---

## 🔧 解析フロー

### 1. データ前処理
- ND 表記（nd, tr(), --- など）を統一的に数値化  
- 不検出値は **検出下限値 / √2** で代替  
- 対数変換（log）  
- 無限大・NaN を除去  

### 2. 最尤推定法 + AIC による線形トレンド評価
- 1次式モデル（a × 年 + b）  
- 0次式モデル（定数）  
- AIC事後確率 p1 ≥ 0.95 → **傾向あり**

### 3. ブートストラップ検定
- 年度の中央値で前期・後期に分割  
- 平均値差の 95% 信頼区間を推定  
- CI の符号で有意差を判定  

### 4. 二項回帰モデル
- 統一検出下限値を下回る地点を「低濃度地点」と定義  
- 年ごとの割合を二項回帰で解析  
- p1 ≥ 0.95 → **傾向あり**

### 5. 可視化
- 実測 log 濃度の散布図  
- 最尤推定の回帰直線  
- 分割年（ブートストラップ）を縦線で表示  

### 6. 結果の集計
物質ごとに以下をまとめた CSV を出力：

- 傾き（線形モデル）  
- AIC事後確率 p1  
- ブートストラップ CI  
- 二項回帰の傾き・p1  
- 各手法の傾向判定  

出力例：`大気寒冷期_results.csv`

---

## ▶ 実行方法

```bash
python pops_idea.py
```

---

## 📊 想定される用途

- 長期モニタリングデータの傾向解析  
- 汚染物質の減少傾向の科学的評価  
- 行政報告書・学術論文の基礎解析  
- AIC に基づくモデル選択の実践例  

---

## 📄 ライセンス（MIT License）

```
MIT License

Copyright (c) 2026 Hiroaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

（以下 MIT ライセンス全文）
```


## 📘 pops_idea_5.py の詳細解説（統合環境データ解析スクリプト）

このスクリプトは、環境モニタリングデータ（大気・水質・底質・生物など）に対して  
**複数の統計モデル・ベイズ推定・可視化・HTML レポート生成までを一括で行う総合解析ツール**です。

解析の流れは以下の 7 段階で構成されています。

---

### 1. データ読み込みと前処理

- CSV から  
  `substance, year, site_id, concentration, detection_limit` を読み込む  
- ND（不検出値）表記（`nd`, `tr()`, `---` など）を `clean_value_user()` で数値化  
- 不検出値は  
  ```
  detection_limit / sqrt(2)
  ```  
  で代替  
- 対数変換は log10 を使用  
- 変換後データを `data/{filename}_conv.csv` に保存

---

### 2. 最尤推定（MLE）+ AIC による線形トレンド評価

- モデル  
  - **1次式**： y = a(x − x_base) + b  
  - **0次式**： y = b0  
- 残差分布は KDE（カーネル密度推定）で推定  
- AIC 事後確率  
  ```
  p1 = exp(-0.5 * ΔAIC) / (1 + exp(-0.5 * ΔAIC))
  ```
- 判定  
  - p1 ≥ 0.95 → **傾向あり**

---

### 3. ベイズ推定（MCMC + KDE 尤度）

- `emcee` による MCMC サンプリング  
- KDE を誤差分布として固定  
- 事前分布は広い正規分布（N(0, 10²））  
- 出力  
  - a の事後平均  
  - b の事後平均  
  - 95% CI  
  - **P(a < 0)**（減少傾向の確率）  
- 判定  
  - a < 0 かつ P(a < 0) ≥ 0.95 → **傾向あり**

---

### 4. ブートストラップ検定（前期 vs 後期）

- 年度の中央値で前期・後期に分割  
- 平均値差の 95% CI を推定  
- 判定  
  - CI > 0 → 後期は有意に低濃度  
  - CI < 0 → 後期は有意に高濃度  
  - 0 をまたぐ → 有意差なし

---

### 5. 二項回帰モデル（低濃度地点割合の傾向）

- 統一検出下限値 = 期間中の最大 detection_limit  
- 低濃度地点フラグ  
  ```
  low_flag = (concentration < unified_dl)
  ```
- GLM（Binomial）で割合の経年変化を推定  
- AIC 事後確率 p1 ≥ 0.95 → **傾向あり**

---

### 6. 回帰直線 + 検出限界の可視化

- 実測 log10 濃度  
- MLE 回帰直線  
- 検出限界（横棒）  
- 分割年（縦線）  
- タイトルに AIC 事後確率 p1 を表示  
- 保存先：  
  ```
  plots_regression/{substance}_{filename}_regression.png
  ```

---

### 7. HTML レポート生成

- 出力ファイル：  
  ```
  outputs/{filename}_regression_report_2024_01.html
  ```
- 内容  
  - 物質名  
  - 年リスト  
  - MLE 傾き・p1  
  - ベイズ推定結果  
  - ブートストラップ CI  
  - 二項回帰結果  
  - グラフ画像

---

## 🔍 このスクリプトの特徴

- KDE を使った柔軟な誤差モデル  
- ベイズ推定（MCMC）による確率的評価  
- MLE / ブートストラップ / 二項回帰の **3 系統の傾向判定**  
- HTML レポートまで自動生成  
- 検出限界をプロットに表示  
- pops_idea_5.pyは、実務レベルの環境統計解析ツールとして完成度が高い

---


---

## 📘 pops_idea_7.py の詳細解説（最新版・改良版）

*pops_idea_7.py* は、環境モニタリングデータのトレンド解析を  
**最尤推定（MLE）・ベイズ推定（MCMC）・ブートストラップ・二項回帰・可視化・HTML レポート生成**  
まで一括で行う総合解析スクリプトです。

前バージョン（pops_idea_5.py）を完全に上回る改良が加えられ、  
**解析の安定性・速度・信頼性が大幅に向上**しています。

---

# 🔷 1. 全体構造

pops_idea_7.py は以下の 6 つの解析モジュールで構成されています。

1. **データ前処理（ND処理・log10変換）**  
2. **最尤推定（MLE）+ AIC による線形トレンド評価**  
3. **KDE 尤度ベイズ推定（MCMC）— 改良版**  
4. **ブートストラップ検定（前期 vs 後期）**  
5. **二項回帰モデル（低濃度地点割合の傾向）**  
6. **可視化（回帰直線・検出限界）と HTML レポート生成**

---

# 🔷 2. 改良されたベイズ推定（MCMC + KDE 尤度）

pops_idea_7.py の最大の特徴は、  
**KDE 尤度ベイズ推定の大幅な強化**です。

### 🔹 主な改良点

- **自動早期収束判定（auto-stop）**  
  信用区間の幅と符号が安定すると自動停止し、計算時間を短縮。

- **KDE の安定化（safe_gaussian_kde）**  
  データが定数に近い場合でも落ちない堅牢な KDE。

- **Gelman–Rubin Rhat の導入**  
  MCMC の収束診断が可能。

- **自己相関時間（autocorr time）の計算**  
  サンプルの独立性を評価。

- **データスケールに応じた事前分布（log_prior_informative）**  
  a（傾き）と b（切片）の事前分布がデータに適応。

- **chunk 実行で途中経過を表示**  
  長時間の MCMC でも進捗が分かりやすい。

### 🔹 出力されるベイズ結果

- a の事後平均  
- b の事後平均  
- 95% 信用区間  
- **P(a < 0)**（減少傾向の確率）  
- Rhat（収束診断）  
- 自己相関時間  
- 自動停止の有無  

---

# 🔷 3. 最尤推定（MLE）+ AIC

- KDE を誤差分布とする柔軟な MLE  
- 1次式モデル vs 0次式モデル  
- AIC 事後確率 p1 による傾向判定  
- 中心化（x_base）を導入し数値安定性を向上

---

# 🔷 4. ブートストラップ検定

- 年度の中央値で前期・後期に分割  
- 平均値差の 95% CI を推定  
- CI の符号で有意差を判定  
- pops_idea_5.py より高速化

---

# 🔷 5. 二項回帰モデル（低濃度地点割合の傾向）

- 統一検出下限値（最大 DL）を採用  
- GLM（Binomial）で割合の経年変化を推定  
- AIC 事後確率 p1 による判定  
- PerfectSeparationWarning に対応

---

# 🔷 6. 可視化と HTML レポート生成

### 🔹 プロット内容

- 実測 log10 濃度  
- MLE 回帰直線  
- 検出限界（横棒）  
- 分割年（縦線）  
- タイトルに  
  - MLE p1  
  - ベイズ P(a<0)  
  を表示

### 🔹 保存先

```
outputs/plots_regression/{substance}_{filename}_regression.png
```

### 🔹 HTML レポート

- 各物質ごとの解析結果をまとめて出力  
- グラフ画像も自動挿入  
- 実務レベルの報告書として利用可能

---

# 🔷 7. pops_idea_7.py の強み（pops_idea_5.py からの進化）

| 項目 | pops_idea_5.py | pops_idea_7.py |
|------|----------------|----------------|
| KDE ベイズ推定 | 基本版 | **自動収束・安定化・診断付き** |
| KDE | 標準 | **safe_gaussian_kde で堅牢化** |
| MCMC | 固定ステップ | **chunk 実行 + 途中経過表示** |
| 事前分布 | 単純ガウス | **データスケール適応型** |
| 診断 | なし | **Rhat / autocorr** |
| HTML レポート | 基本版 | **改善版** |
| 安定性 | 中 | **高い（落ちにくい）** |
| 実行速度 | 中 | **高速（auto-stop）** |

---

# 🔷 8. まとめ：pops_idea_7.py は正式版として採用

- 解析の安定性  
- ベイズ推定の信頼性  
- 実行速度  
- コード構造  
- 実務での運用性  

*pops_idea_7.py* は、pops_trend_analysis.py とする。
