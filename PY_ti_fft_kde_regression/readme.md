# POPs Bayesian Fast FFT-KDE Trend Analysis

環境モニタリングデータの長期トレンド解析のために開発された、FFT-KDE（Fast Fourier Transform Kernel Density Estimation）を用いたベイズ推定フレームワークです。

本ソフトウェアは、従来の正規分布誤差を仮定した線形回帰では十分に対応できない環境データに対し、

- 非正規分布
- 外れ値の存在
- 欠測年
- 不規則な観測間隔

を考慮したトレンド解析を可能にします。

また、

- FFT-KDE尤度
- ベイズ推定（MCMC）
- 熱力学積分（Thermodynamic Integration; TI）
- ベイズファクター（Bayes Factor）
- R-hat収束診断
- OLS診断
- 二項ロジスティック回帰解析

を統合し、環境データの長期変動を多面的に評価できます。

---

## 主な機能

### FFT-KDEによる高速尤度評価

残差分布を正規分布と仮定せず、FFTを利用したカーネル密度推定（FFT-KDE）によって尤度を評価します。 

特徴：

- 非正規分布への対応
- 外れ値に対するロバスト性
- データ駆動型の尤度構築
- FFTルックアップテーブルによる高速計算

---

### ベイズ線形回帰

傾向解析はMCMC（`emcee`）によって行われます。 

推定対象パラメータ：

- 傾き（Slope） `a`
- 切片（Intercept） `b`

事前分布を考慮した事後分布推定により、不確実性を含めたトレンド評価が可能です。 

---

### 自動収束判定（Auto Stop）

MCMC計算では自動停止機能を搭載しています。 

収束後に不要な計算を継続しないため、

- 計算時間の短縮
- メモリ使用量の削減

が可能です。 

---

### R-hat収束診断

Gelman-Rubin統計量（R-hat）を用いてMCMC連鎖の収束状況を評価します。 

一般的な解釈：

| R-hat | 判定 |
|---------|---------|
| < 1.05 | 良好に収束 |
| 1.05～1.10 | 要注意 |
| > 1.10 | 未収束の可能性が高い |

---

### 熱力学積分（Thermodynamic Integration）

周辺尤度（Marginal Likelihood）の推定に熱力学積分（TI）を利用します。 

周辺尤度は

$$
\log Z =
\int_0^1 E_{\beta}[\log L] d\beta
$$


として計算されます。 

対応モデル：

- 線形モデル
- 定数モデル（傾きを0に固定） 

---

### ベイズファクターによるモデル比較

線形モデルと定数モデルの比較をベイズファクターで行います。 


$$
BF=\exp(\log Z_{linear}-\log Z_{constant})
$$


判定にはJeffreysの基準を採用しています。 】

| Bayes Factor | 解釈 |
|--------------|------|
| BF < 1 | 定数モデル支持 |
| 1 ≤ BF < 3 | 弱い証拠 |
| 3 ≤ BF < 10 | 中程度の証拠 |
| 10 ≤ BF < 30 | 強い証拠 |
| 30 ≤ BF < 100 | 非常に強い証拠 |
| BF ≥ 100 | 決定的な証拠 |

#### Thermodynamic Integration はオプション機能であり、Bayes Factor によるモデル比較が必要な場合のみ実行を推奨する。
---

### KDE最尤推定（KDE-MLE）

ベイズ推定に加えて、KDE尤度による最尤推定（Maximum Likelihood Estimation）にも対応しています。 

これにより、

- 古典的統計解析
- ベイズ解析

の結果を比較できます。 

---

### OLS診断

通常の最小二乗回帰（OLS）による解析と診断指標を提供します。 

主な診断項目：

- 決定係数（R²）
- Durbin-Watson検定
- Breusch-Pagan検定
- Ljung-Box検定
- Jarque-Bera検定
- 歪度（Skewness）
- 尖度（Kurtosis）
- VIF（分散拡大係数）

---

### 二項ロジスティック回帰

出現・非出現などの二値データに対する傾向解析にも対応しています。 

利用例：

- 検出率の経年変化
- 出現頻度の増減
- 発生確率のトレンド評価

---

## 解析フロー

```text
観測データ
      │
      ▼
FFT-KDE尤度構築
      │
      ▼
MCMC推定
      │
      ▼
R-hat収束診断
      │
      ▼
Thermodynamic Integration
      │
      ▼
logZ推定
      │
      ▼
Bayes Factor計算
      │
      ▼
トレンド判定
```

---

## 出力内容

解析結果として以下を出力します。

- トレンド推定値
- 95%信用区間
- 傾きの事後分布
- 傾向確率
- R-hat収束指標
- 対数周辺尤度（logZ）
- Bayes Factor
- OLS診断結果
- モデル比較結果
- 図表およびHTMLレポート

---

## 想定利用分野

本プログラムは以下のような環境モニタリングデータを対象としています。

- POPs（残留性有機汚染物質）
- PCB
- ダイオキシン類
- 農薬モニタリング
- 河川・湖沼環境調査
- 生物モニタリング
- 長期環境観測データ

---

## 必要ライブラリ

- Python 3.9+
- numpy
- pandas
- scipy
- matplotlib
- statsmodels
- scikit-learn
- emcee
- japanize-matplotlib

---

## ライセンス

MIT License

Copyright (c) Hiroaki Shiraishi

---

## 引用

本プログラムを研究・報告書等で利用する場合は、ソフトウェア名およびバージョンを引用してください。

> POPs Bayesian Fast FFT-KDE Trend Analysis  
> Version 3.0  
> Hiroaki Shiraishi