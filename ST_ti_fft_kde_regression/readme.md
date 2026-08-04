 
**FFT‑KDE Bayesian Regression + Thermodynamic Integration (TI) Streamlit GUI** は、 POPs 時系列解析ツール群の中でも **最も総合的で、統計手法


---

# README.md

```markdown
# 📊 ベイズ傾向分析 Web GUI App (Bayesian Trend Analysis)

化学物質などの環境モニタリングデータ（濃度・不検出データ含む）に対し、最小二乗法 (OLS)、最大尤度推定 (MLE)、および FFT-KDE ベイズ推定を用いた傾向分析をブラウザ上で行える Streamlit アプリケーションです。

---

## 🌟 主な機能

1. **データ前処理・不検出対応**
   - データ内の文字表記（`tr()`, `nd`, `---` など）を数値化・クレンジング。
   - 不検出（ND）データを判定し、検出限界（$DL$）の $\frac{DL}{\sqrt{2}}$ で自動置換して対数変換 ($\log_{10}$) を実施。
2. **複数の回帰・推論手法**
   - **OLS**（普通最小二乗法）：基本的な線形トレンドの把握
   - **MLE + AIC**（環境省の標準手法等に基づく最大尤度推定）：尤度に基づくモデル比較
   - **FFT-KDE ベイズ推定**（MCMC解析）：高速フーリエ変換と核密度推定を活用した高速ベイズ推定、信頼区間の推定
   - **熱力学積分 (TI)**：周辺尤度（モデル証拠）の算出による客観的なモデル選択
3. **可視化とレポート出力**
   - 解析結果をまとめた HTML レポートの自動生成
   - 傾き（$a$）や切片（$b$）の後験分布（ヒストグラム、Corner plot、Joint posterior plot）の描画とダウンロード

---

## 🚀 起動方法

### 1. 必要なライブラリのインストール
事前に以下のパッケージをインストールしてください。

```bash
pip install streamlit numpy pandas matplotlib seaborn corner

```

### 2. アプリの起動

ターミナルで以下のコマンドを実行します。

```bash
streamlit run ti_fft_kde_regression_v1e.py

---

## 📖 画面構成と使い方

### 📁 左側サイドバー (Sidebar)

* **Data Entry**: 解析対象の CSV ファイルをアップロードします。
* **必要な列**: `substance`（物質名）, `year`（年）, `concentration`（測定濃度）, `detection_limit`（検出限界）


* **MCMC settings**: MCMCのチェイン数（`n_walkers`）、ステップ数（`n_steps`）、バーンイン割合（`burn_frac`）、TIの $\beta$ スケールなどを設定します。

### 📊 メイン画面 (Main Dashboard)

* **📁 Data Preview**: アップロードデータの確認、欠損値チェック。
* **📊 Run Analysis**: 分析ボタンを押して各種計算を実行（OLS, MLE, ベイズ, TI, HTMLレポート出力）。
* **📈 Plot**: 計算された傾き $a$ や切片 $b$ の後験分布（パラメータの確定度合い）をグラフィカルに確認。

---

## 📚 統計用語と出力の解説

### 1. 95%区間について（OLS/MLE vs ベイズ）

本アプリでは、手法によって「区間」の定義や計算方法が異なります。

* **OLS / MLE (95% 信頼区間: Confidence Interval)**
* **計算方法**: 点推定値 $\hat{\theta}$ と標準誤差 $SE$ をもとに、正規分布や $t$ 分布を仮定して計算します。

$$\text{CI} = \hat{\theta} \pm t_{0.025} \times SE$$


* **意味**: 「同じ条件で実験・調査を100回繰り返したとき、95回の確率で真のパラメータが含まれる範囲」を意味します。


* **ベイズ推定 (95% 信用区間 / 確信区間: Credible Interval)**
* **計算方法**: MCMC サンプル列の後験分布（Posterior Distribution）から、下位 2.5% と上位 97.5% のパーセンタイル値を直接算出します（パーセンタイル区間）。
* **意味**: 「**データが得られた下で、真のパラメータがこの範囲に存在する確率が 95% である**」という直感的な確率解釈が可能です。
* **傾き $a$ の評価**: 95% 信用区間に `0` が含まれていない（Crosses zero: False）場合、統計的に意味のある増加・減少傾向（トレンド）が存在すると解釈できます。



---

### 2. 熱力学積分 (TI: Thermodynamic Integration) によるモデル選択

**TI (熱力学積分)** は、2つのモデル（例：「直線傾きモデル」vs「一定・変化なしモデル」）のどちらがよりデータに当てはまっているかを客観的に比較・評価するための高度なベイズ統計手法です。

#### 💡 なぜ TI が必要なのか？

通常のベイズ推定ではパラメータの事後分布を求めますが、「どのモデル自体が最も適切か」を知るには**周辺尤度（Marginal Likelihood / Model Evidence）** $p(y)$ を計算する必要があります。しかし、$p(y)$ の直接計算は高次元積分となり困難です。

#### 🔬 TI の仕組み

逆温度パラメータ $\beta \in [0, 1]$ を導入し、事前分布（$\beta=0$）から事後分布（$\beta=1$）まで徐々に状態を変化させながら対数尤度の期待値を積分計算します。

$$\log p(y) = \int_{0}^{1} \mathbb{E}_{\beta} \left[ \log p(y \mid \theta) \right] d\beta$$

#### 📊 結果の読み方

本アプリの TI 出力では、**ベイズファクター (Bayes Factor: BF)** や **$\log BF$** を用いて判定します。

$$\Delta \log Z = \log Z(\text{linear}) - \log Z(\text{const})$$

* **$\Delta \log Z > 0$（$\text{BF} > 1$）**: **直線モデル（Linear Model）** を支持（傾向変化あり）
* **$\Delta \log Z < 0$（$\text{BF} < 1$）**: **一定モデル（Constant Model）** を支持（有意な傾向なし）

> ⚠️ **注意**: OLS や MLE の p値（$p < 0.05$）で有意と出ても、データのばらつきが大きい場合は TI のモデル比較によって「一定モデル（Constant Model）」が支持されることがあります。TI は過剰適合（オーバーフィッティング）を防ぎ、より慎重な判断を提供します。

---

## ⚠️ 既知の注意事項 (Known Issue)

* **物質名の同一による上書き**:
同じ CSV 内で媒体違い（例: 水質の `PCB` と 大気の `PCB`）のように全く同じ物質名を使用すると、セッション状態が上書きされて意図しない出力になります。
**対策**: データ内の物質名を `PCB_Water` や `PCB_Air` のように一意な名前に変更して読み込んでください。


---

# README.md（英語）

```markdown
# Bayesian Trend Analysis GUI  
### OLS / MLE / FFT‑KDE Bayesian Regression + Thermodynamic Integration (TI)  
Version: ti_fft_kde_regression_v1f.py

This repository provides an interactive **Streamlit-based GUI** for trend analysis of environmental monitoring data (e.g., POPs, PCBs).  
It integrates **three regression methods** and **Bayesian model evidence estimation**:

1. **OLS (Ordinary Least Squares)**  
2. **MLE (Maximum Likelihood Estimation, MOE-based)**  
3. **FFT‑KDE Bayesian Regression**  
4. **Thermodynamic Integration (TI)** for marginal likelihood

The tool is designed for analysts who need **robust trend estimation**, **model comparison**, and **automated HTML reporting** across multiple substances.

---

## ✨ Key Features

### **1. Streamlit GUI for Complete Workflow**
- Upload CSV data  
- Automatic cleaning of:
  - `tr()`, `---`, `nd`  
  - Non-detect substitution using DL/√2  
- Log10 transformation  
- Safe renaming of substances to avoid session conflicts  
- Preview data, missing values, and metadata  

---

## **2. Three Regression Methods**

### **(A) OLS Regression**
- Simple linear regression  
- Quick baseline trend estimate  
- HTML summary table  

### **(B) MLE + AIC (MOE Method)**
- Maximum likelihood estimation  
- AIC-based model comparison  
- Robust to moderate noise  

### **(C) FFT‑KDE Bayesian Regression**
A fast Bayesian method using:
- FFT‑based likelihood evaluation  
- KDE smoothing  
- MCMC sampling (emcee)  
- Posterior summaries:
  - slope \(a\), intercept \(b\)  
  - 95% credible intervals  
  - posterior probability of slope > 0  
- Corner plots  
- Joint posterior visualization  

---

## **3. Thermodynamic Integration (TI)**
TI computes the **marginal likelihood**:

\[
\log p(y) = \int_0^1 \mathbb{E}_{\beta}[\log p(y|\theta)] \, d\beta
\]

Features:
- Multiple β‑schedules (test, recommended, geometric, coarse, fine, ultra_fine)  
- Progress bar + intermediate table  
- Supports multiple substances  
- Saves TI results into session state  

---

## **4. Automated HTML Report Generation**
For each substance, the tool can generate a complete report containing:
- OLS results  
- MLE results  
- Bayesian posterior summary  
- TI results (optional)  
- Regression plot  
- Metadata and settings  

Reports are saved under:
```
outputs/<substance>_report_<timestamp>.html
```

---

## 📁 Input Data Format

The tool expects a CSV with at least:

| Column | Meaning |
|--------|---------|
| `substance` | Chemical name |
| `year` | Observation year |
| `concentration` | Measured concentration |
| `detection_limit` | Detection limit |

Example:
```csv
substance,year,concentration,detection_limit
PCB-118,2010,0.12,0.05
PCB-118,2011,0.10,0.05
PCB-153,2010,0.03,0.02
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install streamlit numpy pandas matplotlib seaborn corner
```

### 2. Launch the GUI
```bash
streamlit run ti_fft_kde_regression_v1f.py
```

### 3. Open the browser
Usually:
```
http://localhost:8501
```

---

## 🧭 GUI Structure

### **Sidebar**
- CSV upload  
- Data cleaning  
- MCMC settings (n_walkers, n_steps, burn_frac, KDE bandwidth, β‑schedule)  
- Configuration saved to session state  

### **Main Tab 1 — Data Preview**
- File info  
- Column names  
- Missing values  
- First rows  

### **Main Tab 2 — Run Analysis**
Buttons:
- OLS  
- MLE  
- FFT‑KDE Bayesian  
- All three + HTML  
- All four (including TI)  
- Regenerate HTML  

Results stored in:
```
st.session_state["ols_results"]
st.session_state["mle_results"]
st.session_state["bayes_results"]
st.session_state["ti_results"]
```

### **Main Tab 3 — Plot**
- Corner plot  
- Posterior of slope \(a\)  
- Joint posterior (a vs b)  

---

## 🔍 FFT‑KDE Bayesian vs NumPyro / emcee Models

This tool uses a **fast FFT‑KDE Bayesian method**, which differs from your other Bayesian implementations:

| Feature | FFT‑KDE Bayesian (this tool) | emcee Hierarchical Model | NumPyro Hierarchical Model |
|--------|------------------------------|---------------------------|-----------------------------|
| Model | Simple linear regression | Hierarchical Student‑t | Hierarchical Student‑t |
| Speed | **Very fast** | Moderate | Fast (HMC) |
| Robustness | Moderate | High | Very high |
| Handles multiple sites | ❌ No | ✔ Yes | ✔ Yes |
| TI support | ✔ Yes | ✔ Yes | ✔ Yes |
| Posterior shape | 2D (a, b) | High‑dimensional | High‑dimensional |
| GUI support | ✔ Full | No | No |

This GUI is ideal for **single‑site trend estimation** and **model comparison**, while the hierarchical models are ideal for **multi‑site inference**.

---

## 🧪 AI Assistance
This repository was developed with assistance from **Microsoft Copilot** and **Google Gemini**  
All final code, modeling decisions, and validation were performed by the author.

---

## 📄 License (MIT)

```text
MIT License

Copyright (c) 2026 Hiroaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
[... full MIT license text ...]
```

---

## 🙌 Acknowledgements
- FFT‑KDE Bayesian method  
- Thermodynamic Integration literature  
- Streamlit community  
- Environmental monitoring researchers  

```

---
