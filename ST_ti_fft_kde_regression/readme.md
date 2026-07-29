 
**FFT‑KDE Bayesian Regression + Thermodynamic Integration (TI) Streamlit GUI** は、 POPs 時系列解析ツール群の中でも **最も総合的で、統計手法



# 📊 ベイズ傾向分析 Web GUI App (Bayesian Trend Analysis)

化学物質などの環境データ（濃度・不検出データ含む）に対して、最小二乗法 (OLS)、最大尤度推定 (MLE)、および FFT-KDE ベイズ推定を用いた傾向分析をブラウザ上で行える Streamlit アプリケーションです。

---

## 🌟 主な機能

1. **データ処理・不検出対応**
   - 不検出（ND）データを自動判定し、検出限界（$DL$）の $\frac{DL}{\sqrt{2}}$ で自動置換して対数変換 ($\log_{10}$) を行います。
2. **複数の解析手法**
   - **OLS**（普通最小二乗法）
   - **MLE + AIC**（環境省の標準手法等に基づく最大尤度推定）
   - **FFT-KDE ベイズ推定**（高速フーリエ変換・核密度推定を活用した MCMC 解析）
   - **熱力学積分 (TI)**（モデル選択・統計的検証用）
3. **可視化とレポート出力**
   - 解析結果の HTML レポート自動出力
   - 傾き（$a$）や切片（$b$）の後験分布（ヒストグラム、Corner plot、Joint posterior plot）の描画

---

## 🚀 起動方法

### 1. 必要なライブラリのインストール
事前に関連ライブラリ（`streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `corner` など）をインストールしてください。

### 2. アプリの起動
ターミナル（コマンドプロンプト）で以下のコマンドを実行します。

```bash
streamlit run ST_POPs_MOE_ti_fft_kde_regression_v1full.py
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
