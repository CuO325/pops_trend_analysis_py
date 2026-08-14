

# 📊 pops_trend_analysis_py

> **残留性有機汚染物質（POPs）等の環境モニタリングデータに向けた、高度な時系列・トレンド分析＆統計モデリング Toolkit**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UI Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

残留性有機汚染物質（POPs）をはじめとする環境データの時系列解析や統計モデリングを行うためのPythonスクリプトおよびWebアプリケーション群です。  
環境省（MOE）で採用されている統計手法の再現から、MCMCを用いた階層ベイズモデル、FFT（高速フーリエ変換）との融合モデルまで、環境データの解析に必要な手法を網羅しています。

---

## ✨ 主な機能と特徴

* 🧠 **階層ベイズモデル（MCMC）**: `emcee` や `NumPyro` を活用した柔軟で高速なベイズ推論
* 📈 **環境省（MOE）手法の再現＆拡張**: カーネル密度推定（KDE）、AICによるモデル選択、ブートストラップ法、MCMCへの応用
#####       [経年分析の方法等に関する補足説明 環境省]https://www.env.go.jp/chemi/kurohon/2025/shosai/3_8.pdf 
* 🔄 **時系列・周波数解析**: FFTとKDEを組み合わせた高度な回帰分析（TI: Thermodynamic Integration）
* 🎯 **変化点検出 & 予測**: 構造変化点（ブレイクポイント）の検出および SARIMA モデルによる将来予測
* 🖥️ **直感的なWeb UI**: Streamlit を用いたインタラクティブな解析アプリを同梱

---

## 📁 ディレクトリ構成

### 🐍 Pythonスクリプト・モジュール (`PY_`)

| ディレクトリ名 | 概要 | 主な手法・ライブラリ |
| :--- | :--- | :--- |
| **`PY_Hierarchical_emcee`** | 階層ベイズモデルによるトレンド分析 | `emcee` |
| **`PY_Hierarchical_numpyro`** | 高速な階層ベイズモデリング | `NumPyro`, `JAX` |
| **`PY_MOE_KDE_AIC/kde_mle`** | 環境省手法：KDE、AICモデル選択、最尤推定、ブートストラップ | KDE, AIC, MLE |
| **`PY_emcee_kde_regression`** | `emcee` と KDE を組み合わせた回帰分析 | `emcee`, KDE |
| **`PY_pops_moe_analysis_plus`** | 環境省手法をベイズ分析（MCMC）へ拡張した高度モデル | ベイズ統計, MCMC |
| **`PY_ti_fft_kde_regression`** | 高速フーリエ変換（FFT）と KDE を組み合わせた時系列回帰分析 | FFT, KDE |

### 🎈 Streamlit アプリケーション (`ST_`)

| ディレクトリ名 | 概要 |
| :--- | :--- |
| **`ST_BREAK_POINTs`** | 時系列データにおける構造変化点（ブレイクポイント）の検出アプリ |
| **`ST_SARIMA_prediction`** | SARIMAモデルを用いた時系列データの将来予測アプリ |
| **`ST_ti_fft_kde_regression`** | FFT・KDEを用いた時系列回帰分析アプリ（TI実装） |

---

```text
pops_trend_analysis_py/
├── PY_emcee_kde_regression/     # 💡KDEベイズ回帰 (emcee版　遅いです)
│   └── pops_trend_analysis_v1.1a.py  
├── PY_Hierarchical_emcee/       # 階層ベイズモデル (emcee版)　地点別の傾向分析
├── PY_Hierarchical_numpyro/     # 階層ベイズモデル (NumPyro版)　地点別の傾向分析
├── PY_MOE_KDE_AIC/              # MOE / KDE-AIC 比較解析　（環境省の黒本記載の方法です）
├── PY_pops_moe_analysis_plus/   # 💡KDEベイズ回帰 (NumPyro版　速いです)
│ 
├── ST_ti_fft_kde_regression/    # 💡FFTを用いたKDE高速化モデル　(ブラウザ上で動作。TIも選択できます。スクリプトが最新です。)
├── ST_BREAK_POINTs/             # 変化点分析 (Breakpoint)　高頻度測定での季節変動を除外した経年変動を折れ線で解析します。
├── ST_SARIMA_prediction/        # 時系列予測 (SARIMA) 高頻度測定の精度管理、警鐘のためのアプリです。
└── README.md

```

## 🛠 動作環境・依存ライブラリ

* **Python 3.9+**

### 主要依存パッケージ
* **数値計算・データ処理**: `numpy`, `scipy`, `pandas`, `statsmodels`
* **可視化**: `matplotlib`, `seaborn`
* **ベイズ推論・MCMC**: [`emcee`](https://emcee.readthedocs.io/), [`numpyro`](https://num.pyro.ai/), `jax` / `jaxlib`
* **Web UI**: `streamlit`

---

## 🚀 使い方

### 1. リポジトリのクローン
```bash
git clone [https://github.com/CuO325/pops_trend_analysis_py.git](https://github.com/CuO325/pops_trend_analysis_py.git)
cd pops_trend_analysis_py

```

### 2. 仮想環境の構築とライブラリのインストール

JAX等の依存関係を考慮し、仮想環境の利用を強く推奨します。

```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate

# 依存ライブラリを一括インストール
pip install numpy scipy pandas matplotlib seaborn emcee numpyro jax statsmodels streamlit

```

### 3. 解析の実行

* **標準スクリプト (`PY_` ディレクトリ)**
各ディレクトリ内の `README.md` またはメインスクリプト（`.py` / `.ipynb`）を参照して実行してください。
* **Streamlit Webアプリ (`ST_` ディレクトリ)**
対象ディレクトリに移動し、以下のコマンドでブラウザ上に画面を起動します。
```bash
cd ST_BREAK_POINTs
streamlit run app.py  # ※実際のファイル名に合わせて変更してください

```

---

## 📊 データフォーマット

入力CSVの列名、日付フォーマット、サンプルデータにつきましては、各モジュールディレクトリ内の README をご確認ください。

---

## 🤖 AI アシスタンス表記

本リポジトリに含まれるコードおよびドキュメントの生成・最適化にあたっては、支援ツールとして **Google Gemini** および **Microsoft Copilot** を活用しています。

---

## 📝 ライセンス

### 本プロジェクト

本リポジトリのコードは [MIT License](https://www.google.com/search?q=LICENSE) のもとで公開されています。

### サードパーティ・ライブラリ

* **NumPyro / JAX**: Apache License 2.0
* **emcee**: MIT License
* **Streamlit**: Apache License 2.0
* **NumPy / SciPy / Pandas / statsmodels**: BSD License


