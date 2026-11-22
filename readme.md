
# 📊 POPsTAP (pops_trend_analysis_py)

>環境モニタリングデータ向けの
>時系列解析・ベイズモデリング・予測ツール集です。

**推奨アプリ：**
`ST_ti_fft_kde_regression`
`PY_Hierarchical_numpyro`

主な機能：
FFT-KDE / MCMC / Thermodynamic Integration /
Bayes Factor / 階層ベイズ / SARIMA

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UI Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

残留性有機汚染物質（POPs）をはじめとする環境データの時系列解析や統計モデリングを行うためのPythonスクリプトおよびWebアプリケーション群です。  
環境省（MOE）で採用されているいくつかの統計手法(1)に加えて、MCMCを用いた階層ベイズモデル、FFT（高速フーリエ変換）との融合モデルまで、環境データの解析に利用可能なアプリとしてまとめています。

[(1) 環境省(2025): 経年分析の方法等に関する補足説明](https://www.env.go.jp/chemi/kurohon/2025/shosai/3_8.pdf)

<img width="650" height="600" alt="13HCBD_ヘキサクロロブタ-1_3-ジエン_regression_20260806_195424" src="https://github.com/user-attachments/assets/5c13bf30-6dcc-4b47-8158-f778aae32113" />

---
## 🚀 初めて利用する方へ

**環境省のPOPsモニタリングの解析法をお探しの場合は、
    **[PY_MOE_KDE_AIC/kde_mle](./PY_MOE_KDE_AIC/kde_mle/)** をお試しください。

**ファイル名がST_で始まるアプリをお試しください。GUI操作やHTML出力ができます。
    **[ST_ti_fft_kde_regression](./ST_ti_fft_kde_regression)**　が推奨です。


---
## Current Status

### 推奨版（Current implementation）
👉 **[PY_MOE_KDE_AIC/kde_mle](./PY_MOE_KDE_AIC/kde_mle/)** AICによるモデル比較(環境省法に準拠)

👉 **[ST_ti_fft_kde_regression](./ST_ti_fft_kde_regression)** Bayes Factorによるモデル比較の追加
- 全体の傾向解析
- FFT-KDE高速尤度
- Thermodynamic Integration (TI)
- Bayes Factorによるモデル比較
- ブラウザ上で動作
- 最新機能を実装
    バッチ処理には
    > [PY_pops_moe_analysis_plus](./PY_pops_moe_analysis_plus/)　が、ST_ti_fft_kde_regressionとほぼ同機能です。

👉 **[PY_Hierarchical_numpyro](./PY_Hierarchical_numpyro/)**
- 地域別の傾向解析
- 高速な階層ベイズ
  
<img width="650" height="600" alt="水質_α-HCH_44" src="https://github.com/user-attachments/assets/55f39c51-276d-4df0-95c1-c6b0f58367a9" />


### 季節変動を考慮した時系列モニタリングの解析
👉 `ST_BREAK_POINTs`
- 季節変動を示す高頻度モニタリングの解析
- break pointの解析
  
<img width="850" height="600" alt="PCB-Di_mean_fit_20260604_171348" src="https://github.com/user-attachments/assets/0d102940-2505-4c21-b6b9-0e852035b961" />

`ST_SARIMA_prediction`
- 次年度の予測とアラート

<img width="650" height="600" alt="hedo_pcb" src="https://github.com/user-attachments/assets/1d5285b4-d4b4-4cd0-ac57-15ca854df121" />


### 参照実装（Reference implementation）
`PY_ti_fft_kde_regression`などは、手法の検証の目的で保存されています。
- emceeを用いたオリジナル実装　(初期段階で作成されました。遅いです)
- 手法検証および比較用
- 目的に応じたバッチ処理用

---
## ✨ 主な機能と特徴

* 🧠 **階層ベイズモデル（MCMC）**: `emcee` や `NumPyro` を活用した柔軟で高速なベイズ推論
* 📈 **環境省（MOE）手法の再現＆拡張**: カーネル密度推定（KDE）、AICによるモデル選択、ブートストラップ法、MCMCへの応用
* 🔄 **時系列・周波数解析**: FFTとKDEを組み合わせた高度な回帰分析（TI: Thermodynamic Integration）
* 🎯 **変化点検出 & 予測**: 構造変化点（ブレイクポイント）の検出および SARIMA モデルによるアラート機能
* 🖥️ **直感的なWeb UI**: Streamlit を用いたインタラクティブな解析アプリを同梱

---

## 📁 ディレクトリ構成

### 🐍 Pythonスクリプト・モジュール (`PY_`)

| ディレクトリ名 | 概要 | 主な手法・ライブラリ |
| :--- | :--- | :--- |
| **[PY_Hierarchical_emcee](./PY_emcee_kde_regression/)** | 階層ベイズモデルによるトレンド分析 （遅いです）| `emcee` |
| **[PY_Hierarchical_numpyro](./PY_Hierarchical_numpyro/)** | 高速な階層ベイズモデリング （💡階層ベイズの推奨版）| `NumPyro`, `JAX` |
| **[PY_MOE_KDE_AIC/kde_mle](./PY_MOE_KDE_AIC/kde_mle/)** | 環境省手法：KDE、AICモデル選択、最尤推定、ブートストラップ | KDE, AIC, MLE |
| **[PY_emcee_kde_regression](./PY_emcee_kde_regression)** | `emcee` と KDE を組み合わせた回帰分析 （遅いです）| `emcee`, KDE |
| **[PY_pops_moe_analysis_plus](./PY_pops_moe_analysis_plus/)** | 環境省手法をベイズ分析（MCMC）へ拡張した高度モデル　（💡準-推奨版） | ベイズ統計, MCMC |
| **[PY_ti_fft_kde_regression](./PY_ti_fft_kde_regression/)** | 高速フーリエ変換（FFT）と KDE を組み合わせた時系列回帰分析 | FFT, KDE |

### 🎈 Streamlit アプリケーション (`ST_`)

| ディレクトリ名 | 概要 |
| :--- | :--- |
| **[ST_BREAK_POINTs](./ST_BREAK_POINTs/)** | 高頻度測定の時系列データにおける構造変化点（ブレイクポイント）の検出アプリ |
| **[ST_SARIMA_prediction](./ST_SARIMA_prediction/)** | 高頻度測定の時系列データのSARIMAモデルを用いたアラート機能 |
| **[ST_ti_fft_kde_regression](./ST_ti_fft_kde_regression)** | FFT・KDEを用いた時系列回帰分析アプリ、TI実装（💡推奨版） |

---

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

### 本プロジェクトのコード
本リポジトリに含まれるソースコードは [MIT License](https://ja.wikipedia.org/wiki/MIT%E3%83%A9%E3%82%A4%E3%82%BB%E3%83%B3%E3%82%B9) のもとで公開されています。

### デモデータについて（出典・引用）
- **MIT License はサンプルデータには適用されません。**
* **利用規約:** [環境省ホームページ 利用規約（公共データ利用規約（第1.0版））](https://www.env.go.jp/other/kiyaku.html) に基づき出典を明記の上、利用しています。

本アプリケーションのデモデータには、環境省が公開している以下のデータを加工・引用して使用しています。

* **出典:** 
年１回のモニタリンク結果は、
[化学物質と環境](https://www.env.go.jp/chemi/kurohon/)　
[環境省 化学物質と環境 データ検索](https://www.kuro-hon.env.go.jp/search/ja)

高頻度監視データは、
環境省「化学物質と環境（令和7年度版 / 2025）」添付資料5 POPs残留状況の高頻度監視分析機関報告データ
* **URL:** https://www.env.go.jp/chemi/kurohon/2025/shosai/5_5.pdf

※デモ用にデータの一部抽出やフォーマットの変換を行っています。
- サンプルデータの利用条件・二次利用（出典の記載等）については、[環境省ホームページの著作権・リンクについて](https://www.env.go.jp/mail.html) に従ってください。

### サードパーティ・ライブラリ
- **NumPyro / JAX** : Apache License 2.0
- **emcee** : MIT License
- **Streamlit** : Apache License 2.0
- **NumPy / SciPy / Pandas / statsmodels** : BSD License

### サードパーティ・ライブラリ

* **NumPyro / JAX**: Apache License 2.0
* **emcee**: MIT License
* **Streamlit**: Apache License 2.0
* **NumPy / SciPy / Pandas / statsmodels**: BSD License


Hiroaki Shiraishi （2026）
