```markdown
# pops_trend_analysis_py

残留性有機汚染物質（POPs）などの時系列・トレンド分析、統計モデリングを行うためのPythonスクリプト群です。
MCMC（emcee, NumPyro）を用いた階層ベイズモデル、カーネル密度推定（KDE）、SARIMAモデル、構造変化点（ブレイクポイント）検出など、環境モニタリングデータ分析に適した各種解析手法を実装しています。

特に、環境省（MOE）が採用している統計手法の再現や、それをベイズ統計・FFT等へ発展させた高度な解析モジュールを含んでいます。

## 📁 ディレクトリ構成

リポジトリ内の主要なモジュールと機能の概要です。

| ディレクトリ / ファイル名 | 概要 |
| :--- | :--- |
| `PY_Hierarchical_emcee` | `emcee` を用いた階層ベイズモデルによるトレンド分析 |
| `PY_Hierarchical_numpyro` | `NumPyro` を用いた高速な階層ベイズモデリング |
| `PY_MOE_KDE_AIC/kde_mle` | 環境省（MOE）手法：カーネル密度推定（KDE）、AICによるモデル選択・最尤推定、ブートストラップ法など |
| `PY_emcee_kde_regression` | `emcee` とKDEを組み合わせた回帰分析 |
| `PY_pops_moe_analysis_plus` | 環境省（MOE）の手法をベイズ分析（MCMC）へ拡張したモデル |
| `PY_ti_fft_kde_regression` | 高速フーリエ変換（FFT）とKDEを組み合わせた時系列回帰分析 |
| `ST_BREAK_POINTs` | 【Streamlitアプリ】時系列データにおける構造変化点（ブレイクポイント）の検出 |
| `ST_SARIMA_prediction` | 【Streamlitアプリ】SARIMAモデルを用いた時系列データの将来予測 |
| `ST_ti_fft_kde_regression` | 【Streamlitアプリ】FFT・KDEを用いた時系列回帰分析（TI: Thermodynamic Integration 実装） |

## 🛠 必須環境・ライブラリ

本プロジェクトを動作させるには、主に以下のPython環境およびライブラリが必要です。

* **Python 3.9+**
* 主要依存ライブラリ:
  * numpy / scipy / pandas
  * matplotlib / seaborn
  * [emcee](https://emcee.readthedocs.io/)
  * [numpyro](https://num.pyro.ai/) & jax / jaxlib
  * statsmodels
  * streamlit (Web UIアプリの実行に必要)

## 🚀 使い方

### 1. リポジトリのクローン
```bash
git clone [https://github.com/CuO325/pops_trend_analysis_py.git](https://github.com/CuO325/pops_trend_analysis_py.git)
cd pops_trend_analysis_py

```

### 2. 依存ライブラリのインストール

JAXなどの環境依存を考慮し、仮想環境（venvやconda）の作成を推奨します。

```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate

# ライブラリのインストール
pip install numpy scipy pandas matplotlib seaborn emcee numpyro jax statsmodels streamlit

```

### 3. スクリプト・アプリの実行

* **標準スクリプト (`PY_` で始まるディレクトリ)**
各ディレクトリ内のREADME、またはメインスクリプト（`*.py` や `*.ipynb`）の指示に従って実行してください。
* **Streamlit Webアプリ (`ST_` で始まるディレクトリ)**
対象のディレクトリに移動し、以下のコマンドでWebブラウザ上に画面を起動します。
```bash
cd ST_BREAK_POINTs
streamlit run app.py  # (※実際のメインファイル名に書き換えてください)

```



## 📊 データフォーマット

入力データのフォーマット（CSVの列名や日付の形式など）やサンプルデータについては、各ディレクトリ内のREADMEを参照してください。

## 💡 AI アシスタンスに関する表記

本リポジトリに含まれるコードおよびドキュメントの生成・最適化にあたり、補助ツールとして **Google Gemini** および **Microsoft Copilot** を利用しています。

## 📝 ライセンス

### 本プロジェクトのライセンス

本リポジトリのコードは [MIT License](https://www.google.com/search?q=LICENSE) のもとで公開されています。

### サードパーティ・ライブラリのライセンス

本プロジェクトが依存している主要外部ライブラリのライセンスは以下の通りです。

* **NumPyro / JAX**: Apache License 2.0
* **emcee**: MIT License
* **Streamlit**: Apache License 2.0
* **NumPy / SciPy / Pandas / statsmodels**: BSD License

```

```