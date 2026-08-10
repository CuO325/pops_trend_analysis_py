
# Hierarchical Student-t Regression with NumPyro
Robust Multi‑Site Trend Analysis for Environmental and QSAR Data
**多地点・多物質モニタリングデータのための頑健な階層ベイズモデル**

このディレクトリには、NumPyro (NUTS) を用いた高機能な階層ベイズ回帰モデルの実装が含まれています。環境モニタリングデータやQSAR（定量的構造活性相関）など、地点やカテゴリごとに繰り返し測定が行われるデータのトレンド分析に最適化されています。

## ✨ 主な特徴と技術的仕様

このスクリプトは、単一地点の回帰分析では困難な「データが少ない地点の推定」を階層構造によって補完（部分プール）し、高い精度と安定性を両立しています。

- **階層モデル（マルチレベル回帰）**: 各地点に個別の傾きと切片を割り当てつつ、全体の母平均からサンプリングすることで「縮小推定」を実現。
- **非中心化パラメータ化 (Non-centered Parameterization)**: パラメータ間の相関によるサンプリングの停滞を防ぎ、複雑な階層構造でも高速かつ安定して収束します。
- **Student-t分布によるロバスト推定**: 観測データの誤差分布に正規分布ではなく Student-t 分布を採用。外れ値の影響を受けにくい、頑健なトレンド推定が可能です。
- **自動レポート生成**: 各地点ごとの推定値、95%信用区間、半減期（または倍加時間）、収束診断（R-hat）を CSV/JSON 形式で出力します。

## 🚀 使い方

### 1. データの準備
`data/` ディレクトリに、地点名（サイトID）、年度、濃度値を含む CSV ファイルを配置します。

### 2. 設定の変更
`pops_hierarchical_numpyro.py` 冒頭の変数で対象物質や閾値を指定します：
- `filenames`: 対象のファイル名
- `substance_data`: 特定の物質に絞る場合はリストに指定
- `MIN_OBS_FOR_SITE`: サイトごとに最低限必要なデータ数

### 3. 実行
```bash
python pops_hierarchical_numpyro.py
```

📊 出力される成果物
実行後、outputs/（または指定ディレクトリ）に以下が生成されます。

サイト別サマリー (CSV): 地点ごとの傾向判定（上昇・減少の確率）、推定された傾き、信頼区間。

全体統計 (JSON): モデル全体のハイパーパラメータ（μ, σ）やベイズ決定係数（R²）。

事後予測プロット: 実測値とモデルの予測区間を重ね合わせた可視化グラフ。


# Hierarchical Student‑t Regression with NumPyro  （English)
Robust Multi‑Site Trend Analysis for Environmental and QSAR Data

This repository provides a high‑performance hierarchical Bayesian regression model implemented in **NumPyro** (NUTS/HMC).  
It is designed for **multi‑site time‑series analysis**, such as environmental monitoring data, chemical concentration trends, or hierarchical QSAR applications.

The model uses:

- **Hierarchical (multilevel) regression**
- **Non‑centered parameterization** for stable sampling
- **Student‑t likelihood** for robustness against outliers
- **NUTS (No‑U‑Turn Sampler)** for fast and reliable inference
- **Posterior predictive distributions**
- **Bayesian R² (overall and per‑site)**

This implementation is suitable for scientific research, regulatory analysis, and large‑scale environmental datasets.

---

## ✨ Key Features

### 🔹 Hierarchical Bayesian Model
Each site has its own slope and intercept:

$$
y_{i,j} = a_i x_{i,j} + b_i + \epsilon_{i,j}
$$

with partial pooling:

$$
a_i \sim N(\mu_a, \sigma_a), \quad b_i \sim N(\mu_b, \sigma_b)
$$

### 🔹 Robust Student‑t Residuals
The model uses a Student‑t likelihood:

$$
\epsilon_{i,j} \sim t_\nu(0, \sigma)
$$

allowing heavy tails and resistance to outliers.

### 🔹 Non‑Centered Parameterization
Improves convergence and sampling efficiency:

$$
a_i = \mu_a + \sigma_a \cdot a_{\text{raw},i}
$$

### 🔹 NUTS / HMC Sampling
NumPyro provides:

- Automatic step size tuning  
- Mass matrix adaptation  
- Fast JAX‑accelerated inference  

### 🔹 Posterior Predictive Checks
The script generates:

- 95% credible intervals  
- Median and mean prediction curves  
- Faceted plots for all sites  

### 🔹 Bayesian R²
Both global and per‑site Bayesian R² are computed.

---

## 📁 Repository Structure

```text
hierarchical_numpyro/
│
├── pops_hierarchical_numpyro.py        # Main NumPyro hierarchical model
├── README.md                 # This file
├── data/                     # Input CSV files
└── plots/                    # Optional output plots

```

---

## 📊 Input Data Format

The script expects a CSV file with the following columns:

| Column          | Description |
|-----------------|-------------|
| `substance`     | Chemical name |
| `year`          | Observation year (int) |
| `site_id`       | Site identifier (string or int) |
| `concentration` | Measured concentration |
| `detection_limit` | Detection limit |

Non‑detects are automatically replaced using:

### Data preprocessing

- Non-detects are replaced with **DL / √2** (Hornung & Reed method).
- Concentrations are transformed using **log₁₀(concentration)** prior to modeling.

---

## 🚀 How to Run

```bash
python pops_hierarchical_numpyro.py
```

The script will:

1. Load and clean the dataset  
2. Filter sites by minimum observations and end year  
3. Standardize predictors  
4. Run NUTS sampling  
5. Generate summaries and plots  
6. Save results as CSV and JSON  

---

## 📦 Output Files

### 1. **Site‑level summary (CSV)**  
Includes:

- Posterior mean and 95% CI for slopes and intercepts  
- Posterior probability of positive/negative slope  
- Half‑life estimates  
- OLS comparison  
- Observation counts  

### 2. **Global summary (JSON)**  
Includes:

- Posterior summaries for μₐ, μᵦ, σₐ, σᵦ, σ, ν  
- Bayesian R²  
- Site‑level observation counts  
- Model diagnostics  

### 3. **Posterior predictive plots**  
Faceted plots showing:

- Observed data  
- Median prediction  
- 95% credible interval  

---

## 📘 Model Overview

This model is suitable for:

- Environmental monitoring (water quality, air pollution, POPs, PCBs, pesticides)
- Multi‑site chemical trend analysis
- Exposure assessment
- Hierarchical QSAR modeling
- Any dataset with repeated measurements across groups

Advantages:

- Handles sparse sites via partial pooling  
- Robust to outliers  
- Provides uncertainty quantification  
- Scales well with NumPyro/JAX  

---

## 🔧 Dependencies

```
numpy
pandas
jax
jaxlib
numpyro
matplotlib
japanize_matplotlib
```

Install with:

```bash
pip install numpyro jax jaxlib pandas matplotlib japanize-matplotlib
```

---

## 📄 License

MIT License  
© 2026 Hiroaki

---

## 🙌 Acknowledgements

This project uses:

- **NumPyro** (Apache 2.0)  
- **JAX** (Apache 2.0)  
- **Matplotlib** (PSF license)  

---

## 💬 Contact

For questions or collaboration, feel free to open an issue or pull request.


# Hierarchical Student-t regression.pyとの違い

 `Hierarchical Student-t regression.py`（以下、**emcee版**）と、 `pops_hierarchical_numpyro.py`（以下、**NumPyro版**）は、どちらも「階層ベイズ + Student-t分布」という同じ数理モデルを目指していますが、**「エンジン（計算手法）」**と**「実用上の特性」**が大きく異なります。



---

## 1. サンプリング・アルゴリズムの違い
ここが最も根本的な違いです。

* **emcee版 (Ensemble Sampler):**
    「アフィイン不変アンサンブルサンプラー」を使用します。複数の「ウォーカー（探索者）」を放ち、お互いの位置を参考にしながら分布を探ります。パラメータ間の相関に強く、直感的に動かしやすいのが特徴です。
* **NumPyro版 (NUTS/HMC):**
    「ハミルトニアンモンテカルロ（NUTS）」を使用します。物理学の運動方程式をシミュレーションするように、勾配（坂道の傾き）を利用して効率的に次の一歩を決めます。

## 2. 計算速度とスケーラビリティ
* **emcee版:**
    CPUで動作し、実装がシンプルです。数地点〜数十地点程度の解析なら十分ですが、地点数やデータ数が膨大になると計算時間が線形に増えていきます。
* **NumPyro版:**
    JAXをベースにしているため、**並列計算が圧倒的に速い**です。特にCPUのマルチコアやGPUを活かせるため、全国規模の数百地点を一度に解析するようなケースではNumPyro版が劇的に有利です。

---

## 3. 実装上の「工夫」の差

| 特徴 | emcee版 (v4.2) | NumPyro版 |
| :--- | :--- | :--- |
| **パラメータ化** | 通常のパラメータ化 | **非中心化パラメータ化** (より高度) |
| **収束の安定性** | ウォーカー数に依存 | 数学的な工夫で複雑な階層でも詰まりにくい |
| **自動停止機能** | 自己相関時間に基づき、収束したら止まる | 事前に指定したサンプル数を回し切る |
| **コードの複雑さ** | 比較的読みやすく、挙動が追いやすい | JAX/NumPyro特有の記法が必要 |

---

## 4. どちらをいつ使うべきか？

> **emcee版を選ぶべき時**
> - 1物質や少数の地点をじっくり、かつ確実に解析したいとき。
> - 計算の進捗（ウォーカーの動き）をログで細かく追いながら、納得感を持って進めたいとき。
> - JAXなどの重いライブラリをインストールできない環境のとき。

> **NumPyro版を選ぶべき時**
> - **大量の物質や地点をまとめて高速に処理したいとき。**
> - モデルをさらに複雑にする（例：年次だけでなく季節性も階層に入れる等）拡張を予定しているとき。
> - 最新のベイズ推論のスタンダードに合わせたいとき。

**結論として：**
 `Hierarchical Student-t regression.py` は、計算の安定性と診断機能（Rhatや自己相関の表示）が非常に充実した「完成された職人ツール」です。一方で `NumPyro版` は、そのロジックを最新の高速エンジンに積み替えた「次世代の量産機」と言えます。

まずは emcee版でモデルの妥当性を確認し、計算時間が課題になってきたら NumPyro版へ移行するという流れがスムーズです。

    

