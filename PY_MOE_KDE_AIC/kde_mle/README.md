

---

# 📘 README — KDE-MLE Trend Analysis (`kde_mle_iterative_v1_1d.py` / `v1_1e.py`)

## `kde_mle_iterative_v1_1e.py`（回帰係数の信頼区間・AICc 算出対応版）

実行例: 
  結果の詳細はコンソールに表示され、csv形式でファイルに保存されます。
```
=== Running KDE-MLE (simple) ===
Slope (a): -0.0074, Intercept (b): 0.5518, p1: 0.9991 → p1 >= 0.95 傾向あり
ci傾き ci_a_s = (-0.0113, -0.00339), ci切片 ci_b_s = (0.5182, 0.5853)
  [AICc確認] AICc0 = 46.3411, AICc1 = 32.3388, p1_c = 0.9991

=== Running KDE-MLE (iterative) ===
状態: 収束 (反復回数: 1)
傾き a = -0.0072, 切片 b = 0.5359, p1 = 0.9998 → p1 >= 0.95 傾向あり
傾き a = -0.0072 (-0.0101, -0.0041), 切片 b = 0.5359 (0.4966, 0.5451),
  [AICc確認] AICc0 = 46.3411, AICc1 = 29.7422, p1_c = 0.9998

=== Running KDE-MLE (MOE Fig.5 Yearly Pooled) ===
状態: 最大回数到達 (反復回数: 50)
傾き a = -0.0066, 切片 b = -0.6377,  ci_a_moe(-0.0101, -0.0029),  ci_b_moe(-0.6919, -0.6378)　p1 = 0.9882 → p1 >= 0.95 傾向あり
  [AICc確認] AICc0 = 102.9499, AICc1 = 94.1130, p1_c = 0.9881
```
## KDE‑MLE / Bootstrap / Logistic Model　(環境省法)

##### ND（不検出）データの補正、物質別解析、地点別集計、AIC によるモデル選択など、環境データ解析に必要な処理をすべて自動化しています。データ例は環境省の公表値ですが、環境省とは異なる前処理法を経て解析に投入するデータが異なるので、環境省の増減傾向にの公表とは必ずしも一致はしません。全く同じ判定結果を得るには前処理部分を変更する必要があります。
---

## 📌 概要

本ディレクトリは、**POPs（残留性有機汚染物質）等の環境モニタリングデータに対して、年次トレンドを０次と1次回帰モデルのAICによる比較等を用いて多角的に評価する手法**を実行するための解析スクリプトです。  
環境省の化学物質環境実態調査（黒本）における経年変化解析の統計モデルをベースに、**濃度の増減傾向（トレンド）を総合的に判定**します。  
**KDE‑MLE（simple / iterative / 年度別合成）・ブートストラップ検定・二項ロジスティック回帰・ND（検出下限値未満）補正・物質別一括処理**までを一貫して実行できます。MCMCなどの計算負荷の高い処理を含まないため、軽量かつ高速に動作します。

### 実装されている5つの解析アプローチ

1. **KDE‑MLE（simple）**  
   全データの残差から単一のカーネル密度分布（KDE）を構築して最尤推定を行うベースライン手法です。反復計算の初期値としても利用されます。
2. **KDE‑MLE（iterative）｜全体一括KDE（本リポジトリの主軸）**  
   パラメータ更新に合わせて全体残差のKDEを反復更新する手法です。ND値（同値データ）が集中する年度があっても全体の残差分布で平滑化されるため、外れ値や検出下限値の影響を受けにくく、ロバスト（頑健）にトレンドを判定できます。
3. **KDE‑MLE（iterative, MOE Fig.5 Yearly Pooled）｜環境省・図5準拠（年度別残差合成KDE）**  
   環境省の補足説明資料（図5）に基づき、年度ごとの残差分布を作成・合成して共通の残差分布を構築する手法です。年度ごとの分布形状を忠実に反映できる反面、ND値が集中する物質（Parlar-62など）では、特定の年度だけKDEがスパイク化したり、傾き $a=0$ の定数モデル（0次式）が各年度の分布に過剰適合して1次式との対数尤度差がつかなくなる（AIC事後確率 $p_1$ が過小評価される）という注意点があります。本実装では、**バンド幅（`bw_method`）を全年度・全データで共通化（平滑化）**する工夫を取り入れることで、この問題を解消し安定した判定を実現しています。
4. **ブートストラップ検定による前期・後期比較**  
   観測期間を前期・後期に分割し、リサンプリングによって平均値差の95%信頼区間を算出して有意な濃度変化を検定します。
5. **二項ロジスティック回帰（低濃度・ND割合のトレンド）**  
   期間中の最大検出下限値を統一基準とし、検出下限値未満（ND）となる地点割合の経年変化（増加・減少傾向）を二項分布モデルとAICで評価します。

# パラメータ信頼区間 (a, b) と AICc の算出仕様

`kde_mle_iterative_v1_1e.py` では、従来の点推定（傾き $a$・切片 $b$）および通常の AIC に加え、不確実性の評価と小標本バイアスを補正するための指標を算出しています。

---

## 1. 回帰係数 a（傾き）・b（切片）の信頼区間（95% CI）

本スクリプトでは、残差の分布形状（非正規分布や検出下限値 ND の影響）を保持したまま推定値のばらつきを評価するため、**残差ブートストラップ法（Residual Bootstrap）** によって 95% 信頼区間を算出しています。

### 計算アルゴリズム（`calc_coef_ci_bootstrap_yearly` 等の処理）

1. **残差の抽出**

   最尤推定で得られた回帰式 $\hat{y}=\hat{a}x_{\mathrm{centered}}+\hat{b}$ から、 実測値との残差
   $e_i=y_i-\hat{y}_i$
   を計算します

2. **リサンプリングと擬似データの生成**
   得られた残差ベクトル e から、元のサンプル数 n と同じ個数を**重複を許してランダム抽出（復元抽出）**し、ブートストラップ残差  $e^{\ast}$ を作成します。

   これを予測値 $\hat{y}$ に足し合わせ、新たな擬似データセットを生成します。
   
   $y^{\ast} = \hat{y} + e^{\ast}$
   
4. **再推定の反復**
   生成した $y^{\ast}$ に対して、固定した KDE 対数密度のもとで負の対数尤度を最小化（L-BFGS-B 法）し、ブートストラップ推定値 ($a^{\ast}$, $b^{\ast}$) を求めます。この試行を $N_{\text{boot}}$ 回（MOE Fig.5 モデルではデフォルト 250 回）繰り返します
   *(※計算高速化のため、事前に 2048 点のグリッド上で対数確率密度を評価し、線形補間 `np.interp` を用いて最適化を行っています)*
5. **パーセンタイル法による区間決定**
   得られた推定値群 { $a^{\ast}$ }と{ $b^{\ast}$ } の分布から、下側 2.5% 点と上側 97.5% 点を抽出し、**95% 信頼区間** とします。

### 結果の解釈（CSV出力カラム）
* `slope_ci_*`：傾き $a$ の 95% 信頼区間 `[下限, 上限]`
  * 区間に `0` を含まない場合（例：`[-0.0820, -0.0210]`）、統計的に有意な増減傾向があることを裏付けます。
* `intercept_ci_*`：切片 $b$（基準年 $x_{\text{base}}$ における対数濃度）の 95% 信頼区間 `[下限, 上限]`

---
### 1-2. KDE-MLE（simple）における信頼区間（`slope_ci_simple`, `intercept_ci_simple`）

[`module_kde_mle_2a.py`](https://github.com/CuO325/pops_trend_analysis_py/tree/main/PY_MOE_KDE_AIC/kde_mle/module) の `run_mle_and_aic_simple(df)` では、線形モデルの推定値 ($\hat{a}_1$, $\hat{b}_1$) に基づく残差から固定 KDE を構築したうえで、その負の対数尤度関数の**数値ヘッセ行列（`calc_coef_ci_hessian`）**から 95% 信頼区間（`ci_a_s`, `ci_b_s`）を算出しています。

#### 計算アルゴリズム（`run_mle_and_aic_simple` ＋ `calc_coef_ci_hessian`）

**最適解残差からの固定 KDE 構築**
   線形モデル（L-BFGS-B 最適化）から残差を求め、`gaussian_kde(e_opt)` により固定カーネル密度 `kde_fixed` を作成します（`np.std(e_opt) < 1e-6` の場合は `(np.nan, np.nan)`）。

**固定 KDE 下での負の対数尤度（NLL）定義**
   `kde_fixed` を固定した状態で、パラメータ $\theta = (a, b)$ に対する負の対数尤度を定義します。

   $$
   \mathrm{NLL}_{\mathrm{fixed}}(a, b) = -\sum_{i=1}^{n} \ln \left( f_{\mathrm{fixed}}(y_i - (a x_{\text{centered}, i} + b)) + 10^{-10} \right)
   $$
   
**数値ヘッセ行列と標準誤差の算出（`calc_coef_ci_hessian`）**
   推定点 $\hat{\theta}=(\hat{a}_1, \hat{b}_1)$ の周辺で微小ステップ $\varepsilon = 10^{-4}$ を用いた中心差分から $2 \times 2$ の数値ヘッセ行列 $H$ を求め、逆行列 $\text{Cov} = H^{-1}$ の対角成分から標準誤差 $\text{SE}_a$, $\text{SE}_b$ を算出します（逆行列が存在しない場合や対角成分が 0 以下の場合は `(np.nan, np.nan)`）。
   
**正規近似による 95% 信頼区間**
   標準正規分布の上側 2.5\% 点 $z = \Phi^{-1}(1 - \alpha/2)$, $\alpha=0.05$ を用いて信頼区間を求めます。
   
   $$
   \text{CI}_a = \hat{a}_1 \pm z \cdot \text{SE}_a, \qquad \text{CI}_b = \hat{b}_1 \pm z \cdot \text{SE}_b
   $$
   

#### `iterative` 版との違いと CSV 出力カラム
* **手法の違い**：`simple` 版は **数値ヘッセ行列による漸近信頼区間**（`calc_coef_ci_hessian`）を用いるのに対し、`iterative` 版（`run_mle_and_aic_iterative`）は収束後の KDE を用いた **残差ブートストラップ**（`calc_coef_ci_bootstrap_kde`、デフォルト `n_boot=500`）で区間を算出します。
* **CSV 出力カラム**：`slope_simple` / `slope_ci_simple`（傾き $a$ と 95% 信頼区間）、`intercept_simple` / `intercept_ci_simple`（切片 $b$ と 95% 信頼区間）として出力されます。

## 2. 小標本補正赤池情報量基準（AICc）とモデル重み（$p_{1,c}$）

環境モニタリングデータでは物質や媒体によってサンプル数 n が限られる（例：n = 24 付近）ケースがあるため、通常の AIC に加えて **AICc（Corrected AIC）** を計算しています。

### 計算式（`calc_aicc` / `compute_aic_posterior`）

通常の \text{AIC} = -2 \ln L + 2k に対し、サンプルサイズ n とパラメータ数 k を用いて以下の補正項を加えます。

$$
\text{AICc} = \text{AIC} + \frac{2k(k + 1)}{n - k - 1}
$$

**定数モデル（モデル0：傾向なし y = b_0）**：パラメータ数 k = 1
  
  $$
  \text{AICc}_0 = \text{AIC}_0 + \frac{4}{n - 2}
  $$
  
**線形モデル（モデル1：経年傾向あり y = ax + b）**：パラメータ数 k = 2
  
  $$
  \text{AICc}_1 = \text{AIC}_1 + \frac{12}{n - 3}
  $$

(※ $n - k - 1 \le 0$ の場合は $\infty$ を返します)

### モデル重み $p_{1,c}$（`p1_c_*`）の算出

$\text{AICc}_0$ と $\text{AICc}_1$ の差
$\Delta \text{AICc} = \text{AICc}_0 - \text{AICc}_1$ から、線形モデル（傾向あり）が支持される事後確率（赤池重み）を算出します。

$$
p_{1,c} = \frac{1}{1 + \exp\left(-\frac{1}{2}(\text{AICc}_0 - \text{AICc}_1)\right)}
$$

### AIC と AICc の違い・使い分け
* **小標本時の過学習抑制**：サンプル数 n が小さいとき、通常の AIC はパラメータ数の多い線形モデル（モデル1）を過大評価しやすい傾向があります。AICc は k=2 に対してより大きなペナルティを与えるため、**p_{1,c} は p_1 よりも保守的（慎重）な値**になります。
* **大標本での一致**：サンプル数 n が十分に大きい場合、補正項 $\frac{2k(k+1)}{n-k-1} \to 0$ となるため、AICc は AIC に一致します。


---

## 📁 ディレクトリ構成

```
PY_MOE_KDE_AIC
    └──kde_mle/  
        └──kde_mle/                
            ├── data/                # 入力 .csv
            ├── outputs/             # 自動生成される解析結果 .csv
            ├──　examples/
            ├── module/
            │   └── module_kde_mle_2a.py                 # KDE-MLE の実装
            ├── kde_mle_iterative_v1_1d.py               # 本 README のコード
            └── README.md

```
### 📁 全体構成
```text
pops_trend_analysis_py/
├── PY_emcee_kde_regression/     # KDEベイズ回帰 (emcee版　遅いです)
├── PY_Hierarchical_emcee/       # 階層ベイズモデル (emcee版)　地点別の傾向分析
├── PY_Hierarchical_numpyro/     # 階層ベイズモデル (NumPyro版)　地点別の傾向分析
├── PY_MOE_KDE_AIC/              # MOE / KDE-AIC 比較解析　（環境省の黒本記載の方法です）
│   └── kde_mle                  # 💡**ここです**
├── PY_pops_moe_analysis_plus/   # KDEベイズ回帰 (NumPyro版　速いです)
│ 
├── ST_ti_fft_kde_regression/    # FFTを用いたKDE高速化モデル　(ブラウザ上で動作。TIも選択できます。スクリプトが最新です。)
├── ST_BREAK_POINTs/             # 変化点分析 (Breakpoint)　高頻度測定での季節変動を除外した経年変動を折れ線で解析します。
├── ST_SARIMA_prediction/        # 時系列予測 (SARIMA) 高頻度測定の精度管理、警鐘のためのアプリです。
└── README.md

```
---

## 📥 入力データ仕様

`data/<filename>.csv` は以下の列を含む必要があります：

| 列名 | 説明 |
|------|------|
| `substance` | 物質名 |
| `year` | 年（整数） |
| `site_id` | 地点 ID |
| `concentration` | 測定値 |
| `detection_limit` | 検出下限値 |

---

## 🧪 ND（不検出）データの扱い

ND（`concentration` ≤ `detection_limit`）の場合、濃度は以下の式で代替値に置き換えます：

`concentration_sub = detection_limit / sqrt(2)`

その後、この代替値に対して常用対数を取り、`log_conc` を作成します。

---

## 📊 解析フロー

### 1. **データ読み込み & 前処理**
- ND 補正  
- log10 変換  
- 無限大・欠損の除去  
- 地点数が少ない物質はスキップ（例：24 地点未満）

---

### 2. **KDE‑MLE（simple）**

- 残差に KDE を当てた対数尤度を最大化  
- AIC により線形モデル vs 定数モデルを比較  
- AIC 事後確率 p1 を算出  
- **p1 ≥ 0.95 → 傾向あり**

simple 版は **iterative の初期値としても使用**します。

---

### 3. **KDE‑MLE（iterative）**

- simple 版の (a, b) を初期値に使用  
- 残差 KDE を毎回更新しながら MLE を反復  ｜残渣を全データから作成する
- tolerance は自動推定  
- simple 版と結果が一致するかを確認
- KDE‑MLE（iterative, MOE Fig.5 Yearly Pooled）｜環境省・図5準拠（年度別残差合成KDE）による分析　(add 20261115)

---

### 4. **ブートストラップ（前期 vs 後期）**

- 年の中央値で前期・後期に分割  
- 平均値差の 95% CI を resampling（R=10000）で推定  
- CI が 0 を跨がなければ有意差あり

---

### 5. **二項ロジスティック回帰（ND 割合のトレンド）**

- 年ごとの ND 割合を算出  
- GLM（Binomial）で傾向を推定  
- AIC による p1 を算出  
- ND の増減傾向を評価

---

## 📤 出力

### `outputs/<filename>_conv.csv`
ND 補正後のデータ。

### `outputs/<filename>_2025_moe.csv`
物質別の解析結果：

| 列名 | 説明 |
|------|------|
| slope_MLE | KDE‑MLE の傾き |
| p1_MLE | AIC 事後確率 |
| trend_MLE | 傾向の有無 |
| bootstrap_CI | 前期・後期の平均差 CI |
| binomial_slope | ND 割合の傾き |
| binomial_p1 | ND 割合の AIC 事後確率 |
| binomial_trend | ND 割合の傾向 |

---

## ▶ 実行方法

```bash
# 既定（data/大気温暖期.csv を解析）
python kde_mle_iterative_v1_1d.py

# ファイル名を指定（data/<名前>.csv を順次解析）
python kde_mle_iterative_v1_1d.py 水質 底質 生物

# 入出力ディレクトリを指定
python kde_mle_iterative_v1_1d.py 水質 --data-dir data --output-dir outputs
```

引数を省略した場合は `data/大気温暖期.csv` が解析されます。
入力 CSV は `data/<ファイル名>.csv`、出力は `outputs/` に保存されます。

---

## ⚠ 注意点（重要）

### 1. **KDE は残差の分散がゼロだと壊れる**
- ND しかない年  
- log_conc が全て同じ  
- 残差がゼロ  

この場合、KDE の共分散行列が特異になり、  
`LinAlgError` が発生していましたが、

本コードでは：

- simple 版  
- iterative 版  

の両方で **残差の分散が小さい場合はスキップ**して対応しています。

---

### 2. **Excel で文字化けする場合**
CSV 出力は UTF‑8‑SIG を使用：

```python
df.to_csv(..., encoding="utf-8-sig")
```

---

### 3. **地点数が少ない物質は解析しない**
例：

```python
if len(df) < 24:
    continue
```

---


## 🧩 依存ライブラリ

- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `scikit-learn`

### 💡 インストール方法

```bash
pip install pandas numpy scipy statsmodels scikit-learn
```

---

## 📄 ライセンス

MIT

---

## 🙋‍♂️ サポート・お問い合わせ
個人ライブラリのため個別サポートは原則行っておりません。バグ報告や機能要望がある場合は GitHub の Issues よりお知らせください。


---

