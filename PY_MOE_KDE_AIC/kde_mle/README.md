

---

# 📘 README — kde_mle_iterative_v1_1d.py
## KDE‑MLE / Bootstrap / Logistic Model　(環境省法)

##### ND（不検出）データの補正、物質別解析、地点別集計、AIC によるモデル選択など、環境データ解析に必要な処理をすべて自動化しています。データ例は環境省の公表値ですが、環境省とは異なる前処理法を経て解析に投入するデータが異なるので、環境省の増減傾向にの公表とは必ずしも一致はしません。全く同じ判定結果を得るには前処理部分を変更する必要があります。
---

## 📌 概要

本ディレクトリは、**POPs（残留性有機汚染物質）等の環境モニタリングデータに対して、年次トレンドを多角的に評価する解析スクリプト**です。  
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



プロット画像付きの解析モデルのPY版（PY_pops_moe_analysis_plusなど）やブラウザ上で操作するST版（ST_ti_fft_kde_regression）もあります。
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

```
python スクリプト名.py
```

`filenames` に指定した CSV が順次解析されます。

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

