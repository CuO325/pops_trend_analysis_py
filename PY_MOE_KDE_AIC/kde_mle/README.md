

---

# 📘 README — POPs Trend Analysis (KDE‑MLE / Bootstrap / Logistic Model　環境省法)

#### ND（不検出）データの補正、物質別解析、地点別集計、AIC によるモデル選択など、環境データ解析に必要な処理をすべて自動化しています。
#### データ例は環境省の公表値ですが、環境省とは異なる前処理法を経て解析に投入するデータが異なるので、環境省の増減傾向にの公表とは必ずしも一致はしません。
#### 全く同じ判定結果を得るには前処理法を変更する必要があります。
## 📌 概要

本リポジトリは、**POPs（残留性有機汚染物質）等の環境データに対して、年次トレンドを多角的に評価する解析スクリプト**です。
環境省のPOPsモニタリングの増減傾向の判断に利用している統計モデルを組み合わせて、**濃度の増減傾向（トレンド）を判定**します。
**KDE‑MLE（simple / iterative）・ブートストラップ・二項ロジスティック回帰・ND 補正・物質別処理**まで含む総合的な 傾向解析なっています。
MCMCなどの計算負荷の高いジョブは組み込んでいないので、軽量です。

1. **KDE‑MLE（simple）**  
2. **KDE‑MLE（iterative）**  
3. **ブートストラップによる前期・後期比較**  
4. **二項ロジスティック回帰（ND 割合のトレンド）**

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
            ├── kde_mle_iterative_v1_1c.py               # 本 README のコード
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

ND（concentration ≤ detection_limit）は以下で補正します：

$$
\text{concentration\_sub} = \frac{\text{detection\_limit}}{\sqrt{2}}
$$

その後、常用対数を取って `log_conc` を作成します。

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
- 残差 KDE を毎回更新しながら MLE を反復  
- tolerance は自動推定  
- simple 版と結果が一致するかを確認

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

- pandas  
- numpy  
- scipy  
- statsmodels  
- scikit-learn  
- matplotlib  
- emcee  
- tqdm  

---

## 📄 ライセンス

MIT

---

## 🙋‍♂️ サポート

ありません。


---

