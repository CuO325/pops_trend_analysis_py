
# FFT-KDE Based Bayesian Trend Analysis for Environmental Data

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 概要

このPythonスクリプトは環境データの時系列傾向解析を目的としたツールです。FFTを活用したカーネル密度推定（FFT-KDE）をベースにし、最尤推定（MLE）、最小二乗法（OLS）、ベイズ推定（emceeによるMCMC）、および熱力学積分（Thermodynamic Integration; TI）によるベイズ因子計算を組み合わせています。

主な特徴は以下の通りです。

- 環境汚染物質などの検出限界を考慮したデータ前処理
- KDEによる尤度評価を用いた1次式 vs 定数モデルの選択と傾向判定
- emceeを用いた高速なベイズ推定と事後分布の収束判定（R-hat）
- TIによるベイズ因子の計算（モデル比較）
- ブートストラップ検定および二項ロジスティック回帰による傾向評価補助
- 結果のプロット作成と詳細なHTMLレポートの自動生成


<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/0df583b7-5588-4173-ac71-54a573d6ffff" />

---

## 必要環境

- Python 3.8 以上推奨
- 以下のPythonライブラリ
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scipy
  - statsmodels
  - sklearn
  - emcee
  - japanize_matplotlib （日本語フォント対応）
  
インストール例（pip）:
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn emcee japanize-matplotlib
```

---

## 使い方

1. `data`フォルダに対象CSVデータを配置します。CSVは以下の列が含まれている必要があります：
   - `substance`（物質名）
   - `year`（観測年）
   - `site_id`（サンプリング地点ID）
   - `concentration`（濃度）
   - `detection_limit`（検出限界）
   
2. スクリプトを実行すると、データに対して傾向解析（MLE, ベイズ, ブートストラップ等）を行い、結果をJSONファイルとHTMLレポートとして出力します。

3. プロット画像は `outputs/plots_fft_kde` に保存されます。

4. `main()`関数の`DEFAULT_FILENAMES`を編集することで、対象ファイルを変更可能です。

### 解析オプション・機能スイッチの設定（スクリプト冒頭）

スクリプト（`pops_fft_kde_analysis_v1.2a.py`）冒頭の定数および `DEFAULT_BAYES_CFG`（`BayesConfig`）を編集することで、計算精度や実行する解析機能を切り替えられます。

#### 1. パイプライン全体スイッチ
* **`RUN_TI`**（デフォルト: `True`）
  * `True`: 熱力学積分（Thermodynamic Integration; TI）を実行し、対数周辺尤度（`logZ`）とベイズ因子（`BF` / `logBF`）によるモデル比較を行います。
  * `False`: TI をスキップし、MCMC による傾き・切片の事後分布推定のみを実行します（計算時間を大幅に短縮したい場合に有効です）。
* **`MIN_SAMPLES`**（デフォルト: `24`）
  * 解析対象とする最小データ数です。これ未満のサンプル数しかない物質はスキップされます。
* **`PROB_THRESHOLD`**（デフォルト: `0.95`）
  * MLE のモデル選択確率 `p1` やベイズ事後確率 `P(a < 0)` / `P(a > 0)` で「傾向あり」と判定するしきい値です。

#### 2. ベイズ推定・熱力学積分（TI）の詳細設定（`DEFAULT_BAYES_CFG`）
`BayesConfig` のパラメータで、FFT-KDE の評価方法や MCMC・TI のサンプリング条件を調整できます。

* **FFT-KDE 尤度設定**
  * **`mode_loo`**（デフォルト: `True`）: `True` で Leave-One-Out（LOO）対数尤度を使用し、過学習を抑制します。
  * **`bw_factor`**（デフォルト: `1.0`）: カーネル密度推定のバンド幅倍率です。
  * **`grid_size_lookup`**（デフォルト: `1024`）: FFT-KDE ルックアップテーブルのグリッド分割数です（大きいほど高精度、小さいほど高速）。
* **MCMC（事後分布推定）設定**
  * **`mcmc_walkers`**（デフォルト: `32`） / **`mcmc_steps`**（デフォルト: `2000`）: ウォーカー数と最大ステップ数。
  * **`mcmc_burn_frac`**（デフォルト: `0.5`）: バーンインとして破棄する前半ステップの割合。
  * **`mcmc_ci_width_threshold`**（デフォルト: `0.01`）: 傾き `a` の 95% 信用区間幅がこの値未満かつ符号が確定した場合に早期終了（Auto-Stop）します。
  * **`mcmc_stable_rounds`**: int = 5             # 1 → 5 に増やす（最低500ステップ保証）
  * **`mcmc_seed`**（デフォルト: `None`）: 乱数シード（再現性を確保したい場合は整数を指定）。
* **熱力学積分（TI）設定・βスケジュール**
  * **`ti_walkers`**（デフォルト: `32`） / **`ti_steps`**（デフォルト: `2000`） / **`ti_burn_frac`**（デフォルト: `0.5`）: 各温度 $\beta$ における MCMC サンプリング条件。
  * **`beta_schedule`**（デフォルト: `"default"`）: 逆温度 $\beta \in [0, 1]$ の分割スケジュールを指定します。
    * `"default"`（計 51 点）: $[0, 0.9]$ を 30 点、 $[0.9, 1.0]$ を 21 点に分割（事後分布付近を密に評価する標準設定）。
    * `"coarse"`（計 21 点）: $[0, 1.0]$ を等間隔に 21 点分割（動作確認や高速計算向け）。
    * `"dense"`（計 70 点）: $[0, 0.8]$ を 25 点、 $[0.8, 0.95]$ を 25 点、 $[0.95, 1.0]$ を 20 点に分割（高精度なベイズ因子評価向け）。
---

## 機能概要

### 主要な解析

- **MLEによる1次モデル vs 定数モデルのAIC比較**
- **ベイズ推定（FFT-KDE + MCMC）による傾向の事後分布推定**
- **熱力学積分（TI）によるベイズ因子（モデル比較）推定**
- **ブートストラップによる期間分割の平均値差検定**
- **二項ロジスティック回帰による検出限界以下割合の傾向解析**

### 1. MLE-AIC（最尤推定・赤池情報量規準）による傾向評価

頻度主義的アプローチとして、**最尤推定（MLE: Maximum Likelihood Estimation）** と **赤池情報量規準（AIC）** を組み合わせたモデル選択を行っています（プロット内の `MLE回帰直線` および `MLE p1` に相当）。

- **2モデルの比較**: 「傾向なしモデル（$H_0$: 傾き $a = 0$）」と「傾向ありモデル（$H_1$: 傾き $a \neq 0$）」のそれぞれについて最尤推定を行い、AIC を算出します。
- **AIC事後確率（ $p_1$ / Akaike Weight）**: 2モデル間の $\Delta \text{AIC} = \text{AIC}_0 - \text{AIC}_1$ から、傾向ありモデル $H_1$ の相対的な支持確率 $p_1$ を次式で算出します。
  $$p_1 = \frac{1}{1 + \exp(-\Delta \text{AIC} / 2)}$$


### 2. ベイズファクター（ $BF_{10}$ ）の解釈基準

### 1. MLE-AIC（最尤推定・赤池情報量規準）による傾向評価

頻度主義的アプローチとして、**最尤推定（MLE: Maximum Likelihood Estimation）** と **赤池情報量規準（AIC）** を組み合わせたモデル選択を行っています（プロット内の `MLE回帰直線` および `MLE p1` に相当）。

- **2モデルの比較**: 「傾向なしモデル（$H_0$: 傾き $a = 0$）」と「傾向ありモデル（$H_1$: 傾き $a \neq 0$）」のそれぞれについて最尤推定を行い、AIC を算出します。
- **AIC事後確率（ $p_1$ / Akaike Weight）**: 2モデル間の $\Delta \text{AIC} = \text{AIC}_0 - \text{AIC}_1$ から、傾向ありモデル $H_1$ の相対的な支持確率 $p_1$ を次式で算出します。
  $$p_1 = \frac{1}{1 + \exp(-\Delta \text{AIC} / 2)}$$
-  $p_1$　が 0.95以上で傾向ありモデルを採用します。



### 2. ベイズファクター（ $BF_{10}$ ）の解釈基準
 `emcee` によるベイズ事後確率 $P(a < 0)$ やベイズファクター（ $BF_{10}$ ）と並置して比較することで、偽陽性・偽陰性を抑えた堅牢な傾向判定を実現しています。
- 算出されたベイズファクター $BF_{10}$ は、以下の基準（Kass and Raftery, 1995 準拠）に従ってモデルの証拠強度を評価します。

| $BF_{10}$ の値 | 傾向の証拠（ $H_1$: 傾向あり） | 判定 |
| :--- | :--- | :--- |
| $> 100$ | **極めて強い証拠 (Decisive)** | 傾向あり（確定） |
| $10 \sim 100$ | **強い証拠 (Strong)** | 傾向あり |
| $3 \sim 10$ | **中程度の証拠 (Substantial)** | 傾向あり（示唆） |
| $1 \sim 3$ | **弱い証拠 (Anecdotal)** | 傾向なし / 不明瞭 |
| $< 1$ | **$H_0$（傾向なし）を支持** | 傾向なし |



---
### レポート

- JSON形式で結果データを保存
- 詳細なHTMLレポートで解析結果とグラフをわかりやすく表示

---

## ライセンス

このソフトウェアはMITライセンスの下で公開されています。詳細は [LICENSE](./LICENSE) ファイルを参照してください。

---

## 連絡先・作成者

- 作成者: H.S.
- 更新年: 2026年

---

## 参考資料

- Jeffreys, H. (1961). Theory of Probability.
- emcee: https://emcee.readthedocs.io
- KDE、熱力学積分、ベイズモデル比較関連の文献

---

## 注意事項

- データの質や量によっては解析が安定しない場合があります。
- MCMCの収束診断（R-hat）やCI幅に注意してください。
- 検出限界以下の処理は代替値√2分の1を用いています。
- 高速計算のためFFTベースのKDEを利用していますが、理論的理解を伴った利用を推奨します。

---

### 実行例

```bash
python pops_fft_kde_analysis_v1.2a.py
```

このコマンドにより、`data`フォルダ内の指定ファイルを解析し、`outputs/plots_fft_kde`にグラフ画像が作成され、HTMLレポートおよびJSON結果が生成されます。

---

**Thank you for using this tool!**  
If you find issues or want to suggest features, please open an issue on GitHub.

```
