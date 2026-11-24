```python
# We need to generate a refined README.md incorporating MIT License, 
# Library credits, and AI usage credits.
# Since it's a markdown file generation, we must use the python interpreter.

readme_content = """# Hierarchical Student-t Regression with emcee

MCMC（マルコフ連鎖モンテカルロ法）サンプラーである `emcee` を使用し、観測地点（サイト）ごとに異なる傾き（スロープ）と切片を持つ**階層ベイズ型 Student-t 回帰分析**を行うための Python スクリプトです。

通常の正規分布ではなく Student-t 分布（ $t$ 分布）を尤度関数に採用しているため、外れ値に対してロバスト（頑健）なトレンド解析が可能です。水質データなどの環境モニタリングデータの長期的な傾向分析などに適しています。

## 📌 主な機能

- **階層ベイズモデル**: 各サイトの傾きと切片が、共通のグローバル分布（ハイパーパラメータ）から生成されると仮定し、データが少ないサイトの推定精度を補います。
- **ロバスト回帰**: 尤度に Student-t 分布を採用。異常値や突発的な高濃度データに引きずられない安定したトレンド（傾き）を推定します。
- **データ自動前処理**: 検出下限（ND）未満のデータに対して $1/\\sqrt{2}$ を代入する処理や、対数変換（$\\log_{10}$）を自動で行います。
- **充実した診断機能と可視化**: 
  - 各パラメータの収束指標 $R\\mathchar`-hat$（Gelman-Rubin統計量）の自動計算
  - ウォーカー（Walkers）のトレースプロットによる収束確認
  - 各サイトの事後平均傾きと 95% 信用区間（CI）のバーグラフ出力
  - サイトごとの「OLS（最小二乗法） vs 階層ベイズ」の回帰直線、95% 信用区間（CI）、95% 予測区間（PI）のプロット（オプション）
- **半減時間の算出**: 傾きの事後分布から、物質の濃度が半分になるまでの「半減時間（年）」とその 95% 信用区間を自動計算します。

## 🛠 必要要件（前提ライブラリ）

実行には以下の Python ライブラリが必要です。


```

```text
README.md generated successfully.

```bash
pip install numpy pandas emcee matplotlib scipy japanize-matplotlib statsmodels

```

## 📂 準備するデータ

実行前に、スクリプトと同じディレクトリに `data/` フォルダを作成し、その中に解析対象の CSV ファイルを配置してください。

**構成例:**

```text
.
├── Hierarchical Student-t_emcee.py
├── data/
│   └── 水質.csv
└── outputs/          # 実行後に自動生成されます

```

**CSVの必須カラム:**

* `substance`: 物質名（例: `α-HCH`）
* `year`: 観測年（西暦）
* `site_id`: 観測地点のID（文字列または数値）
* `concentration`: 観測濃度値（NDの場合は `nd` や `---` など、または検出下限値以下の数値）
* `detection_limit`: 検出下限値

## 🚀 使い方

### 1. 設定の変更

スクリプト内の上部にある各種設定項目を、目的に応じて書き換えます。

```python
# 入力設定
filenames = ["水質"]         # data/<name>.csv の名前
substance_data = ["α-HCH"]   # [] にするとCSV内の全物質をループ処理します

# フィルタリング条件
MIN_OBS_FOR_SITE = 5        # サイトごとの最小データ数（これ以下は除外）
MIN_END_YEAR = 2010         # 調査最終年度がこの年以下のサイトを除外
MIN_ANALYSIS_YEAR = 1999    # 解析に含める開始年（この年より大きいデータを採用）
MAX_ANALYSIS_YEAR = 2030    # 解析に含める終了年（この年以下のデータを採用）

# MCMCサンプリング設定
N_WALKERS = 360             # ウォーカー数（2 × ndim 以上が必須）
STEPS     = 10000           # サンプリングステップ数

```

### 2. スクリプトの実行

ターミナルやコマンドプロンプトからスクリプトを実行します。

```bash
python "Hierarchical Student-t_emcee.py"

```

## 📊 出力成果物

実行が完了すると、自動的に `outputs/[ファイル名]_[物質名]/` ディレクトリが作成され、以下のファイルが保存されます。

1. **`[ファイル名]_[物質名]_site_summary.csv`** (各サイトごとの傾き・切片・推定半減時間など)
2. **`[ファイル名]_[物質名]_overall_summary.json`** (モデル全体のハイパーパラメータや $R\\mathchar`-hat$ 収束判定)
3. **`[ファイル名]_[物質名]_slopes_t.png`** (全サイトの傾きのエラーバー付きバープロット)
4. **`[ファイル名]_[物質名]_t_half_hist.png`** (サイト別の平均半減時間の分布ヒストグラム)

## ⚠️ 収束（R-hat）に関する注意点

スクリプト実行時、コンソールに各パラメータの $R\mathchar`-hat$（アールハット）値が表示されます。 コンソール上に `収束が不十分なパラメータ: [...]` としてインデックスが表示された場合は、`STEPS`（ステップ数）を増やすか、`BURN_IN`（バーンイン期間）を長くすることを検討してください。

## ✒️ クレジット & 利用ライブラリ (Credits & Libraries)

本スクリプトの構築にあたり、以下の素晴らしいオープンソースライブラリを利用しています。

* **emcee**: [Foreman-Mackey et al. (2013)](https://github.com/dfm/emcee) - The MCMC Hammer (Affine Invariant Markov chain Monte Carlo Ensemble sampler)
* **NumPy & SciPy**: 数値計算および統計関数の提供
* **Pandas**: データ構造およびデータ前処理
* **Matplotlib**: グラフ・プロットの描画
* **japanize-matplotlib**: グラフ内での日本語フォント表示サポート
* **Statsmodels**: 比較用 OLS（最小二乗法）の計算

### 🤖 AIの利用について (AI Assistance Credit)

本スクリプトのコード最適化、高速化・安定化（v4.2a）およびドキュメントの生成にあたっては、生成AI（Google Gemini）による開発支援を活用しています。

## 📄 ライセンス (License)

このプロジェクトは **MITライセンス** のもとで公開されています。商用利用、修正、配布、サブライセンスを含め、どなたでも無償で自由に使用することができます。詳細についてはリポジトリ内の `LICENSE` ファイル（または以下）を参照してください。

```text
MIT License

Copyright (c) 2026 Hiroaki Shiraishi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

---

**Author:** Hiroaki Shiraishi

**Version:** 高速化・安定化版 v4.2a
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md generated successfully.")

```
ご指定いただいた内容（MITライセンス、利用ライブラリへの謝辞、およびコード生成へのAI活用のクレジット）を追加した `README.md` を作成しました。

この修正により、オープンソースプロジェクトとしての透明性が高まり、他の研究者や開発者が安心して利用・引用できるようになります。

Markdown形式のファイルを生成しましたので、以下のファイルチップからダウンロードするか、下部のテキストエリアから直接コピーしてご利用ください。

Your Markdown file is ready
[file-tag: code-generated-file-0-1780795113347128095]

---

### README.md の内容

```markdown
# Hierarchical Student-t Regression with emcee

MCMC（マルコフ連鎖モンテカルロ法）サンプラーである `emcee` を使用し、観測地点（サイト）ごとに異なる傾き（スロープ）と切片を持つ**階層ベイズ型 Student-t 回帰分析**を行うための Python スクリプトです。

通常の正規分布ではなく Student-t 分布（ $t$ 分布）を尤度関数に採用しているため、外れ値に対してロバスト（頑健）なトレンド解析が可能です。水質データなどの環境モニタリングデータの長期的な傾向分析などに適しています。

## 📌 主な機能

- **階層ベイズモデル**: 各サイトの傾きと切片が、共通のグローバル分布（ハイパーパラメータ）から生成されると仮定し、データが少ないサイトの推定精度を補います。
- **ロバスト回帰**: 尤度に Student-t 分布を採用。異常値や突発的な高濃度データに引きずられない安定したトレンド（傾き）を推定します。
- **データ自動前処理**: 検出下限（ND）未満のデータに対して $1/\sqrt{2}$ を代入する処理や、対数変換（$\log_{10}$）を自動で行います。
- **充実した診断機能と可視化**: 
  - 各パラメータの収束指標 $R\mathchar`-hat$（Gelman-Rubin統計量）の自動計算
  - ウォーカー（Walkers）のトレースプロットによる収束確認
  - 各サイトの事後平均傾きと 95% 信用区間（CI）のバーグラフ出力
  - サイトごとの「OLS（最小二乗法） vs 階層ベイズ」の回帰直線、95% 信用区間（CI）、95% 予測区間（PI）のプロット（オプション）
- **半減時間の算出**: 傾きの事後分布から、物質の濃度が半分になるまでの「半減時間（年）」とその 95% 信用区間を自動計算します。

## 🛠 必要要件（前提ライブラリ）

実行には以下の Python ライブラリが必要です。

```bash
pip install numpy pandas emcee matplotlib scipy japanize-matplotlib statsmodels

```

## 📂 準備するデータ

実行前に、スクリプトと同じディレクトリに `data/` フォルダを作成し、その中に解析対象の CSV ファイルを配置してください。

**構成例:**

```text
.
├── Hierarchical Student-t_emcee.py
├── data/
│   └── 水質.csv
└── outputs/          # 実行後に自動生成されます

```

**CSVの必須カラム:**

* `substance`: 物質名（例: `α-HCH`）
* `year`: 観測年（西暦）
* `site_id`: 観測地点のID（文字列または数値）
* `concentration`: 観測濃度値（NDの場合は `nd` や `---` など、または検出下限値以下の数値）
* `detection_limit`: 検出下限値

## 🚀 使い方

### 1. 設定の変更

スクリプト内の上部にある各種設定項目を、目的に応じて書き換えます。

```python
# 入力設定
filenames = ["水質"]         # data/<name>.csv の名前
substance_data = ["α-HCH"]   # [] にするとCSV内の全物質をループ処理します

# フィルタリング条件
MIN_OBS_FOR_SITE = 5        # サイトごとの最小データ数（これ以下は除外）
MIN_END_YEAR = 2010         # 調査最終年度がこの年以下のサイトを除外
MIN_ANALYSIS_YEAR = 1999    # 解析に含める開始年（この年より大きいデータを採用）
MAX_ANALYSIS_YEAR = 2030    # 解析に含める終了年（この年以下のデータを採用）

# MCMCサンプリング設定
N_WALKERS = 360             # ウォーカー数（2 × ndim 以上が必須）
STEPS     = 10000           # サンプリングステップ数

```

### 2. スクリプトの実行

ターミナルやコマンドプロンプトからスクリプトを実行します。

```bash
python "Hierarchical Student-t_emcee.py"

```

## 📊 出力成果物

実行が完了すると、自動的に `outputs/[ファイル名]_[物質名]/` ディレクトリが作成され、以下のファイルが保存されます。

1. **`[ファイル名]_[物質名]_site_summary.csv`** (各サイトごとの傾き・切片・推定半減時間など)
2. **`[ファイル名]_[物質名]_overall_summary.json`** (モデル全体のハイパーパラメータや $R\mathchar`-hat$ 収束判定)
3. **`[ファイル名]_[物質名]_slopes_t.png`** (全サイトの傾きのエラーバー付きバープロット)
4. **`[ファイル名]_[物質名]_t_half_hist.png`** (サイト別の平均半減時間の分布ヒストグラム)

## ⚠️ 収束（R-hat）に関する注意点

スクリプト実行時、コンソールに各パラメータの $R\mathchar`-hat$（アールハット）値が表示されます。 コンソール上に `収束が不十分なパラメータ: [...]` としてインデックスが表示された場合は、`STEPS`（ステップ数）を増やすか、`BURN_IN`（バーンイン期間）を長くすることを検討してください。

## ✒️ クレジット & 利用ライブラリ (Credits & Libraries)

本スクリプトの構築にあたり、以下の素晴らしいオープンソースライブラリを利用しています。

* **emcee**: [Foreman-Mackey et al. (2013)](https://github.com/dfm/emcee) - The MCMC Hammer (Affine Invariant Markov chain Monte Carlo Ensemble sampler)
* **NumPy & SciPy**: 数値計算および統計関数の提供
* **Pandas**: データ構造およびデータ前処理
* **Matplotlib**: グラフ・プロットの描画
* **japanize-matplotlib**: グラフ内での日本語フォント表示サポート
* **Statsmodels**: 比較用 OLS（最小二乗法）の計算

### 🤖 AIの利用について (AI Assistance Credit)

本スクリプトのコード最適化、高速化・安定化（v4.2a）およびドキュメントの生成にあたっては、生成AI（Google Gemini、Microsoft Copilot）による開発支援を活用しています。

## 📄 ライセンス (License)

このプロジェクトは **MITライセンス** のもとで公開されています。商用利用、修正、配布、サブライセンスを含め、どなたでも無償で自由に使用することができます。詳細についてはリポジトリ内の `LICENSE` ファイル（または以下）を参照してください。

```text
MIT License

Copyright (c) 2026 Hiroaki Shiraishi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

---

**Author:** Hiroaki Shiraishi

**Version:** 高速化・安定化版 v1

```

```