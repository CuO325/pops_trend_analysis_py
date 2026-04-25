

# 📘 環境データ傾向分析ツール（Python）

このリポジトリは、大気・水質・底質・生物などの環境モニタリングデータに対して、**経年変化（トレンド）を統計的に評価するための統合解析ツール**です。  
最尤推定法、AIC、ブートストラップ検定、二項回帰モデルなど複数の手法を組み合わせ、物質ごとの濃度変化を多角的に評価します。

---

## 🚀 主な特徴

- **不検出値（ND）の統一処理**：検出下限値の 1/√2 による代替  
- **対数線形回帰（最尤推定 + AIC）**：AIC事後確率 p1 による傾向判定  
- **ブートストラップ検定**：前期・後期の平均値差の信頼区間を推定  
- **二項回帰モデル**：低濃度地点割合の経年変化を評価  
- **可視化**：回帰直線とデータ点をプロット  
- **結果の自動集計と CSV 出力**

---

## 📁 入力データ形式

CSV は以下の列を含む必要があります：

| 列名 | 説明 |
|------|------|
| `substance` | 物質名 |
| `year` | 年度（整数） |
| `site_id` | 測定地点 ID |
| `concentration` | 濃度（ND 表記含む） |
| `detection_limit` | 検出下限値 |

例：`data/大気寒冷期.csv`

---

## 🔧 解析フロー

### 1. データ前処理
- ND 表記（nd, tr(), --- など）を統一的に数値化  
- 不検出値は **検出下限値 / √2** で代替  
- 対数変換（log）  
- 無限大・NaN を除去  

### 2. 最尤推定法 + AIC による線形トレンド評価
- 1次式モデル（a × 年 + b）  
- 0次式モデル（定数）  
- AIC事後確率 p1 ≥ 0.95 → **傾向あり**

### 3. ブートストラップ検定
- 年度の中央値で前期・後期に分割  
- 平均値差の 95% 信頼区間を推定  
- CI の符号で有意差を判定  

### 4. 二項回帰モデル
- 統一検出下限値を下回る地点を「低濃度地点」と定義  
- 年ごとの割合を二項回帰で解析  
- p1 ≥ 0.95 → **傾向あり**

### 5. 可視化
- 実測 log 濃度の散布図  
- 最尤推定の回帰直線  
- 分割年（ブートストラップ）を縦線で表示  

### 6. 結果の集計
物質ごとに以下をまとめた CSV を出力：

- 傾き（線形モデル）  
- AIC事後確率 p1  
- ブートストラップ CI  
- 二項回帰の傾き・p1  
- 各手法の傾向判定  

出力例：`大気寒冷期_results.csv`

---

## ▶ 実行方法

```bash
python pops_monitoring.py
```

---

## 📊 想定される用途

- 長期モニタリングデータの傾向解析  
- 汚染物質の減少傾向の科学的評価  
- 行政報告書・学術論文の基礎解析  
- AIC に基づくモデル選択の実践例  

---

## 📄 ライセンス（MIT License）

```
MIT License

Copyright (c) 2026 Hiroaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

（以下 MIT ライセンス全文）
```

