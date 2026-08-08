```markdown
# POPs Monthly Monitoring & Alert System

This repository provides a Streamlit web application for **monthly monitoring of POPs (Persistent Organic Pollutants)**, focusing on **PCB concentrations**.  
The app fits **SARIMA models** to log-transformed monthly concentration data, visualizes trends and seasonality, and performs **anomaly/alert detection** based on prediction intervals.

---

## Features

- **Interactive data upload**
  - Upload PCB monitoring data as a CSV file.
  - Select target substance from a dropdown list.

- **Preprocessing & transformation**
  - Handles missing values (`'---'` → `NaN`, forward fill).
  - Averages daily measurements (`1日目`, `2日目`, `3日目`) into a single monthly concentration.
  - Constructs a monthly time index from `年` and `月`.
  - Applies natural log transform to concentrations.

- **Modeling with SARIMA**
  - Uses `statsmodels` `SARIMAX` for time series modeling.
  - Optional **auto parameter search** with `pmdarima.auto_arima`:
    - Seasonal period fixed at 12 (monthly data).
    - Reasonable bounds on `(p, d, q)` and `(P, D, Q)`.
  - Manual override of `(p, d, q, P, D, Q)` via sidebar controls.
  - Displays **AIC** of the fitted model.

- **Time series diagnostics**
  - Seasonal decomposition (trend, seasonal, residual).
  - ADF test for stationarity with interpretation.
  - ACF/PACF plots after differencing.

- **Forecasting & alert logic**
  - Forecasts future concentrations for a user-selected horizon (1–36 months).
  - Computes **95% prediction intervals** on the log scale and back-transforms to the original scale.
  - Visualizes:
    - Recent observed concentrations.
    - Forecast mean.
    - 95% prediction band.
    - Upper prediction bound as **alert threshold**.
  - Alert rule:
    - If a future observed value exceeds the **upper 95% prediction bound**, it is flagged as an **anomalous increase (alert)**.

---

## Input data format

The app expects a CSV file with at least the following columns (Japanese headers):

- **`調査対象物質`**: Target substance name (string).
- **`年`**: Year (integer).
- **`月`**: Month (integer).
- **`1日目`**, **`2日目`**, **`3日目`**: Daily concentration values (numeric, may contain `'---'` for missing).

Example (header row):

```text
調査対象物質,年,月,1日目,2日目,3日目
PCB-118,2010,1,0.12,0.10,0.11
PCB-118,2010,2,0.09,0.08,---
...
```

The app:

- Replaces `'---'` with missing values.
- Converts daily columns to numeric.
- Averages the available daily values into a single monthly concentration.

---

## Requirements

Main Python dependencies:

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `japanize-matplotlib`
- `statsmodels`
- `pmdarima`

You can install them via:

```bash
pip install streamlit pandas numpy matplotlib japanize-matplotlib statsmodels pmdarima
```

---

## How to run

1. **Clone this repository**:

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install dependencies** (see above).

3. **Run the Streamlit app**:

   ```bash
   streamlit run st_pops_mon_v1a.py
   ```

4. Open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

5. In the **sidebar**:
   - Upload your PCB CSV file.
   - Select the target substance.
   - Choose the analysis start year.
   - Optionally click **“パラメータの自動設定”** to run `auto_arima`.
   - Adjust `(p, d, q, P, D, Q)` if needed.
   - Set the forecast horizon (months).

6. Explore the three tabs:
   - **📊 データ確認・前処理**: Data overview and log-transformed time series.
   - **📈 時系列分解・定常性確認**: Decomposition, ADF test, ACF/PACF.
   - **🚨 モデル予測・警鐘シミュレーション**: Forecasts, prediction intervals, and alert threshold.

---

## Alert interpretation

- **Alert threshold**: The **upper bound of the 95% prediction interval** for each future month.
- **Alert condition**:  
  When a future observed concentration exceeds this upper bound, it is interpreted as a **potential abnormal increase** and should trigger further investigation or risk assessment.

This system is intended as a **screening and monitoring tool**; final regulatory or scientific judgments should be made with expert review and additional context.

---

## License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2025 Hiroaki

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
```