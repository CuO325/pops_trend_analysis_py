  
 **Hierarchical Student‑t Regression（emcee版）** について、環境統計・階層ベイズ・ロバスト回帰の “研究者向け・実務者向け” の英語版readmeです。


---

# README.md

```markdown
# Hierarchical Student‑t Regression (Per‑Site Slopes and Intercepts)  
### Robust Bayesian Trend Estimation with emcee  
Version: v4.2 (site_t_std_update_fixed_v4_2a.py)

This repository implements a **hierarchical Bayesian regression model** with **Student‑t likelihood**, estimated using the **emcee ensemble MCMC sampler**.  
The model is designed for **multi‑site environmental monitoring data**, where each site has its own slope and intercept, while sharing hierarchical hyperparameters.

The Student‑t likelihood provides **robustness to outliers**, and the hierarchical structure stabilizes estimation across sites with sparse data.

---

## ✨ Key Features

### **1. Hierarchical Student‑t Regression**
Each site \(i\) has:
\[
y_{i,t} = a_i x_{i,t} + b_i + \epsilon_{i,t}, \quad \epsilon_{i,t} \sim t_\nu(0, \sigma)
\]

Hierarchical priors:
\[
a_i \sim \mathcal{N}(\mu_a, \sigma_a), \quad b_i \sim \mathcal{N}(\mu_b, \sigma_b)
\]

Global parameters:
- \( \mu_a, \sigma_a \) — hierarchical mean & scale for slopes  
- \( \mu_b, \sigma_b \) — hierarchical mean & scale for intercepts  
- \( \nu \) — degrees of freedom of Student‑t  
- \( \sigma \) — residual scale  

Supports both **centered** and **non‑centered parameterizations**.

---

## **2. Robust MCMC with emcee**
- Uses **emcee.EnsembleSampler** with:
  - StretchMove  
  - DEMove  
  - DESnookerMove  
- Configurable:
  - Number of walkers  
  - Steps  
  - Burn‑in  
  - Thinning  
  - Initial spread  

The sampler is optimized for **high‑dimensional hierarchical models**.

---

## **3. Convergence Diagnostics**
Includes:
- **Gelman–Rubin R‑hat** (custom implementation)
- Walker trace plots
- Autocorrelation time estimation
- Automatic detection of problematic parameters:
  - Flags parameters with R‑hat > 1.05

---

## **4. Data Filtering & Preprocessing**
Built‑in filters:
- Minimum number of observations per site  
- Minimum end year  
- Analysis year range  
- Cleaning of censored values (`tr()`, `nd`, `---`)  

---

## **5. Posterior Summaries**
For each site:
- Posterior mean & 95% CI of slope/intercept (standardized & original scale)
- Posterior probability that slope is negative/positive
- Half‑life estimation:
  \[
  t_{1/2} = \frac{\log_{10}(2)}{|a|}
  \]
- OLS comparison (slope, intercept, p‑value)
- Observation year statistics

Global summaries:
- Posterior of \( \mu_a, \mu_b, \sigma_a, \sigma_b, \nu, \sigma \)
- Probability that global slope \( \mu_a \) is negative/positive
- Acceptance fraction
- Autocorrelation diagnostics

---

## **6. Output & Reporting**
The script generates:
- Site‑level summary CSV  
- Global summary JSON  
- Optional diagnostic plots:
  - Trace plots  
  - Walker traces  
  - Site‑level regression plots  
  - Posterior distributions  

All outputs are organized by:
```
{output_prefix}/
    ├── _site_summary.csv
    ├── overall_summary.json
    ├── plots_regression_t/
```

---

## 📁 Input Data Format

The script expects a CSV with at least:

| Column | Meaning |
|--------|---------|
| `site_id` | Site identifier |
| `year` | Observation year |
| `log_conc` | Log‑transformed concentration |

Example:
```csv
site_id,year,log_conc
A01,2001,1.23
A01,2002,1.10
B03,1999,0.85
```

---

## 🚀 How to Run

### 1. Prepare your data
Place CSV files under:
```
data/<filename>.csv
```

### 2. Configure settings
Edit:
```python
filenames = ["水質"]
substance_data = ["α-HCH"]
```

### 3. Run the script
```bash
python site_t_emcee.py
```

### 4. Outputs will be saved automatically
- Summary tables  
- Diagnostic plots  
- JSON metadata  

---

## 🧠 Model Structure Overview

### Posterior log‑probability
The core log‑posterior is implemented in:

```python
log_prob(params, x_data, y_data, site_idx, n_sites)
```

Includes:
- Hierarchical priors  
- Weakly informative priors for global parameters  
- Student‑t likelihood (custom stable implementation)  

### Non‑centered version
Improves sampling stability for hierarchical models:
```python
log_prob_non_centered(...)
```

---

## 📊 Diagnostics & Visualization Tools

### Included utilities:
- `plot_walker_traces()`  
- `compute_rhat_from_sampler()`  
- `split_walkers_into_groups()`  
- `gelman_rhat()`  
- `summarize_and_save()`  

These allow deep inspection of:
- Convergence  
- Posterior shape  
- Site‑level behavior  
- Global trends  

---

## 🧪 AI Assistance
This repository was developed with assistance from **Microsoft Copilot** and **Google Gemini**.  
All final code, modeling decisions, and validation were performed by the author.

---

## 📄 License (MIT)

```text
MIT License

Copyright (c) 2026 Hiroaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[... full MIT license text ...]
```

---

## 🙌 Acknowledgements
- emcee development team  
- SciPy & NumPy  
- statsmodels  
- Environmental monitoring researchers  

---

```

---

