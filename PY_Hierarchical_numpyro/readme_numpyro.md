---

# README.md

```markdown
# Hierarchical Student‑t Regression with NumPyro (NUTS)
### Robust Bayesian Trend Estimation for Multi‑Site Environmental Monitoring  
Version: pops_hierarchical_numpyro.py

This repository implements a **hierarchical Bayesian regression model** with a **Student‑t likelihood** using **NumPyro + NUTS**.  
It is designed for **multi‑site POPs (Persistent Organic Pollutants) monitoring**, where each site has its own slope and intercept, while sharing hierarchical hyperparameters.

This NumPyro version provides **high‑performance HMC sampling**, **automatic differentiation**, and **robust inference** even for large datasets.

---

## ✨ Key Features

### **1. Hierarchical Student‑t Regression (Non‑Centered)**
For each site \(i\):

\[
y_{i,t} = a_i x_{i,t} + b_i + \epsilon_{i,t}, \quad 
\epsilon_{i,t} \sim t_{\nu}(0, \sigma)
\]

Hierarchical priors:
\[
a_i = \mu_a + \sigma_a \cdot a_{i,\text{raw}}, \quad 
b_i = \mu_b + \sigma_b \cdot b_{i,\text{raw}}
\]

Global parameters:
- \( \mu_a, \sigma_a \) — hierarchical mean & scale of slopes  
- \( \mu_b, \sigma_b \) — hierarchical mean & scale of intercepts  
- \( \nu \) — degrees of freedom (heavy‑tail control)  
- \( \sigma \) — observation noise  

Uses **non‑centered parameterization** for stable sampling.

---

## **2. High‑Performance Inference with NumPyro**
- **NUTS (No‑U‑Turn Sampler)**  
- JAX‑accelerated HMC  
- Automatic differentiation  
- Multi‑chain parallel sampling  
- Fast convergence even in high‑dimensional hierarchical models  

Compared to emcee:
- Much **faster** for large datasets  
- Better **geometry‑aware exploration**  
- Provides **exact HMC diagnostics** (divergences, ESS, R‑hat)  

---

## **3. Data Cleaning & Preprocessing**
Built‑in handling of:
- Non‑detects (`ND`, `tr()`, `---`)  
- Substitution using DL/√2  
- Log10 transformation  
- Site filtering:
  - Minimum number of observations  
  - Minimum end year  
  - Analysis year range  

---

## **4. Posterior Summaries**
Global parameters:
- Posterior mean, SD, 95% CI  
- Posterior probability of positive/negative slope  
- Degrees of freedom \( \nu \)  
- Observation noise \( \sigma \)  

Site‑level parameters:
- Posterior mean & CI of slope/intercept  
- Posterior probability slope > 0  
- Bayesian R² per site  
- Half‑life estimation (if needed)

---

## **5. Posterior Predictive Distribution**
Functions included:
- `posterior_predictive_intervals()`  
- `plot_posterior_predictive_facet()`  

Outputs:
- Median prediction  
- 95% credible interval  
- Mean posterior fit  
- Faceted plots across all sites  

---

## **6. Output Structure**
The script generates:
```
results/
    ├── site_summary.csv
    ├── global_summary.json
    ├── predictive_plots/
    └── facet_plots/
```

---

## 📁 Input Data Format

Expected columns:

| Column | Meaning |
|--------|---------|
| `substance` | Chemical name |
| `year` | Observation year |
| `site_id` | Site identifier |
| `concentration` | Measured concentration |
| `detection_limit` | Detection limit |

Example:
```csv
substance,year,site_id,concentration,detection_limit
PCB-118,2010,A01,0.12,0.05
PCB-118,2011,A01,0.10,0.05
PCB-118,2010,B03,0.03,0.02
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
substance_data = ["総PCB"]
```

### 3. Run the script
```bash
python pops_hierarchical_numpyro.py
```

### 4. Outputs will be saved automatically.

---

## 🔍 Comparison with the emcee Version

| Feature | NumPyro (NUTS) | emcee (Ensemble MCMC) |
|--------|----------------|------------------------|
| Sampling method | Hamiltonian Monte Carlo (NUTS) | Random‑walk ensemble sampling |
| Speed | **Fast (JAX‑accelerated)** | Slower for high‑dimensional models |
| Convergence | Strong diagnostics (ESS, R‑hat, divergences) | Requires manual R‑hat implementation |
| Parameterization | Non‑centered hierarchical model | Centered / non‑centered both possible |
| Robustness | Excellent for hierarchical models | May struggle with correlations |
| Posterior predictive | Built‑in Predictive API | Manual implementation |
| Heavy‑tail modeling | Student‑t with learned ν | Student‑t with fixed or sampled ν |

**Summary:**  
NumPyro version is **faster**, **more stable**, and **more scalable**, especially for many sites or long time series.  
emcee version is still valuable when:
- You want ensemble‑based sampling  
- You need full control over the log‑posterior  
- You prefer pure NumPy/Python without JAX  

---

## 🧪 AI Assistance
This repository was developed with assistance from **Microsoft Copilot**.  
All final modeling decisions and validation were performed by the author.

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
- NumPyro & JAX developers  
- Environmental monitoring researchers  
- Bayesian modeling community  

```

---