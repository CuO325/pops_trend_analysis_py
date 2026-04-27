

---

# 📄 **README.md　**

```markdown
# Hierarchical Student‑t Regression with NumPyro  
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

\[
y_{i,j} = a_i x_{i,j} + b_i + \epsilon_{i,j}
\]

with partial pooling:

\[
a_i \sim N(\mu_a, \sigma_a), \quad b_i \sim N(\mu_b, \sigma_b)
\]

### 🔹 Robust Student‑t Residuals
The model uses a Student‑t likelihood:

\[
\epsilon_{i,j} \sim t_\nu(0, \sigma)
\]

allowing heavy tails and resistance to outliers.

### 🔹 Non‑Centered Parameterization
Improves convergence and sampling efficiency:

\[
a_i = \mu_a + \sigma_a \cdot a_{\text{raw},i}
\]

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

```
hierarchical_numpyro/
│
├── mymodel_numpyro.py        # Main NumPyro hierarchical model
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

\[
\text{DL} / \sqrt{2}
\]

and log‑transformed:

\[
\log_{10}(\text{concentration})
\]

---

## 🚀 How to Run

```bash
python mymodel_numpyro.py
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

```



