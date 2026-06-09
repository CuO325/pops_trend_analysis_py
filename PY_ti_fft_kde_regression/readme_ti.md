---

# 📄 **README.md（English Version）**

```markdown
# Bayesian Trend Analysis with FFT-KDE Likelihood and Thermodynamic Integration

This repository provides a complete framework for Bayesian trend estimation using  
**FFT-based Kernel Density Estimation (FFT-KDE)** for likelihood construction and  
**Thermodynamic Integration (TI)** for marginal likelihood (logZ) computation and  
Bayes Factor model comparison.

The tool is designed for environmental time‑series data where:
- Observations may be irregularly spaced,
- Measurement noise is non-Gaussian,
- Missing years are common,
- Classical linear regression assumptions do not hold.

The framework includes:
- Bayesian linear regression with FFT-KDE likelihood  
- MCMC sampling (emcee) with R-hat diagnostics  
- Thermodynamic Integration for logZ estimation  
- Model comparison (linear vs constant) via Bayes Factor  
- Automatic convergence detection  
- HTML report generation with plots (E[logL], Rhat, posterior summaries)

---

## Features

### 🔹 FFT-KDE Likelihood
Instead of assuming Gaussian residuals, the likelihood is estimated using  
FFT-based kernel density estimation. This allows flexible, data-driven likelihoods.

### 🔹 Bayesian Linear Regression
Posterior sampling is performed using **emcee** (affine-invariant ensemble sampler).  
The posterior includes:
- Slope `a`
- Intercept `b`
- Informative priors with optional scaling

### 🔹 R-hat Convergence Diagnostics
For both posterior sampling and TI sampling, R-hat is computed to ensure  
chain convergence.

### 🔹 Thermodynamic Integration (TI)
Marginal likelihood is computed via:

\[
\log Z = \int_0^1 \mathbb{E}_{\beta}[\log L] \, d\beta
\]

Supports:
- Linear model
- Constant model (slope fixed to zero)
- Fine or custom β-schedules
- E[logL] plots saved to `output/ti/`

### 🔹 Bayes Factor Model Comparison
\[
\text{BF} = \exp(\log Z_{\text{linear}} - \log Z_{\text{const}})
\]

Interpretation follows Jeffreys scale.

### 🔹 HTML Report Generation
The script outputs a complete HTML report including:
- Posterior means and credible intervals  
- R-hat values  
- TI β–E[logL] plots  
- Model comparison summary  
- Diagnostic messages  

---

## Directory Structure

```
project/
│
├── src/
│   ├── run_analysis.py
│   ├── bayes_core.py
│   ├── ti.py
│   ├── kde_fft.py
│   └── utils.py
│
├── output/
│   ├── reports/
│   └── ti/
│       ├── ti_plot_YYYYMMDD_HHMMSS.png
│       └── ...
│
└── README.md
```

---

## Usage

Run the main analysis script:

```bash
python run_analysis.py --input data.csv
```

Outputs:
- HTML report in `output/reports/`
- TI diagnostic plots in `output/ti/`

---

## Example Output

- Posterior slope: negative trend with 95% CI not crossing zero  
- R-hat < 1.05 for all parameters  
- TI logZ(linear) ≪ logZ(const)  
- Bayes Factor → decisive support for linear model  

---

## Requirements

- Python 3.9+
- numpy  
- scipy  
- pandas  
- matplotlib  
- emcee  
- numba  
- streamlit (optional for UI)

Install:

```bash
pip install -r requirements.txt
```

---

## Notes

- The constant model is implemented as a linear model with slope `a = 0` fixed.
- FFT-KDE lookup tables are reused for efficiency.
- TI supports both LOO and non-LOO KDE likelihoods.

---

## License

MIT License  
© 2026 Hiroaki Shiraishi

```

---



