"""
scripts/run_kde_mle.py

Example script for running KDE-MLE trend estimation.

This script:
  1. Loads a CSV file containing year, concentration, detection_limit
  2. Applies non-detect substitution (DL / sqrt(2))
  3. Computes log10 concentration
  4. Runs KDE-MLE (simple or iterative)
  5. Prints results

This script is intended to call:
    from models.kde_mle import run_mle_and_aic_simple, run_mle_and_aic_iterative
"""

import pandas as pd
import numpy as np
import os

from module_kde_mle import (
    run_mle_and_aic_simple,
    run_mle_and_aic_iterative
)


# -----------------------------
# Utility: clean numeric values
# -----------------------------
def clean_value_user(val):
    """Convert strings like 'tr(0.1)', 'nd', '---' into numeric values."""
    if isinstance(val, str):
        val = val.replace("tr(", "").replace(")", "")
        val = val.replace("---", "")
        val = val.replace("nd", "0")
    try:
        if val == "":
            return np.nan
        return float(val)
    except Exception:
        return np.nan


# -----------------------------
# Main execution
# -----------------------------
def main():

    # === 1. Select input file ===
    filename = "HCB大気温暖期2013.csv"
    filepath = os.path.join("data", filename)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"\n=== Loading data: {filename} ===")

    # === 2. Load data ===
    df = pd.read_csv(
        filepath,
        usecols=["year", "concentration", "detection_limit"]
    )

    # === 3. Clean values ===
    df["concentration"] = df["concentration"].apply(clean_value_user)
    df["detection_limit"] = df["detection_limit"].apply(clean_value_user)

    # === 4. Non-detect substitution ===
    is_nd = df["concentration"] <= df["detection_limit"]
    substitute = df["detection_limit"] / np.sqrt(2)

    df["concentration_sub"] = df["concentration"].copy()
    df.loc[is_nd, "concentration_sub"] = substitute[is_nd]

    # === 5. log10 transform ===
    df["log_conc"] = np.log10(df["concentration_sub"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_conc"])

    print(f"Data points after cleaning: {len(df)}")

    # === 6. Run KDE-MLE (simple version) ===
    print("\n=== Running KDE-MLE (simple) ===")
    a, b, aic0, aic1, p1, x_base = run_mle_and_aic_simple(df)

    print(f"Slope (a): {a:.4f}")
    print(f"Intercept (b): {b:.4f}")
    print(f"AIC0 (constant): {aic0:.3f}")
    print(f"AIC1 (linear):   {aic1:.3f}")
    print(f"AIC posterior weight p1: {p1:.4f}")

    # === 7. Run KDE-MLE (iterative version) ===
    print("\n=== Running KDE-MLE (iterative) ===")
    a2, b2, aic0_2, aic1_2, p1_2, x_base2 = run_mle_and_aic_iterative(df,max_iter=50, tolerance=5e-3)

    print(f"Slope (a): {a2:.4f}")
    print(f"Intercept (b): {b2:.4f}")
    print(f"AIC0 (constant): {aic0_2:.3f}")
    print(f"AIC1 (linear):   {aic1_2:.3f}")
    print(f"AIC posterior weight p1: {p1_2:.4f}")

    print("\n=== Finished KDE-MLE analysis ===")


if __name__ == "__main__":
    main()
