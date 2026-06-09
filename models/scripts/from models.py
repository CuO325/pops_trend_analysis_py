import os
os.chdir("..")

from models.kde_mle import (
    run_mle_and_aic_simple,
    run_mle_and_aic_iterative
)
#from models.kde_mle import run_mle_and_aic_simple, run_mle_and_aic_iterative
import pandas as pd

filename = "13HCBD_conv.csv"
filepath = os.path.join("data", filename)
df = pd.read_csv(filepath)

print("Simple MLE and AIC:")
a, b, aic0, aic1, p1, x_base = run_mle_and_aic_simple(df)

print("Slope:", a)
print("Intercept:", b)
print("AIC0:", aic0)
print("AIC1:", aic1)
print("Posterior weight p1:", p1)

print("\nIterative MLE and AIC:")
a, b, aic0, aic1, p1, x_base = run_mle_and_aic_iterative(df)

print("Slope:", a)
print("Intercept:", b)
print("AIC0:", aic0)
print("AIC1:", aic1)
print("Posterior weight p1:", p1)
