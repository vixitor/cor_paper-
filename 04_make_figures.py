# 04_make_figures.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    pack = joblib.load("outputs/models/models.joblib")
    ols = pack["ols"]
    ridge = pack["ridge"]
    X_cols = pack["X_cols"]

    # OLS params include intercept at index 0
    ols_beta = np.array(ols.params[1:], dtype=float)
    ridge_beta = np.array(ridge.coef_, dtype=float)

    labels = ["Profitability (ROA)", "Size (ln assets)", "Tangibility", "Market-to-Book", "Industry Median Lev."]
    x = np.arange(len(X_cols))
    w = 0.38

    plt.figure()
    plt.bar(x - w/2, np.abs(ols_beta), width=w, label="OLS")
    plt.bar(x + w/2, np.abs(ridge_beta), width=w, label="Ridge")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("|Coefficient|")
    plt.title("OLS vs Ridge Coefficients (Absolute Value)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figure1_coefficients.png", dpi=200)
    print(f"Saved: {OUT/'figure1_coefficients.png'}")

if __name__ == "__main__":
    main()

