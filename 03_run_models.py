# 03_run_models.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import joblib

DATA = Path("data/processed/dataset.parquet")
OUT = Path("outputs")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main(seed=42):
    df = pd.read_parquet(DATA)

    y = df["leverage"].astype(float).values
    X_cols = ["roa", "size_ln_assets", "tangibility", "market_to_book", "industry_median_leverage"]
    X = df[X_cols].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )

    # ---------- OLS baseline ----------
    X_train_ols = sm.add_constant(X_train)
    X_test_ols = sm.add_constant(X_test)

    ols = sm.OLS(y_train, X_train_ols).fit()
    yhat_train_ols = ols.predict(X_train_ols)
    yhat_test_ols = ols.predict(X_test_ols)

    ols_train_r2 = r2_score(y_train, yhat_train_ols)
    ols_test_r2 = r2_score(y_test, yhat_test_ols)
    ols_test_rmse = rmse(y_test, yhat_test_ols)

    # ---------- Ridge with 5-fold CV ----------
    lambdas = np.logspace(-3, 3, 100)  # wide grid
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    cv_mse = []
    for lam in lambdas:
        fold_mse = []
        for tr_idx, va_idx in kf.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model = Ridge(alpha=lam, fit_intercept=True)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            fold_mse.append(mean_squared_error(y_va, pred))
        cv_mse.append(np.mean(fold_mse))

    best_idx = int(np.argmin(cv_mse))
    best_lambda = float(lambdas[best_idx])

    ridge = Ridge(alpha=best_lambda, fit_intercept=True)
    ridge.fit(X_train, y_train)
    yhat_train_ridge = ridge.predict(X_train)
    yhat_test_ridge = ridge.predict(X_test)

    ridge_train_r2 = r2_score(y_train, yhat_train_ridge)
    ridge_test_r2 = r2_score(y_test, yhat_test_ridge)
    ridge_test_rmse = rmse(y_test, yhat_test_ridge)

    # ---------- Save Table 1 ----------
    table = pd.DataFrame([
        {"Model": "OLS (baseline)", "Training R2": ols_train_r2, "Test R2": ols_test_r2, "Test RMSE": ols_test_rmse},
        {"Model": f"Ridge (alpha={best_lambda:.4g})", "Training R2": ridge_train_r2, "Test R2": ridge_test_r2, "Test RMSE": ridge_test_rmse},
    ])

    table.to_csv(OUT / "tables" / "table1_performance.csv", index=False)

    # Save models + metadata
    joblib.dump({"ols": ols, "ridge": ridge, "X_cols": X_cols, "best_lambda": best_lambda}, OUT / "models" / "models.joblib")

    print("Best lambda (ridge):", best_lambda)
    print(table)

if __name__ == "__main__":
    main()

