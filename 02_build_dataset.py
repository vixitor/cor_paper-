# 02_build_dataset.py
# Build modeling dataset from locally-extracted fundamentals (data/raw/yf_meta.csv).
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def winsorize(s, p=0.01):
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lo, hi)

def main():
    meta_path = RAW / "yf_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run 01_download_fundamentals.py first.")

    meta = pd.read_csv(meta_path)
    required = ["ticker", "sector", "assets", "liabilities", "long_debt", "current_debt", "ppe", "equity", "net_income"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Missing columns in yf_meta.csv: {missing}")

    records = []
    for _, row in meta.iterrows():
        try:
            assets = row["assets"]
            if pd.isna(assets) or assets <= 0:
                continue

            long_debt = row.get("long_debt", np.nan)
            current_debt = row.get("current_debt", np.nan)
            liabilities = row.get("liabilities", np.nan)
            equity = row.get("equity", np.nan)
            net_income = row.get("net_income", np.nan)
            ppe = row.get("ppe", np.nan)
            sector = row.get("sector", "Unknown")

            total_debt = np.nan
            if pd.notna(long_debt) or pd.notna(current_debt):
                total_debt = (long_debt if pd.notna(long_debt) else 0.0) + (current_debt if pd.notna(current_debt) else 0.0)
            elif pd.notna(liabilities):
                total_debt = liabilities

            if pd.isna(total_debt):
                continue

            leverage = total_debt / assets
            roa = np.nan if pd.isna(net_income) else net_income / assets
            size = np.log(assets) if assets > 0 else np.nan
            tang = np.nan if pd.isna(ppe) else ppe / assets

            mtb = np.nan
            if pd.notna(equity) and equity != 0:
                mtb = equity / assets  # proxy since market cap not available

            if any(pd.isna(v) for v in [leverage, roa, size, tang, mtb, sector]):
                continue

            records.append(
                {
                    "ticker": row["ticker"],
                    "sector": sector if isinstance(sector, str) else "Unknown",
                    "leverage": leverage,
                    "roa": roa,
                    "size_ln_assets": size,
                    "tangibility": tang,
                    "market_to_book": mtb,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(records)
    if df.empty:
        print("No usable fundamentals found. Check data/raw/yf_meta.csv for issues.")
        return

    for c in ["leverage", "roa", "size_ln_assets", "tangibility", "market_to_book"]:
        df[c] = winsorize(df[c], p=0.01)

    df["industry_median_leverage"] = df.groupby("sector")["leverage"].transform("median")

    X_cols = ["roa", "size_ln_assets", "tangibility", "market_to_book", "industry_median_leverage"]
    for c in X_cols:
        mu = df[c].mean()
        sd = df[c].std(ddof=0)
        df[c] = (df[c] - mu) / (sd if sd > 0 else 1.0)

    df.to_parquet(OUT / "dataset.parquet", index=False)
    df.to_csv(OUT / "dataset.csv", index=False)
    print(f"Saved dataset: {len(df)} rows -> {OUT/'dataset.csv'}")

if __name__ == "__main__":
    main()
