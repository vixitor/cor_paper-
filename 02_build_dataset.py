# 02_build_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
YF = Path("data/raw/yf")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# Common line items in yfinance balance_sheet / financials
BS_TOTAL_ASSETS = ["Total Assets"]
BS_TOTAL_LIAB = ["Total Liab", "Total Liabilities Net Minority Interest"]
BS_SHORT_DEBT = ["Short Long Term Debt", "Current Debt"]
BS_LONG_DEBT = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
BS_PPE = ["Property Plant Equipment", "Net PPE"]
BS_EQUITY = ["Total Stockholder Equity", "Stockholders Equity"]

IS_NET_INCOME = ["Net Income", "Net Income Common Stockholders"]

def pick_first_available(df, candidates):
    for c in candidates:
        if df is not None and not df.empty and c in df.index:
            return c
    return None

def most_recent_col(df):
    # yfinance columns are timestamps; take max
    if df is None or df.empty:
        return None
    return max(df.columns)

def value_at(df, rowname, col):
    if rowname is None or col is None:
        return None
    try:
        v = df.loc[rowname, col]
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

def winsorize(s, p=0.01):
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lo, hi)

def main():
    meta = pd.read_csv(RAW / "yf_meta.csv")
    tickers = meta["ticker"].astype(str).tolist()

    records = []
    for tkr in tickers:
        try:
            bs_path = YF / f"{tkr}_balance_sheet.parquet"
            is_path = YF / f"{tkr}_income_statement.parquet"
            if not bs_path.exists() or not is_path.exists():
                continue

            bs = pd.read_parquet(bs_path)
            is_ = pd.read_parquet(is_path)

            col_bs = most_recent_col(bs)
            col_is = most_recent_col(is_)

            row_assets = pick_first_available(bs, BS_TOTAL_ASSETS)
            row_liab = pick_first_available(bs, BS_TOTAL_LIAB)
            row_sd = pick_first_available(bs, BS_SHORT_DEBT)
            row_ld = pick_first_available(bs, BS_LONG_DEBT)
            row_ppe = pick_first_available(bs, BS_PPE)
            row_eq = pick_first_available(bs, BS_EQUITY)
            row_ni = pick_first_available(is_, IS_NET_INCOME)

            total_assets = value_at(bs, row_assets, col_bs)
            total_liab = value_at(bs, row_liab, col_bs)
            short_debt = value_at(bs, row_sd, col_bs)
            long_debt = value_at(bs, row_ld, col_bs)
            ppe = value_at(bs, row_ppe, col_bs)
            equity = value_at(bs, row_eq, col_bs)
            net_income = value_at(is_, row_ni, col_is)

            mcap = meta.loc[meta["ticker"] == tkr, "marketCap"].iloc[0] if "marketCap" in meta.columns else None
            sector = meta.loc[meta["ticker"] == tkr, "sector"].iloc[0] if "sector" in meta.columns else None

            # Construct debt: prefer short+long; fallback to total liabilities as proxy (not ideal)
            if short_debt is not None or long_debt is not None:
                total_debt = (short_debt or 0.0) + (long_debt or 0.0)
            else:
                total_debt = total_liab  # rough proxy if debt items missing

            if total_assets is None or total_assets == 0:
                continue

            leverage = total_debt / total_assets
            roa = None if net_income is None else net_income / total_assets
            size = np.log(total_assets) if total_assets > 0 else None
            tang = None if ppe is None else ppe / total_assets

            # Market-to-book proxy: market cap / book equity
            mtb = None
            if mcap is not None and equity is not None and equity != 0:
                mtb = float(mcap) / float(equity)

            records.append({
                "ticker": tkr,
                "sector": sector,
                "date_assets": str(col_bs),
                "leverage": leverage,
                "roa": roa,
                "size_ln_assets": size,
                "tangibility": tang,
                "market_to_book": mtb,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)

    # Basic cleaning
    df = df.dropna(subset=["leverage", "roa", "size_ln_assets", "tangibility", "market_to_book", "sector"])
    # Winsorize to reduce outliers (finance data often heavy-tailed)
    for c in ["leverage", "roa", "size_ln_assets", "tangibility", "market_to_book"]:
        df[c] = winsorize(df[c], p=0.01)

    # Industry median leverage: use sector as proxy "industry"
    df["industry_median_leverage"] = df.groupby("sector")["leverage"].transform("median")

    # Standardize predictors (NOT y), per your paper
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

