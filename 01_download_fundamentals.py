# 01_download_fundamentals.py
import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

RAW = Path("data/raw")
OUT = Path("data/raw/yf")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    uni = pd.read_csv(RAW / "universe_sp500_sample.csv")
    rows = []

    for tkr in tqdm(uni["ticker"].tolist()):
        try:
            tk = yf.Ticker(tkr)

            bs = tk.balance_sheet  # columns are periods (usually year-end)
            is_ = tk.financials
            info = tk.info or {}

            if bs is None or bs.empty or is_ is None or is_.empty:
                rows.append({"ticker": tkr, "error": "missing fundamentals"})
                continue

            # Store raw tables for debugging / audit
            bs.to_parquet(OUT / f"{tkr}_balance_sheet.parquet")
            is_.to_parquet(OUT / f"{tkr}_income_statement.parquet")

            rows.append({
                "ticker": tkr,
                "sector": info.get("sector", None),
                "industry": info.get("industry", None),
                "marketCap": info.get("marketCap", None),
                "currency": info.get("currency", None),
            })
        except Exception as e:
            rows.append({"ticker": tkr, "error": str(e)})

    meta = pd.DataFrame(rows)
    meta.to_csv(RAW / "yf_meta.csv", index=False)
    print(f"Saved meta to {RAW/'yf_meta.csv'}")

if __name__ == "__main__":
    main()
