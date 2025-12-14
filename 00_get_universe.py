import pandas as pd
import numpy as np
import requests
import yfinance as yf
from io import StringIO
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def fetch_sp500_tickers():
    """Fetch S&P 500 constituents with a user-agent and a yfinance fallback."""
    errors = []

    try:
        resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        tickers = table["Symbol"].tolist()
        return [t.replace(".", "-") for t in tickers]
    except Exception as e:
        errors.append(f"wiki fetch failed: {e}")

    try:
        tickers = yf.tickers_sp500()
        return [t.replace(".", "-") for t in tickers]
    except Exception as e:
        errors.append(f"yfinance fallback failed: {e}")

    raise RuntimeError("Unable to fetch S&P 500 constituents; " + " | ".join(errors))

def main(n=120, seed=42):
    tickers = fetch_sp500_tickers()

    if n > len(tickers):
        raise ValueError(
            f"Requested {n} tickers but only {len(tickers)} available from S&P 500 list."
        )

    rng = np.random.default_rng(seed)
    sample = rng.choice(tickers, size=n, replace=False)

    pd.DataFrame({"ticker": sample}).to_csv(
        RAW / "universe_sp500_sample.csv", index=False
    )

if __name__ == "__main__":
    main()
