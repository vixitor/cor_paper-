# 01_download_fundamentals.py
# Build fundamentals from local SEC FSD (data/2025q3), no web calls.
import pandas as pd
from pathlib import Path

SEC_DIR = Path("data/2025q3")
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# Tag candidates for each line item we care about
TAG_GROUPS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "long_debt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "current_debt": [
        "LongTermDebtCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
    ],
    "ppe": [
        "PropertyPlantAndEquipmentNet",
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
}

TAG_SET = {t for tags in TAG_GROUPS.values() for t in tags}

FORM_PRIORITY = {"10-K": 2, "20-F": 2, "40-F": 2, "10-Q": 1}

def sic_to_sector(sic):
    if pd.isna(sic):
        return "Unknown"
    try:
        sic = int(sic)
    except Exception:
        return "Unknown"
    if 1 <= sic <= 999:
        return "Agriculture"
    if 1000 <= sic <= 1499:
        return "Mining"
    if 1500 <= sic <= 1799:
        return "Construction"
    if 2000 <= sic <= 3999:
        return "Manufacturing"
    if 4000 <= sic <= 4999:
        return "Transport/Utilities"
    if 5000 <= sic <= 5199:
        return "Wholesale"
    if 5200 <= sic <= 5999:
        return "Retail"
    if 6000 <= sic <= 6799:
        return "Finance/RealEstate"
    if 7000 <= sic <= 8999:
        return "Services"
    if 9100 <= sic <= 9999:
        return "PublicAdmin"
    return "Other"

def pick_first(row, candidates):
    for c in candidates:
        v = row.get(c)
        if pd.notna(v):
            return v
    return None

def load_latest_filings():
    sub_cols = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp"]
    sub = pd.read_csv(SEC_DIR / "sub.txt", delimiter="\t", usecols=sub_cols, dtype={"adsh": str})
    sub["form_rank"] = sub["form"].map(FORM_PRIORITY).fillna(0)
    sub = sub[sub["form_rank"] > 0]
    sub = sub.sort_values(["cik", "form_rank", "period"], ascending=[True, False, False])
    latest = sub.groupby("cik", as_index=False).head(1).copy()
    latest["cik_str"] = latest["cik"].apply(lambda x: str(int(x)).zfill(10))
    latest["sector"] = latest["sic"].apply(sic_to_sector)
    return latest[["adsh", "cik_str", "sector", "sic", "period", "form", "fy", "fp"]]

def load_numbers(adsh_list):
    num = pd.read_csv(
        SEC_DIR / "num.txt",
        delimiter="\t",
        usecols=["adsh", "tag", "ddate", "qtrs", "value"],
        dtype={"adsh": str, "tag": str, "ddate": int, "qtrs": int, "value": float},
    )
    num = num[num["adsh"].isin(adsh_list)]
    num = num[num["tag"].isin(TAG_SET)]
    num = num.sort_values(["adsh", "tag", "qtrs", "ddate"], ascending=[True, True, False, False])
    best = num.groupby(["adsh", "tag"], as_index=False).head(1)
    wide = best.pivot(index="adsh", columns="tag", values="value").reset_index()
    return wide

def main():
    latest = load_latest_filings()
    if latest.empty:
        raise RuntimeError("No filings found in SEC data.")

    numbers = load_numbers(latest["adsh"].tolist())
    meta = latest.merge(numbers, on="adsh", how="left")

    records = []
    for _, row in meta.iterrows():
        assets = pick_first(row, TAG_GROUPS["assets"])
        liabilities = pick_first(row, TAG_GROUPS["liabilities"])
        long_debt = pick_first(row, TAG_GROUPS["long_debt"])
        current_debt = pick_first(row, TAG_GROUPS["current_debt"])
        ppe = pick_first(row, TAG_GROUPS["ppe"])
        equity = pick_first(row, TAG_GROUPS["equity"])
        net_income = pick_first(row, TAG_GROUPS["net_income"])

        if pd.isna(assets):
            continue

        records.append(
            {
                "ticker": row["cik_str"],  # use CIK as identifier
                "sector": row["sector"],
                "sic": row.get("sic"),
                "period": row.get("period"),
                "form": row.get("form"),
                "assets": assets,
                "liabilities": liabilities,
                "long_debt": long_debt,
                "current_debt": current_debt,
                "ppe": ppe,
                "equity": equity,
                "net_income": net_income,
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(RAW / "yf_meta.csv", index=False)
    print(f"Saved fundamentals from SEC data: {len(out)} firms -> {RAW/'yf_meta.csv'}")

if __name__ == "__main__":
    main()
