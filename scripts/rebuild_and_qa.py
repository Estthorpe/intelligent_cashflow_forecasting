# scripts/rebuild_and_qa.py
# Regenerates the dataset using correct date parsing for AR (Train.csv),
# generates synthetic AP to target 300k rows, saves to data/processed/clean_ap_ar.csv,
# and runs automated QA checks.

import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from faker import Faker

# ---------------------------
# Env & constants
# ---------------------------
load_dotenv()

RAW_INPUT_PATH = os.getenv("filepath")  # Train.csv
PROCESSED_OUTPUT_PATH = os.getenv("processed_outputpath", "data/processed/clean_ap_ar.csv")
TARGET_SIZE = 300_000  # final target row count (AR + AP)

faker = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ---------------------------
# Utils
# ---------------------------
def ensure_dir(path_str: str):
    out_dir = Path(path_str).parent
    out_dir.mkdir(parents=True, exist_ok=True)

def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    """
    Robustly parse YYYYMMDD values that may be numeric or strings.
    """
    s = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    # Keep only digits (drop stray chars)
    s = s.str.replace(r"\D", "", regex=True)
    # Enforce 8 chars (YYYYMMDD); invalids become NaT
    s = s.where(s.str.len() == 8, None)
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

# ---------------------------
# Step 1 — Load & normalize AR (Train.csv) with correct date parsing
# ---------------------------
def load_and_normalize_ar(train_csv_path: str) -> pd.DataFrame:
    if not train_csv_path or not Path(train_csv_path).exists():
        raise FileNotFoundError(f"Train.csv not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path, low_memory=False)

    # Expected columns from your Train.csv:
    # ['business_code', 'cust_number', 'name_customer', 'clear_date', 'buisness_year',
    #  'doc_id', 'posting_date', 'document_create_date', 'document_create_date.1',
    #  'due_in_date', 'invoice_currency', 'document type', 'posting_id', 'area_business',
    #  'total_open_amount', 'baseline_create_date', 'cust_payment_terms', 'invoice_id', 'isOpen']

    rename_map = {
        "invoice_id": "transaction_id",
        "baseline_create_date": "transaction_date",  # YYYYMMDD
        "due_in_date": "due_date",                  # YYYYMMDD
        "total_open_amount": "amount",
        "name_customer": "vendor_customer",
    }
    missing = [k for k in rename_map.keys() if k not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in Train.csv: {missing}")

    df = df.rename(columns=rename_map)

    # ✅ Correct date parsing (YYYYMMDD)
    df["transaction_date"] = parse_yyyymmdd(df["transaction_date"])
    df["due_date"] = parse_yyyymmdd(df["due_date"])

    # AR -> positive amounts
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df["transaction_type"] = "AR"
    df["category"] = "AR"

    # Compute payment_terms in days (may be NaN if dates are missing)
    df["payment_terms"] = (df["due_date"] - df["transaction_date"]).dt.days

    # Keep standard schema
    cols = [
        "transaction_id", "transaction_date", "due_date", "vendor_customer",
        "amount", "transaction_type", "category", "payment_terms"
    ]
    return df[cols]

# ---------------------------
# Step 2 — Clean AR
# ---------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="transaction_id")
    df = df.dropna(subset=["transaction_date", "due_date", "amount", "vendor_customer"])
    df = df[df["amount"] != 0]
    # Business logic: due_date >= transaction_date
    df = df[df["due_date"] >= df["transaction_date"]]
    # Reasonable payment terms (0–120); fill missing with median 30
    df["payment_terms"] = pd.to_numeric(df["payment_terms"], errors="coerce")
    df.loc[df["payment_terms"].isna() | (df["payment_terms"] < 0) | (df["payment_terms"] > 120), "payment_terms"] = 30
    df["payment_terms"] = df["payment_terms"].astype(int)
    return df

# ---------------------------
# Step 3 — Generate synthetic AP up to TARGET_SIZE
# ---------------------------
def generate_synthetic_ap(ar_df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    current = len(ar_df)
    needed = max(0, target_rows - current)
    if needed == 0:
        return pd.DataFrame(columns=ar_df.columns)

    vendors = ar_df["vendor_customer"].dropna().unique().tolist()
    start_date = ar_df["transaction_date"].min()
    end_date = ar_df["transaction_date"].max()

    recs = []
    for _ in range(needed):
        txn_date = faker.date_between(start_date=start_date.to_pydatetime(),
                                      end_date=end_date.to_pydatetime())
        terms = random.choice([15, 30, 45, 60])
        due_date = txn_date + timedelta(days=terms)
        amount = round(random.uniform(1_000, 50_000), 2)  # AP scale
        vendor = random.choice(vendors) if random.random() < 0.8 else faker.company()

        recs.append({
            "transaction_id": faker.uuid4(),
            "transaction_date": pd.Timestamp(txn_date),
            "due_date": pd.Timestamp(due_date),
            "vendor_customer": vendor,
            "amount": -abs(amount),          # AP = outflow
            "transaction_type": "AP",
            "category": "AP",
            "payment_terms": terms,
        })

    return pd.DataFrame(recs)

# ---------------------------
# Step 4 — Save and QA
# ---------------------------
def run_qa_and_report(df: pd.DataFrame) -> dict:
    info = {}

    # Basic integrity
    info["row_count"] = len(df)
    info["duplicate_ids"] = int(df["transaction_id"].duplicated().sum())
    info["nulls_pct"] = (df.isna().mean() * 100).round(3).to_dict()

    # Dates
    info["start_date"] = str(df["transaction_date"].min())
    info["end_date"] = str(df["transaction_date"].max())
    info["pct_due_ge_txn"] = float((df["due_date"] >= df["transaction_date"]).mean() * 100)

    # AP/AR balance
    type_counts = df["transaction_type"].value_counts(dropna=False)
    info["type_counts"] = type_counts.to_dict()
    info["type_pct"] = ((type_counts / len(df)) * 100).round(2).to_dict()

    # Amount sign vs type
    ap = df["transaction_type"].str.upper().eq("AP")
    ar = df["transaction_type"].str.upper().eq("AR")
    mismatches = ((ap & (df["amount"] > 0)) | (ar & (df["amount"] < 0))).mean() * 100
    info["pct_amount_sign_mismatch"] = round(float(mismatches), 3)

    # Prophet readiness — daily aggregate
    daily = (
        df.groupby("transaction_date", as_index=False)["amount"]
          .sum()
          .rename(columns={"transaction_date": "ds", "amount": "y"})
          .sort_values("ds")
    )
    info["prophet_daily_rows"] = int(len(daily))
    info["prophet_start"] = str(daily["ds"].min()) if len(daily) else None
    info["prophet_end"] = str(daily["ds"].max()) if len(daily) else None

    # Date continuity (missing days in range)
    if len(daily):
        full = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
        info["missing_days_in_range"] = int(len(full.difference(daily["ds"])))
        info["y_nulls"] = int(daily["y"].isna().sum())
        info["y_zero_pct"] = float((daily["y"] == 0).mean() * 100)

    # Quick assertions (raise if broken)
    assert info["duplicate_ids"] == 0, "Duplicate transaction_id found."
    assert info["pct_due_ge_txn"] >= 99.0, "Too many rows where due_date < transaction_date."
    assert info["pct_amount_sign_mismatch"] < 0.1, "Amount sign vs transaction_type mismatch too high."
    assert len(daily) > 300, "Too few daily points for Prophet."

    # Save QA snapshots
    qa_dir = Path("data/processed/_qa")
    qa_dir.mkdir(parents=True, exist_ok=True)
    df.head(1000).to_csv(qa_dir / "sample_rows.csv", index=False)
    daily.head(50).to_csv(qa_dir / "prophet_daily_preview.csv", index=False)
    pd.Series(info).to_csv(qa_dir / "qa_summary.csv")

    return info

def main():
    print("📥 Loading Train.csv and parsing dates as YYYYMMDD...")
    ar = load_and_normalize_ar(RAW_INPUT_PATH)
    print(f"✅ AR loaded: {len(ar):,} rows. Date range: {ar['transaction_date'].min()} → {ar['transaction_date'].max()}")

    print("🧼 Cleaning AR...")
    ar_clean = clean_data(ar)
    print(f"✅ AR cleaned: {len(ar_clean):,} rows remain.")

    print(f"🧪 Generating synthetic AP to reach {TARGET_SIZE:,} total rows...")
    ap_syn = generate_synthetic_ap(ar_clean, TARGET_SIZE)
    print(f"✅ AP synthetic: {len(ap_syn):,} rows.")

    print("🔗 Combining AR + AP...")
    full = pd.concat([ar_clean, ap_syn], ignore_index=True)

    print(f"💾 Saving processed dataset → {PROCESSED_OUTPUT_PATH}")
    ensure_dir(PROCESSED_OUTPUT_PATH)
    full.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    print("🔍 Running automated QA...")
    info = run_qa_and_report(full)

    print("\n✅ QA PASSED")
    print(f"- Rows: {info['row_count']:,}")
    print(f"- Dates: {info['start_date']} → {info['end_date']}")
    print(f"- AP/AR: {info['type_counts']} ({info['type_pct']})")
    print(f"- Due ≥ Txn: {info['pct_due_ge_txn']:.2f}%")
    print(f"- Prophet daily rows: {info['prophet_daily_rows']}, missing days: {info.get('missing_days_in_range', 'n/a')}")
    print(f"- File saved: {PROCESSED_OUTPUT_PATH}")
    print("🔎 QA artifacts: data/processed/_qa/*")

if __name__ == "__main__":
    main()
