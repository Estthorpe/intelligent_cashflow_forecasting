# scripts/generate_full_dataset.py

import os
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import timedelta
from dotenv import load_dotenv

# --------------------------------------
# STEP 0 – Load environment variables
# --------------------------------------
load_dotenv()
print("ENV filepath:", os.getenv("filepath"))
print("ENV outputpath:", os.getenv("outputpath"))

RAW_INPUT_PATH = os.getenv("filepath")
RAW_OUTPUT_PATH = os.getenv("outputpath")
TARGET_SIZE = 300_000

# Setup
faker = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------------------
# STEP 1 – Load and Normalize Train.csv
# --------------------------------------
def normalize_train_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Could not find file: {file_path}")

    df = pd.read_csv(file_path)

    df = df.rename(columns={
        'invoice_id': 'transaction_id',
        'baseline_create_date': 'transaction_date',
        'due_in_date': 'due_date',
        'total_open_amount': 'amount',
        'name_customer': 'vendor_customer'
    })

    df['transaction_type'] = 'AR'
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
    df['payment_terms'] = (df['due_date'] - df['transaction_date']).dt.days
    df['category'] = df['transaction_type']

    df = df[['transaction_id', 'transaction_date', 'due_date', 'vendor_customer',
             'amount', 'transaction_type', 'category', 'payment_terms']]

    return df

# --------------------------------------
# STEP 2 – Clean and Validate the Data
# --------------------------------------
def clean_data(df):
    df = df.drop_duplicates(subset='transaction_id')
    df = df.dropna(subset=['transaction_date', 'amount', 'vendor_customer'])
    df = df[df['amount'] != 0]
    df = df[df['due_date'] >= df['transaction_date']]
    df['amount'] = df['amount'].abs()
    df['payment_terms'] = df['payment_terms'].fillna(30).astype(int)
    return df

# --------------------------------------
# STEP 3 – Generate Synthetic AP Data
# --------------------------------------
def generate_synthetic_ap_data(real_df, target_rows):
    existing_count = len(real_df)
    needed = target_rows - existing_count
    print(f"🔧 Generating {needed:,} synthetic AP records...")

    synthetic_records = []
    vendors = real_df['vendor_customer'].dropna().unique().tolist()
    start_date = real_df['transaction_date'].min()
    end_date = real_df['transaction_date'].max()

    for _ in range(needed):
        txn_date = faker.date_between(start_date=start_date, end_date=end_date)
        amount = round(random.uniform(1000, 50000), 2)
        vendor = random.choice(vendors) if random.random() < 0.8 else faker.company()
        terms = random.choice([15, 30, 45, 60])
        due_date = txn_date + timedelta(days=terms)

        synthetic_records.append({
            'transaction_id': faker.uuid4(),
            'transaction_date': txn_date,
            'due_date': due_date,
            'vendor_customer': vendor,
            'amount': -abs(amount),  # AP = cash outflow
            'transaction_type': 'AP',
            'category': 'AP',
            'payment_terms': terms
        })

    return pd.DataFrame(synthetic_records)

# --------------------------------------
# STEP 4 – Combine & Save Final Dataset
# --------------------------------------
def create_and_save_dataset():
    print("📥 Loading and normalizing real AR data...")
    ar_df = normalize_train_csv(RAW_INPUT_PATH)
    ar_df = clean_data(ar_df)
    print(f"✅ Loaded {len(ar_df):,} cleaned AR records.")

    ap_df = generate_synthetic_ap_data(ar_df, target_rows=TARGET_SIZE)

    full_df = pd.concat([ar_df, ap_df], ignore_index=True)
    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)
    full_df.to_csv(RAW_OUTPUT_PATH, index=False)

    print(f"✅ Final dataset saved to {RAW_OUTPUT_PATH} with {len(full_df):,} rows.")

# --------------------------------------
# Run the Script
# --------------------------------------
if __name__ == "__main__":
    create_and_save_dataset()
