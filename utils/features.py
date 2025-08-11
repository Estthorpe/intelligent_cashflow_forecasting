import numpy as np
import pandas as pd
from sklearn.utils import resample

# -------- Daily aggregation for Prophet --------
def to_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    daily = (df.groupby("transaction_date", as_index=False)["amount"]
               .sum()
               .rename(columns={"transaction_date": "ds", "amount": "y"})
               .sort_values("ds"))
    return daily

# -------- Outlier handling (optional) --------
def winsorize_series(y: pd.Series, p_low=0.01, p_high=0.99):
    lo, hi = y.quantile(p_low), y.quantile(p_high)
    return y.clip(lo, hi)

def maybe_log_transform(y: pd.Series, enable=False):
    if not enable:
        return y, False
    # Shift to positive if needed
    shift = 0
    if (y <= 0).any():
        shift = abs(y.min()) + 1.0
    return np.log(y + shift), True

# -------- AP/AR balancing (for classification tasks only) --------
def balance_ap_ar(df: pd.DataFrame, target_col="transaction_type", ap_label="AP", ar_label="AR"):
    ap = df[df[target_col] == ap_label]
    ar = df[df[target_col] == ar_label]
    if len(ap) == 0 or len(ar) == 0: 
        return df
    minority = ar if len(ar) < len(ap) else ap
    majority = ap if len(ap) > len(ar) else ar
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced = pd.concat([majority, minority_up], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)
