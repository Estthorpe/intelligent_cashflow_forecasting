# scripts/modelling/backtest_and_features.py
# Rolling-origin backtesting with feature enrichment for Prophet.
# - Adds day-of-week, month (sin/cos), and AP/AR (7d rolling mean, shifted) regressors
# - Runs multiple windows
# - Saves stability table + plots
# - Saves best model artifacts for Streamlit

from __future__ import annotations

import sys
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pickle

# ---------- Path safety: add project root ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- Local config ----------
from config.config import (
    DATA_PROCESSED,         # Path to data/processed/clean_ap_ar.csv
    OUTPUT_FORECASTS_DIR,   # Path to outputs/forecasts
    TRAIN_END, VAL_END,     # Not used directly here; we do rolling windows
)

# Extra output dirs
BACKTEST_DIR = PROJECT_ROOT / "outputs" / "backtests"
MODELS_DIR   = PROJECT_ROOT / "outputs" / "models"
for d in (OUTPUT_FORECASTS_DIR, BACKTEST_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Utilities ----------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def iqr_cap(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(q1 - k*iqr, q3 + k*iqr)

def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # basic schema guard
    need = {"transaction_date", "amount", "transaction_type"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {missing} in {path}. Found: {list(df.columns)}")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date", "amount"])
    # cap extreme transactions to stabilize aggregation
    df["amount"] = iqr_cap(df["amount"].astype(float))
    return df

def to_daily_with_ap_ar(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily net cash (y) plus AP/AR daily sums."""
    # AR positive, AP negative already; we separate by sign
    ar = df.copy()
    ar["ar_amount"] = ar["amount"].where(ar["amount"] > 0, 0.0)

    ap = df.copy()
    ap["ap_amount"] = ap["amount"].where(ap["amount"] < 0, 0.0)

    daily_ar = ar.groupby("transaction_date", as_index=False)["ar_amount"].sum()
    daily_ap = ap.groupby("transaction_date", as_index=False)["ap_amount"].sum()
    daily_y  = df.groupby("transaction_date", as_index=False)["amount"].sum()

    daily = daily_y.merge(daily_ar, on="transaction_date", how="left") \
                   .merge(daily_ap, on="transaction_date", how="left")

    daily = daily.rename(columns={"transaction_date": "ds", "amount": "y"})
    daily = daily.sort_values("ds")
    # fill missing ar/ap columns with 0
    daily["ar_amount"] = daily["ar_amount"].fillna(0.0)
    daily["ap_amount"] = daily["ap_amount"].fillna(0.0)
    return daily

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["ds"].dt.dayofweek  # 0=Mon
    # Month as cyclic features
    month = df["ds"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    # Day-of-week one-hot to feed Prophet as regressors (leak-safe)
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=False)
    df = pd.concat([df, dow_dummies], axis=1)
    return df

def add_ap_ar_roll_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Create safe regressors using rolling means shifted by 1 day (no leakage)."""
    df = df.copy()
    df["ar_7d_mean_lag1"] = df["ar_amount"].rolling(window, min_periods=1).mean().shift(1)
    df["ap_7d_mean_lag1"] = df["ap_amount"].rolling(window, min_periods=1).mean().shift(1)
    # Fill early NaNs with 0 (no prior info)
    df[["ar_7d_mean_lag1", "ap_7d_mean_lag1"]] = df[["ar_7d_mean_lag1", "ap_7d_mean_lag1"]].fillna(0.0)
    return df

def build_feature_frame(df_txn: pd.DataFrame) -> pd.DataFrame:
    daily = to_daily_with_ap_ar(df_txn)
    daily = add_calendar_features(daily)
    daily = add_ap_ar_roll_features(daily, window=7)
    return daily

def add_regressors(m: Prophet, reg_cols: list[str]):
    for col in reg_cols:
        m.add_regressor(col)

@dataclass
class WindowConfig:
    train_days: int = 365
    val_days:   int = 60
    test_days:  int = 60
    step_days:  int = 30  # move start by this many days per window

def iter_windows(dates: pd.Series, cfg: WindowConfig):
    """Yield (train_start, train_end, val_end, test_end)."""
    start = dates.min()
    last  = dates.max()
    while True:
        train_start = start
        train_end   = train_start + timedelta(days=cfg.train_days - 1)
        val_end     = train_end + timedelta(days=cfg.val_days)
        test_end    = val_end + timedelta(days=cfg.test_days)

        if test_end > last:
            break

        yield train_start, train_end, val_end, test_end
        start = start + timedelta(days=cfg.step_days)

def slice_period(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    return df[(df["ds"] >= start) & (df["ds"] <= end)].copy()

def train_and_eval(daily_feat: pd.DataFrame, train_start, train_end, val_end, test_end):
    # Split windows
    train_df = slice_period(daily_feat, train_start, train_end)
    val_df   = slice_period(daily_feat, train_end + timedelta(days=1), val_end)
    test_df  = slice_period(daily_feat, val_end + timedelta(days=1), test_end)

    # Prophet fit with regressors
    reg_cols = [
        "month_sin", "month_cos",
        "ar_7d_mean_lag1", "ap_7d_mean_lag1",
        "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"
    ]
    m = Prophet(
        changepoint_prior_scale=0.5,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    add_regressors(m, reg_cols)
    m.fit(train_df[["ds", "y"] + reg_cols])

    def _eval_on(df_actual: pd.DataFrame):
        future = df_actual[["ds"] + reg_cols]
        fcst = m.predict(future)
        merged = df_actual.merge(fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")
        y_true = merged["y"].values
        y_pred = merged["yhat"].values
        return {
            "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            "rmse": rmse(y_true, y_pred),
            "merged": merged,
            "fcst": fcst
        }

    val_metrics  = _eval_on(val_df)
    test_metrics = _eval_on(test_df)

    # Build full history for this window (train+val+test) to store residuals if needed
    hist_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_fcst = m.predict(hist_df[["ds"] + reg_cols])
    hist = full_fcst.merge(hist_df[["ds", "y"]], on="ds", how="left")
    hist["residual"] = hist["y"] - hist["yhat"]

    info = {
        "model": m,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "val_mape": val_metrics["mape"],
        "val_rmse": val_metrics["rmse"],
        "test_mape": test_metrics["mape"],
        "test_rmse": test_metrics["rmse"],
        "val_merged": val_metrics["merged"],
        "test_merged": test_metrics["merged"],
        "hist": hist,
        "reg_cols": reg_cols
    }
    return info

def plot_stability(table: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(table["window_index"], table["val_mape"], marker="o", label="Val MAPE")
    plt.plot(table["window_index"], table["test_mape"], marker="o", label="Test MAPE")
    plt.title("Rolling Backtest Stability (MAPE by window)")
    plt.xlabel("Backtest Window Index")
    plt.ylabel("MAPE (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

def save_best_artifacts(best: dict):
    """Persist best model artifacts for Streamlit & downstream scripts."""
    # 1) Save model pickle
    model_path = MODELS_DIR / "best_prophet.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)

    # 2) Save forecasts & residuals like the training script expects
    hist = best["hist"][["ds", "y", "yhat", "yhat_lower", "yhat_upper", "residual"]]
    hist.to_csv(OUTPUT_FORECASTS_DIR / "history_with_residuals.csv", index=False)

    # For a “full” forecast, we just store the same hist frame (Streamlit uses it for history)
    best["hist"].drop(columns=["residual"]).to_csv(
        OUTPUT_FORECASTS_DIR / "prophet_full_forecast.csv", index=False
    )

    # Validation joined (handy for diagnostics)
    best["val_merged"].to_csv(OUTPUT_FORECASTS_DIR / "validation_forecast.csv", index=False)

    # 3) Save run summary
    summary = {
        "train_start": str(best["train_df"]["ds"].min().date()),
        "train_end":   str(best["train_df"]["ds"].max().date()),
        "val_start":   str(best["val_df"]["ds"].min().date()),
        "val_end":     str(best["val_df"]["ds"].max().date()),
        "test_start":  str(best["test_df"]["ds"].min().date()),
        "test_end":    str(best["test_df"]["ds"].max().date()),
        "val_mape": round(best["val_mape"], 2),
        "val_rmse": round(best["val_rmse"], 2),
        "test_mape": round(best["test_mape"], 2),
        "test_rmse": round(best["test_rmse"], 2),
        "regressors": best["reg_cols"],
        "run_timestamp": datetime.now().isoformat()
    }
    with open(OUTPUT_FORECASTS_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 4) Save plots for report
    # Forecast plot across hist
    plt.figure(figsize=(12, 6))
    plt.plot(best["hist"]["ds"], best["hist"]["yhat"], label="Forecast")
    plt.fill_between(best["hist"]["ds"], best["hist"]["yhat_lower"], best["hist"]["yhat_upper"], alpha=0.3, label="CI")
    plt.plot(best["hist"]["ds"], best["hist"]["y"], label="Actual")
    plt.title("Best Model — Forecast vs Actuals (hist)")
    plt.xlabel("Date"); plt.ylabel("Amount")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(OUTPUT_FORECASTS_DIR / "best_forecast_plot.png", dpi=300)
    plt.show()

    # Residuals plot
    plt.figure(figsize=(12, 4))
    plt.plot(best["hist"]["ds"], best["hist"]["y"] - best["hist"]["yhat"], label="Residuals")
    plt.axhline(0, linestyle="--")
    plt.title("Best Model — Residuals Over Time")
    plt.xlabel("Date"); plt.ylabel("Residual")
    plt.grid(True); plt.tight_layout()
    plt.savefig(OUTPUT_FORECASTS_DIR / "best_residuals_plot.png", dpi=300)
    plt.show()

def main():
    print(f"📥 Loading: {DATA_PROCESSED}")
    tx = load_transactions(DATA_PROCESSED)
    daily_feat = build_feature_frame(tx)

    # Rolling window config
    cfg = WindowConfig(train_days=365, val_days=60, test_days=60, step_days=30)

    results = []
    best = None
    window_index = 0

    for train_start, train_end, val_end, test_end in iter_windows(daily_feat["ds"], cfg):
        window_index += 1
        info = train_and_eval(daily_feat, train_start, train_end, val_end, test_end)

        row = {
            "window_index": window_index,
            "train_start": str(train_start.date()),
            "train_end":   str(train_end.date()),
            "val_end":     str(val_end.date()),
            "test_end":    str(test_end.date()),
            "val_mape":    info["val_mape"],
            "val_rmse":    info["val_rmse"],
            "test_mape":   info["test_mape"],
            "test_rmse":   info["test_rmse"],
        }
        results.append(row)

        # Track best by validation MAPE (primary), then test MAPE as tie-breaker
        if (best is None) or (info["val_mape"] < best["val_mape"] - 1e-9) or \
           (abs(info["val_mape"] - best["val_mape"]) < 1e-9 and info["test_mape"] < best["test_mape"]):
            best = info
            best["window_index"] = window_index
            best["train_start"] = train_start
            best["train_end"]   = train_end
            best["val_end"]     = val_end
            best["test_end"]    = test_end

        print(f"[{window_index}] Train {row['train_start']}→{row['train_end']} | "
              f"Val MAPE {row['val_mape']:.2f}% | Test MAPE {row['test_mape']:.2f}%")

    # Save stability table
    table = pd.DataFrame(results)
    table_path = BACKTEST_DIR / "stability_table.csv"
    table.to_csv(table_path, index=False)
    print(f"✅ Stability table → {table_path}")

    # Stability plots
    plot_stability(table, BACKTEST_DIR / "stability_mape.png")

    # Persist best model + artifacts for Streamlit
    if best is not None:
        print(f"🌟 Best window index: {best['window_index']} | "
              f"Val MAPE {best['val_mape']:.2f}% | Test MAPE {best['test_mape']:.2f}%")
        save_best_artifacts(best)
        print(f"✅ Best model saved to {MODELS_DIR}/best_prophet.pkl and forecast artifacts to {OUTPUT_FORECASTS_DIR}")
    else:
        print("No valid backtest windows produced results. Check your date range and window configuration.")

if __name__ == "__main__":
    main()
