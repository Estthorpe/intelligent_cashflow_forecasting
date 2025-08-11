# scripts/anomaly/detect.py
# Detect anomalies on Prophet residuals, classify (Liquidity Gap / Unexpected Inflow),
# score severity, and write an email-ready JSON for n8n.

from __future__ import annotations

# --- Make script runnable from anywhere (adds project root to sys.path) ---
import sys
from pathlib import Path

# detect.py is at: <project>/scripts/anomaly/detect.py
# project_root = parents[2] => <project>
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Standard libs & third-party ---
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --- Local imports (now resolvable because we added project_root) ---
from config.config import (
    OUTPUT_FORECASTS_DIR,
    OUTPUT_ALERTS_DIR,
    ZSCORE_THRESHOLD,
    ISOF_CONTAM,
    RANDOM_STATE,
)
from utils.io_utils import ensure_dir, save_csv, save_json


def rolling_zscore(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add a rolling z-score on residuals."""
    s = df["residual"].astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    df["zscore"] = (s - mu) / sd
    return df


def isolation_forest_flag(
    df: pd.DataFrame, contam: float = 0.01, random_state: int = 42
) -> pd.DataFrame:
    """Add IsolationForest anomaly flag on residuals."""
    X = df[["residual"]].fillna(0.0).values
    iso = IsolationForest(contamination=contam, random_state=random_state)
    df["iso_pred"] = iso.fit_predict(X)  # -1 anomaly, 1 normal
    df["anomaly_iso"] = df["iso_pred"].eq(-1)
    return df


def classify_type_and_severity(row: pd.Series, z_threshold: float = 3.0):
    """Return (anomaly_type, severity, pct_deviation) for a row."""
    resid = row.get("residual", np.nan)
    z = row.get("zscore", np.nan)
    yhat = row.get("yhat", np.nan)

    # Type by direction of residual
    if pd.isna(resid) or resid == 0:
        a_type = None
    elif resid < 0:
        a_type = "Liquidity Gap"          # actual below forecast (extra outflow / shortfall)
    else:
        a_type = "Unexpected Inflow"      # actual above forecast

    # How far from forecast in % terms (fallback if forecast nearly zero)
    denom = max(1.0, abs(yhat)) if pd.notna(yhat) else 1.0
    pct_dev = abs(resid) / denom  # e.g., 0.35 == 35%

    # Severity: primarily z-score; pct_dev as a secondary heuristic
    if pd.isna(z):
        severity = "info"
    else:
        if abs(z) >= 6 or pct_dev >= 0.75:
            severity = "high"
        elif abs(z) >= 4 or pct_dev >= 0.40:
            severity = "medium"
        elif abs(z) >= z_threshold or pct_dev >= 0.25:
            severity = "low"
        else:
            severity = "info"

    return a_type, severity, pct_dev


def main():
    hist_path = OUTPUT_FORECASTS_DIR / "history_with_residuals.csv"
    if not hist_path.exists():
        raise FileNotFoundError(
            f"Residuals not found: {hist_path}\n"
            "Run: python scripts/modeling/prophet_train.py first to generate forecasts & residuals."
        )

    # Load & sort
    df = pd.read_csv(hist_path, parse_dates=["ds"])
    df = df.sort_values("ds")

    # Add anomaly features
    df = rolling_zscore(df, window=14)
    df = isolation_forest_flag(df, contam=ISOF_CONTAM, random_state=RANDOM_STATE)
    df["anomaly_zscore"] = df["zscore"].abs() >= ZSCORE_THRESHOLD
    df["anomaly_flag"] = df["anomaly_zscore"] | df["anomaly_iso"]

    # Classify each row
    types, severities, pct_devs = [], [], []
    for _, row in df.iterrows():
        a_type, severity, pct_dev = classify_type_and_severity(row, ZSCORE_THRESHOLD)
        types.append(a_type)
        severities.append(severity)
        pct_devs.append(pct_dev)

    df["anomaly_type"] = types
    df["severity"] = severities
    df["pct_deviation"] = pct_devs

    # Build alert for the latest date
    latest = df.iloc[-1]
    alert = {
        "ds": str(latest["ds"]),
        "y": None if pd.isna(latest.get("y")) else float(latest["y"]),
        "yhat": float(latest["yhat"]),
        "yhat_lower": float(latest["yhat_lower"]),
        "yhat_upper": float(latest["yhat_upper"]),
        "residual": None if pd.isna(latest.get("residual")) else float(latest["residual"]),
        "zscore": None if pd.isna(latest.get("zscore")) else float(latest["zscore"]),
        "anomaly_zscore": bool(latest.get("anomaly_zscore", False)),
        "anomaly_iso": bool(latest.get("anomaly_iso", False)),
        "anomaly_type": latest.get("anomaly_type"),
        "severity": latest.get("severity"),
        "pct_deviation": (
            None if pd.isna(latest.get("pct_deviation")) else float(latest["pct_deviation"])
        ),
    }

    # Decide if we should trigger an alert
    trigger = bool(latest.get("anomaly_flag", False)) and alert["severity"] in {"low", "medium", "high"}
    alert["anomaly_trigger"] = trigger

    # Construct a clear subject/body for email (n8n)
    if trigger:
        dir_word = "below" if (alert["residual"] is not None and alert["residual"] < 0) else "above"
        sev = (alert["severity"] or "info").upper()
        a_type = alert["anomaly_type"] or "Anomaly"
        pct = f"{round(alert['pct_deviation']*100,1)}%" if alert["pct_deviation"] is not None else "n/a"
        subject = f"[{sev}] {a_type} on {alert['ds']}: actual {dir_word} forecast ({pct})"
        body = (
            f"Date: {alert['ds']}\n"
            f"Type: {a_type}\n"
            f"Severity: {sev}\n"
            f"Actual (y): {alert['y']}\n"
            f"Forecast (yhat): {alert['yhat']}\n"
            f"Residual: {alert['residual']}\n"
            f"Z-score: {alert['zscore']}\n"
            f"Pct Deviation vs Forecast: {pct}\n"
            f"Band: [{alert['yhat_lower']} — {alert['yhat_upper']}]\n"
        )
    else:
        subject = f"[INFO] No anomaly trigger on {alert['ds']}"
        body = f"Forecast within expected range on {alert['ds']}."

    alert["email_subject"] = subject
    alert["email_body"] = body

    # Save artifacts
    ensure_dir(OUTPUT_ALERTS_DIR)
    save_csv(df, OUTPUT_ALERTS_DIR / "anomaly_series.csv")
    save_json(alert, OUTPUT_ALERTS_DIR / "latest_alert.json")

    print(subject)
    print(body)
    print(f"Alert JSON → {OUTPUT_ALERTS_DIR / 'latest_alert.json'}")


if __name__ == "__main__":
    main()
