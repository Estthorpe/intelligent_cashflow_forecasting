# scripts/automation/run_daily.py
import sys, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Path-safe import of config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import OUTPUT_FORECASTS_DIR, OUTPUT_ALERTS_DIR, ZSCORE_THRESHOLD

HIST_PATH = OUTPUT_FORECASTS_DIR / "history_with_residuals.csv"
ALERT_JSON = OUTPUT_ALERTS_DIR / "latest_alert.json"

def rolling_zscore(series: pd.Series, window: int = 14) -> pd.Series:
    s = series.astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    return (s - mu) / sd

def classify(residual, z, yhat, z_thresh):
    if pd.isna(residual) or residual == 0:
        a_type = None
    elif residual < 0:
        a_type = "Liquidity Gap"
    else:
        a_type = "Unexpected Inflow"
    denom = max(1.0, abs(yhat)) if pd.notna(yhat) else 1.0
    pct_dev = abs(residual) / denom
    if pd.isna(z): severity = "info"
    else:
        if abs(z) >= 6 or pct_dev >= 0.75:   severity = "high"
        elif abs(z) >= 4 or pct_dev >= 0.40: severity = "medium"
        elif abs(z) >= z_thresh or pct_dev >= 0.25: severity = "low"
        else: severity = "info"
    return a_type, severity, pct_dev

def json_default(o):
    if isinstance(o, (pd.Timestamp, )):
        return o.isoformat()
    if isinstance(o, (np.integer, )):
        return int(o)
    if isinstance(o, (np.floating, )):
        return float(o)
    if isinstance(o, (np.ndarray, )):
        return o.tolist()
    return str(o)

def main():
    if not HIST_PATH.exists():
        raise FileNotFoundError(f"Missing {HIST_PATH}. Run training/backtest first.")
    df = pd.read_csv(HIST_PATH, parse_dates=["ds"])
    if "residual" not in df.columns:
        df["residual"] = df["y"] - df["yhat"]
    df["zscore"] = rolling_zscore(df["residual"], window=14)
    latest = df.dropna(subset=["yhat"]).iloc[-1]  # last available forecasted day
    a_type, severity, pct = classify(latest["residual"], latest["zscore"], latest["yhat"], float(ZSCORE_THRESHOLD))
    trigger = bool((abs(latest["zscore"]) >= float(ZSCORE_THRESHOLD)) and (severity in ["low","medium","high"]))
    dir_word = "below" if (latest["residual"] < 0) else "above"
    pct_str = f"{round(pct*100,1)}%"
    subject = f"[{severity.upper()}] {a_type or 'No anomaly'} on {latest['ds'].date()}: actual {dir_word} forecast ({pct_str})" if trigger \
              else f"[INFO] No anomaly trigger on {latest['ds'].date()}"
    body = (
        f"Date: {latest['ds'].date()}\n"
        f"Type: {a_type}\n"
        f"Severity: {severity.upper()}\n"
        f"Actual (y): {latest.get('y')}\n"
        f"Forecast (yhat): {latest.get('yhat')}\n"
        f"Residual: {latest.get('residual')}\n"
        f"Z-score: {round(latest['zscore'],2) if pd.notna(latest['zscore']) else 'n/a'}\n"
        f"Pct Deviation vs Forecast: {pct_str}\n"
        f"Band: [{latest.get('yhat_lower')} — {latest.get('yhat_upper')}]"
    )
    payload = {
        "anomaly_trigger": trigger,
        "email_subject": subject,
        "email_body": body,
        "latest": {
            "ds": latest["ds"],
            "y": latest.get("y"),
            "yhat": latest.get("yhat"),
            "residual": latest.get("residual"),
            "zscore": latest.get("zscore"),
            "severity": severity,
            "anomaly_type": a_type,
            "pct_deviation": pct
        }
    }
    ALERT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ALERT_JSON.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    print(f"✅ Wrote alert → {ALERT_JSON}")

if __name__ == "__main__":
    main()
