# scripts/automation/replay_step.py
import sys, json
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIST = PROJECT_ROOT / "outputs" / "forecasts" / "history_with_residuals.csv"
STATE = PROJECT_ROOT / "outputs" / "alerts" / "replay_state.json"
ALERT_JSON = PROJECT_ROOT / "outputs" / "alerts" / "latest_alert.json"
DEMO_LOG = PROJECT_ROOT / "outputs" / "alerts" / "demo_alert_log.csv"

# Basic thresholds
ZSCORE_THRESHOLD = 3.0
WINDOW = 14

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

def rolling_zscore(series: pd.Series, window: int = 14) -> pd.Series:
    s = series.astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    return (s - mu) / sd

def classify(residual, z, yhat):
    if pd.isna(residual) or residual == 0:
        a_type = None
    elif residual < 0:
        a_type = "Liquidity Gap"
    else:
        a_type = "Unexpected Inflow"
    denom = max(1.0, abs(yhat)) if pd.notna(yhat) else 1.0
    pct_dev = abs(residual) / denom
    if pd.isna(z):
        severity = "info"
    else:
        if abs(z) >= 6 or pct_dev >= 0.75:   severity = "high"
        elif abs(z) >= 4 or pct_dev >= 0.40: severity = "medium"
        elif abs(z) >= ZSCORE_THRESHOLD or pct_dev >= 0.25: severity = "low"
        else: severity = "info"
    return a_type, severity, pct_dev

def maybe_inject_demo_anomaly(row, idx, inject, every_n):
    """Optionally force an anomaly on every Nth 'day' for demos."""
    if not inject or every_n <= 0:
        return row
    if idx % every_n == 0:
        bump = 0.5 * (abs(row["yhat"]) if pd.notna(row["yhat"]) else 1.0)
        row["y"] = (row.get("y") or 0.0) + bump * (1 if (idx//every_n) % 2 == 0 else -1)
        row["residual"] = row["y"] - row["yhat"]
    return row

def main():
    if not HIST.exists():
        raise FileNotFoundError(f"Missing {HIST}. Run training/backtest first to create it.")

    df = pd.read_csv(HIST, parse_dates=["ds"]).sort_values("ds")
    if "residual" not in df.columns:
        df["residual"] = df["y"] - df["yhat"]

    # load or init state
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
    else:
        state = {"pointer_iso": None, "step_days": 1, "inject_demo_anomalies": False, "inject_every_n_days": 7, "step_idx": 0}

    ds_min, ds_max = df["ds"].min(), df["ds"].max()
    pointer = pd.to_datetime(state["pointer_iso"]) if state["pointer_iso"] else ds_min
    step_idx = int(state.get("step_idx", 0)) + 1

    # advance pointer
    next_pointer = pointer + timedelta(days=int(state.get("step_days", 1)))
    if next_pointer > ds_max:
        next_pointer = ds_min
        step_idx = 1  # restart loop at beginning

    # slice up to "today"
    dfc = df[df["ds"] <= next_pointer].copy()
    # inject demo anomaly on last row if toggled
    dfc.iloc[-1] = maybe_inject_demo_anomaly(
        dfc.iloc[-1].copy(),
        idx=step_idx,
        inject=bool(state.get("inject_demo_anomalies", False)),
        every_n=int(state.get("inject_every_n_days", 7))
    )

    # recompute z-scores on this slice
    dfc["zscore"] = rolling_zscore(dfc["residual"], window=WINDOW)

    latest = dfc.iloc[-1]
    a_type, severity, pct = classify(latest["residual"], latest["zscore"], latest["yhat"])
    trigger = bool((abs(latest["zscore"]) >= ZSCORE_THRESHOLD) and (severity in ["low","medium","high"]))
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
    print(f"✅ Wrote alert for {latest['ds'].date()} → {ALERT_JSON}")

    # append demo log
    row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "ds": latest["ds"],
        "anomaly_trigger": trigger,
        "severity": severity,
        "anomaly_type": a_type,
        "subject": subject
    }
    if DEMO_LOG.exists():
        pd.DataFrame([row]).to_csv(DEMO_LOG, mode="a", header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(DEMO_LOG, index=False)

    # persist state
    state["pointer_iso"] = str(next_pointer)
    state["step_idx"] = step_idx
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
