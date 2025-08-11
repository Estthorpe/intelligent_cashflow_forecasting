# app/streamlit_app.py
# Streamlit dashboard for Intelligent Cash Flow Forecasting & Anomaly Alerts
from __future__ import annotations

import sys, json, pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---- Path-safe imports ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    OUTPUT_FORECASTS_DIR, OUTPUT_ALERTS_DIR, ZSCORE_THRESHOLD
)

# Derived artifact paths
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"
BACKTEST_DIR  = PROJECT_ROOT / "outputs" / "backtests"
HIST_PATH     = OUTPUT_FORECASTS_DIR / "history_with_residuals.csv"
RUN_SUMMARY   = OUTPUT_FORECASTS_DIR / "run_summary.json"
BEST_MODEL_PKL= MODELS_DIR / "best_prophet.pkl"
STABILITY_CSV = BACKTEST_DIR / "stability_table.csv"
APP_ALERT_JSON= OUTPUT_ALERTS_DIR / "latest_alert_from_app.json"

# ======================================================
# JSON helpers (Fix #1)
# ======================================================
def _json_default(o):
    """Safe JSON encoder for pandas/NumPy objects."""
    if isinstance(o, (pd.Timestamp, )):
        return o.isoformat()
    if isinstance(o, (np.integer, )):
        return int(o)
    if isinstance(o, (np.floating, )):
        return float(o)
    if isinstance(o, (np.ndarray, )):
        return o.tolist()
    return str(o)

def save_latest_alert(payload: Dict[str, Any], path: Path):
    """Save alert payload to JSON (Fix #2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

# ======================================================
# Data loaders
# ======================================================
@st.cache_data
def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ds"])
    need = {"ds","yhat","yhat_lower","yhat_upper"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{path.name} missing required columns: {miss}")
    # y/residual may be NaN on future rows; compute residual if missing
    if "residual" not in df.columns:
        df["residual"] = df["y"] - df["yhat"]
    return df.sort_values("ds")

@st.cache_data
def load_run_summary(path: Path) -> dict | None:
    return None if not path.exists() else json.loads(path.read_text(encoding="utf-8"))

@st.cache_data
def load_stability(path: Path) -> pd.DataFrame | None:
    return None if not path.exists() else pd.read_csv(path)

@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# ======================================================
# Anomaly utilities
# ======================================================
def rolling_zscore(series: pd.Series, window: int = 14) -> pd.Series:
    s = series.astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0, np.nan)
    return (s - mu) / sd

def classify(row, z_thresh: float) -> tuple[str|None, str, float]:
    resid = row.get("residual", np.nan)
    z = row.get("zscore", np.nan)
    yhat = row.get("yhat", np.nan)
    # direction → type
    if pd.isna(resid) or resid == 0:
        a_type = None
    elif resid < 0:
        a_type = "Liquidity Gap"
    else:
        a_type = "Unexpected Inflow"
    # % deviation vs forecast magnitude
    denom = max(1.0, abs(yhat)) if pd.notna(yhat) else 1.0
    pct_dev = abs(resid) / denom
    # severity by z + pct deviation
    if pd.isna(z):
        severity = "info"
    else:
        if abs(z) >= 6 or pct_dev >= 0.75:   severity = "high"
        elif abs(z) >= 4 or pct_dev >= 0.40: severity = "medium"
        elif abs(z) >= z_thresh or pct_dev >= 0.25: severity = "low"
        else: severity = "info"
    return a_type, severity, pct_dev

def build_anomalies(hist: pd.DataFrame, z_thresh: float, window: int) -> pd.DataFrame:
    df = hist.copy()
    df["zscore"] = rolling_zscore(df["residual"], window=window)
    df["anomaly_zscore"] = df["zscore"].abs() >= z_thresh
    a_type, severity, pct = [], [], []
    for _, r in df.iterrows():
        t, s, p = classify(r, z_thresh)
        a_type.append(t); severity.append(s); pct.append(p)
    df["anomaly_type"] = a_type
    df["severity"] = severity
    df["pct_deviation"] = pct
    df["anomaly_flag"] = df["anomaly_zscore"] & df["severity"].isin(["low","medium","high"])
    return df

def build_status_payload(df: pd.DataFrame) -> dict:
    latest = df.dropna(subset=["yhat"]).iloc[-1]
    trigger = bool(latest.get("anomaly_flag", False))
    if trigger:
        dir_word = "below" if (latest["residual"] < 0) else "above"
        pct = f"{round(latest['pct_deviation']*100,1)}%"
        subject = f"[{latest['severity'].upper()}] {latest['anomaly_type']} on {latest['ds'].date()}: actual {dir_word} forecast ({pct})"
        body = (
            f"Date: {latest['ds'].date()}\n"
            f"Type: {latest['anomaly_type']}\n"
            f"Severity: {latest['severity'].upper()}\n"
            f"Actual (y): {latest.get('y')}\n"
            f"Forecast (yhat): {latest.get('yhat')}\n"
            f"Residual: {latest.get('residual')}\n"
            f"Z-score: {round(latest['zscore'],2) if pd.notna(latest['zscore']) else 'n/a'}\n"
            f"Pct Deviation vs Forecast: {pct}\n"
            f"Band: [{latest.get('yhat_lower')} — {latest.get('yhat_upper')}]"
        )
    else:
        subject = f"[INFO] No anomaly trigger on {latest['ds'].date()}"
        body = f"Forecast within expected range on {latest['ds'].date()}."
    return {"trigger": trigger, "subject": subject, "body": body, "latest": latest.to_dict()}

# ======================================================
# UI
# ======================================================
st.set_page_config(page_title="Cash Flow Forecast & Anomalies", layout="wide")
st.title("💸 Intelligent Cash Flow Forecasting & Anomaly Alerts")
st.caption("CFO • Treasury • FP&A — Daily forecast, residual anomalies, and alert readiness")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    z = st.number_input("Z‑score threshold", min_value=1.0, max_value=6.0,
                        value=float(ZSCORE_THRESHOLD), step=0.5)
    window = st.slider("Rolling window (days)", min_value=7, max_value=60, value=14, step=1)
    lookback_days = st.slider("Chart lookback (days)", min_value=30, max_value=720, value=180, step=30)
    st.divider()
    st.markdown("**Artifacts**")
    st.write(f"History: `{HIST_PATH.name}`" if HIST_PATH.exists() else "_Missing_")
    st.write(f"Model: `{BEST_MODEL_PKL.name}`" if BEST_MODEL_PKL.exists() else "_Missing_")
    st.write(f"Run summary: `{RUN_SUMMARY.name}`" if RUN_SUMMARY.exists() else "_Missing_")

# Load data
if not HIST_PATH.exists():
    st.error("history_with_residuals.csv not found. Run training/backtest to generate artifacts.")
    st.stop()

history = load_history(HIST_PATH)
summary = load_run_summary(RUN_SUMMARY)
_ = load_model(BEST_MODEL_PKL)  # optional; charts don't need the model object

# Build anomalies
df = build_anomalies(history, z_thresh=float(z), window=int(window))

# KPIs
cols = st.columns(4)
with cols[0]:
    st.metric("Latest Date", value=str(pd.to_datetime(df["ds"]).max().date()))
with cols[1]:
    st.metric("Latest Forecast (yhat)", value=f"{df.iloc[-1]['yhat']:,.0f}")
with cols[2]:
    # Count of anomalies in the last 30 days
    recent_cut = pd.to_datetime(df["ds"]).max() - pd.Timedelta(days=30)
    st.metric("Anomalies (30d)", int(df[(df["ds"]>=recent_cut) & (df["anomaly_flag"])].shape[0]))
with cols[3]:
    st.metric("Z-threshold", value=str(z))

# Save an email-ready payload for n8n
status = build_status_payload(df)
st.info(status["subject"])
with st.expander("Email preview / alert payload"):
    st.code(status["body"])
# Persist the alert JSON using fixed helper
save_latest_alert({
    "anomaly_trigger": status["trigger"],
    "email_subject": status["subject"],
    "email_body": status["body"],
    "latest": status["latest"]
}, APP_ALERT_JSON)
st.caption(f"Saved alert payload → {APP_ALERT_JSON}")

# Lookback filter
cutoff = pd.to_datetime(df["ds"]).max() - pd.Timedelta(days=int(lookback_days))
dfc = df[df["ds"] >= cutoff].copy()

# Chart 1: Forecast vs Actuals with anomalies
st.subheader("Forecast vs Actuals")
fig1, ax1 = plt.subplots(figsize=(11,4))
ax1.plot(dfc["ds"], dfc["yhat"], label="Forecast (yhat)")
ax1.fill_between(dfc["ds"], dfc["yhat_lower"], dfc["yhat_upper"], alpha=0.2, label="Confidence Band")
ax1.plot(dfc["ds"], dfc["y"], label="Actual (y)")
flagged = dfc[dfc["anomaly_flag"]]
if not flagged.empty:
    for atype, sub in flagged.groupby("anomaly_type"):
        ax1.scatter(sub["ds"], sub["y"], s=25, label=f"Anomaly: {atype}")
ax1.set_xlabel("Date"); ax1.set_ylabel("Amount")
ax1.legend(); ax1.grid(True); ax1.set_title("Daily Net Cash — Forecast vs Actuals")
st.pyplot(fig1)

# Chart 2: Residuals
st.subheader("Residuals Over Time")
fig2, ax2 = plt.subplots(figsize=(11,3))
ax2.plot(dfc["ds"], dfc["residual"], label="Residuals")
ax2.axhline(0, color="black", linewidth=1)
ax2.legend(); ax2.grid(True)
st.pyplot(fig2)

# Chart 3: Z-score
st.subheader("Rolling Z-score of Residuals")
fig3, ax3 = plt.subplots(figsize=(11,2.8))
ax3.plot(dfc["ds"], dfc["zscore"], label="Rolling Z")
ax3.axhline(float(z), color="red", linestyle="--", label="Threshold")
ax3.axhline(-float(z), color="red", linestyle="--")
ax3.legend(); ax3.grid(True)
st.pyplot(fig3)

# Recent anomalies table + download
st.subheader("Recent Anomalies")
recent = dfc[dfc["anomaly_flag"]].sort_values("ds", ascending=False).head(50)[
    ["ds","y","yhat","residual","zscore","anomaly_type","severity","pct_deviation"]
]
st.dataframe(recent, use_container_width=True)
st.download_button("Download recent anomalies (CSV)", data=recent.to_csv(index=False),
                   file_name="recent_anomalies.csv")

# Stability table (from backtests)
st.subheader("Model Stability (Backtests)")
stab = load_stability(STABILITY_CSV)
if stab is None or stab.empty:
    st.caption("Run backtesting to populate stability_table.csv")
else:
    st.dataframe(stab, use_container_width=True)
    st.download_button("Download stability table (CSV)", data=stab.to_csv(index=False),
                       file_name="stability_table.csv")

# Footer
st.caption("© Intelligent Cash Flow Forecasting — Streamlit Demo")
