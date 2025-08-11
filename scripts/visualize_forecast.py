# visualize_forecast.py
# Path-safe visualizer for Prophet outputs: forecast + residuals (+ anomaly overlay if available)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- Resolve paths relative to project root (parent of this file if run from root) ---
here = Path(__file__).resolve()
project_root = here.parent  # assumes file lives at project root; change to parents[2] if moved into scripts/
fc_path = project_root / "outputs" / "forecasts" / "prophet_full_forecast.csv"
hist_path = project_root / "outputs" / "forecasts" / "history_with_residuals.csv"
anomaly_path = project_root / "outputs" / "alerts" / "anomaly_series.csv"  # optional overlay if present

# --- Load data ---
forecast_df = pd.read_csv(fc_path, parse_dates=["ds"])
hist_df = pd.read_csv(hist_path, parse_dates=["ds"])

# Optional: load anomalies for overlay
anoms = None
if anomaly_path.exists():
    anoms = pd.read_csv(anomaly_path, parse_dates=["ds"])

# -------- Plot 1: Forecast vs Actuals --------
plt.figure(figsize=(14, 6))
plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast (yhat)")
plt.fill_between(
    forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], alpha=0.2, label="Confidence band"
)
# Only historical dates have 'y' available (model appends future rows without 'y')
hist_merge = forecast_df.merge(hist_df[["ds", "y"]], on="ds", how="left")
plt.plot(hist_merge["ds"], hist_merge["y"], label="Actual (y)")
plt.title("Prophet Forecast vs Actuals")
plt.xlabel("Date")
plt.ylabel("Net Cash Flow (daily)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Plot 2: Residuals over time --------
plt.figure(figsize=(14, 4))
plt.plot(hist_df["ds"], hist_df["residual"], label="Residual (y - yhat)")
plt.axhline(0, linestyle="--")
plt.title("Residuals Over Time")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Plot 3: Residuals histogram --------
plt.figure(figsize=(8, 4))
hist_df["residual"].dropna().plot(kind="hist", bins=50)
plt.title("Residuals Distribution")
plt.xlabel("Residual")
plt.tight_layout()
plt.show()

# -------- Optional overlay: anomalies on forecast (if anomaly_series.csv exists) --------
if anoms is not None and "anomaly_flag" in anoms.columns:
    # Only mark rows that are flagged anomalies
    flagged = anoms[anoms["anomaly_flag"] == True]
    if not flagged.empty:
        plt.figure(figsize=(14, 6))
        plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Forecast (yhat)")
        plt.fill_between(
            forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], alpha=0.2, label="Confidence band"
        )
        plt.plot(hist_merge["ds"], hist_merge["y"], label="Actual (y)")
        # Overlay anomalies colored by type if present
        colors = None
        if "anomaly_type" in flagged.columns:
            # matplotlib will auto-assign colors per category, no manual palette set
            for atype, sub in flagged.groupby("anomaly_type"):
                plt.scatter(sub["ds"], sub["y"], label=f"Anomaly: {atype}", s=40)
        else:
            plt.scatter(flagged["ds"], flagged["y"], label="Anomaly", s=40)
        plt.title("Anomalies Overlay")
        plt.xlabel("Date")
        plt.ylabel("Net Cash Flow (daily)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No anomalies flagged yet; run the anomaly detection script first.")
