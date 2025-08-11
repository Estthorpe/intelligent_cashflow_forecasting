import os
import pandas as pd
import numpy as np

# Paths
residuals_path = os.path.join("outputs", "forecasts", "history_with_residuals.csv")

# Load residuals
df = pd.read_csv(residuals_path, parse_dates=['ds'])

# Z-score anomaly detection
threshold = 3  # >3 standard deviations = anomaly
df['zscore'] = (df['residual'] - df['residual'].mean()) / df['residual'].std()
df['anomaly_flag'] = df['zscore'].abs() > threshold

# Save anomalies
anomalies = df[df['anomaly_flag']]
anomaly_path = os.path.join("outputs", "anomalies", "detected_anomalies.csv")
os.makedirs(os.path.dirname(anomaly_path), exist_ok=True)
anomalies.to_csv(anomaly_path, index=False)

# Print summary
print(f"✅ Anomaly detection complete. {len(anomalies)} anomalies found.")
print(f"Results saved to: {anomaly_path}")

# Optional: print a few anomalies
print(anomalies[['ds', 'y', 'yhat', 'residual', 'zscore']].head())
