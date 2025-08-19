# Intelligent Cash Flow Forecasting & Anomaly Alert System

A machine learning solution for forecasting daily AP/AR cash flow and detecting anomalies for Treasury and FP&A teams.

## Features

- Forecasting with Prophet (seasonality + AP/AR cadence features).

- Backtesting for stability checks.

- Anomaly detection using rolling Z-score.

- KPIs, charts, and tables in a Streamlit dashboard.

- JSON alert payload ready for n8n to email and log.

- GitHub-ready folder structure with config & reusable modules.

# 💰 Intelligent Cash Flow Forecasting & Anomaly Alert System

> *We turn raw AP/AR transactions into daily cash flow, forecast the next days with Prophet, measure how wrong we are each day, and alert finance leaders when the miss is statistically unusual—then we visualize all of it in a simple web app.*

---

## 📌 Overview
Finance teams often struggle to **predict cash flow** accurately and spot **critical anomalies** like sudden outflows or liquidity gaps.  
This project automates the process end-to-end:

1. **Data Pipeline** – Cleans & aggregates AP/AR transaction data to daily net cash flow.
2. **Forecasting** – Uses Facebook Prophet to forecast 30–90 days.
3. **Anomaly Detection** – Flags unusual deviations using rolling Z-score logic.
4. **Visualization** – An Interactive dashboard that visualizes anomalies interactively via Streamlit.
5. **Automation** – n8n workflow to send email alerts and log forecasts daily.
6. **n8n workflow automation** to:
  - Pull anomaly results daily from GitHub.
  - Send Gmail alerts when anomalies are detected.
  - Append both anomalies and daily “no anomaly” heartbeats into Google Sheets for auditing.

---

## 🛠️ Tech Stack
- **Python (Prophet, Pandas, NumPy)** – forecasting & anomaly detection
- **Streamlit** – dashboard interface
- **n8n** – workflow automation
- **Google Sheets & Gmail APIs** – logging and alerting

---

## Architecture

    A[CSV / Google Sheets] --> B[Prophet Forecast Model]
    B --> C[Anomaly Detection]
    C -->|Residuals + Z-score| D[n8n Automation]
    D -->|Email Alerts| E[Finance Team]
    B --> F[Streamlit Dashboard]
    C --> F
    D -->|Google Sheets Log| G[Daily Forecast Log]

 How to Run
1. Install Requirements
pip install -r requirements.txt

2. Prepare Data
python scripts/generate_full_dataset.py
python scripts/rebuild_and_qa.py

3. Train & Backtest
python scripts/modelling/backtest_and_features.py

4. Run Streamlit Dashboard
bash

streamlit run app/streamlit_app.py
5. (Next) Automate with n8n
Use the JSON alert in outputs/alerts/latest_alert_from_app.json.

Send via Gmail/Brevo.

Log forecasts to Google Sheets.

📊 Example Dashboard

📈 Key Metrics (Best Model)
Validation MAPE: ~8.0%

Test MAPE: ~10.7%

Rolling stability: verified with backtests
