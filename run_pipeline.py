import os
import subprocess

# Run training
subprocess.run(["python", "scripts/modelling/prophet_train.py"], check=True)

# Run anomaly detection
subprocess.run(["python", "scripts/anomaly/detect.py"], check=True)
