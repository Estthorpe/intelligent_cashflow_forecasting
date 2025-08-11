# scripts/automation/replay_init.py
import json
from pathlib import Path

STATE = Path("outputs/alerts/replay_state.json")

def main():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pointer_iso": None,           # start at the first date in history
        "step_days": 1,                # advance by 1 day per run
        "inject_demo_anomalies": False,# True = force a visible anomaly every N days
        "inject_every_n_days": 7,      # used only if inject_demo_anomalies=True
        "step_idx": 0
    }
    STATE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Replay state initialized → {STATE}")

if __name__ == "__main__":
    main()
