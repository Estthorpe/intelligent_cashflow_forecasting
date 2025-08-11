# utils/io_utils.py
from pathlib import Path
import pandas as pd

def ensure_dir(p: Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)

def _resolve_path(p) -> Path:
    p = Path(p)
    if not p.is_absolute():
        # Make it absolute relative to the project root (utils/ is under ROOT)
        root = Path(__file__).resolve().parents[1]
        p = root / p
    return p

def read_processed_df(path: Path) -> pd.DataFrame:
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at:\n  {path}\n"
            f"Tip: run `python scripts/rebuild_and_qa.py` to generate it."
        )
    df = pd.read_csv(path, low_memory=False)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
    return df

def save_csv(df: pd.DataFrame, path: Path):
    path = _resolve_path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
