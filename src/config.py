from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "reports" / "figures"

# Ensure folders exist safely
for p in [RAW, PROCESSED, FIGURES]:
    if p.exists():
        if not p.is_dir():
            raise RuntimeError(f"Path exists but is not a directory: {p}")
    else:
        p.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
SESSION_NAME = os.getenv("SESSION_NAME", "tg_session")
