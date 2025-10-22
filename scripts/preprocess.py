import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
from src.preprocess import apply_clean
from src.config import PROCESSED

p = argparse.ArgumentParser()
p.add_argument("--infile", required=True)
p.add_argument("--outfile", default=None)
args = p.parse_args()

df = pd.read_csv(args.infile)
df = apply_clean(df, text_col="text")
out = Path(args.outfile) if args.outfile else (PROCESSED / "cleaned.csv")
df.to_csv(out, index=False)
print(f"saved: {out} ({len(df)} rows)")
