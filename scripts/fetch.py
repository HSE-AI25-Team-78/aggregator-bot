import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import asyncio
from src.fetch_messages import fetch_channel_messages
from src.config import RAW

p = argparse.ArgumentParser()
p.add_argument("--channel", required=True, help="username/url/id канала")
p.add_argument("--limit", type=int, default=1000)
p.add_argument("--out", default=None, help="путь к csv")
args = p.parse_args()

async def main():
    df = await fetch_channel_messages(args.channel, args.limit)
    out = Path(args.out) if args.out else RAW / f"{str(args.channel).strip('@').replace('/', '_')}.csv"
    df.to_csv(out, index=False)
    print(f"saved: {out} ({len(df)} rows)")

asyncio.run(main())
