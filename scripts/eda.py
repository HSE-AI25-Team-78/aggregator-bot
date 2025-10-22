import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
from src.eda import top_words, top_ngrams, save_wordcloud
from src.config import FIGURES

p = argparse.ArgumentParser()
p.add_argument("--infile", required=True)
p.add_argument("--out_dir", default=str(FIGURES))
p.add_argument("--n", type=int, default=20)
args = p.parse_args()

df = pd.read_csv(args.infile)
print("top words:", top_words(df, n=args.n))
print("top bigrams:", top_ngrams(df, k=2, n=args.n))
png = save_wordcloud(df, fname="wordcloud.png")
print(f"wordcloud: {png}")
