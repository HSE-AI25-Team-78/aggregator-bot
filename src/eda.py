import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from .config import FIGURES


def top_words(df: pd.DataFrame, col="clean_text", n=30):
    texts = df[col].fillna("").astype(str)
    corpus = " ".join(texts).strip()
    if not corpus:
        return []
    words = corpus.split()
    return Counter(words).most_common(n)


def top_ngrams(df: pd.DataFrame, k=2, col="clean_text", n=20):
    texts = df[col].fillna("").astype(str)
    corpus = " ".join(texts).strip()
    if not corpus:
        return []
    words = corpus.split()
    grams = ngrams(words, k)
    return Counter(grams).most_common(n)


def save_wordcloud(df: pd.DataFrame, col="clean_text", fname="wordcloud.png"):
    texts = df[col].fillna("").astype(str)
    text = " ".join(texts).strip()
    if not text:
        # создаём заглушку, чтобы не падать на пустом тексте
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No text", ha="center", va="center")
        plt.axis("off")
        out = FIGURES / fname
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out
    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(wc)
    plt.axis("off")
    out = FIGURES / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out