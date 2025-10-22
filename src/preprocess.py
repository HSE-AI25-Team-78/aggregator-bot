import re
import pandas as pd

URL_RE = re.compile(r"(http|https)://\S+")
NON_LETTERS_RE = re.compile(r"[^a-zA-Zа-яА-ЯёЁ\s]")

def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = URL_RE.sub("", s)
    s = NON_LETTERS_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def apply_clean(df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df[text_col].astype(str).apply(clean_text)
    df["len_words"] = df["clean_text"].apply(lambda x: len(x.split()))
    return df
