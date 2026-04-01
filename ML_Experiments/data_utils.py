import re
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import Counter


def clean_text(text: str) -> str:
    """
    Простая очистка текста:
      - убираем ссылки, @упоминания, #хэштеги
      - приводим к нижнему регистру
      - выкидываем цифры, лишние символы и пунктуацию
      - схлопываем лишние пробелы
    """
    URL_RE = re.compile(r"http\S+|www\.\S+|t\.me/\S+")
    MENTION_RE = re.compile(r"@\w+")
    HASHTAG_RE = re.compile(r"#\w+")
    NON_LETTERS_RE = re.compile(r"[^a-zA-Zа-яА-ЯёЁ\s]")

    if not isinstance(text, str):
        return ""

    # нижний регистр
    t = text.lower()

    # ссылки, упоминания, хэштеги
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)

    # убираем всё, кроме букв и пробелов
    t = NON_LETTERS_RE.sub(" ", t)

    # схлопываем пробелы
    t = re.sub(r"\s+", " ", t).strip()

    return t


def preprocess_data(df: pd.DataFrame, min_len: int = 10):
    """
    Загружаем сырой датасет → чистим тексты → фильтруем совсем короткие →
    сохраняем processed_posts.csv
    """

    # Проверим, что нужные колонки есть
    if "text" not in df.columns:
        raise ValueError("В '{file_path}' нет колонки 'text'")

    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    print(f"[*] Строк после удаления пустых текстов: {len(df)}")

    print("[*] Очищаем тексты...")
    df["text_clean"] = df["text"].astype(str).apply(clean_text)

    # Фильтруем слишком короткие очищенные тексты
    before_len = len(df)
    df = df[df["text_clean"].str.len() >= min_len]
    print(f"[*] Убрали слишком короткие тексты: {before_len} → {len(df)}")

    df = df.drop_duplicates(subset=["text_clean", "topic"], keep="first")

    # Сохраняем только нужные колонки
    keep_cols = []
    for col in ["id", "date", "channel", "topic", "text", "text_clean"]:
        if col in df.columns:
            keep_cols.append(col)
    df = df[keep_cols]
    return df


def calc_metrics(y_true, y_pred):
    metric_macro = lambda f, **kwargs: f(y_true, y_pred, average="macro", **kwargs)
    return {
        "f1_macro": metric_macro(f1_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": metric_macro(precision_score, zero_division=0),
        "recall_macro": metric_macro(recall_score, zero_division=0)
    }


def extract_ngrams(text, n_range=(3, 8)):
        words = text.split()
        ngrams = []
        for n in range(n_range[0], min(n_range[1], len(words)) + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.append(ngram)
        return ngrams

def find_templates(channel_texts: list, threshold=0.5):
    all_ngrams = []
    for text in channel_texts:
        all_ngrams.extend(extract_ngrams(text))

    ngram_counts = Counter(all_ngrams)

    template_phrases = []

    for phrase, count in ngram_counts.items():
        if count / len(channel_texts) > threshold and len(phrase.split()) >= 3:
            template_phrases.append(phrase)

    return template_phrases


def remove_templates(text, templates):
    clean_text = text
    templates_sorted = sorted(templates, key=len, reverse=True)
    for template in templates_sorted:
        clean_text = clean_text.replace(template, '').strip()
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def clean_templates_in_df(df_origin):
    df = df_origin.copy()
    for channel in df['channel'].unique():
        templates = find_templates(df[df['channel'] == channel]['text_clean'].tolist())
        mask = df['channel'] == channel
        df.loc[mask, 'text_clean2'] = df.loc[mask, 'text_clean'].apply(
            lambda x: remove_templates(x, templates)
        )
    return df
