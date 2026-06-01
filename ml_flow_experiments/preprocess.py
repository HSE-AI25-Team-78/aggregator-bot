from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "raw_posts_labeled.csv"
PROCESSED_PATH = DATA_DIR / "processed_posts.csv"


URL_RE = re.compile(r"http\S+|www\.\S+|t\.me/\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_LETTERS_RE = re.compile(r"[^a-zA-Zа-яА-ЯёЁ\s]")


def clean_text(text: str) -> str:
    """
    Простая очистка текста:
      - убираем ссылки, @упоминания, #хэштеги
      - приводим к нижнему регистру
      - выкидываем цифры, лишние символы и пунктуацию
      - схлопываем лишние пробелы
    """
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


def main(min_len: int = 10) -> None:
    """
    Загружаем сырой датасет → чистим тексты → фильтруем совсем короткие →
    сохраняем processed_posts.csv
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Не найден файл {RAW_PATH}")

    print(f"[*] Загружаем сырой датасет: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # Проверим, что нужные колонки есть
    if "text" not in df.columns:
        raise ValueError("В raw_posts_labeled.csv нет колонки 'text'")

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

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"[+] Обработанный датасет сохранён: {PROCESSED_PATH} (строк: {len(df)})")


if __name__ == "__main__":
    main(min_len=15)