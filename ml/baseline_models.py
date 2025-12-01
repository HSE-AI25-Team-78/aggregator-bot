from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder


# Пути к данным
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_DIR = DATA_DIR / "splits"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VAL_PATH = SPLIT_DIR / "val.csv"

RESULTS_DIR = DATA_DIR / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = RESULTS_DIR / "baseline_results.csv"


def load_data():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Не найдены train/val в data/splits. Сначала запусти ml.dataset_split.")

    print(f"[*] Загружаем train: {TRAIN_PATH}")
    df_train = pd.read_csv(TRAIN_PATH)

    print(f"[*] Загружаем val:   {VAL_PATH}")
    df_val = pd.read_csv(VAL_PATH)

    if "topic" not in df_train.columns:
        raise ValueError("В train.csv нет 'topic'")

    # Используем cleaned текст если есть
    if "text_clean" in df_train.columns:
        text_col = "text_clean"
    else:
        print("[!] Нет text_clean — используем сырой text")
        text_col = "text"

    return df_train, df_val, text_col


def vectorize_text(df_train, df_val, text_col: str):
    print(f"[*] TF-IDF vectorizer (1-2 граммы) по колонке '{text_col}'")

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=20000,
    )

    X_train = tfidf.fit_transform(df_train[text_col].astype(str))
    X_val = tfidf.transform(df_val[text_col].astype(str))

    print(f"[*] Размерность признаков: {X_train.shape[1]}")

    return X_train, X_val, tfidf


def encode_labels(df_train, df_val):
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["topic"])
    y_val = le.transform(df_val["topic"])

    print("[*] Найденные классы:", list(le.classes_))
    return y_train, y_val, le


def evaluate_model(name: str, model, X_train, y_train, X_val, y_val):
    print(f"\n[***] Обучаем модель: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    result = {
        "model": name,
        "f1_macro": f1_score(y_val, y_pred, average="macro"),
        "accuracy": accuracy_score(y_val, y_pred),
        "precision_macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
    }

    print(result)
    return result


def main():
    df_train, df_val, text_col = load_data()

    X_train, X_val, tfidf = vectorize_text(df_train, df_val, text_col)

    y_train, y_val, le = encode_labels(df_train, df_val)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)),
        ("LinearSVC", LinearSVC(C=1.0)),
        ("MultinomialNB", MultinomialNB()),
        ("KNN_k5", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]

    results = []

    for name, model in models:
        res = evaluate_model(name, model, X_train, y_train, X_val, y_val)
        results.append(res)

    res_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)

    res_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n[+] Результаты сохранены: {RESULTS_PATH}")

    print("\n[+] Таблица результатов:")
    print(res_df)


if __name__ == "__main__":
    main()