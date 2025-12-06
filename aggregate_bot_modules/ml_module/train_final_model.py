from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLIT_DIR = DATA_DIR / "splits"

TRAIN_PATH = SPLIT_DIR / "train.csv"
VAL_PATH = SPLIT_DIR / "val.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

MODELS_DIR = DATA_DIR / "models" / "logreg_tfidf"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_PATH = MODELS_DIR / "tfidf.joblib"
MODEL_PATH = MODELS_DIR / "logreg_model.joblib"
LE_PATH = MODELS_DIR / "label_encoder.joblib"


def load_splits():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Нет train/val/test. Сначала запусти ml.dataset_split.")

    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    text_col = "text_clean" if "text_clean" in df_train.columns else "text"

    if "topic" not in df_train.columns:
        raise ValueError("В train.csv нет 'topic'")

    return df_train, df_val, df_test, text_col


def main():
    df_train, df_val, df_test, text_col = load_splits()

    # Объединяем train+val для финального обучения
    df_train_full = pd.concat([df_train, df_val], ignore_index=True)

    le = LabelEncoder()
    y_train_full = le.fit_transform(df_train_full["topic"])
    y_test = le.transform(df_test["topic"])

    # ВАЖНО: сюда можно подставить лучшие параметры из experiments_logreg
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=20000,
    )

    X_train_full = tfidf.fit_transform(df_train_full[text_col].astype(str))
    X_test = tfidf.transform(df_test[text_col].astype(str))

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        n_jobs=-1,
    )

    print("[*] Обучаем финальную модель на train+val...")
    model.fit(X_train_full, y_train_full)

    print("[*] Оцениваем на test...")
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"[+] Test F1_macro = {f1:.4f}\n")

    print("[+] Classification report:")
    print(classification_report(y_test, y_pred, target_names=list(le.classes_)))

    # Сохраняем артефакты
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LE_PATH)

    print(f"\n[+] TF-IDF сохранён в: {TFIDF_PATH}")
    print(f"[+] Модель сохранена в: {MODEL_PATH}")
    print(f"[+] LabelEncoder сохранён в: {LE_PATH}")


if __name__ == "__main__":
    main()