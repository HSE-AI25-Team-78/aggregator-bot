from __future__ import annotations

import pickle

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from .paths import (
    DEPLOY_MODELS_DIR,
    LOGREG_MODELS_DIR,
    SERVICE_CONFIG_DIR,
    SERVICE_MODELS_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    ensure_artifact_dirs,
)


LOGREG_TFIDF_PATH = LOGREG_MODELS_DIR / "tfidf.joblib"
LOGREG_MODEL_PATH = LOGREG_MODELS_DIR / "logreg_model.joblib"
LOGREG_LE_PATH = LOGREG_MODELS_DIR / "label_encoder.joblib"

DEPLOY_TFIDF_PATH = DEPLOY_MODELS_DIR / "vectorizer.pkl"
DEPLOY_MODEL_PATH = DEPLOY_MODELS_DIR / "MultinomialNB.pkl"
DEPLOY_LE_PATH = DEPLOY_MODELS_DIR / "label_encoder.pkl"

SERVICE_TFIDF_PATH = SERVICE_CONFIG_DIR / "vectorizer.pkl"
SERVICE_LE_PATH = SERVICE_CONFIG_DIR / "label_encoder.pkl"
SERVICE_MODEL_PATH = SERVICE_MODELS_DIR / "MultinomialNB.pkl"


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


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
        sublinear_tf=True,
    )


def save_pickle(path, obj) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def main():
    ensure_artifact_dirs()
    df_train, df_val, df_test, text_col = load_splits()

    df_train_full = pd.concat([df_train, df_val], ignore_index=True)

    le = LabelEncoder()
    y_train_full = le.fit_transform(df_train_full["topic"])
    y_test = le.transform(df_test["topic"])

    tfidf = build_vectorizer()
    X_train_full = tfidf.fit_transform(df_train_full[text_col].astype(str))
    X_test = tfidf.transform(df_test[text_col].astype(str))

    logreg = LogisticRegression(
        max_iter=3000,
        C=2.0,
        n_jobs=-1,
    )
    deploy_nb = MultinomialNB()

    print("[*] Обучаем LogisticRegression (аналитическая модель)...")
    logreg.fit(X_train_full, y_train_full)
    logreg_pred = logreg.predict(X_test)
    logreg_f1 = f1_score(y_test, logreg_pred, average="macro")
    print(f"[+] LogisticRegression test F1_macro = {logreg_f1:.4f}")
    print(classification_report(y_test, logreg_pred, target_names=[str(label) for label in le.classes_]))

    print("\n[*] Обучаем MultinomialNB (deploy-модель для service/bot)...")
    deploy_nb.fit(X_train_full, y_train_full)
    nb_pred = deploy_nb.predict(X_test)
    nb_f1 = f1_score(y_test, nb_pred, average="macro")
    print(f"[+] MultinomialNB test F1_macro = {nb_f1:.4f}")
    print(classification_report(y_test, nb_pred, target_names=[str(label) for label in le.classes_]))

    joblib.dump(tfidf, LOGREG_TFIDF_PATH)
    joblib.dump(logreg, LOGREG_MODEL_PATH)
    joblib.dump(le, LOGREG_LE_PATH)

    save_pickle(DEPLOY_TFIDF_PATH, tfidf)
    save_pickle(DEPLOY_MODEL_PATH, deploy_nb)
    save_pickle(DEPLOY_LE_PATH, le)

    save_pickle(SERVICE_TFIDF_PATH, tfidf)
    save_pickle(SERVICE_LE_PATH, le)
    save_pickle(SERVICE_MODEL_PATH, deploy_nb)

    print(f"\n[+] LogReg TF-IDF сохранён в: {LOGREG_TFIDF_PATH}")
    print(f"[+] LogReg модель сохранена в: {LOGREG_MODEL_PATH}")
    print(f"[+] LogReg LabelEncoder сохранён в: {LOGREG_LE_PATH}")
    print(f"[+] Deploy TF-IDF сохранён в: {DEPLOY_TFIDF_PATH}")
    print(f"[+] Deploy NB сохранён в: {DEPLOY_MODEL_PATH}")
    print(f"[+] Deploy LabelEncoder сохранён в: {DEPLOY_LE_PATH}")
    print(f"[+] Service vectorizer экспортирован в: {SERVICE_TFIDF_PATH}")
    print(f"[+] Service label encoder экспортирован в: {SERVICE_LE_PATH}")
    print(f"[+] Service MultinomialNB экспортирован в: {SERVICE_MODEL_PATH}")


if __name__ == "__main__":
    main()
