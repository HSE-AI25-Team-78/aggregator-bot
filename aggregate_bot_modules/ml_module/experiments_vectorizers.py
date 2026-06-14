from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from .paths import EXPERIMENTS_RESULTS_DIR, TRAIN_PATH, VAL_PATH, ensure_artifact_dirs

RESULTS_PATH = EXPERIMENTS_RESULTS_DIR / "vectorizers.csv"


def load_data():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Сначала запусти ml.dataset_split")

    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)

    text_col = "text_clean" if "text_clean" in df_train.columns else "text"

    le = LabelEncoder()
    y_train = le.fit_transform(df_train["topic"])
    y_val = le.transform(df_val["topic"])

    return df_train[text_col].astype(str), df_val[text_col].astype(str), y_train, y_val


def main():
    ensure_artifact_dirs()
    X_train_text, X_val_text, y_train, y_val = load_data()

    configs = []

    # 1) CountVectorizer + LogReg
    configs.append(
        (
            "CountVectorizer",
            CountVectorizer(ngram_range=(1, 1), min_df=5, max_features=20000),
            "LogisticRegression",
            LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        )
    )

    # 2) TF-IDF (1,1) + LogReg
    configs.append(
        (
            "TfidfVectorizer",
            TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_features=20000),
            "LogisticRegression",
            LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        )
    )

    # 3) TF-IDF (1,2) + LogReg
    configs.append(
        (
            "TfidfVectorizer",
            TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000),
            "LogisticRegression",
            LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        )
    )

    # 4) TF-IDF (1,2) + LinearSVC
    configs.append(
        (
            "TfidfVectorizer",
            TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000),
            "LinearSVC",
            LinearSVC(C=1.0),
        )
    )

    # 5) HashingVectorizer + LogReg (без vocab, просто для идеи)
    configs.append(
        (
            "HashingVectorizer",
            HashingVectorizer(n_features=20000, alternate_sign=False),
            "LogisticRegression",
            LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
        )
    )

    results = []

    for vec_name, vectorizer, model_name, model in configs:
        print(f"\n[*] {vec_name} + {model_name}")
        X_train = vectorizer.fit_transform(X_train_text) if hasattr(vectorizer, "fit_transform") else vectorizer.transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred, average="macro")
        acc = accuracy_score(y_val, y_pred)

        print(f"    F1-macro = {f1:.4f}, accuracy = {acc:.4f}")

        # попытаемся вытащить параметры векторизатора
        ngram_range = getattr(vectorizer, "ngram_range", None)
        min_df = getattr(vectorizer, "min_df", None)
        max_features = getattr(vectorizer, "max_features", getattr(vectorizer, "n_features", None))

        results.append(
            {
                "vectorizer": vec_name,
                "model": model_name,
                "ngram_range": str(ngram_range),
                "min_df": min_df,
                "max_features": max_features,
                "f1_macro": f1,
                "accuracy": acc,
            }
        )

    df_res = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    df_res.to_csv(RESULTS_PATH, index=False)
    print(f"\n[+] Результаты сравнения векторизаций сохранены: {RESULTS_PATH}")
    print(df_res.head())


if __name__ == "__main__":
    main()
