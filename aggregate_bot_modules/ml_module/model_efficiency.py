from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from .paths import EXPERIMENTS_RESULTS_DIR, TRAIN_PATH, VAL_PATH, ensure_artifact_dirs

RESULTS_PATH = EXPERIMENTS_RESULTS_DIR / "model_efficiency.csv"


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

    # Один общий TF-IDF, чтобы фича-пространство было одинаковым
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000)
    X_train = tfidf.fit_transform(X_train_text)
    X_val = tfidf.transform(X_val_text)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)),
        ("LinearSVC", LinearSVC(C=1.0)),
        ("MultinomialNB", MultinomialNB()),
        ("KNN_k5", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]

    # возьмём подмножество для честного измерения инференса
    n_infer = min(1000, X_val.shape[0])
    idx = np.random.choice(X_val.shape[0], size=n_infer, replace=False)
    X_val_small = X_val[idx]
    y_val_small = y_val[idx]

    results = []

    for name, model in models:
        print(f"\n[*] {name}: измеряем train / inference")

        # train time
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # inference time на n_infer примеров
        t0 = time.perf_counter()
        y_pred = model.predict(X_val_small)
        infer_time = time.perf_counter() - t0

        f1 = f1_score(y_val_small, y_pred, average="macro")

        per_sample = infer_time / n_infer

        print(f"    train_time = {train_time:.3f} s")
        print(f"    infer_time_total = {infer_time:.4f} s ({per_sample*1000:.4f} ms / sample)")
        print(f"    F1_macro (на подмножестве) = {f1:.4f}")

        results.append(
            {
                "model": name,
                "train_time_sec": train_time,
                "infer_time_sec_total": infer_time,
                "infer_time_ms_per_sample": per_sample * 1000,
                "f1_macro_subset": f1,
                "n_infer_samples": int(n_infer),
            }
        )

    df_res = pd.DataFrame(results).sort_values("infer_time_ms_per_sample")
    df_res.to_csv(RESULTS_PATH, index=False)

    print(f"\n[+] Результаты эффективности моделей сохранены: {RESULTS_PATH}")
    print(df_res)


if __name__ == "__main__":
    main()
