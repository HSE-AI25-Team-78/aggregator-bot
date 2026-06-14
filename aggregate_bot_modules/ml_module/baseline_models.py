from __future__ import annotations

import json

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
from .mlflow_utils import log_artifact_if_exists, log_metrics, log_params, start_mlflow_run
from .paths import BASELINE_RESULTS_DIR, PROJECT_ROOT, TRAIN_PATH, VAL_PATH, ensure_artifact_dirs

RESULTS_PATH = BASELINE_RESULTS_DIR / "baseline_results.csv"
SUMMARY_PATH = BASELINE_RESULTS_DIR / "baseline_summary.json"
VECTORIZER_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_features": 20000,
}


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

    tfidf = TfidfVectorizer(**VECTORIZER_PARAMS)

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
    ensure_artifact_dirs()
    df_train, df_val, text_col = load_data()

    X_train, X_val, tfidf = vectorize_text(df_train, df_val, text_col)

    y_train, y_val, le = encode_labels(df_train, df_val)

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=2000, C=1.0)),
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
    summary = {
        "text_column": text_col,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "class_count": len(le.classes_),
        "classes": [str(label) for label in le.classes_],
        "vectorizer": {
            **VECTORIZER_PARAMS,
            "vocabulary_size": X_train.shape[1],
        },
        "best_model": res_df.iloc[0]["model"],
        "results": res_df.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Результаты сохранены: {RESULTS_PATH}")

    print("\n[+] Таблица результатов:")
    print(res_df)

    with start_mlflow_run(
        project_root=PROJECT_ROOT,
        experiment_name="aggregator_bot_baselines",
        run_name="baseline_models",
        tags={
            "pipeline": "aggregate_bot_modules.ml_module.baseline_models",
            "stage": "baseline",
        },
    ) as mlflow_cfg:
        if mlflow_cfg:
            log_params(
                {
                    "text_column": text_col,
                    "train_rows": len(df_train),
                    "val_rows": len(df_val),
                    "class_count": len(le.classes_),
                    "tfidf_ngram_range": VECTORIZER_PARAMS["ngram_range"],
                    "tfidf_min_df": VECTORIZER_PARAMS["min_df"],
                    "tfidf_max_features": VECTORIZER_PARAMS["max_features"],
                    "vocabulary_size": X_train.shape[1],
                }
            )
            best_row = res_df.iloc[0]
            log_metrics(
                {
                    "best_f1_macro": float(best_row["f1_macro"]),
                    "best_accuracy": float(best_row["accuracy"]),
                    "best_precision_macro": float(best_row["precision_macro"]),
                    "best_recall_macro": float(best_row["recall_macro"]),
                }
            )
            for row in res_df.to_dict(orient="records"):
                prefix = str(row["model"]).lower().replace(".", "").replace("-", "_")
                log_metrics(
                    {
                        f"{prefix}_f1_macro": float(row["f1_macro"]),
                        f"{prefix}_accuracy": float(row["accuracy"]),
                        f"{prefix}_precision_macro": float(row["precision_macro"]),
                        f"{prefix}_recall_macro": float(row["recall_macro"]),
                    }
                )
            log_artifact_if_exists(RESULTS_PATH)
            log_artifact_if_exists(SUMMARY_PATH)
            print(f"[+] MLflow run logged to {mlflow_cfg['tracking_uri']} ({mlflow_cfg['experiment_name']}).")


if __name__ == "__main__":
    main()
