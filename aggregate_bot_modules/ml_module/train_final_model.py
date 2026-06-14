from __future__ import annotations

import json
import os
import pickle
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .paths import (
    DEPLOY_MODELS_DIR,
    FINAL_RESULTS_DIR,
    LOGREG_MODELS_DIR,
    MLFLOW_RUNS_DIR,
    SERVICE_CONFIG_DIR,
    SERVICE_MODELS_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    ensure_artifact_dirs,
)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except ImportError:  # pragma: no cover - optional dependency path
    mlflow = None
    infer_signature = None


LOGREG_TFIDF_PATH = LOGREG_MODELS_DIR / "tfidf.joblib"
LOGREG_MODEL_PATH = LOGREG_MODELS_DIR / "logreg_model.joblib"
LOGREG_LE_PATH = LOGREG_MODELS_DIR / "label_encoder.joblib"

DEPLOY_TFIDF_PATH = DEPLOY_MODELS_DIR / "vectorizer.pkl"
DEPLOY_MODEL_PATH = DEPLOY_MODELS_DIR / "MultinomialNB.pkl"
DEPLOY_LE_PATH = DEPLOY_MODELS_DIR / "label_encoder.pkl"

SERVICE_TFIDF_PATH = SERVICE_CONFIG_DIR / "vectorizer.pkl"
SERVICE_LE_PATH = SERVICE_CONFIG_DIR / "label_encoder.pkl"
SERVICE_MODEL_PATH = SERVICE_MODELS_DIR / "MultinomialNB.pkl"
SERVICE_MANIFEST_PATH = SERVICE_CONFIG_DIR / "model_manifest.json"

TRAINING_SUMMARY_PATH = FINAL_RESULTS_DIR / "training_summary.json"
LOGREG_REPORT_PATH = FINAL_RESULTS_DIR / "logreg_classification_report.csv"
DEPLOY_NB_REPORT_PATH = FINAL_RESULTS_DIR / "deploy_nb_classification_report.csv"
LOGREG_CM_PATH = FINAL_RESULTS_DIR / "logreg_confusion_matrix.png"
DEPLOY_NB_CM_PATH = FINAL_RESULTS_DIR / "deploy_nb_confusion_matrix.png"

VECTORIZER_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_features": 10000,
    "sublinear_tf": True,
}
LOGREG_PARAMS = {
    "max_iter": 3000,
    "C": 2.0,
}
DEPLOY_NB_PARAMS = {
    "alpha": 1.0,
}


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
    return TfidfVectorizer(**VECTORIZER_PARAMS)


def save_pickle(path: Path, obj) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def build_logreg() -> LogisticRegression:
    return LogisticRegression(**LOGREG_PARAMS)


def build_deploy_nb() -> MultinomialNB:
    return MultinomialNB(**DEPLOY_NB_PARAMS)


def build_text_pipeline(vectorizer: TfidfVectorizer, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", estimator),
        ]
    )


def evaluate_model(model_name: str, model, X_test, y_test, label_encoder: LabelEncoder) -> dict[str, Any]:
    predictions = model.predict(X_test)
    labels = [str(label) for label in label_encoder.classes_]
    report_dict = classification_report(
        y_test,
        predictions,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model_name": model_name,
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
    }
    return {
        "predictions": predictions,
        "metrics": metrics,
        "report_df": pd.DataFrame(report_dict).transpose(),
        "labels": labels,
    }


def save_confusion_matrix(y_true, y_pred, labels: list[str], output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(cmap="Blues", xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title(title)
    ax.tick_params(axis="both", which="major", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_training_summary(
    *,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_train_full: pd.DataFrame,
    text_col: str,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    logreg_metrics: dict[str, Any],
    nb_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "text_column": text_col,
        "row_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
            "train_full": len(df_train_full),
        },
        "class_count": len(label_encoder.classes_),
        "classes": [str(label) for label in label_encoder.classes_],
        "vectorizer": {
            **VECTORIZER_PARAMS,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "models": {
            "logreg": {
                "params": LOGREG_PARAMS,
                "metrics": logreg_metrics,
            },
            "deploy_nb": {
                "params": DEPLOY_NB_PARAMS,
                "metrics": nb_metrics,
            },
        },
        "comparison": {
            "logreg_minus_nb_f1_macro": logreg_metrics["f1_macro"] - nb_metrics["f1_macro"],
        },
        "git": get_git_context(),
    }


def build_service_manifest(
    *,
    text_col: str,
    label_encoder: LabelEncoder,
    vectorizer: TfidfVectorizer,
    nb_metrics: dict[str, Any],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": "MultinomialNB",
        "task": "news_topic_classification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_column": text_col,
        "class_count": len(label_encoder.classes_),
        "classes": [str(label) for label in label_encoder.classes_],
        "vectorizer": {
            **VECTORIZER_PARAMS,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "model_params": DEPLOY_NB_PARAMS,
        "test_metrics": nb_metrics,
        "artifacts": {
            "vectorizer": str(SERVICE_TFIDF_PATH.resolve()),
            "label_encoder": str(SERVICE_LE_PATH.resolve()),
            "model": str(SERVICE_MODEL_PATH.resolve()),
        },
        "git": training_summary.get("git", {}),
    }


def get_git_context() -> dict[str, str | None]:
    def read_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return result.stdout.strip() or None

    project_root = SERVICE_CONFIG_DIR.parent.parent
    return {
        "branch": read_git(["git", "-C", str(project_root), "branch", "--show-current"]),
        "commit": read_git(["git", "-C", str(project_root), "rev-parse", "HEAD"]),
    }


def mlflow_enabled() -> bool:
    raw_value = os.getenv("ENABLE_MLFLOW", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def configure_mlflow() -> dict[str, str] | None:
    if not mlflow_enabled():
        print("[*] MLflow отключён через ENABLE_MLFLOW=0.")
        return None
    if mlflow is None:
        print("[warn] mlflow не установлен, пропускаем experiment tracking.")
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_RUNS_DIR.resolve().as_uri())
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "aggregator_bot_training")
    run_name = os.getenv("MLFLOW_RUN_NAME", "train_final_model")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_name": run_name,
    }


def log_metrics_to_mlflow(prefix: str, metrics: dict[str, Any]) -> None:
    if mlflow is None:
        return
    for metric_name, value in metrics.items():
        if metric_name == "model_name":
            continue
        mlflow.log_metric(f"{prefix}_{metric_name}", float(value))


def log_artifact_if_exists(path: Path) -> None:
    if mlflow is None or not path.exists():
        return
    mlflow.log_artifact(str(path))


def main():
    ensure_artifact_dirs()
    df_train, df_val, df_test, text_col = load_splits()
    df_train_full = pd.concat([df_train, df_val], ignore_index=True)

    label_encoder = LabelEncoder()
    y_train_full = label_encoder.fit_transform(df_train_full["topic"])
    y_test = label_encoder.transform(df_test["topic"])

    tfidf = build_vectorizer()
    X_train_full = tfidf.fit_transform(df_train_full[text_col].astype(str))
    X_test = tfidf.transform(df_test[text_col].astype(str))

    logreg = build_logreg()
    deploy_nb = build_deploy_nb()

    print("[*] Обучаем LogisticRegression (аналитическая модель)...")
    logreg.fit(X_train_full, y_train_full)
    logreg_eval = evaluate_model("LogisticRegression", logreg, X_test, y_test, label_encoder)
    logreg_metrics = logreg_eval["metrics"]
    print(f"[+] LogisticRegression test F1_macro = {logreg_metrics['f1_macro']:.4f}")
    print(logreg_eval["report_df"])

    print("\n[*] Обучаем MultinomialNB (deploy-модель для service/bot)...")
    deploy_nb.fit(X_train_full, y_train_full)
    nb_eval = evaluate_model("MultinomialNB", deploy_nb, X_test, y_test, label_encoder)
    nb_metrics = nb_eval["metrics"]
    print(f"[+] MultinomialNB test F1_macro = {nb_metrics['f1_macro']:.4f}")
    print(nb_eval["report_df"])

    joblib.dump(tfidf, LOGREG_TFIDF_PATH)
    joblib.dump(logreg, LOGREG_MODEL_PATH)
    joblib.dump(label_encoder, LOGREG_LE_PATH)

    save_pickle(DEPLOY_TFIDF_PATH, tfidf)
    save_pickle(DEPLOY_MODEL_PATH, deploy_nb)
    save_pickle(DEPLOY_LE_PATH, label_encoder)

    save_pickle(SERVICE_TFIDF_PATH, tfidf)
    save_pickle(SERVICE_LE_PATH, label_encoder)
    save_pickle(SERVICE_MODEL_PATH, deploy_nb)

    logreg_eval["report_df"].to_csv(LOGREG_REPORT_PATH, encoding="utf-8")
    nb_eval["report_df"].to_csv(DEPLOY_NB_REPORT_PATH, encoding="utf-8")
    save_confusion_matrix(
        y_test,
        logreg_eval["predictions"],
        logreg_eval["labels"],
        LOGREG_CM_PATH,
        "LogisticRegression Confusion Matrix",
    )
    save_confusion_matrix(
        y_test,
        nb_eval["predictions"],
        nb_eval["labels"],
        DEPLOY_NB_CM_PATH,
        "Deploy MultinomialNB Confusion Matrix",
    )

    training_summary = build_training_summary(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        df_train_full=df_train_full,
        text_col=text_col,
        vectorizer=tfidf,
        label_encoder=label_encoder,
        logreg_metrics=logreg_metrics,
        nb_metrics=nb_metrics,
    )
    TRAINING_SUMMARY_PATH.write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    service_manifest = build_service_manifest(
        text_col=text_col,
        label_encoder=label_encoder,
        vectorizer=tfidf,
        nb_metrics=nb_metrics,
        training_summary=training_summary,
    )
    SERVICE_MANIFEST_PATH.write_text(
        json.dumps(service_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mlflow_cfg = configure_mlflow()
    run_context = nullcontext(None)
    if mlflow_cfg and mlflow is not None:
        run_context = mlflow.start_run(run_name=mlflow_cfg["run_name"])

    with run_context:
        if mlflow_cfg and mlflow is not None:
            mlflow.set_tag("pipeline", "aggregate_bot_modules.ml_module.train_final_model")
            mlflow.set_tag("deploy_model", "MultinomialNB")
            mlflow.set_tag("analysis_model", "LogisticRegression")
            git_context = training_summary["git"]
            if git_context.get("branch"):
                mlflow.set_tag("git_branch", git_context["branch"])
            if git_context.get("commit"):
                mlflow.set_tag("git_commit", git_context["commit"])

            mlflow.log_param("text_column", text_col)
            mlflow.log_param("train_rows", len(df_train))
            mlflow.log_param("val_rows", len(df_val))
            mlflow.log_param("test_rows", len(df_test))
            mlflow.log_param("train_full_rows", len(df_train_full))
            mlflow.log_param("class_count", len(label_encoder.classes_))

            for key, value in VECTORIZER_PARAMS.items():
                mlflow.log_param(f"tfidf_{key}", value)
            for key, value in LOGREG_PARAMS.items():
                mlflow.log_param(f"logreg_{key}", value)
            for key, value in DEPLOY_NB_PARAMS.items():
                mlflow.log_param(f"deploy_nb_{key}", value)

            log_metrics_to_mlflow("logreg_test", logreg_metrics)
            log_metrics_to_mlflow("deploy_nb_test", nb_metrics)
            mlflow.log_metric("logreg_minus_nb_f1_macro", training_summary["comparison"]["logreg_minus_nb_f1_macro"])
            mlflow.log_metric("vocabulary_size", len(tfidf.vocabulary_))

            for artifact_path in [
                LOGREG_REPORT_PATH,
                DEPLOY_NB_REPORT_PATH,
                LOGREG_CM_PATH,
                DEPLOY_NB_CM_PATH,
                TRAINING_SUMMARY_PATH,
                SERVICE_MANIFEST_PATH,
                LOGREG_TFIDF_PATH,
                LOGREG_MODEL_PATH,
                LOGREG_LE_PATH,
                DEPLOY_TFIDF_PATH,
                DEPLOY_MODEL_PATH,
                DEPLOY_LE_PATH,
                SERVICE_TFIDF_PATH,
                SERVICE_LE_PATH,
                SERVICE_MODEL_PATH,
            ]:
                log_artifact_if_exists(artifact_path)

            if infer_signature is not None:
                sample_input = df_test[text_col].astype(str).head(5).tolist()
                logreg_pipeline = build_text_pipeline(tfidf, logreg)
                deploy_nb_pipeline = build_text_pipeline(tfidf, deploy_nb)
                signature = infer_signature(sample_input, logreg_pipeline.predict(sample_input))
                try:
                    mlflow.sklearn.log_model(
                        sk_model=logreg_pipeline,
                        artifact_path="logreg_pipeline_model",
                        signature=signature,
                    )
                    mlflow.sklearn.log_model(
                        sk_model=deploy_nb_pipeline,
                        artifact_path="deploy_nb_pipeline_model",
                        signature=signature,
                    )
                except Exception as exc:  # pragma: no cover - defensive guard for optional logging
                    print(f"[warn] Не удалось залогировать sklearn pipeline в MLflow: {exc}")

            print(f"[+] MLflow run logged to {mlflow_cfg['tracking_uri']} ({mlflow_cfg['experiment_name']}).")

    print(f"\n[+] LogReg TF-IDF сохранён в: {LOGREG_TFIDF_PATH}")
    print(f"[+] LogReg модель сохранена в: {LOGREG_MODEL_PATH}")
    print(f"[+] LogReg LabelEncoder сохранён в: {LOGREG_LE_PATH}")
    print(f"[+] Deploy TF-IDF сохранён в: {DEPLOY_TFIDF_PATH}")
    print(f"[+] Deploy NB сохранён в: {DEPLOY_MODEL_PATH}")
    print(f"[+] Deploy LabelEncoder сохранён в: {DEPLOY_LE_PATH}")
    print(f"[+] Service vectorizer экспортирован в: {SERVICE_TFIDF_PATH}")
    print(f"[+] Service label encoder экспортирован в: {SERVICE_LE_PATH}")
    print(f"[+] Service MultinomialNB экспортирован в: {SERVICE_MODEL_PATH}")
    print(f"[+] Service manifest экспортирован в: {SERVICE_MANIFEST_PATH}")
    print(f"[+] Итоговый summary сохранён в: {TRAINING_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
