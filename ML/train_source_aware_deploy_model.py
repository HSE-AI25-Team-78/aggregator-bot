from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregate_bot_modules.ml_module.mlflow_utils import (  # noqa: E402
    log_artifact_if_exists,
    log_metrics,
    log_params,
    start_mlflow_run,
)
from ML.pseudo_label_pipeline import (  # noqa: E402
    LABELS,
    RAW_LABELS_PATH,
    SERVICE_CONFIG_DIR,
    SERVICE_MODELS_DIR,
    clean_text,
    read_previous_best_nb_f1,
    save_pickle,
)


AUGMENTED_TRAIN_PATH = PROJECT_ROOT / "ML" / "targeted_weak_class_artifacts" / "augmented_train_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "ML" / "source_aware_deploy_model_artifacts"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
MANIFEST_PATH = SERVICE_CONFIG_DIR / "model_manifest.json"
MODEL_OUTPUT_PATH = SERVICE_MODELS_DIR / "SourceAwareCalibratedLinearSVC.pkl"
VECTORIZER_OUTPUT_PATH = SERVICE_CONFIG_DIR / "vectorizer.pkl"
LABEL_ENCODER_OUTPUT_PATH = SERVICE_CONFIG_DIR / "label_encoder.pkl"


def load_seed_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_LABELS_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].map(clean_text)
    df["label_name"] = df["topic"].astype(int).map(lambda idx: LABELS[idx])
    df["channel_short"] = df["channel_short"].fillna("unknown").astype(str)
    df = df[df["clean_text"].str.len() >= 15].copy()
    return df


def load_augmented_train() -> pd.DataFrame:
    df = pd.read_csv(AUGMENTED_TRAIN_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["clean_text"].astype(str)
    df["label_name"] = df["label_name"].astype(str)
    df["channel_short"] = df["channel_short"].fillna("unknown").astype(str)
    return df


def build_source_aware_text(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda row: f"__source__{str(row['channel_short']).strip().lower()} {str(row['clean_text']).strip()}",
        axis=1,
    )


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
        sublinear_tf=True,
    )


def evaluate_model(model, X_test, y_test, label_encoder: LabelEncoder) -> tuple[dict, str]:
    predictions = model.predict(X_test)
    metrics = {
        "model_name": "SourceAwareCalibratedLinearSVC",
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
    }
    report = classification_report(
        y_test,
        predictions,
        target_names=[str(label) for label in label_encoder.classes_],
        zero_division=0,
    )
    return metrics, report


def write_manifest(label_encoder: LabelEncoder, vectorizer: TfidfVectorizer, metrics: dict) -> None:
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.update(
        {
            "model_name": metrics["model_name"],
            "task": "news_topic_classification",
            "source_pipeline": "ML.train_source_aware_deploy_model",
            "class_count": len(label_encoder.classes_),
            "classes": [str(label) for label in label_encoder.classes_],
            "vectorizer": {
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_features": 20000,
                "sublinear_tf": True,
                "vocabulary_size": len(vectorizer.vocabulary_),
                "source_aware": True,
            },
            "test_metrics": {
                "f1_macro": metrics["f1_macro"],
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
            },
            "artifacts": {
                "vectorizer": str(VECTORIZER_OUTPUT_PATH.resolve()),
                "label_encoder": str(LABEL_ENCODER_OUTPUT_PATH.resolve()),
                "model": str(MODEL_OUTPUT_PATH.resolve()),
            },
        }
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_dataset()
    augmented_train_df = load_augmented_train()
    previous_best_nb_f1 = read_previous_best_nb_f1()

    train_seed, test_df = train_test_split(
        seed_df,
        test_size=0.15,
        stratify=seed_df["label_name"],
        random_state=42,
    )

    # Use the best augmented train corpus but build source-aware features.
    train_text = build_source_aware_text(augmented_train_df)
    test_text = build_source_aware_text(test_df)

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(augmented_train_df["label_name"])
    y_test = label_encoder.transform(test_df["label_name"])

    model = CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, class_weight="balanced"),
        cv=3,
        method="sigmoid",
    )
    model.fit(X_train, y_train)
    metrics, report = evaluate_model(model, X_test, y_test, label_encoder)

    improved = float(metrics["f1_macro"]) > previous_best_nb_f1
    if improved:
        save_pickle(VECTORIZER_OUTPUT_PATH, vectorizer)
        save_pickle(LABEL_ENCODER_OUTPUT_PATH, label_encoder)
        save_pickle(MODEL_OUTPUT_PATH, model)
        write_manifest(label_encoder, vectorizer, metrics)

    summary = {
        "seed_rows": len(seed_df),
        "augmented_train_rows": len(augmented_train_df),
        "test_rows": len(test_df),
        "previous_best_nb_f1_macro": previous_best_nb_f1,
        "f1_macro": metrics["f1_macro"],
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "service_export_updated": improved,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Source-Aware Deploy Model Report",
                "",
                f"- Seed rows: **{len(seed_df)}**",
                f"- Augmented train rows: **{len(augmented_train_df)}**",
                f"- Test rows: **{len(test_df)}**",
                f"- Previous best NB F1_macro: **{previous_best_nb_f1:.4f}**",
                f"- Source-aware F1_macro: **{metrics['f1_macro']:.4f}**",
                f"- Accuracy: **{metrics['accuracy']:.4f}**",
                f"- Precision macro: **{metrics['precision_macro']:.4f}**",
                f"- Recall macro: **{metrics['recall_macro']:.4f}**",
                f"- Service export updated: **{'yes' if improved else 'no'}**",
                "",
                "## Classification report",
                "",
                "```text",
                report,
                "```",
            ]
        ),
        encoding="utf-8",
    )

    with start_mlflow_run(
        project_root=PROJECT_ROOT,
        experiment_name="aggregator_bot_source_aware_deploy_model",
        run_name="train_source_aware_deploy_model",
        tags={
            "pipeline": "ML.train_source_aware_deploy_model",
            "stage": "deploy_model_search",
        },
    ) as mlflow_cfg:
        if mlflow_cfg:
            log_params(
                {
                    "seed_rows": len(seed_df),
                    "augmented_train_rows": len(augmented_train_df),
                    "test_rows": len(test_df),
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "source_aware": True,
                }
            )
            log_metrics(
                {
                    "previous_best_nb_f1_macro": previous_best_nb_f1,
                    "f1_macro": metrics["f1_macro"],
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "service_export_updated": float(improved),
                }
            )
            for artifact_path in [SUMMARY_PATH, REPORT_PATH, MANIFEST_PATH]:
                log_artifact_if_exists(artifact_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
