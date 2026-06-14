from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
OUTPUT_DIR = PROJECT_ROOT / "ML" / "stronger_deploy_model_artifacts"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
MANIFEST_PATH = SERVICE_CONFIG_DIR / "model_manifest.json"
MODEL_OUTPUT_PATH = SERVICE_MODELS_DIR / "CalibratedLinearSVC.pkl"
VECTORIZER_OUTPUT_PATH = SERVICE_CONFIG_DIR / "vectorizer.pkl"
LABEL_ENCODER_OUTPUT_PATH = SERVICE_CONFIG_DIR / "label_encoder.pkl"


def load_seed_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_LABELS_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= 15].copy()
    df["label_name"] = df["topic"].astype(int).map(lambda idx: LABELS[idx])
    return df


def load_augmented_train() -> pd.DataFrame:
    df = pd.read_csv(AUGMENTED_TRAIN_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["clean_text"].astype(str)
    df["label_name"] = df["label_name"].astype(str)
    return df


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=15000,
        sublinear_tf=True,
    )


def evaluate_model(name: str, model, X_test, y_test, label_encoder: LabelEncoder) -> tuple[dict, str]:
    predictions = model.predict(X_test)
    metrics = {
        "model_name": name,
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
            "source_pipeline": "ML.train_stronger_deploy_model",
            "class_count": len(label_encoder.classes_),
            "classes": [str(label) for label in label_encoder.classes_],
            "vectorizer": {
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_features": 15000,
                "sublinear_tf": True,
                "vocabulary_size": len(vectorizer.vocabulary_),
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

    _, test_df = train_test_split(
        seed_df,
        test_size=0.15,
        stratify=seed_df["label_name"],
        random_state=42,
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(augmented_train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(augmented_train_df["label_name"])
    y_test = label_encoder.transform(test_df["label_name"])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=4000, C=2.0, class_weight="balanced"),
        "SGDClassifier_log_loss": SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=4000, class_weight="balanced", random_state=42),
        "CalibratedLinearSVC": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced"),
            cv=3,
            method="sigmoid",
        ),
    }

    results: list[dict] = []
    reports: dict[str, str] = {}
    fitted_models: dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        metrics, report = evaluate_model(name, model, X_test, y_test, label_encoder)
        results.append(metrics)
        reports[name] = report

    results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    best = results_df.iloc[0].to_dict()
    best_name = str(best["model_name"])
    best_model = fitted_models[best_name]

    improved = float(best["f1_macro"]) > previous_best_nb_f1
    if improved:
        save_pickle(VECTORIZER_OUTPUT_PATH, vectorizer)
        save_pickle(LABEL_ENCODER_OUTPUT_PATH, label_encoder)
        save_pickle(MODEL_OUTPUT_PATH, best_model)
        write_manifest(label_encoder, vectorizer, best)

    summary = {
        "seed_rows": len(seed_df),
        "augmented_train_rows": len(augmented_train_df),
        "test_rows": len(test_df),
        "previous_best_nb_f1_macro": previous_best_nb_f1,
        "best_model": best_name,
        "best_f1_macro": float(best["f1_macro"]),
        "service_export_updated": improved,
        "results": results_df.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stronger Deploy Model Report",
        "",
        f"- Seed rows: **{len(seed_df)}**",
        f"- Augmented train rows: **{len(augmented_train_df)}**",
        f"- Test rows: **{len(test_df)}**",
        f"- Previous best NB F1_macro: **{previous_best_nb_f1:.4f}**",
        f"- Best model: **{best_name}**",
        f"- Best F1_macro: **{float(best['f1_macro']):.4f}**",
        f"- Service export updated: **{'yes' if improved else 'no'}**",
        "",
        "## Results",
        "",
    ]
    for row in results_df.to_dict(orient="records"):
        lines.append(
            f"- {row['model_name']}: f1_macro={row['f1_macro']:.4f}, accuracy={row['accuracy']:.4f}, "
            f"precision_macro={row['precision_macro']:.4f}, recall_macro={row['recall_macro']:.4f}"
        )
    lines.extend(["", "## Best model report", "", "```text", reports[best_name], "```"])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    with start_mlflow_run(
        project_root=PROJECT_ROOT,
        experiment_name="aggregator_bot_stronger_deploy_model",
        run_name="train_stronger_deploy_model",
        tags={
            "pipeline": "ML.train_stronger_deploy_model",
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
                }
            )
            for row in results_df.to_dict(orient="records"):
                prefix = str(row["model_name"]).lower().replace(".", "").replace("-", "_")
                log_metrics(
                    {
                        f"{prefix}_f1_macro": float(row["f1_macro"]),
                        f"{prefix}_accuracy": float(row["accuracy"]),
                        f"{prefix}_precision_macro": float(row["precision_macro"]),
                        f"{prefix}_recall_macro": float(row["recall_macro"]),
                    }
                )
            log_metrics(
                {
                    "previous_best_nb_f1_macro": previous_best_nb_f1,
                    "best_f1_macro": float(best["f1_macro"]),
                    "service_export_updated": float(improved),
                }
            )
            for artifact_path in [SUMMARY_PATH, REPORT_PATH, MANIFEST_PATH]:
                log_artifact_if_exists(artifact_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
