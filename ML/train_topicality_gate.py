from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregate_bot_modules.ml_module.mlflow_utils import (  # noqa: E402
    log_artifact_if_exists,
    log_metrics,
    log_params,
    start_mlflow_run,
)
from ML.pseudo_label_pipeline import LABELS, RAW_LABELS_PATH, clean_text  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "ML" / "topicality_gate_artifacts"
SERVICE_CONFIG_DIR = PROJECT_ROOT / "service" / "config"
GATE_MODEL_PATH = SERVICE_CONFIG_DIR / "topicality_gate.pkl"
GATE_VECTORIZER_PATH = SERVICE_CONFIG_DIR / "topicality_gate_vectorizer.pkl"
MANIFEST_PATH = SERVICE_CONFIG_DIR / "model_manifest.json"
SUMMARY_PATH = OUTPUT_DIR / "topicality_gate_summary.json"
REPORT_PATH = OUTPUT_DIR / "topicality_gate_report.md"

TOPICAL_THRESHOLD = 0.55
GENERAL_THRESHOLD = 0.35


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_LABELS_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= 15].copy()
    df["label_name"] = df["topic"].astype(int).map(lambda idx: LABELS[idx])
    df["is_topical"] = (df["topic"].astype(int) != 0).astype(int)
    return df


def update_manifest(summary: dict) -> None:
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["topicality_gate"] = {
        "model_name": "LogisticRegression",
        "task": "topical_vs_general_gate",
        "general_threshold": GENERAL_THRESHOLD,
        "topical_threshold": TOPICAL_THRESHOLD,
        "metrics": summary["metrics"],
        "artifacts": {
            "vectorizer": str(GATE_VECTORIZER_PATH.resolve()),
            "model": str(GATE_MODEL_PATH.resolve()),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["is_topical"],
        random_state=42,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=12000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])
    y_train = train_df["is_topical"].astype(int)
    y_test = test_df["is_topical"].astype(int)

    model = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_binary": float(f1_score(y_test, pred, average="binary")),
        "precision_binary": float(precision_score(y_test, pred, average="binary", zero_division=0)),
        "recall_binary": float(recall_score(y_test, pred, average="binary", zero_division=0)),
    }
    report = classification_report(y_test, pred, target_names=["general", "topical"], zero_division=0)

    save_pickle(GATE_VECTORIZER_PATH, vectorizer)
    save_pickle(GATE_MODEL_PATH, model)

    summary = {
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "general_threshold": GENERAL_THRESHOLD,
        "topical_threshold": TOPICAL_THRESHOLD,
        "metrics": metrics,
        "topical_rate_train": round(float(y_train.mean()), 4),
        "topical_rate_test": round(float(y_test.mean()), 4),
        "mean_topical_probability": round(float(proba.mean()), 4),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Topicality Gate Report",
                "",
                f"- Rows: **{len(df)}**",
                f"- Accuracy: **{metrics['accuracy']:.4f}**",
                f"- F1(binary): **{metrics['f1_binary']:.4f}**",
                f"- Precision(binary): **{metrics['precision_binary']:.4f}**",
                f"- Recall(binary): **{metrics['recall_binary']:.4f}**",
                f"- General threshold: **{GENERAL_THRESHOLD}**",
                f"- Topical threshold: **{TOPICAL_THRESHOLD}**",
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

    update_manifest(summary)

    with start_mlflow_run(
        project_root=PROJECT_ROOT,
        experiment_name="aggregator_bot_topicality_gate",
        run_name="train_topicality_gate",
        tags={
            "pipeline": "ML.train_topicality_gate",
            "stage": "topicality_gate",
        },
    ) as mlflow_cfg:
        if mlflow_cfg:
            log_params(
                {
                    "rows": len(df),
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "general_threshold": GENERAL_THRESHOLD,
                    "topical_threshold": TOPICAL_THRESHOLD,
                    "vectorizer_max_features": 12000,
                }
            )
            log_metrics(metrics)
            for artifact_path in [SUMMARY_PATH, REPORT_PATH, GATE_MODEL_PATH, GATE_VECTORIZER_PATH, MANIFEST_PATH]:
                log_artifact_if_exists(artifact_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
