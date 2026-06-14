from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggregate_bot_modules.ml_module.mlflow_utils import (  # noqa: E402
    log_artifact_if_exists,
    log_metrics,
    log_params,
    start_mlflow_run,
)

ML_DIR = PROJECT_ROOT / "ML"
PSEUDO_DIR = ML_DIR / "pseudo_label_artifacts"
OUTPUT_DIR = ML_DIR / "targeted_weak_class_artifacts"

RAW_LABELS_PATH = ML_DIR / "raw_posts_labeled.csv"
SERVICE_CONFIG_DIR = PROJECT_ROOT / "service" / "config"
SERVICE_MODELS_DIR = SERVICE_CONFIG_DIR / "models"
PSEUDO_REPORT_PATH = ML_DIR / "pseudo_label_artifacts" / "report.md"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
MANIFEST_PATH = SERVICE_CONFIG_DIR / "model_manifest.json"

LABELS = [
    "Общее",
    "Наука и техника",
    "ИТ и телекоммуникации",
    "Общество, государство, политика",
    "Экономика",
    "Медицина",
    "Искусство и культура",
    "Развлечения",
    "Спорт",
    "История",
    "Происшествия",
]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

CAP_SELECTED = {
    "ИТ и телекоммуникации": 350,
    "Экономика": 300,
    "Наука и техника": 300,
    "Происшествия": 220,
    "Искусство и культура": 220,
    "Спорт": 180,
}

WEAK_RULES = {
    "Общество, государство, политика": {
        "min_confidence": 0.34,
        "min_margin": 0.17,
        "require_proposed_match": True,
        "limit": 220,
    },
    "Медицина": {
        "min_confidence": 0.29,
        "min_margin": 0.09,
        "require_proposed_match": True,
        "limit": 70,
    },
    "История": {
        "min_confidence": 0.30,
        "min_margin": 0.10,
        "require_proposed_match": False,
        "limit": 35,
    },
    "Развлечения": {
        "min_confidence": 0.36,
        "min_margin": 0.12,
        "require_proposed_match": True,
        "limit": 80,
    },
    "Общее": {
        "min_confidence": 0.20,
        "min_margin": 0.04,
        "require_proposed_match": False,
        "limit": 20,
    },
}


def clean_text(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|t\.me/\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_seed_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_LABELS_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= 15].copy()
    df["label_name"] = df["topic"].astype(int).map(lambda idx: LABELS[idx])
    return df


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=15000,
        sublinear_tf=True,
    )


def build_models(train_df: pd.DataFrame):
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["clean_text"].astype(str))

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label_name"])

    logreg = LogisticRegression(
        max_iter=4000,
        C=2.0,
        class_weight="balanced",
    )
    logreg.fit(X_train, y_train)

    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)

    return vectorizer, logreg, nb, le


def evaluate_model(vectorizer, model, le, test_df: pd.DataFrame) -> tuple[float, str]:
    X_test = vectorizer.transform(test_df["clean_text"].astype(str))
    y_test = le.transform(test_df["label_name"])
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(label) for label in le.classes_],
        zero_division=0,
    )
    return f1, report


def cap_selected(selected_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for label, group in selected_df.groupby("final_label"):
        limit = CAP_SELECTED.get(label, len(group))
        group = group.sort_values(["confidence", "margin"], ascending=False).head(limit)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def select_weak_examples(rejected_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for label, rules in WEAK_RULES.items():
        group = rejected_df[rejected_df["final_label"] == label].copy()
        if group.empty:
            continue
        group = group[group["confidence"] >= rules["min_confidence"]]
        group = group[group["margin"] >= rules["min_margin"]]
        if rules["require_proposed_match"]:
            group = group[group["proposed_label"] == label]
        group = group.sort_values(["confidence", "margin"], ascending=False).head(rules["limit"])
        if not group.empty:
            parts.append(group)
    if not parts:
        return pd.DataFrame(columns=rejected_df.columns)
    return pd.concat(parts, ignore_index=True)


def to_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"final_topic_id": "topic", "final_label": "label_name"})[
        ["channel_short", "date", "topic", "text", "clean_text", "label_name"]
    ]


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def read_previous_best_nb_f1() -> float:
    if not PSEUDO_REPORT_PATH.exists():
        return 0.0
    text = PSEUDO_REPORT_PATH.read_text(encoding="utf-8")
    marker = "Augmented MultinomialNB F1_macro: **"
    for line in text.splitlines():
        if marker in line:
            value = line.split(marker, 1)[1].split("**", 1)[0]
            try:
                return float(value)
            except ValueError:
                return 0.0
    return 0.0


def write_service_manifest(*, label_encoder: LabelEncoder, vectorizer: TfidfVectorizer, nb_f1: float) -> None:
    manifest = {
        "model_name": "MultinomialNB",
        "task": "news_topic_classification",
        "source_pipeline": "ML.targeted_weak_class_pipeline",
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
            "f1_macro": nb_f1,
        },
        "artifacts": {
            "vectorizer": str((SERVICE_CONFIG_DIR / "vectorizer.pkl").resolve()),
            "label_encoder": str((SERVICE_CONFIG_DIR / "label_encoder.pkl").resolve()),
            "model": str((SERVICE_MODELS_DIR / "MultinomialNB.pkl").resolve()),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_dataset()
    selected_df = pd.read_csv(PSEUDO_DIR / "pseudo_labeled_selected.csv")
    rejected_df = pd.read_csv(PSEUDO_DIR / "pseudo_labeled_rejected.csv")

    train_df, test_df = train_test_split(
        seed_df,
        test_size=0.15,
        stratify=seed_df["label_name"],
        random_state=42,
    )

    baseline_vectorizer, baseline_logreg, baseline_nb, baseline_le = build_models(train_df)
    baseline_nb_f1, baseline_nb_report = evaluate_model(baseline_vectorizer, baseline_nb, baseline_le, test_df)
    previous_best_nb_f1 = read_previous_best_nb_f1()

    capped_selected_df = cap_selected(selected_df)
    weak_selected_df = select_weak_examples(rejected_df)

    augmented_train = pd.concat(
        [
            train_df[["channel_short", "date", "topic", "text", "clean_text", "label_name"]],
            to_train_frame(capped_selected_df),
            to_train_frame(weak_selected_df),
        ],
        ignore_index=True,
    )

    final_vectorizer, final_logreg, final_nb, final_le = build_models(augmented_train)
    final_logreg_f1, final_logreg_report = evaluate_model(final_vectorizer, final_logreg, final_le, test_df)
    final_nb_f1, final_nb_report = evaluate_model(final_vectorizer, final_nb, final_le, test_df)

    selected_capped_path = OUTPUT_DIR / "selected_capped.csv"
    weak_selected_path = OUTPUT_DIR / "weak_class_selected.csv"
    augmented_train_path = OUTPUT_DIR / "augmented_train_dataset.csv"

    capped_selected_df.to_csv(selected_capped_path, index=False, encoding="utf-8")
    weak_selected_df.to_csv(weak_selected_path, index=False, encoding="utf-8")
    augmented_train.to_csv(augmented_train_path, index=False, encoding="utf-8")

    updated = final_nb_f1 >= previous_best_nb_f1
    if updated:
        save_pickle(SERVICE_CONFIG_DIR / "vectorizer.pkl", final_vectorizer)
        save_pickle(SERVICE_CONFIG_DIR / "label_encoder.pkl", final_le)
        save_pickle(SERVICE_MODELS_DIR / "MultinomialNB.pkl", final_nb)
        write_service_manifest(
            label_encoder=final_le,
            vectorizer=final_vectorizer,
            nb_f1=final_nb_f1,
        )

    lines = [
        "# Targeted Weak Class Pipeline Report",
        "",
        f"- Seed labeled rows: **{len(seed_df)}**",
        f"- Capped selected rows: **{len(capped_selected_df)}**",
        f"- Weak-class targeted rows: **{len(weak_selected_df)}**",
        f"- Augmented train rows: **{len(augmented_train)}**",
        "",
        "## Weak-class additions by label",
        "",
    ]
    if len(weak_selected_df):
        for label, count in weak_selected_df["final_label"].value_counts().items():
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Metrics on original labeled holdout",
            "",
            f"- Baseline MultinomialNB F1_macro: **{baseline_nb_f1:.4f}**",
            f"- Previous best exported MultinomialNB F1_macro: **{previous_best_nb_f1:.4f}**",
            f"- Targeted MultinomialNB F1_macro: **{final_nb_f1:.4f}**",
            f"- Targeted LogisticRegression F1_macro: **{final_logreg_f1:.4f}**",
            f"- Service export updated: **{'yes' if updated else 'no'}**",
            "",
            "## Targeted NB report",
            "",
            "```text",
            final_nb_report,
            "```",
        ]
    )
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "seed_rows": len(seed_df),
        "selected_capped_rows": len(capped_selected_df),
        "weak_selected_rows": len(weak_selected_df),
        "augmented_train_rows": len(augmented_train),
        "previous_best_nb_f1_macro": previous_best_nb_f1,
        "baseline_nb_f1_macro": baseline_nb_f1,
        "final_nb_f1_macro": final_nb_f1,
        "final_logreg_f1_macro": final_logreg_f1,
        "service_export_updated": updated,
        "weak_rules": WEAK_RULES,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with start_mlflow_run(
        project_root=PROJECT_ROOT,
        experiment_name="aggregator_bot_targeted_weak_classes",
        run_name="targeted_weak_class_pipeline",
        tags={
            "pipeline": "ML.targeted_weak_class_pipeline",
            "stage": "weak_class_targeting",
        },
    ) as mlflow_cfg:
        if mlflow_cfg:
            log_params(
                {
                    "seed_rows": len(seed_df),
                    "selected_capped_rows": len(capped_selected_df),
                    "weak_selected_rows": len(weak_selected_df),
                    "cap_selected_count": len(CAP_SELECTED),
                    "weak_rule_count": len(WEAK_RULES),
                }
            )
            log_metrics(
                {
                    "baseline_nb_f1_macro": baseline_nb_f1,
                    "previous_best_nb_f1_macro": previous_best_nb_f1,
                    "final_nb_f1_macro": final_nb_f1,
                    "final_logreg_f1_macro": final_logreg_f1,
                    "augmented_train_rows": len(augmented_train),
                    "service_export_updated": float(updated),
                }
            )
            for artifact_path in [
                selected_capped_path,
                weak_selected_path,
                augmented_train_path,
                report_path,
                SUMMARY_PATH,
            ]:
                log_artifact_if_exists(artifact_path)
            print(f"[+] MLflow run logged to {mlflow_cfg['tracking_uri']} ({mlflow_cfg['experiment_name']}).")

    print(f"[+] Capped selected rows: {len(capped_selected_df)}")
    print(f"[+] Weak-class targeted rows: {len(weak_selected_df)}")
    print(f"[+] Baseline NB F1_macro: {baseline_nb_f1:.4f}")
    print(f"[+] Targeted NB F1_macro: {final_nb_f1:.4f}")
    print(f"[+] Service export updated: {'yes' if updated else 'no'}")


if __name__ == "__main__":
    main()
