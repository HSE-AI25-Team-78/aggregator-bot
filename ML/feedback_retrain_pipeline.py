from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
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
from ML.build_feedback_training_candidates import (  # noqa: E402
    ACCEPTED_CSV_PATH,
    OUTPUT_DIR,
    REPORT_PATH as FEEDBACK_REPORT_PATH,
    SUMMARY_PATH as FEEDBACK_SUMMARY_PATH,
)
from ML.pseudo_label_pipeline import (  # noqa: E402
    LABELS,
    RAW_LABELS_PATH,
    REPORT_PATH as PSEUDO_REPORT_PATH,
    SUMMARY_PATH as PSEUDO_SUMMARY_PATH,
    build_models,
    clean_text,
    evaluate_model,
    read_previous_best_nb_f1,
    save_pickle,
    write_service_manifest,
    SERVICE_CONFIG_DIR,
    SERVICE_MODELS_DIR,
)


REPORT_PATH = OUTPUT_DIR / "feedback_retrain_report.md"
SUMMARY_PATH = OUTPUT_DIR / "feedback_retrain_summary.json"
AUGMENTED_TRAIN_PATH = OUTPUT_DIR / "feedback_augmented_train_dataset.csv"


def load_seed_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_LABELS_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= 15].copy()
    df["label_name"] = df["topic"].astype(int).map(lambda idx: LABELS[idx])
    return df


def load_feedback_candidates() -> pd.DataFrame:
    if not ACCEPTED_CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ACCEPTED_CSV_PATH)
    if df.empty:
        return df
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["clean_text"].astype(str)
    df["label_name"] = df["predicted_label"].astype(str)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_dataset()
    feedback_df = load_feedback_candidates()
    previous_best_nb_f1 = read_previous_best_nb_f1()

    train_df, test_df = train_test_split(
        seed_df,
        test_size=0.15,
        stratify=seed_df["label_name"],
        random_state=42,
    )

    baseline_models = build_models(train_df)
    baseline_logreg_f1, _ = evaluate_model(
        baseline_models.vectorizer, baseline_models.logreg, baseline_models.label_encoder, test_df
    )
    baseline_nb_f1, baseline_nb_report = evaluate_model(
        baseline_models.vectorizer, baseline_models.nb, baseline_models.label_encoder, test_df
    )

    if feedback_df.empty:
        summary = {
            "seed_rows": len(seed_df),
            "feedback_rows": 0,
            "baseline_nb_f1_macro": baseline_nb_f1,
            "final_nb_f1_macro": baseline_nb_f1,
            "service_export_updated": False,
            "reason": "no_feedback_candidates",
        }
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_PATH.write_text(
            "# Feedback Retrain Report\n\nПока нет принятых feedback-кандидатов для дообучения.\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    feedback_train = feedback_df[
        [
            "source",
            "published_at",
            "topic",
            "text",
            "clean_text",
            "label_name",
            "weak_feedback_score",
            "predicted_confidence",
        ]
    ].rename(columns={"source": "channel_short", "published_at": "date"})

    feedback_train["replicas"] = feedback_train["weak_feedback_score"].map(lambda value: 2 if float(value) >= 1.2 else 1)
    replicated_feedback = feedback_train.loc[feedback_train.index.repeat(feedback_train["replicas"])].drop(columns=["replicas"])

    augmented_train = pd.concat(
        [
            train_df[["channel_short", "date", "topic", "text", "clean_text", "label_name"]],
            replicated_feedback[["channel_short", "date", "topic", "text", "clean_text", "label_name"]],
        ],
        ignore_index=True,
    )
    augmented_train.to_csv(AUGMENTED_TRAIN_PATH, index=False, encoding="utf-8")

    final_models = build_models(augmented_train)
    final_logreg_f1, _ = evaluate_model(
        final_models.vectorizer, final_models.logreg, final_models.label_encoder, test_df
    )
    final_nb_f1, final_nb_report = evaluate_model(
        final_models.vectorizer, final_models.nb, final_models.label_encoder, test_df
    )

    improved = final_nb_f1 >= max(baseline_nb_f1, previous_best_nb_f1)
    if improved:
        save_pickle(SERVICE_CONFIG_DIR / "vectorizer.pkl", final_models.vectorizer)
        save_pickle(SERVICE_CONFIG_DIR / "label_encoder.pkl", final_models.label_encoder)
        save_pickle(SERVICE_MODELS_DIR / "MultinomialNB.pkl", final_models.nb)
        write_service_manifest(
            label_encoder=final_models.label_encoder,
            vectorizer=final_models.vectorizer,
            nb_f1=final_nb_f1,
        )

    summary = {
        "seed_rows": len(seed_df),
        "feedback_rows": int(len(feedback_df)),
        "replicated_feedback_rows": int(len(replicated_feedback)),
        "augmented_train_rows": int(len(augmented_train)),
        "baseline_logreg_f1_macro": baseline_logreg_f1,
        "baseline_nb_f1_macro": baseline_nb_f1,
        "previous_best_nb_f1_macro": previous_best_nb_f1,
        "final_logreg_f1_macro": final_logreg_f1,
        "final_nb_f1_macro": final_nb_f1,
        "service_export_updated": improved,
        "feedback_by_label": dict(Counter(feedback_df["label_name"])),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Feedback Retrain Report",
        "",
        f"- Seed rows: **{len(seed_df)}**",
        f"- Accepted feedback rows: **{len(feedback_df)}**",
        f"- Replicated feedback rows: **{len(replicated_feedback)}**",
        f"- Augmented train rows: **{len(augmented_train)}**",
        "",
        "## Feedback by label",
        "",
    ]
    for label, count in Counter(feedback_df["label_name"]).most_common():
        report_lines.append(f"- {label}: {count}")
    report_lines.extend(
        [
            "",
            "## Metrics on original labeled holdout",
            "",
            f"- Baseline MultinomialNB F1_macro: **{baseline_nb_f1:.4f}**",
            f"- Previous best exported MultinomialNB F1_macro: **{previous_best_nb_f1:.4f}**",
            f"- Feedback-augmented MultinomialNB F1_macro: **{final_nb_f1:.4f}**",
            f"- Service export updated: **{'yes' if improved else 'no'}**",
            "",
            "## Baseline NB report",
            "",
            "```text",
            baseline_nb_report,
            "```",
            "",
            "## Feedback-augmented NB report",
            "",
            "```text",
            final_nb_report,
            "```",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    with start_mlflow_run(
        project_root=PROJECT_ROOT,
        experiment_name="aggregator_bot_feedback_loop",
        run_name="feedback_retrain_pipeline",
        tags={
            "pipeline": "ML.feedback_retrain_pipeline",
            "stage": "feedback_loop",
        },
    ) as mlflow_cfg:
        if mlflow_cfg:
            log_params(
                {
                    "seed_rows": len(seed_df),
                    "feedback_rows": len(feedback_df),
                    "replicated_feedback_rows": len(replicated_feedback),
                    "augmented_train_rows": len(augmented_train),
                }
            )
            log_metrics(
                {
                    "baseline_nb_f1_macro": baseline_nb_f1,
                    "previous_best_nb_f1_macro": previous_best_nb_f1,
                    "final_nb_f1_macro": final_nb_f1,
                    "service_export_updated": float(improved),
                }
            )
            for artifact_path in [
                ACCEPTED_CSV_PATH,
                FEEDBACK_SUMMARY_PATH,
                FEEDBACK_REPORT_PATH,
                AUGMENTED_TRAIN_PATH,
                SUMMARY_PATH,
                REPORT_PATH,
                PSEUDO_SUMMARY_PATH,
                PSEUDO_REPORT_PATH,
            ]:
                log_artifact_if_exists(artifact_path)
            print(f"[+] MLflow run logged to {mlflow_cfg['tracking_uri']} ({mlflow_cfg['experiment_name']}).")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
