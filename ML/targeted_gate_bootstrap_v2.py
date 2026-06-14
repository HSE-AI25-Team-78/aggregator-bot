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
from bot.recommender import NewsRecommender, UNCERTAIN_LABEL  # noqa: E402
from ML.pseudo_label_pipeline import (  # noqa: E402
    LABELS,
    RAW_LABELS_PATH,
    SERVICE_CONFIG_DIR,
    SERVICE_MODELS_DIR,
    build_models,
    clean_text,
    evaluate_model,
    read_previous_best_nb_f1,
    save_pickle,
    write_service_manifest,
)


DATASET_PATH = PROJECT_ROOT / "data"
IMPORTED_DIR = PROJECT_ROOT / "bot_data" / "imported_channels"
OUTPUT_DIR = PROJECT_ROOT / "ML" / "targeted_gate_bootstrap_v2_artifacts"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
SELECTED_PATH = OUTPUT_DIR / "selected_rows.csv"
AUGMENTED_TRAIN_PATH = OUTPUT_DIR / "augmented_train_dataset.csv"

TARGET_RULES = {
    "ИТ и телекоммуникации": {
        "sources": {"habr_com", "codeblog", "devschacht", "d_code", "xakaton_it"},
        "min_confidence": 0.72,
        "limit": 700,
    },
    "История": {
        "sources": {"historyrussi", "postnauka", "nplusone", "bbcrussian", "meduzalive"},
        "min_confidence": 0.78,
        "limit": 300,
    },
    "Развлечения": {
        "sources": {"kinostro4ka", "karoartcinema", "mosrutop", "meduzalive", "bbcrussian"},
        "min_confidence": 0.72,
        "limit": 450,
    },
    "Искусство и культура": {
        "sources": {"karoartcinema", "mosrutop", "postnauka", "bbcrussian", "meduzalive"},
        "min_confidence": 0.72,
        "limit": 450,
    },
}


def load_seed_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_LABELS_PATH)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= 15].copy()
    df["label_name"] = df["topic"].astype(int).map(lambda idx: LABELS[idx])
    return df


def existing_clean_texts(seed_df: pd.DataFrame) -> set[str]:
    return set(seed_df["clean_text"].astype(str))


def collect_targeted_rows(seed_df: pd.DataFrame) -> pd.DataFrame:
    recommender = NewsRecommender(DATASET_PATH, imported_dir=IMPORTED_DIR)
    existing = existing_clean_texts(seed_df)

    rows: list[dict] = []
    for item in recommender.items:
        if item.predicted_label == UNCERTAIN_LABEL:
            continue
        if item.clean_text in existing:
            continue
        rules = TARGET_RULES.get(item.predicted_label)
        if not rules:
            continue
        if item.source not in rules["sources"]:
            continue
        if item.predicted_confidence < rules["min_confidence"]:
            continue
        rows.append(
            {
                "item_id": item.item_id,
                "channel_short": item.source,
                "date": item.published_at,
                "text": item.text,
                "clean_text": item.clean_text,
                "label_name": item.predicted_label,
                "topic": LABELS.index(item.predicted_label),
                "predicted_confidence": round(float(item.predicted_confidence), 6),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    capped_parts = []
    for label, group in df.groupby("label_name"):
        limit = TARGET_RULES[label]["limit"]
        capped_parts.append(group.sort_values("predicted_confidence", ascending=False).head(limit))
    return pd.concat(capped_parts, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_dataset()
    selected_df = collect_targeted_rows(seed_df)

    train_df, test_df = train_test_split(
        seed_df,
        test_size=0.15,
        stratify=seed_df["label_name"],
        random_state=42,
    )

    baseline_models = build_models(train_df)
    baseline_nb_f1, baseline_nb_report = evaluate_model(
        baseline_models.vectorizer, baseline_models.nb, baseline_models.label_encoder, test_df
    )
    previous_best_nb_f1 = read_previous_best_nb_f1()

    if selected_df.empty:
        summary = {
            "seed_rows": len(seed_df),
            "selected_rows": 0,
            "baseline_nb_f1_macro": baseline_nb_f1,
            "final_nb_f1_macro": baseline_nb_f1,
            "service_export_updated": False,
            "reason": "no_targeted_rows",
        }
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_PATH.write_text("# Targeted Gate Bootstrap V2\n\nНе удалось собрать новые targeted rows.\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    selected_df.to_csv(SELECTED_PATH, index=False, encoding="utf-8")
    augmented_train = pd.concat(
        [
            train_df[["channel_short", "date", "topic", "text", "clean_text", "label_name"]],
            selected_df[["channel_short", "date", "topic", "text", "clean_text", "label_name"]],
        ],
        ignore_index=True,
    )
    augmented_train.to_csv(AUGMENTED_TRAIN_PATH, index=False, encoding="utf-8")

    final_models = build_models(augmented_train)
    final_nb_f1, final_nb_report = evaluate_model(
        final_models.vectorizer, final_models.nb, final_models.label_encoder, test_df
    )

    improved = final_nb_f1 > max(previous_best_nb_f1, baseline_nb_f1)
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
        "selected_rows": int(len(selected_df)),
        "augmented_train_rows": int(len(augmented_train)),
        "baseline_nb_f1_macro": baseline_nb_f1,
        "previous_best_nb_f1_macro": previous_best_nb_f1,
        "final_nb_f1_macro": final_nb_f1,
        "service_export_updated": improved,
        "selected_by_label": {str(label): int(count) for label, count in selected_df["label_name"].value_counts().items()},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Targeted Gate Bootstrap V2",
        "",
        f"- Seed rows: **{len(seed_df)}**",
        f"- Selected rows: **{len(selected_df)}**",
        f"- Augmented train rows: **{len(augmented_train)}**",
        f"- Baseline NB F1_macro: **{baseline_nb_f1:.4f}**",
        f"- Previous best NB F1_macro: **{previous_best_nb_f1:.4f}**",
        f"- Final NB F1_macro: **{final_nb_f1:.4f}**",
        f"- Service export updated: **{'yes' if improved else 'no'}**",
        "",
        "## Selected by label",
        "",
    ]
    for label, count in Counter(selected_df["label_name"]).most_common():
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Baseline NB report",
            "",
            "```text",
            baseline_nb_report,
            "```",
            "",
            "## Final NB report",
            "",
            "```text",
            final_nb_report,
            "```",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    with start_mlflow_run(
        project_root=PROJECT_ROOT,
        experiment_name="aggregator_bot_targeted_gate_bootstrap_v2",
        run_name="targeted_gate_bootstrap_v2",
        tags={
            "pipeline": "ML.targeted_gate_bootstrap_v2",
            "stage": "targeted_self_training",
        },
    ) as mlflow_cfg:
        if mlflow_cfg:
            log_params(
                {
                    "seed_rows": len(seed_df),
                    "selected_rows": len(selected_df),
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
            for artifact_path in [SELECTED_PATH, AUGMENTED_TRAIN_PATH, SUMMARY_PATH, REPORT_PATH]:
                log_artifact_if_exists(artifact_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
