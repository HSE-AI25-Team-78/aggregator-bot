from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.recommender import NewsRecommender  # noqa: E402

ML_DIR = PROJECT_ROOT / "ML"
RAW_LABELS_PATH = ML_DIR / "raw_posts_labeled.csv"
OUTPUT_DIR = ML_DIR / "recommendation_eval_artifacts"
TEMP_CORPUS_PATH = OUTPUT_DIR / "eval_corpus.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
DEFAULT_RECOMMENDER_CONFIG = PROJECT_ROOT / "bot" / "config" / "recommender_config.json"

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
K = 5
MAX_SEEDS_PER_TOPIC = 15
UNCERTAIN_LABEL = "Неуверенно"


def load_labeled_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_LABELS_PATH)
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["topic_name"] = df["topic"].astype(int).map(lambda idx: LABELS[idx])
    df["id"] = df.index.astype(str)
    return df


def build_temp_corpus(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "id": row["id"],
                "channel_short": row.get("channel_short", "labeled_seed"),
                "date": row.get("date", ""),
                "text": row["text"],
            }
        )
    with TEMP_CORPUS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "channel_short", "date", "text"])
        writer.writeheader()
        writer.writerows(rows)


def precision_at_k(labels: list[str], expected: str) -> float:
    if not labels:
        return 0.0
    return sum(1 for label in labels if label == expected) / len(labels)


def evaluate_recommender(config_path: Path | None = None) -> dict:
    df = load_labeled_dataset()
    build_temp_corpus(df)
    recommender = NewsRecommender(TEMP_CORPUS_PATH, config_path=config_path or DEFAULT_RECOMMENDER_CONFIG)

    true_topic_by_item_id: dict[str, str] = {}
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        item_id = f"{row['channel_short']}:{row['id']}"
        topic = row["topic_name"]
        true_topic_by_item_id[item_id] = topic
        grouped_ids[topic].append(item_id)

    per_topic_feed_precision: list[tuple[str, float]] = []
    per_topic_similarity_precision: list[tuple[str, float]] = []
    topic_feed_scores: list[float] = []
    similarity_scores: list[float] = []
    uncertain_predictions = 0

    for item in recommender.items:
        if item.predicted_label == UNCERTAIN_LABEL:
            uncertain_predictions += 1

    for topic in LABELS:
        recs = recommender.recommend_for_topics(
            limit=K,
            topics={topic},
            min_confidence=0.0,
            diversify=False,
        )
        topic_labels = [true_topic_by_item_id.get(item.item_id, "") for item in recs]
        topic_score = precision_at_k(topic_labels, topic)
        per_topic_feed_precision.append((topic, round(topic_score, 4)))
        if recs:
            topic_feed_scores.append(topic_score)

        seed_scores: list[float] = []
        for seed_item_id in grouped_ids.get(topic, [])[:MAX_SEEDS_PER_TOPIC]:
            similar = recommender.similar_to_item(
                seed_item_id,
                limit=K,
                min_confidence=0.0,
                diversify=False,
            )
            similar_labels = [true_topic_by_item_id.get(item.item_id, "") for item in similar]
            if similar_labels:
                seed_scores.append(precision_at_k(similar_labels, topic))
        avg_seed_score = sum(seed_scores) / len(seed_scores) if seed_scores else 0.0
        per_topic_similarity_precision.append((topic, round(avg_seed_score, 4)))
        if seed_scores:
            similarity_scores.append(avg_seed_score)

    summary = {
        "overall": {
            "topic_feed_precision_at_5": round(sum(topic_feed_scores) / len(topic_feed_scores), 4) if topic_feed_scores else 0.0,
            "similarity_precision_at_5": round(sum(similarity_scores) / len(similarity_scores), 4) if similarity_scores else 0.0,
            "topic_coverage": round(sum(1 for _, score in per_topic_feed_precision if score > 0) / len(LABELS), 4),
            "uncertain_prediction_rate": round(uncertain_predictions / len(recommender.items), 4) if recommender.items else 0.0,
            "topics_evaluated": len(LABELS),
            "k": K,
        },
        "per_topic_topic_feed_precision": per_topic_feed_precision,
        "per_topic_similarity_precision": per_topic_similarity_precision,
    }
    if config_path is not None:
        summary["config_path"] = str(Path(config_path).resolve())
    return summary


def write_artifacts(summary: dict) -> None:
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Offline Recommendation Evaluation",
        "",
        "Оценка считается на `ML/raw_posts_labeled.csv` как offline proxy для двух сценариев:",
        "- `recommend_for_topics`: precision@5 по истинной теме",
        "- `similar_to_item`: average precision@5 по истинной теме для seed-новостей",
        "",
        f"- Topic feed precision@5: **{summary['overall']['topic_feed_precision_at_5']:.4f}**",
        f"- Similarity precision@5: **{summary['overall']['similarity_precision_at_5']:.4f}**",
        f"- Topic coverage: **{summary['overall']['topic_coverage']:.4f}**",
        f"- Uncertain prediction rate: **{summary['overall']['uncertain_prediction_rate']:.4f}**",
        "",
        "## Per-topic topic feed precision@5",
        "",
    ]
    for topic, score in summary["per_topic_topic_feed_precision"]:
        lines.append(f"- {topic}: {score:.4f}")

    lines.extend(["", "## Per-topic similarity precision@5", ""])
    for topic, score in summary["per_topic_similarity_precision"]:
        lines.append(f"- {topic}: {score:.4f}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = evaluate_recommender()
    write_artifacts(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
