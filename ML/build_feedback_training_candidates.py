from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.analytics_report import load_events  # noqa: E402
from bot.recommender import NewsRecommender  # noqa: E402


ML_DIR = PROJECT_ROOT / "ML"
OUTPUT_DIR = ML_DIR / "feedback_loop_artifacts"
EVENTS_PATH = PROJECT_ROOT / "bot_data" / "events.jsonl"
PROFILES_PATH = PROJECT_ROOT / "bot_data" / "user_profiles.json"
DATASET_PATH = PROJECT_ROOT / "data"
IMPORTED_CHANNELS_PATH = PROJECT_ROOT / "bot_data" / "imported_channels"

CANDIDATES_CSV_PATH = OUTPUT_DIR / "feedback_training_candidates.csv"
ACCEPTED_CSV_PATH = OUTPUT_DIR / "feedback_training_accepted.csv"
REJECTED_CSV_PATH = OUTPUT_DIR / "feedback_training_rejected.csv"
SUMMARY_PATH = OUTPUT_DIR / "feedback_summary.json"
REPORT_PATH = OUTPUT_DIR / "feedback_report.md"


TOPIC_TO_ID = {
    "Общее": 0,
    "Наука и техника": 1,
    "ИТ и телекоммуникации": 2,
    "Общество, государство, политика": 3,
    "Экономика": 4,
    "Медицина": 5,
    "Искусство и культура": 6,
    "Развлечения": 7,
    "Спорт": 8,
    "История": 9,
    "Происшествия": 10,
}

HARD_POSITIVE_CONFIDENCE = 0.24
SOFT_POSITIVE_CONFIDENCE = 0.72
SOFT_POSITIVE_MIN_SCORE = 0.65


def load_profiles() -> dict[str, dict]:
    if not PROFILES_PATH.exists():
        return {}
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def build_recommender() -> NewsRecommender:
    recommender = NewsRecommender(DATASET_PATH)
    if IMPORTED_CHANNELS_PATH.exists():
        recommender.dataset_path = DATASET_PATH
        recommender.reload()
    return recommender


def score_candidate(
    *,
    item,
    like_count: int,
    dislike_count: int,
    shown_count: int,
    selected_topic_matches: int,
    search_shown_count: int,
) -> tuple[float, str, bool]:
    positive_score = 0.0
    reason_parts: list[str] = []

    if like_count > 0:
        positive_score += 1.2 * like_count
        reason_parts.append("liked")
    if selected_topic_matches > 0:
        positive_score += 0.35 * selected_topic_matches
        reason_parts.append("shown_in_selected_topic")
    if search_shown_count > 0:
        positive_score += 0.15 * search_shown_count
        reason_parts.append("search_related")
    if shown_count > 1:
        positive_score += min(0.2, 0.05 * (shown_count - 1))
        reason_parts.append("repeat_exposure")

    negative_score = 1.1 * dislike_count
    net_score = positive_score - negative_score

    hard_positive = like_count > 0 and dislike_count == 0 and item.predicted_confidence >= HARD_POSITIVE_CONFIDENCE
    soft_positive = (
        like_count == 0
        and dislike_count == 0
        and selected_topic_matches > 0
        and item.predicted_confidence >= SOFT_POSITIVE_CONFIDENCE
        and net_score >= SOFT_POSITIVE_MIN_SCORE
    )
    accepted = hard_positive or soft_positive
    reason = ", ".join(reason_parts) if reason_parts else "weak_signal"
    return round(net_score, 4), reason, accepted


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events(EVENTS_PATH)
    profiles = load_profiles()
    recommender = build_recommender()

    per_item: dict[str, dict] = defaultdict(
        lambda: {
            "like_count": 0,
            "dislike_count": 0,
            "shown_count": 0,
            "search_shown_count": 0,
            "selected_topic_matches": 0,
            "user_ids": set(),
            "queries": Counter(),
        }
    )

    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        user_id = str(event.get("user_id"))
        profile = profiles.get(user_id, {})
        selected_topics = set(profile.get("selected_topics", []))

        if event_type == "feed_shown":
            item_ids = payload.get("item_ids", []) or []
            item_topics = payload.get("item_topics", []) or []
            query_text = payload.get("query_text")
            for index, item_id in enumerate(item_ids):
                record = per_item[str(item_id)]
                record["shown_count"] += 1
                record["user_ids"].add(user_id)
                topic = item_topics[index] if index < len(item_topics) else None
                if topic and topic in selected_topics:
                    record["selected_topic_matches"] += 1
                if query_text:
                    record["search_shown_count"] += 1
                    record["queries"][str(query_text)] += 1
        elif event_type == "feedback_like":
            item_id = payload.get("item_id")
            if item_id:
                record = per_item[str(item_id)]
                record["like_count"] += 1
                record["user_ids"].add(user_id)
        elif event_type == "feedback_dislike":
            item_id = payload.get("item_id")
            if item_id:
                record = per_item[str(item_id)]
                record["dislike_count"] += 1
                record["user_ids"].add(user_id)

    rows: list[dict] = []
    for item_id, signals in per_item.items():
        item = recommender.get_item(item_id)
        if item is None:
            continue
        if item.predicted_label not in TOPIC_TO_ID:
            continue

        net_score, reason, accepted = score_candidate(
            item=item,
            like_count=signals["like_count"],
            dislike_count=signals["dislike_count"],
            shown_count=signals["shown_count"],
            selected_topic_matches=signals["selected_topic_matches"],
            search_shown_count=signals["search_shown_count"],
        )
        rows.append(
            {
                "item_id": item.item_id,
                "source": item.source,
                "published_at": item.published_at,
                "predicted_label": item.predicted_label,
                "topic": TOPIC_TO_ID[item.predicted_label],
                "predicted_confidence": round(float(item.predicted_confidence), 6),
                "text": item.text,
                "clean_text": item.clean_text,
                "like_count": signals["like_count"],
                "dislike_count": signals["dislike_count"],
                "shown_count": signals["shown_count"],
                "search_shown_count": signals["search_shown_count"],
                "selected_topic_matches": signals["selected_topic_matches"],
                "unique_users": len(signals["user_ids"]),
                "top_queries": " | ".join(query for query, _ in signals["queries"].most_common(3)),
                "weak_feedback_score": net_score,
                "accept_reason": reason,
                "accepted_for_training": accepted,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        summary = {
            "total_candidates": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 0,
        }
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT_PATH.write_text("# Feedback Loop Report\n\nПока нет кандидатов из событий бота.\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    df = df.sort_values(
        by=["accepted_for_training", "weak_feedback_score", "predicted_confidence", "like_count", "shown_count"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    accepted_df = df[df["accepted_for_training"]].copy()
    rejected_df = df[~df["accepted_for_training"]].copy()

    df.to_csv(CANDIDATES_CSV_PATH, index=False, encoding="utf-8")
    accepted_df.to_csv(ACCEPTED_CSV_PATH, index=False, encoding="utf-8")
    rejected_df.to_csv(REJECTED_CSV_PATH, index=False, encoding="utf-8")

    summary = {
        "total_candidates": int(len(df)),
        "accepted_candidates": int(len(accepted_df)),
        "rejected_candidates": int(len(rejected_df)),
        "accepted_by_label": (
            {str(label): int(count) for label, count in accepted_df["predicted_label"].value_counts().items()}
            if not accepted_df.empty
            else {}
        ),
        "rejected_by_label": (
            {str(label): int(count) for label, count in rejected_df["predicted_label"].value_counts().items()}
            if not rejected_df.empty
            else {}
        ),
        "mean_feedback_score": round(float(df["weak_feedback_score"].mean()), 4),
        "mean_accepted_confidence": round(float(accepted_df["predicted_confidence"].mean()), 4) if not accepted_df.empty else 0.0,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Feedback Loop Report",
        "",
        f"- Total candidates: **{len(df)}**",
        f"- Accepted for training: **{len(accepted_df)}**",
        f"- Rejected: **{len(rejected_df)}**",
        f"- Mean feedback score: **{summary['mean_feedback_score']}**",
        f"- Mean accepted confidence: **{summary['mean_accepted_confidence']}**",
        "",
        "## Accepted by label",
        "",
    ]
    for label, count in Counter(accepted_df["predicted_label"]).most_common():
        report_lines.append(f"- {label}: {count}")
    report_lines.extend(
        [
            "",
            "## Top accepted examples",
            "",
        ]
    )
    for row in accepted_df.head(10).itertuples(index=False):
        report_lines.append(
            f"- `{row.item_id}` | {row.predicted_label} | conf={row.predicted_confidence} | "
            f"score={row.weak_feedback_score} | reason={row.accept_reason}"
        )
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
