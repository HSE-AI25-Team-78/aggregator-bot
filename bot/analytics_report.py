from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import csv


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EVENTS_PATH = ROOT_DIR / "bot_data" / "events.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "bot_data" / "analytics_summary.json"
DEFAULT_REFRESH_STATUS_PATH = ROOT_DIR / "bot_data" / "refresh_status.json"
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_IMPORTED_DIR = ROOT_DIR / "bot_data" / "imported_channels"


def load_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []
    events: list[dict] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def parse_event_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return sum(1 for _ in reader)


def build_corpus_health() -> dict:
    base_files = [path for path in sorted(DEFAULT_DATA_DIR.glob("*.csv")) if path.name != "all_channels_combined.csv"]
    imported_files = sorted(DEFAULT_IMPORTED_DIR.glob("*.csv")) if DEFAULT_IMPORTED_DIR.exists() else []
    refresh_status = load_json(DEFAULT_REFRESH_STATUS_PATH)
    entries = refresh_status.get("entries", [])

    total_base_items = sum(_count_csv_rows(path) for path in base_files)
    total_imported_items = sum(_count_csv_rows(path) for path in imported_files)
    retention_windows = Counter()
    active_sources = 0
    error_sources = 0
    for entry in entries:
        retention_days = entry.get("retention_days")
        if retention_days is not None:
            retention_windows[str(retention_days)] += 1
        kept = entry.get("kept_after_retention")
        if isinstance(kept, int) and kept > 0:
            active_sources += 1
        if entry.get("status") not in {"imported", "skipped"}:
            error_sources += 1

    return {
        "last_refresh_at": refresh_status.get("completed_at"),
        "refresh_summary": refresh_status.get("summary", {}),
        "base_sources": len(base_files),
        "custom_sources": len(imported_files),
        "active_sources": active_sources,
        "error_sources": error_sources,
        "total_items": total_base_items + total_imported_items,
        "base_items": total_base_items,
        "custom_items": total_imported_items,
        "retention_windows": sorted(retention_windows.items(), key=lambda pair: int(pair[0])),
        "largest_sources": sorted(
            (
                (path.stem, _count_csv_rows(path))
                for path in base_files + imported_files
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:12],
    }


def build_summary(events: list[dict]) -> dict:
    event_counts = Counter(event.get("event_type", "unknown") for event in events)
    unique_users = {event.get("user_id") for event in events if event.get("user_id") is not None}
    dated_events = [(event, parse_event_timestamp(event.get("timestamp"))) for event in events]
    dated_events = [(event, ts) for event, ts in dated_events if ts is not None]

    feed_events = [event for event in events if event.get("event_type") == "feed_shown"]
    total_shown = sum(int(event.get("payload", {}).get("item_count", 0)) for event in feed_events)
    like_count = event_counts.get("feedback_like", 0)
    dislike_count = event_counts.get("feedback_dislike", 0)
    similar_count = event_counts.get("open_similar", 0)
    search_count = event_counts.get("search_query", 0)

    ctr_like = like_count / total_shown if total_shown else 0.0
    ctr_dislike = dislike_count / total_shown if total_shown else 0.0
    ctr_similar = similar_count / total_shown if total_shown else 0.0

    top_queries = Counter(
        event.get("payload", {}).get("query", "").strip()
        for event in events
        if event.get("event_type") == "search_query" and event.get("payload", {}).get("query")
    )
    imported_channels = Counter(
        event.get("payload", {}).get("channel", "")
        for event in events
        if event.get("event_type") == "custom_channel_import" and event.get("payload", {}).get("status") == "imported"
    )
    shown_sources = Counter()
    shown_topics = Counter()
    for event in feed_events:
        payload = event.get("payload", {})
        shown_sources.update(source for source in payload.get("item_sources", []) if source)
        shown_topics.update(topic for topic in payload.get("item_topics", []) if topic)

    likes_by_topic = Counter(
        event.get("payload", {}).get("topic", "")
        for event in events
        if event.get("event_type") == "feedback_like" and event.get("payload", {}).get("topic")
    )
    dislikes_by_topic = Counter(
        event.get("payload", {}).get("topic", "")
        for event in events
        if event.get("event_type") == "feedback_dislike" and event.get("payload", {}).get("topic")
    )
    onboarding_started = event_counts.get("onboarding_started", 0)
    onboarding_completed = event_counts.get("onboarding_completed", 0)
    onboarding_completion_rate = onboarding_completed / onboarding_started if onboarding_started else 0.0
    feedback_events = like_count + dislike_count + similar_count
    feed_to_feedback_rate = feedback_events / len(feed_events) if feed_events else 0.0

    user_days: dict[int, set[str]] = {}
    first_seen_by_user: dict[int, str] = {}
    last_seen_by_user: dict[int, str] = {}
    daily_active_users: Counter[str] = Counter()
    for event, ts in dated_events:
        user_id = event.get("user_id")
        if user_id is None:
            continue
        day_key = ts.date().isoformat()
        daily_active_users[day_key] += 0
        user_days.setdefault(int(user_id), set()).add(day_key)
        daily_active_users[day_key] = len({uid for uid, days in user_days.items() if day_key in days})
        first_seen_by_user[user_id] = min(first_seen_by_user.get(user_id, day_key), day_key)
        last_seen_by_user[user_id] = max(last_seen_by_user.get(user_id, day_key), day_key)

    sorted_days = sorted(daily_active_users)
    latest_day = sorted_days[-1] if sorted_days else None
    latest_day_dt = datetime.fromisoformat(latest_day) if latest_day else None
    dau = daily_active_users.get(latest_day, 0) if latest_day else 0
    wau = 0
    mau = 0
    if latest_day_dt:
        week_start = (latest_day_dt - timedelta(days=6)).date().isoformat()
        month_start = (latest_day_dt - timedelta(days=29)).date().isoformat()
        wau_users = set()
        mau_users = set()
        for user_id, days in user_days.items():
            for day in days:
                if day >= week_start:
                    wau_users.add(user_id)
                if day >= month_start:
                    mau_users.add(user_id)
        wau = len(wau_users)
        mau = len(mau_users)

    one_day_returned = 0
    seven_day_returned = 0
    cohort_size = len(user_days)
    for _, days in user_days.items():
        ordered = sorted(datetime.fromisoformat(day).date() for day in days)
        if len(ordered) < 2:
            continue
        first_day = ordered[0]
        deltas = {(day - first_day).days for day in ordered[1:]}
        if any(delta >= 1 for delta in deltas):
            one_day_returned += 1
        if any(delta >= 7 for delta in deltas):
            seven_day_returned += 1
    day1_retention = one_day_returned / cohort_size if cohort_size else 0.0
    day7_retention = seven_day_returned / cohort_size if cohort_size else 0.0

    corpus_health = build_corpus_health()

    return {
        "total_events": len(events),
        "unique_users": len(unique_users),
        "event_counts": dict(event_counts),
        "recommendation_metrics": {
            "feeds_shown": len(feed_events),
            "items_shown": total_shown,
            "like_events": like_count,
            "dislike_events": dislike_count,
            "similar_open_events": similar_count,
            "search_events": search_count,
            "like_rate_per_item": round(ctr_like, 4),
            "dislike_rate_per_item": round(ctr_dislike, 4),
            "similar_open_rate_per_item": round(ctr_similar, 4),
            "feed_to_feedback_rate": round(feed_to_feedback_rate, 4),
        },
        "onboarding_metrics": {
            "started": onboarding_started,
            "completed": onboarding_completed,
            "completion_rate": round(onboarding_completion_rate, 4),
        },
        "retention_metrics": {
            "dau": dau,
            "wau": wau,
            "mau": mau,
            "day1_retention": round(day1_retention, 4),
            "day7_retention": round(day7_retention, 4),
        },
        "daily_active_users": sorted(daily_active_users.items()),
        "content_breakdown": {
            "shown_sources": shown_sources.most_common(15),
            "shown_topics": shown_topics.most_common(15),
            "likes_by_topic": likes_by_topic.most_common(15),
            "dislikes_by_topic": dislikes_by_topic.most_common(15),
        },
        "top_queries": top_queries.most_common(10),
        "imported_channels": imported_channels.most_common(10),
        "corpus_health": corpus_health,
    }


def main() -> None:
    events = load_events(DEFAULT_EVENTS_PATH)
    summary = build_summary(events)
    DEFAULT_OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
