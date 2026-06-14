from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable


NON_TEXT_RE = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]")


@dataclass(slots=True)
class EventCluster:
    event_id: str
    topic: str
    title: str
    summary: str
    anchor_item_id: str
    item_ids: list[str]
    sources: list[str]
    source_count: int
    item_count: int
    published_from: str
    published_to: str


def build_event_clusters(
    items: list,
    item_positions: dict[str, int],
    similarity_between: Callable[[str, str], float],
    date_key: Callable[[str], datetime],
    max_window_hours: int = 36,
    similarity_threshold: float = 0.72,
) -> list[EventCluster]:
    events: list[EventCluster] = []
    items_by_topic: dict[str, list] = {}
    for item in items:
        items_by_topic.setdefault(item.predicted_label, []).append(item)

    window = timedelta(hours=max_window_hours)
    for topic, topic_items in items_by_topic.items():
        ordered = sorted(topic_items, key=lambda item: _safe_date(date_key, item.published_at), reverse=True)
        assigned: set[str] = set()
        for index, item in enumerate(ordered):
            if item.item_id in assigned:
                continue
            anchor_dt = _safe_date(date_key, item.published_at)
            member_items = [item]
            assigned.add(item.item_id)
            for candidate in ordered[index + 1:]:
                if candidate.item_id in assigned:
                    continue
                candidate_dt = _safe_date(date_key, candidate.published_at)
                if anchor_dt != datetime.min.replace(tzinfo=timezone.utc) and candidate_dt != datetime.min.replace(tzinfo=timezone.utc):
                    if anchor_dt - candidate_dt > window:
                        break
                if _same_event_heuristic(item, candidate, similarity_between, similarity_threshold):
                    member_items.append(candidate)
                    assigned.add(candidate.item_id)

            event = _make_event_cluster(member_items, topic, date_key)
            events.append(event)

    events.sort(key=lambda event: _safe_date(date_key, event.published_to), reverse=True)
    return events


def save_event_clusters(path: Path, events: Iterable[EventCluster]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "events": [asdict(event) for event in events]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _same_event_heuristic(anchor, candidate, similarity_between: Callable[[str, str], float], threshold: float) -> bool:
    if anchor.item_id == candidate.item_id:
        return True
    if anchor.predicted_label != candidate.predicted_label:
        return False
    similarity = similarity_between(anchor.item_id, candidate.item_id)
    if similarity >= threshold:
        return True
    anchor_prefix = " ".join(anchor.clean_text.split()[:16])
    candidate_prefix = " ".join(candidate.clean_text.split()[:16])
    if anchor_prefix and anchor_prefix == candidate_prefix:
        return True
    overlap = _token_overlap(anchor.clean_text, candidate.clean_text)
    return overlap >= 0.62


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.split() if len(token) >= 4}
    right_tokens = {token for token in right.split() if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens.intersection(right_tokens)
    base = min(len(left_tokens), len(right_tokens))
    return len(intersection) / base if base else 0.0


def _make_event_cluster(member_items: list, topic: str, date_key: Callable[[str], datetime]) -> EventCluster:
    ordered = sorted(
        member_items,
        key=lambda item: (
            -float(item.predicted_confidence),
            -_safe_date(date_key, item.published_at).timestamp() if _safe_date(date_key, item.published_at) != datetime.min.replace(tzinfo=timezone.utc) else 0.0,
        ),
    )
    anchor = ordered[0]
    timeline = sorted(member_items, key=lambda item: _safe_date(date_key, item.published_at))
    published_from = timeline[0].published_at if timeline else ""
    published_to = timeline[-1].published_at if timeline else ""
    title = _build_title(anchor.text, anchor.clean_text)
    summary = _build_summary(anchor.clean_text)
    sources = []
    for item in member_items:
        if item.source not in sources:
            sources.append(item.source)
    digest = hashlib.sha1(f"{topic}|{anchor.item_id}|{published_to}".encode("utf-8")).hexdigest()[:16]
    return EventCluster(
        event_id=f"event:{digest}",
        topic=topic,
        title=title,
        summary=summary,
        anchor_item_id=anchor.item_id,
        item_ids=[item.item_id for item in member_items],
        sources=sources,
        source_count=len(sources),
        item_count=len(member_items),
        published_from=published_from,
        published_to=published_to,
    )


def _build_title(raw_text: str, clean_text: str) -> str:
    candidate = raw_text.strip().splitlines()[0].strip() if raw_text else ""
    candidate = re.sub(r"https?://\S+", "", candidate)
    candidate = NON_TEXT_RE.sub(" ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -")
    if len(candidate) < 24:
        words = clean_text.split()
        candidate = " ".join(words[:14]).strip()
    candidate = candidate[:120].rstrip(" .,;:")
    if not candidate:
        return "Событие"
    return candidate[:1].upper() + candidate[1:] + ("..." if len(candidate) >= 120 else "")


def _build_summary(clean_text: str) -> str:
    text = re.sub(r"\s+", " ", clean_text).strip()
    if len(text) > 220:
        text = text[:217].rstrip(" .,;:") + "..."
    return text


def _safe_date(date_key: Callable[[str], datetime], value: str) -> datetime:
    dt = date_key(value)
    if dt == datetime.min:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
