from __future__ import annotations

import csv
import hashlib
import math
import pickle
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sklearn.exceptions import InconsistentVersionWarning


URL_RE = re.compile(r"http\S+|www\.\S+|t\.me/\S+")
NON_LETTERS_RE = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]")
MODEL_LABELS = [
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


@dataclass(slots=True)
class NewsItem:
    item_id: str
    source: str
    published_at: str
    text: str
    clean_text: str
    predicted_label: str
    predicted_confidence: float


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.lower()
    normalized = URL_RE.sub(" ", normalized)
    normalized = NON_LETTERS_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class NewsRecommender:
    def __init__(self, dataset_path: str | Path, imported_dir: str | Path | None = None) -> None:
        self.dataset_path = Path(dataset_path)
        self.imported_dir = Path(imported_dir) if imported_dir else None
        self.root_dir = Path(__file__).resolve().parent.parent
        self.items: list[NewsItem] = []
        self.item_positions: dict[str, int] = {}
        self.doc_vectors: list[dict[str, float]] = []
        self.doc_norms: list[float] = []
        self.idf: dict[str, float] = {}
        self.sources: list[str] = []
        self.available_topics: list[str] = MODEL_LABELS[:]
        self._classifier = self._load_classifier()
        self._dataset_snapshot: tuple[tuple[str, int, int], ...] = ()
        self._last_change_check_at = 0.0
        self.reload()

    def reload(self) -> None:
        dataset_files = self._dataset_files()
        self.items = self._load_items(dataset_files)
        if not self.items:
            raise ValueError(f"No news items found in {self.dataset_path}")

        self.item_positions = {item.item_id: index for index, item in enumerate(self.items)}
        self.doc_vectors, self.doc_norms, self.idf = self._build_tfidf(self.items)
        self.sources = sorted({item.source for item in self.items})
        self._dataset_snapshot = self._build_snapshot(dataset_files)
        self._last_change_check_at = time.time()

    def reload_if_changed(self, min_check_interval_seconds: int = 60) -> bool:
        now = time.time()
        if now - self._last_change_check_at < min_check_interval_seconds:
            return False
        self._last_change_check_at = now

        dataset_files = self._dataset_files()
        snapshot = self._build_snapshot(dataset_files)
        if snapshot == self._dataset_snapshot:
            return False

        self.reload()
        return True

    def _load_items(self, dataset_files: list[Path]) -> list[NewsItem]:
        items: list[NewsItem] = []
        seen_ids: set[str] = set()
        for csv_path in dataset_files:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                fallback_source = csv_path.stem
                for row in reader:
                    text = (row.get("text") or "").strip()
                    source = (
                        row.get("channel_short")
                        or row.get("channel")
                        or row.get("source")
                        or fallback_source
                    ).strip()
                    published_at = (row.get("date") or "").strip()
                    cleaned = clean_text(text)
                    if not cleaned:
                        continue
                    item_id = self._make_item_id(source or fallback_source, row, published_at, cleaned)
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)
                    predicted_label, predicted_confidence = self._predict_label(cleaned)
                    items.append(
                        NewsItem(
                            item_id=item_id,
                            source=source or fallback_source,
                            published_at=published_at,
                            text=text,
                            clean_text=cleaned,
                            predicted_label=predicted_label,
                            predicted_confidence=predicted_confidence,
                        )
                    )
        return items

    def _dataset_files(self) -> list[Path]:
        if self.dataset_path.is_dir():
            files = [
                path
                for path in sorted(self.dataset_path.glob("*.csv"))
                if path.name != "all_channels_combined.csv"
            ]
        else:
            files = [self.dataset_path]
        if self.imported_dir and self.imported_dir.exists():
            files.extend(sorted(self.imported_dir.glob("*.csv")))
        return files

    def _build_tfidf(
        self,
        items: list[NewsItem],
    ) -> tuple[list[dict[str, float]], list[float], dict[str, float]]:
        tokenized_docs = [item.clean_text.split() for item in items]
        doc_freq: Counter[str] = Counter()

        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))

        total_docs = len(tokenized_docs)
        idf = {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
        }

        vectors: list[dict[str, float]] = []
        norms: list[float] = []

        for tokens in tokenized_docs:
            tf = Counter(tokens)
            total_terms = sum(tf.values()) or 1
            vector = {
                token: (count / total_terms) * idf[token]
                for token, count in tf.items()
                if token in idf
            }
            norm = math.sqrt(sum(weight * weight for weight in vector.values()))
            vectors.append(vector)
            norms.append(norm)

        return vectors, norms, idf

    def get_item(self, item_id: str) -> NewsItem | None:
        position = self.item_positions.get(item_id)
        if position is not None:
            return self.items[position]
        return None

    def latest(
        self,
        limit: int = 5,
        sources: set[str] | None = None,
        boosted_sources: set[str] | None = None,
        source_boost: float = 0.0,
        diversify: bool = True,
    ) -> list[NewsItem]:
        filtered = [item for item in self.items if not sources or item.source in sources]
        boosted_sources = boosted_sources or set()
        scored = []
        for item in filtered:
            score = self._freshness_score(item) + 0.35 * item.predicted_confidence
            if item.source in boosted_sources:
                score += source_boost
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        candidates = [(score, self.item_positions[item.item_id]) for score, item in scored]
        if diversify:
            return self._diversified_top_items(candidates, limit=limit)
        return [self.items[position] for _, position in candidates[:limit]]

    def recommend_for_query(
        self,
        query: str,
        limit: int = 5,
        sources: set[str] | None = None,
        topics: set[str] | None = None,
        min_confidence: float = 0.0,
        boosted_sources: set[str] | None = None,
        source_boost: float = 0.0,
        exclude_ids: set[str] | None = None,
        diversify: bool = True,
    ) -> list[NewsItem]:
        query_vector, query_norm = self._vectorize_query(query)
        if not query_vector or not query_norm:
            return self.recommend_for_topics(
                limit=limit,
                sources=sources,
                topics=topics,
                min_confidence=min_confidence,
                boosted_sources=boosted_sources,
                source_boost=source_boost,
                exclude_ids=exclude_ids,
                diversify=diversify,
            )

        scores = self._score_against_corpus(query_vector, query_norm)
        return self._top_items(
            scores,
            limit=limit,
            sources=sources,
            topics=topics,
            min_confidence=min_confidence,
            boosted_sources=boosted_sources,
            source_boost=source_boost,
            exclude_ids=exclude_ids,
            diversify=diversify,
        )

    def recommend_for_topics(
        self,
        limit: int = 5,
        sources: set[str] | None = None,
        topics: set[str] | None = None,
        min_confidence: float = 0.0,
        boosted_sources: set[str] | None = None,
        source_boost: float = 0.0,
        exclude_ids: set[str] | None = None,
        diversify: bool = True,
    ) -> list[NewsItem]:
        if not topics:
            return self.latest(
                limit=limit,
                sources=sources,
                boosted_sources=boosted_sources,
                source_boost=source_boost,
                diversify=diversify,
            )
        scores = [0.0] * len(self.items)
        return self._top_items(
            scores,
            limit=limit,
            sources=sources,
            topics=topics,
            min_confidence=min_confidence,
            boosted_sources=boosted_sources,
            source_boost=source_boost,
            exclude_ids=exclude_ids,
            topical_mode=True,
            diversify=diversify,
        )

    def recommend_for_profile(
        self,
        liked_ids: Iterable[str],
        limit: int = 5,
        sources: set[str] | None = None,
        topics: set[str] | None = None,
        min_confidence: float = 0.0,
        boosted_sources: set[str] | None = None,
        source_boost: float = 0.0,
        exclude_ids: set[str] | None = None,
        diversify: bool = True,
    ) -> list[NewsItem]:
        liked_ids = [item_id for item_id in liked_ids if item_id in self.item_positions]
        if not liked_ids:
            return self.recommend_for_topics(
                limit=limit,
                sources=sources,
                topics=topics,
                min_confidence=min_confidence,
                boosted_sources=boosted_sources,
                source_boost=source_boost,
                exclude_ids=exclude_ids,
                diversify=diversify,
            )

        profile_vector = self._mean_vector(liked_ids)
        profile_norm = math.sqrt(sum(weight * weight for weight in profile_vector.values()))
        scores = self._score_against_corpus(profile_vector, profile_norm)

        return self._top_items(
            scores,
            limit=limit,
            sources=sources,
            topics=topics,
            min_confidence=min_confidence,
            boosted_sources=boosted_sources,
            source_boost=source_boost,
            exclude_ids=set(exclude_ids or set()).union(liked_ids),
            diversify=diversify,
        )

    def similar_to_item(
        self,
        item_id: str,
        limit: int = 5,
        sources: set[str] | None = None,
        topics: set[str] | None = None,
        min_confidence: float = 0.0,
        boosted_sources: set[str] | None = None,
        source_boost: float = 0.0,
        exclude_ids: set[str] | None = None,
        diversify: bool = True,
    ) -> list[NewsItem]:
        position = self.item_positions.get(item_id)
        if position is None:
            return []

        vector = self.doc_vectors[position]
        norm = self.doc_norms[position]
        scores = self._score_against_corpus(vector, norm)
        all_excluded = set(exclude_ids or set())
        all_excluded.add(item_id)
        return self._top_items(
            scores,
            limit=limit,
            sources=sources,
            topics=topics,
            min_confidence=min_confidence,
            boosted_sources=boosted_sources,
            source_boost=source_boost,
            exclude_ids=all_excluded,
            diversify=diversify,
        )

    def similarity_between(self, left_item_id: str, right_item_id: str) -> float:
        left = self.item_positions.get(left_item_id)
        right = self.item_positions.get(right_item_id)
        if left is None or right is None:
            return 0.0
        return self._vector_similarity(left, right)

    def _latest_filtered(
        self,
        limit: int,
        sources: set[str] | None,
        topics: set[str] | None,
        min_confidence: float,
    ) -> list[NewsItem]:
        filtered = [
            item for item in self.items
            if (not sources or item.source in sources)
            and (not topics or item.predicted_label in topics)
            and item.predicted_confidence >= min_confidence
        ]
        scored = [(self._freshness_score(item) + 0.45 * item.predicted_confidence, item) for item in filtered]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def _vectorize_query(self, query: str) -> tuple[dict[str, float], float]:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return {}, 0.0

        tokens = cleaned_query.split()
        tf = Counter(tokens)
        total_terms = sum(tf.values()) or 1
        vector = {
            token: (count / total_terms) * self.idf[token]
            for token, count in tf.items()
            if token in self.idf
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        return vector, norm

    def _mean_vector(self, item_ids: list[str]) -> dict[str, float]:
        aggregated: dict[str, float] = {}
        for item_id in item_ids:
            position = self.item_positions[item_id]
            for token, weight in self.doc_vectors[position].items():
                aggregated[token] = aggregated.get(token, 0.0) + weight

        count = len(item_ids) or 1
        return {token: weight / count for token, weight in aggregated.items()}

    def _score_against_corpus(self, vector: dict[str, float], norm: float) -> list[float]:
        scores: list[float] = []
        if not vector or not norm:
            return [0.0] * len(self.items)

        for doc_vector, doc_norm in zip(self.doc_vectors, self.doc_norms):
            if not doc_norm:
                scores.append(0.0)
                continue
            dot_product = 0.0
            for token, weight in vector.items():
                dot_product += weight * doc_vector.get(token, 0.0)
            scores.append(dot_product / (norm * doc_norm))
        return scores

    def _top_items(
        self,
        scores: list[float],
        limit: int,
        sources: set[str] | None,
        topics: set[str] | None,
        min_confidence: float,
        boosted_sources: set[str] | None,
        source_boost: float,
        exclude_ids: set[str] | None,
        topical_mode: bool = False,
        diversify: bool = True,
    ) -> list[NewsItem]:
        exclude_ids = exclude_ids or set()
        boosted_sources = boosted_sources or set()
        candidates: list[tuple[float, int]] = []

        for position, item in enumerate(self.items):
            if item.item_id in exclude_ids:
                continue
            if sources and item.source not in sources:
                continue
            if topics and item.predicted_label not in topics:
                continue
            if item.predicted_confidence < min_confidence:
                continue
            score = self._final_rank_score(
                item=item,
                base_score=scores[position],
                boosted_sources=boosted_sources,
                source_boost=source_boost,
                topical_mode=topical_mode,
            )
            candidates.append((score, position))

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        if diversify:
            return self._diversified_top_items(candidates, limit=limit)
        return [self.items[position] for _, position in candidates[:limit]]

    def _final_rank_score(
        self,
        item: NewsItem,
        base_score: float,
        boosted_sources: set[str],
        source_boost: float,
        topical_mode: bool,
    ) -> float:
        freshness = self._freshness_score(item)
        confidence = item.predicted_confidence
        boost = source_boost if item.source in boosted_sources else 0.0
        if topical_mode:
            return 0.58 * confidence + 0.32 * freshness + boost
        return 0.62 * base_score + 0.20 * confidence + 0.12 * freshness + boost

    def _diversified_top_items(self, candidates: list[tuple[float, int]], limit: int) -> list[NewsItem]:
        results: list[NewsItem] = []
        selected_positions: list[int] = []
        source_counts: Counter[str] = Counter()
        overflow: list[int] = []

        for _, position in candidates:
            item = self.items[position]
            if self._looks_like_duplicate(position, selected_positions):
                continue
            if source_counts[item.source] >= 1:
                overflow.append(position)
                continue
            results.append(item)
            selected_positions.append(position)
            source_counts[item.source] += 1
            if len(results) >= limit:
                return results

        for position in overflow:
            item = self.items[position]
            if self._looks_like_duplicate(position, selected_positions):
                continue
            results.append(item)
            selected_positions.append(position)
            if len(results) >= limit:
                break

        return results

    def _looks_like_duplicate(self, position: int, selected_positions: list[int]) -> bool:
        for selected_position in selected_positions:
            if self._is_near_duplicate(position, selected_position):
                return True
        return False

    def _is_near_duplicate(self, left: int, right: int) -> bool:
        left_item = self.items[left]
        right_item = self.items[right]
        if left_item.predicted_label != right_item.predicted_label:
            return False

        similarity = self._vector_similarity(left, right)
        if similarity >= 0.84:
            return True

        left_prefix = " ".join(left_item.clean_text.split()[:14])
        right_prefix = " ".join(right_item.clean_text.split()[:14])
        return bool(left_prefix and left_prefix == right_prefix)

    def _vector_similarity(self, left: int, right: int) -> float:
        left_vector = self.doc_vectors[left]
        right_vector = self.doc_vectors[right]
        left_norm = self.doc_norms[left]
        right_norm = self.doc_norms[right]
        if not left_norm or not right_norm:
            return 0.0

        if len(left_vector) > len(right_vector):
            left_vector, right_vector = right_vector, left_vector
        dot_product = 0.0
        for token, weight in left_vector.items():
            dot_product += weight * right_vector.get(token, 0.0)
        return dot_product / (left_norm * right_norm)

    @staticmethod
    def _build_snapshot(dataset_files: list[Path]) -> tuple[tuple[str, int, int], ...]:
        snapshot: list[tuple[str, int, int]] = []
        for path in dataset_files:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            snapshot.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(snapshot)

    @staticmethod
    def _make_item_id(source: str, row: dict[str, str], published_at: str, cleaned_text: str) -> str:
        raw_id = (row.get("id") or row.get("message_id") or "").strip()
        if raw_id:
            return f"{source}:{raw_id}"
        digest = hashlib.sha1(f"{source}|{published_at}|{cleaned_text[:200]}".encode("utf-8")).hexdigest()[:16]
        return f"{source}:{digest}"

    @staticmethod
    def _freshness_score(item: NewsItem) -> float:
        published_at = NewsRecommender._date_key(item.published_at)
        if published_at == datetime.min:
            return 0.0
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_days = max(0.0, (now - published_at).total_seconds() / 86400)
        return 1.0 / (1.0 + age_days / 5.0)

    def _load_classifier(self):
        config_dir = self.root_dir / "service" / "config"
        vectorizer_path = config_dir / "vectorizer.pkl"
        model_path = config_dir / "models" / "MultinomialNB.pkl"
        label_encoder_path = config_dir / "label_encoder.pkl"

        if not (vectorizer_path.exists() and model_path.exists() and label_encoder_path.exists()):
            return None

        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        with vectorizer_path.open("rb") as file:
            vectorizer = pickle.load(file)
        with model_path.open("rb") as file:
            model = pickle.load(file)
        with label_encoder_path.open("rb") as file:
            label_encoder = pickle.load(file)
        return vectorizer, model, label_encoder

    def _predict_label(self, cleaned_text: str) -> tuple[str, float]:
        if not self._classifier:
            return "Общее", 0.0

        vectorizer, model, label_encoder = self._classifier
        features = vectorizer.transform([cleaned_text])
        raw_label = str(label_encoder.inverse_transform(model.predict(features))[0])
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(features).max())
        try:
            label_index = int(float(raw_label))
            if 0 <= label_index < len(MODEL_LABELS):
                return MODEL_LABELS[label_index], confidence
        except ValueError:
            pass
        return raw_label, confidence

    @staticmethod
    def _date_key(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min
