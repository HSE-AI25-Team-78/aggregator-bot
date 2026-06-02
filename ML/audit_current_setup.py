from __future__ import annotations

import csv
import pickle
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ML_DATASET_PATH = PROJECT_ROOT / "ML" / "raw_posts_labeled.csv"
SERVICE_CONFIG_DIR = PROJECT_ROOT / "service" / "config"
SERVICE_LABEL_ENCODER_PATH = SERVICE_CONFIG_DIR / "label_encoder.pkl"
SERVICE_VECTORIZER_PATH = SERVICE_CONFIG_DIR / "vectorizer.pkl"
SERVICE_MODEL_DIR = SERVICE_CONFIG_DIR / "models"
PIPELINE_DATA_DIR = PROJECT_ROOT / "aggregate_bot_modules" / "data"
PIPELINE_RAW_PATH = PIPELINE_DATA_DIR / "raw_posts_labeled.csv"
PIPELINE_PROCESSED_PATH = PIPELINE_DATA_DIR / "processed_posts.csv"

SERVICE_LABELS = [
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


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def inspect_ml_dataset() -> None:
    print_section("1. Dataset in ML/")

    if not ML_DATASET_PATH.exists():
        print(f"[missing] {ML_DATASET_PATH}")
        return

    with ML_DATASET_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Path: {ML_DATASET_PATH}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {reader.fieldnames}")

    topic_counts = Counter(row.get("topic", "") for row in rows)
    print("Topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0])):
        label = SERVICE_LABELS[int(topic)] if str(topic).isdigit() and int(topic) < len(SERVICE_LABELS) else topic
        print(f"  {topic:>2} -> {label:<35} {count}")


def inspect_service_artifacts() -> None:
    print_section("2. Service model artifacts")

    if not SERVICE_LABEL_ENCODER_PATH.exists():
        print(f"[missing] {SERVICE_LABEL_ENCODER_PATH}")
        return

    with SERVICE_LABEL_ENCODER_PATH.open("rb") as f:
        label_encoder = pickle.load(f)

    with SERVICE_VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)

    print(f"Label encoder path: {SERVICE_LABEL_ENCODER_PATH}")
    print(f"Label encoder classes: {list(label_encoder.classes_)}")
    print(f"Vectorizer path: {SERVICE_VECTORIZER_PATH}")
    print(f"Vectorizer vocabulary size: {len(getattr(vectorizer, 'vocabulary_', {}))}")

    print("Available service models:")
    if not SERVICE_MODEL_DIR.exists():
        print(f"  [missing] {SERVICE_MODEL_DIR}")
        return

    for model_path in sorted(SERVICE_MODEL_DIR.glob("*.pkl")):
        with model_path.open("rb") as f:
            model = pickle.load(f)
        print(f"  {model_path.name:<24} {type(model).__name__}")


def inspect_reproducible_pipeline() -> None:
    print_section("3. Reproducible pipeline state")

    print(f"Expected pipeline data dir: {PIPELINE_DATA_DIR}")
    print(f"raw_posts_labeled.csv exists: {PIPELINE_RAW_PATH.exists()}")
    print(f"processed_posts.csv exists: {PIPELINE_PROCESSED_PATH.exists()}")
    print(f"ML/raw_posts_labeled.csv exists: {ML_DATASET_PATH.exists()}")

    if not PIPELINE_RAW_PATH.exists() and ML_DATASET_PATH.exists():
        print(
            "[warning] The scripted ML pipeline points to aggregate_bot_modules/data, "
            "but the only real labeled dataset currently lives in ML/."
        )


def print_conclusion() -> None:
    print_section("4. Main conclusion")
    print(
        "The project currently has three divergent ML contours:\n"
        "  1) ML/raw_posts_labeled.csv -> real 11-class labeled dataset.\n"
        "  2) aggregate_bot_modules/ml_module -> scripted pipeline, but wired for a different data location.\n"
        "  3) service/config -> deployed inference artifacts used by the bot.\n"
    )
    print(
        "Before retraining for product quality, the first engineering step is to make these three contours a single source of truth:\n"
        "  - one taxonomy,\n"
        "  - one canonical dataset path,\n"
        "  - one reproducible training script,\n"
        "  - one export target for service/bot inference.\n"
    )


def main() -> None:
    inspect_ml_dataset()
    inspect_service_artifacts()
    inspect_reproducible_pipeline()
    print_conclusion()


if __name__ == "__main__":
    main()
