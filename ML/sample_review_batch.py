from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_QUEUE_PATH = PROJECT_ROOT / "ML" / "bootstrap" / "review_queue.csv"
OUTPUT_PATH = PROJECT_ROOT / "ML" / "bootstrap" / "review_batch_v1.csv"


def main(per_label: int = 25) -> None:
    if not REVIEW_QUEUE_PATH.exists():
        raise FileNotFoundError(f"Review queue not found: {REVIEW_QUEUE_PATH}")

    with REVIEW_QUEUE_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["proposed_label"]].append(row)

    sampled: list[dict[str, str]] = []
    for label in sorted(grouped):
        sampled.extend(grouped[label][:per_label])

    fieldnames = [
        "review_status",
        "final_label",
        "review_comment",
        *rows[0].keys(),
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sampled:
            writer.writerow(
                {
                    "review_status": "",
                    "final_label": "",
                    "review_comment": "",
                    **row,
                }
            )

    print(f"[+] Review batch saved to: {OUTPUT_PATH}")
    print(f"    rows={len(sampled)}")
    print(f"    labels={len(grouped)}")


if __name__ == "__main__":
    main()
