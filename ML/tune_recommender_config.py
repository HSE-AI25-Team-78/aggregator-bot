from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ML.offline_recommendation_eval import evaluate_recommender  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "ML" / "recommender_tuning_artifacts"
BEST_CONFIG_PATH = OUTPUT_DIR / "best_recommender_config.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "bot" / "config" / "recommender_config.json"


def load_base_config() -> dict:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def composite_score(summary: dict) -> float:
    overall = summary["overall"]
    return (
        0.45 * overall["topic_feed_precision_at_5"]
        + 0.55 * overall["similarity_precision_at_5"]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_base_config()

    candidates = []
    general_base_weights = [0.42, 0.50, 0.58]
    general_topic_affinity_weights = [0.12, 0.18, 0.24]
    general_source_affinity_weights = [0.04, 0.06, 0.10]
    topical_topic_affinity_weights = [0.12, 0.18, 0.24]
    profile_min_weights = [0.1, 0.2, 0.3]

    for base_w, topic_w, source_w, topical_topic_w, profile_min in itertools.product(
        general_base_weights,
        general_topic_affinity_weights,
        general_source_affinity_weights,
        topical_topic_affinity_weights,
        profile_min_weights,
    ):
        config = json.loads(json.dumps(base_config))
        config["general"]["base_score_weight"] = base_w
        config["general"]["topic_affinity_weight"] = topic_w
        config["general"]["source_affinity_weight"] = source_w
        config["topical"]["topic_affinity_weight"] = topical_topic_w
        config["profile"]["min_item_preference_weight"] = profile_min

        temp_config_path = OUTPUT_DIR / "candidate_config.json"
        temp_config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = evaluate_recommender(temp_config_path)
        score = composite_score(summary)
        candidates.append(
            {
                "score": round(score, 6),
                "summary": summary,
                "config": config,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]

    BEST_CONFIG_PATH.write_text(json.dumps(best["config"], ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Recommender Ranking Tuning",
        "",
        f"- Composite score: **{best['score']:.6f}**",
        f"- Topic feed precision@5: **{best['summary']['overall']['topic_feed_precision_at_5']:.4f}**",
        f"- Similarity precision@5: **{best['summary']['overall']['similarity_precision_at_5']:.4f}**",
        f"- Topic coverage: **{best['summary']['overall']['topic_coverage']:.4f}**",
        "",
        "## Best config",
        "",
        "```json",
        json.dumps(best["config"], ensure_ascii=False, indent=2),
        "```",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
