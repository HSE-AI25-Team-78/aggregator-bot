# Recommender Ranking Tuning

- Composite score: **0.712930**
- Topic feed precision@5: **0.8600**
- Similarity precision@5: **0.5926**
- Topic coverage: **0.9091**

## Best config

```json
{
  "latest": {
    "freshness_weight": 1.0,
    "confidence_weight": 0.35
  },
  "topical": {
    "confidence_weight": 0.52,
    "freshness_weight": 0.24,
    "topic_affinity_weight": 0.12,
    "source_affinity_weight": 0.06
  },
  "general": {
    "base_score_weight": 0.42,
    "confidence_weight": 0.16,
    "freshness_weight": 0.1,
    "topic_affinity_weight": 0.18,
    "source_affinity_weight": 0.04
  },
  "profile": {
    "min_item_preference_weight": 0.1
  }
}
```