from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.analytics_report import build_summary, load_events  # noqa: E402

BOT_DATA_DIR = ROOT_DIR / "bot_data"
EVENTS_PATH = BOT_DATA_DIR / "events.jsonl"
ANALYTICS_JSON_PATH = BOT_DATA_DIR / "analytics_summary.json"
DASHBOARD_HTML_PATH = BOT_DATA_DIR / "dashboard.html"
REFRESH_STATUS_PATH = BOT_DATA_DIR / "refresh_status.json"
MODEL_MANIFEST_PATH = ROOT_DIR / "service" / "config" / "model_manifest.json"
RECOMMENDATION_EVAL_PATH = ROOT_DIR / "ML" / "recommendation_eval_artifacts" / "summary.json"
FEEDBACK_SUMMARY_PATH = ROOT_DIR / "ML" / "feedback_loop_artifacts" / "feedback_summary.json"
FEEDBACK_RETRAIN_SUMMARY_PATH = ROOT_DIR / "ML" / "feedback_loop_artifacts" / "feedback_retrain_summary.json"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_metric_cards(summary: dict) -> str:
    rec = summary.get("recommendation_metrics", {})
    onboarding = summary.get("onboarding_metrics", {})
    retention = summary.get("retention_metrics", {})
    cards = [
        ("Всего событий", summary.get("total_events", 0)),
        ("Уникальных пользователей", summary.get("unique_users", 0)),
        ("Показов ленты", rec.get("feeds_shown", 0)),
        ("Показано новостей", rec.get("items_shown", 0)),
        ("Like rate / item", rec.get("like_rate_per_item", 0.0)),
        ("Feed→feedback", rec.get("feed_to_feedback_rate", 0.0)),
        ("Dislike rate / item", rec.get("dislike_rate_per_item", 0.0)),
        ("Open similar / item", rec.get("similar_open_rate_per_item", 0.0)),        
        ("Onboarding completion", onboarding.get("completion_rate", 0.0)),
        ("DAU", retention.get("dau", 0)),
        ("WAU", retention.get("wau", 0)),
        ("MAU", retention.get("mau", 0)),
        ("Day-1 retention", retention.get("day1_retention", 0.0)),
    ]
    return "\n".join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in cards
    )


def render_pairs(title: str, pairs: list[list] | list[tuple]) -> str:
    items = "".join(f"<li><span>{name}</span><strong>{value}</strong></li>" for name, value in pairs)
    return f'<section class="panel"><h3>{title}</h3><ul class="pairs">{items}</ul></section>'


def render_bar_pairs(title: str, pairs: list[list] | list[tuple], max_items: int = 12) -> str:
    limited = list(pairs)[:max_items]
    if not limited:
        return f'<section class="panel"><h3>{title}</h3><p class="muted">Пока нет данных.</p></section>'
    max_value = max((value for _, value in limited), default=1) or 1
    rows = []
    for name, value in limited:
        width = max(4, int((float(value) / max_value) * 100))
        rows.append(
            f'<li><span>{name}</span><div class="bar-wrap"><div class="bar" style="width:{width}%"></div></div><strong>{value}</strong></li>'
        )
    return f'<section class="panel"><h3>{title}</h3><ul class="bars">{"".join(rows)}</ul></section>'


def main() -> None:
    BOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events(EVENTS_PATH)
    summary = build_summary(events)
    ANALYTICS_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    model_manifest = load_json(MODEL_MANIFEST_PATH) or {}
    refresh_status = load_json(REFRESH_STATUS_PATH) or {}
    recommendation_eval = load_json(RECOMMENDATION_EVAL_PATH) or {}
    feedback_summary = load_json(FEEDBACK_SUMMARY_PATH) or {}
    feedback_retrain_summary = load_json(FEEDBACK_RETRAIN_SUMMARY_PATH) or {}

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>Aggregator Bot Dashboard</title>
    <style>
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 24px;
      background: #0f172a;
      color: #e2e8f0;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .muted {{ color: #94a3b8; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 18px 0 28px;
    }}
    .card, .panel {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 14px;
      padding: 16px;
    }}
    .card .label {{ color: #94a3b8; font-size: 13px; margin-bottom: 8px; }}
    .card .value {{ font-size: 28px; font-weight: 700; }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    ul.pairs {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    ul.pairs li {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 0;
      border-bottom: 1px solid #1f2937;
    }}
    ul.pairs li:last-child {{ border-bottom: none; }}
    ul.bars {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    ul.bars li {{
      display: grid;
      grid-template-columns: minmax(120px, 1.4fr) minmax(120px, 3fr) auto;
      align-items: center;
      gap: 12px;
      padding: 8px 0;
      border-bottom: 1px solid #1f2937;
    }}
    ul.bars li:last-child {{ border-bottom: none; }}
    .bar-wrap {{
      width: 100%;
      height: 10px;
      background: #0b1220;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      background: linear-gradient(90deg, #22c55e, #38bdf8);
      border-radius: 999px;
    }}
    code {{
      background: #0b1220;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>Aggregator Bot Dashboard</h1>
  <p class="muted">Локальный продуктовый dashboard по событиям бота, состоянию refresh-пайплайна, deploy-модели и offline recommendation evaluation.</p>

  <div class="grid">
    {render_metric_cards(summary)}
  </div>

  <div class="two-col">
    <section class="panel">
      <h3>Deploy Model</h3>
      <p><strong>Pipeline:</strong> <code>{model_manifest.get('source_pipeline', 'unknown')}</code></p>
      <p><strong>Model:</strong> {model_manifest.get('model_name', 'unknown')}</p>
      <p><strong>F1_macro:</strong> {model_manifest.get('test_metrics', {}).get('f1_macro', 'n/a')}</p>
      <p><strong>Classes:</strong> {model_manifest.get('class_count', 'n/a')}</p>
    </section>

    <section class="panel">
      <h3>Refresh Status</h3>
      <p><strong>Status:</strong> {refresh_status.get('status', 'unknown')}</p>
      <p><strong>Last run:</strong> {refresh_status.get('last_run_at', 'n/a')}</p>
      <p><strong>Refreshed:</strong> {refresh_status.get('refreshed_count', 'n/a')}</p>
      <p><strong>Errors:</strong> {refresh_status.get('error_count', 'n/a')}</p>
    </section>
  </div>

  <div class="two-col">
    {render_pairs("Top Queries", summary.get("top_queries", []))}
    {render_pairs("Imported Channels", summary.get("imported_channels", []))}
  </div>

  <div class="two-col">
    {render_bar_pairs("Shown Topics", summary.get("content_breakdown", {}).get("shown_topics", []))}
    {render_bar_pairs("Likes by Topic", summary.get("content_breakdown", {}).get("likes_by_topic", []))}
  </div>

  <div class="two-col">
    {render_bar_pairs("Daily Active Users", summary.get("daily_active_users", []))}
    <section class="panel">
      <h3>Retention</h3>
      <p><strong>DAU:</strong> {summary.get('retention_metrics', {}).get('dau', 0)}</p>
      <p><strong>WAU:</strong> {summary.get('retention_metrics', {}).get('wau', 0)}</p>
      <p><strong>MAU:</strong> {summary.get('retention_metrics', {}).get('mau', 0)}</p>
      <p><strong>Day-1 retention:</strong> {summary.get('retention_metrics', {}).get('day1_retention', 0.0)}</p>
      <p><strong>Day-7 retention:</strong> {summary.get('retention_metrics', {}).get('day7_retention', 0.0)}</p>
      <p><strong>Feed to feedback:</strong> {summary.get('recommendation_metrics', {}).get('feed_to_feedback_rate', 0.0)}</p>
    </section>
  </div>

  <div class="two-col">
    <section class="panel">
      <h3>Offline Recommendation Eval</h3>
      <p><strong>Topic precision@5:</strong> {recommendation_eval.get('overall', {}).get('topic_feed_precision_at_5', 'n/a')}</p>
      <p><strong>Similar precision@5:</strong> {recommendation_eval.get('overall', {}).get('similarity_precision_at_5', 'n/a')}</p>
      <p><strong>Coverage:</strong> {recommendation_eval.get('overall', {}).get('topic_coverage', 'n/a')}</p>
      <p><strong>Uncertain rate:</strong> {recommendation_eval.get('overall', {}).get('uncertain_prediction_rate', 'n/a')}</p>
    </section>
    {render_pairs("Per-topic Topic Feed Precision@5", recommendation_eval.get("per_topic_topic_feed_precision", []))}
  </div>

  <div class="two-col">
    <section class="panel">
      <h3>Feedback Candidates</h3>
      <p><strong>Total:</strong> {feedback_summary.get('total_candidates', 0)}</p>
      <p><strong>Accepted:</strong> {feedback_summary.get('accepted_candidates', 0)}</p>
      <p><strong>Rejected:</strong> {feedback_summary.get('rejected_candidates', 0)}</p>
      <p><strong>Mean feedback score:</strong> {feedback_summary.get('mean_feedback_score', 0.0)}</p>
      <p><strong>Mean accepted confidence:</strong> {feedback_summary.get('mean_accepted_confidence', 0.0)}</p>
    </section>
    <section class="panel">
      <h3>Feedback Retrain</h3>
      <p><strong>Feedback rows:</strong> {feedback_retrain_summary.get('feedback_rows', 0)}</p>
      <p><strong>Previous best NB F1:</strong> {feedback_retrain_summary.get('previous_best_nb_f1_macro', 'n/a')}</p>
      <p><strong>Final NB F1:</strong> {feedback_retrain_summary.get('final_nb_f1_macro', 'n/a')}</p>
      <p><strong>Export updated:</strong> {feedback_retrain_summary.get('service_export_updated', False)}</p>
    </section>
  </div>
</body>
</html>
"""
    DASHBOARD_HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {DASHBOARD_HTML_PATH}")


if __name__ == "__main__":
    main()
