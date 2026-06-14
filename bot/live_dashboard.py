from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from bot.analytics_report import build_summary, load_events
from bot.build_dashboard import (
    EVENTS_PATH,
    FEEDBACK_RETRAIN_SUMMARY_PATH,
    FEEDBACK_SUMMARY_PATH,
    MODEL_MANIFEST_PATH,
    RECOMMENDATION_EVAL_PATH,
    load_json,
)


HOST = "127.0.0.1"
PORT = 8765


def fmt_metric(value, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_badge(label: str, tone: str = "default") -> str:
    return f'<span class="badge badge-{tone}">{html.escape(label)}</span>'


def render_stat_card(label: str, value: str, hint: str = "", tone: str = "default") -> str:
    return f"""
    <article class="stat-card stat-{tone}">
      <div class="stat-label">{html.escape(label)}</div>
      <div class="stat-value">{html.escape(value)}</div>
      <div class="stat-hint">{html.escape(hint)}</div>
    </article>
    """


def render_pairs_panel(title: str, pairs: list[tuple[str, object]], empty_text: str = "Пока нет данных") -> str:
    rows = []
    for key, value in pairs:
        rows.append(
            f"<li><span>{html.escape(str(key))}</span><strong>{html.escape(str(value))}</strong></li>"
        )
    if not rows:
        rows.append(f"<li class='empty-row'>{html.escape(empty_text)}</li>")
    return f"""
    <section class="panel">
      <h3>{html.escape(title)}</h3>
      <ul class="pairs-list">
        {''.join(rows)}
      </ul>
    </section>
    """


def render_bar_panel(title: str, pairs: list[tuple[str, object]], empty_text: str = "Пока нет данных") -> str:
    max_value = max((float(value) for _, value in pairs), default=1.0)
    rows = []
    for label, value in pairs:
        numeric = float(value)
        width = 0 if max_value <= 0 else max(6.0, (numeric / max_value) * 100.0)
        rows.append(
            "<li>"
            f"<span class='bar-label'>{html.escape(str(label))}</span>"
            f"<span class='bar-track'><span class='bar-fill' style='width:{width:.2f}%'></span></span>"
            f"<strong>{html.escape(fmt_metric(value, digits=3))}</strong>"
            "</li>"
        )
    if not rows:
        rows.append(f"<li class='empty-row'>{html.escape(empty_text)}</li>")
    return f"""
    <section class="panel">
      <h3>{html.escape(title)}</h3>
      <ul class="bars-list">
        {''.join(rows)}
      </ul>
    </section>
    """


def build_dashboard_html() -> str:
    events = load_events(EVENTS_PATH)
    summary = build_summary(events)
    model_manifest = load_json(MODEL_MANIFEST_PATH) or {}
    recommendation_eval = load_json(RECOMMENDATION_EVAL_PATH) or {}
    feedback_summary = load_json(FEEDBACK_SUMMARY_PATH) or {}
    feedback_retrain_summary = load_json(FEEDBACK_RETRAIN_SUMMARY_PATH) or {}

    recommendation_metrics = summary.get("recommendation_metrics", {})
    onboarding_metrics = summary.get("onboarding_metrics", {})
    retention_metrics = summary.get("retention_metrics", {})
    corpus_health = summary.get("corpus_health", {})
    refresh_summary = corpus_health.get("refresh_summary", {})
    overall_eval = recommendation_eval.get("overall", {})

    model_name = str(model_manifest.get("model_name", "unknown"))
    model_f1 = model_manifest.get("test_metrics", {}).get("f1_macro", "n/a")
    last_refresh = corpus_health.get("last_refresh_at") or "n/a"

    hero_badges = " ".join(
        [
            render_badge(f"Model: {model_name}", "info"),
            render_badge(f"F1_macro: {fmt_metric(model_f1, 4)}", "success"),
            render_badge(f"Refresh: {refresh_summary.get('refreshed', 0)} updated", "default"),
            render_badge(f"Errors: {refresh_summary.get('errors', 0)}", "warning" if refresh_summary.get("errors", 0) else "success"),
        ]
    )

    top_stats = "".join(
        [
            render_stat_card("События", str(summary.get("total_events", 0)), "Всего product-событий в логе"),
            render_stat_card("Пользователи", str(summary.get("unique_users", 0)), "Уникальные активные пользователи"),
            render_stat_card("Показано новостей", str(recommendation_metrics.get("items_shown", 0)), "Суммарно по всем лентам", "info"),
            render_stat_card("Feedback rate", fmt_metric(recommendation_metrics.get("feed_to_feedback_rate", 0.0), 3), "Лайки, дизлайки и похожее на одну ленту", "success"),
            render_stat_card("Corpus size", str(corpus_health.get("total_items", 0)), "Новостей после retention", "info"),
            render_stat_card("Active sources", str(corpus_health.get("active_sources", 0)), "Источники с данными после чистки"),
        ]
    )

    left_col = "".join(
        [
            render_pairs_panel(
                "Deploy Model",
                [
                    ("Pipeline", model_manifest.get("source_pipeline", "unknown")),
                    ("Model", model_name),
                    ("F1_macro", fmt_metric(model_f1, 4)),
                    ("Classes", model_manifest.get("class_count", "n/a")),
                ],
            ),
            render_pairs_panel(
                "Refresh Status",
                [
                    ("Last refresh", last_refresh),
                    ("Refreshed", refresh_summary.get("refreshed", 0)),
                    ("Skipped", refresh_summary.get("skipped", 0)),
                    ("Errors", refresh_summary.get("errors", 0)),
                ],
            ),
            render_pairs_panel(
                "Corpus Health",
                [
                    ("Base sources", corpus_health.get("base_sources", 0)),
                    ("Custom sources", corpus_health.get("custom_sources", 0)),
                    ("Base items", corpus_health.get("base_items", 0)),
                    ("Custom items", corpus_health.get("custom_items", 0)),
                ],
            ),
        ]
    )

    right_col = "".join(
        [
            render_pairs_panel(
                "Onboarding",
                [
                    ("Started", onboarding_metrics.get("started", 0)),
                    ("Completed", onboarding_metrics.get("completed", 0)),
                    ("Completion rate", fmt_metric(onboarding_metrics.get("completion_rate", 0.0), 3)),
                ],
            ),
            render_pairs_panel(
                "Retention",
                [
                    ("DAU", retention_metrics.get("dau", 0)),
                    ("WAU", retention_metrics.get("wau", 0)),
                    ("MAU", retention_metrics.get("mau", 0)),
                    ("Day-1", fmt_metric(retention_metrics.get("day1_retention", 0.0), 3)),
                    ("Day-7", fmt_metric(retention_metrics.get("day7_retention", 0.0), 3)),
                ],
            ),
            render_pairs_panel(
                "Offline Recommendation Eval",
                [
                    ("Topic precision@5", fmt_metric(overall_eval.get("topic_feed_precision_at_5", 0.0), 4)),
                    ("Similar precision@5", fmt_metric(overall_eval.get("similarity_precision_at_5", 0.0), 4)),
                    ("Coverage", fmt_metric(overall_eval.get("topic_coverage", 0.0), 4)),
                    ("Uncertain rate", fmt_metric(overall_eval.get("uncertain_prediction_rate", 0.0), 4)),
                ],
            ),
        ]
    )

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="15" />
  <title>Aggregator Bot Control Room</title>
  <style>
    :root {{
      --bg: #07111f;
      --panel: rgba(10, 22, 39, 0.88);
      --panel-strong: rgba(13, 28, 48, 0.96);
      --line: rgba(148, 163, 184, 0.14);
      --text: #e5eefc;
      --muted: #98a7c1;
      --blue: #56c1ff;
      --green: #4ade80;
      --amber: #fbbf24;
      --pink: #f472b6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(86,193,255,0.16), transparent 24%),
        radial-gradient(circle at top right, rgba(244,114,182,0.12), transparent 22%),
        linear-gradient(180deg, #07111f 0%, #0b1628 100%);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(16,33,57,0.96), rgba(9,21,38,0.92));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px 28px 22px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
      margin-bottom: 22px;
    }}
    .eyebrow {{
      color: var(--blue);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      margin-bottom: 10px;
      font-weight: 700;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.08;
    }}
    .hero p {{
      margin: 0 0 16px;
      color: var(--muted);
      max-width: 860px;
      line-height: 1.5;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
    }}
    .badge-info {{ border-color: rgba(86,193,255,0.28); color: #aee6ff; }}
    .badge-success {{ border-color: rgba(74,222,128,0.28); color: #b9f7cc; }}
    .badge-warning {{ border-color: rgba(251,191,36,0.32); color: #ffe2a1; }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .stat-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      min-height: 124px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }}
    .stat-label {{ color: var(--muted); font-size: 13px; margin-bottom: 14px; }}
    .stat-value {{ font-size: 30px; font-weight: 700; margin-bottom: 10px; }}
    .stat-hint {{ color: var(--muted); font-size: 12px; line-height: 1.4; }}
    .columns {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 16px;
      margin-bottom: 18px;
    }}
    .stack {{
      display: grid;
      gap: 16px;
    }}
    .panel {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: 0 14px 36px rgba(0,0,0,0.18);
    }}
    .panel h3 {{
      margin: 0 0 14px;
      font-size: 18px;
    }}
    .pairs-list, .bars-list {{
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .pairs-list li {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      padding: 11px 0;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
    }}
    .pairs-list li strong {{ color: var(--text); font-weight: 700; text-align: right; }}
    .pairs-list li:last-child,
    .bars-list li:last-child {{ border-bottom: none; }}
    .bars-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .bars-list li {{
      display: grid;
      grid-template-columns: minmax(120px, 1.3fr) minmax(120px, 2.8fr) auto;
      align-items: center;
      gap: 12px;
      padding: 11px 0;
      border-bottom: 1px solid var(--line);
    }}
    .bar-label {{
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .bar-track {{
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      overflow: hidden;
    }}
    .bar-fill {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--blue), var(--green));
      border-radius: 999px;
    }}
    .bars-list strong {{ font-size: 13px; }}
    .section-title {{
      margin: 6px 0 14px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .footer-note {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .empty-row {{
      color: var(--muted);
      padding: 10px 0;
    }}
    code {{
      background: rgba(255,255,255,0.05);
      padding: 2px 6px;
      border-radius: 8px;
    }}
    @media (max-width: 980px) {{
      .columns, .bars-grid {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        padding: 18px;
      }}
      h1 {{
        font-size: 28px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Control Room</div>
      <h1>Aggregator Bot Product Dashboard</h1>
      <p>Живая панель продукта: качество рекомендаций, здоровье корпуса, refresh-конвейер, пользовательские сигналы и feedback loop в одном экране. Страница обновляется каждые 15 секунд.</p>
      <div class="badge-row">{hero_badges}</div>
      <div class="footer-note">Events file: <code>{html.escape(str(EVENTS_PATH))}</code></div>
    </section>

    <section class="stats-grid">
      {top_stats}
    </section>

    <div class="columns">
      <div class="stack">
        {left_col}
      </div>
      <div class="stack">
        {right_col}
      </div>
    </div>

    <div class="bars-grid">
      {render_bar_panel("Shown Topics", summary.get("content_breakdown", {}).get("shown_topics", []))}
      {render_bar_panel("Shown Sources", summary.get("content_breakdown", {}).get("shown_sources", []))}
      {render_bar_panel("Likes by Topic", summary.get("content_breakdown", {}).get("likes_by_topic", []))}
      {render_bar_panel("Dislikes by Topic", summary.get("content_breakdown", {}).get("dislikes_by_topic", []))}
      {render_bar_panel("Daily Active Users", summary.get("daily_active_users", []))}
      {render_bar_panel("Retention Windows", corpus_health.get("retention_windows", []))}
    </div>

    <div class="columns">
      <div class="stack">
        {render_pairs_panel("Top Queries", summary.get("top_queries", []))}
        {render_pairs_panel("Imported Channels", summary.get("imported_channels", []))}
        {render_pairs_panel("Largest Sources", corpus_health.get("largest_sources", []))}
      </div>
      <div class="stack">
        {render_pairs_panel(
            "Feedback Candidates",
            [
                ("Total", feedback_summary.get("total_candidates", 0)),
                ("Accepted", feedback_summary.get("accepted_candidates", 0)),
                ("Rejected", feedback_summary.get("rejected_candidates", 0)),
                ("Mean feedback score", fmt_metric(feedback_summary.get("mean_feedback_score", 0.0), 3)),
                ("Mean accepted confidence", fmt_metric(feedback_summary.get("mean_accepted_confidence", 0.0), 3)),
            ],
        )}
        {render_pairs_panel(
            "Feedback Retrain",
            [
                ("Feedback rows", feedback_retrain_summary.get("feedback_rows", 0)),
                ("Previous best NB F1", fmt_metric(feedback_retrain_summary.get("previous_best_nb_f1_macro", 0.0), 4)),
                ("Final NB F1", fmt_metric(feedback_retrain_summary.get("final_nb_f1_macro", 0.0), 4)),
                ("Export updated", feedback_retrain_summary.get("service_export_updated", False)),
            ],
        )}
        {render_pairs_panel(
            "Per-topic Topic Precision@5",
            recommendation_eval.get("per_topic_topic_feed_precision", []),
        )}
      </div>
    </div>
  </div>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            content = build_dashboard_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/analytics.json":
            payload = json.dumps(build_summary(load_events(EVENTS_PATH)), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Live dashboard running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
