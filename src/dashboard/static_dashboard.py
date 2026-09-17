from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _table(frame: pd.DataFrame, limit: int = 12) -> str:
    if frame.empty:
        return "<p>No data available.</p>"
    visible = frame.head(limit)
    header = "".join(f"<th>{escape(str(column))}</th>" for column in visible.columns)
    rows = []
    for _, row in visible.iterrows():
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_dashboard_html(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ai_summary = summary[summary["controller"] == "ai_controller"]
    total_throughput = int(ai_summary["total_throughput"].sum()) if not ai_summary.empty else 0
    sim_duration_seconds = decisions["green_duration"].sum() if not decisions.empty else 0
    if sim_duration_seconds > 0:
        throughput_per_hour = (total_throughput / sim_duration_seconds) * 3600
        throughput_display = f"{total_throughput} veh ({int(throughput_per_hour)} veh/hr)"
    else:
        throughput_display = f"{total_throughput} vehicles"
    avg_wait = float(ai_summary["average_waiting_time"].mean()) if not ai_summary.empty else 0.0
    max_latency = float(decisions["decision_latency_seconds"].max()) if not decisions.empty else 0.0
    accident_alerts = int(metrics["accident_alert"].sum()) if not metrics.empty else 0
    emergency_rows = metrics[metrics["emergency_count"] > 0]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ITMS Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f7fb; color: #18212f; }}
    header {{ background: #113a5c; color: white; padding: 24px 32px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #dde5ef; border-radius: 8px; padding: 16px; }}
    .metric {{ font-size: 30px; font-weight: 700; margin-top: 8px; }}
    .label {{ color: #56657a; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
    section {{ margin-top: 24px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #dde5ef; }}
    th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #e8edf4; font-size: 13px; }}
    th {{ background: #edf3f9; }}
    .alert {{ color: #9b2c2c; font-weight: 700; }}
  </style>
</head>
<body>
  <header>
    <h1>AI-Enabled Intelligent Traffic Management System</h1>
    <p>Prototype dashboard generated from simulated intersection metrics.</p>
  </header>
  <main>
    <div class="grid">
      <div class="card"><div class="label">AI Throughput</div><div class="metric">{throughput_display}</div></div>
      <div class="card"><div class="label">Average Waiting Time</div><div class="metric">{avg_wait:.2f}s</div></div>
      <div class="card"><div class="label">Max Decision Latency</div><div class="metric">{max_latency:.4f}s</div></div>
      <div class="card"><div class="label">Accident Alerts</div><div class="metric alert">{accident_alerts}</div></div>
    </div>
    <section>
      <h2>Performance Summary</h2>
      {_table(summary, limit=20)}
    </section>
    <section>
      <h2>Latest Congestion Predictions</h2>
      {_table(predictions, limit=20)}
    </section>
    <section>
      <h2>Recent Control Decisions</h2>
      {_table(decisions.tail(20).reset_index(drop=True), limit=20)}
    </section>
    <section>
      <h2>Emergency Vehicle Evidence</h2>
      {_table(emergency_rows.head(12), limit=12)}
    </section>
  </main>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output
