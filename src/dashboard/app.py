from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit is not installed. Use reports/dashboard.html or install "
            "`streamlit` and rerun this app."
        ) from exc

    st.set_page_config(page_title="ITMS Dashboard", layout="wide")
    st.title("AI-Enabled Intelligent Traffic Management System")
    st.caption("Live-style dashboard backed by generated prototype CSVs.")

    summary = pd.read_csv(REPORTS_DIR / "performance_summary.csv")
    predictions = pd.read_csv(DATA_DIR / "predictions.csv")
    decisions = pd.read_csv(DATA_DIR / "control_decisions.csv")
    metrics = pd.read_csv(DATA_DIR / "ai_controller_metrics.csv")

    ai_summary = summary[summary["controller"] == "ai_controller"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AI throughput", int(ai_summary["total_throughput"].sum()))
    col2.metric("Avg waiting time", f"{ai_summary['average_waiting_time'].mean():.2f}s")
    col3.metric("Max latency", f"{decisions['decision_latency_seconds'].max():.4f}s")
    col4.metric("Accident alerts", int(metrics["accident_alert"].sum()))

    st.subheader("Manual Override")
    st.radio("Selected signal phase", ["north", "east", "south", "west"], horizontal=True)

    st.subheader("Performance Summary")
    st.dataframe(summary, use_container_width=True)
    st.subheader("Congestion Predictions")
    st.dataframe(predictions, use_container_width=True)
    st.subheader("Recent Decisions")
    st.dataframe(decisions.tail(40), use_container_width=True)


if __name__ == "__main__":
    main()
