"""Page 3: benchmark results (Phase 6) from the stored files: no LLM call."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(os.environ.get("RESULTS_DIR", "/app/results"))
FIGURES = Path(os.environ.get("FIGURES_DIR", "/app/figures"))
NAMES = {"direct": "Direct Cypher", "A": "A vector", "B": "B GraphRAG", "C": "C Text2Cypher"}

st.set_page_config(page_title="AI-ecosystem KG · Results", layout="wide")
st.title("Benchmark results")
st.caption("30 verified questions × 4 systems × 2 graph versions; correctness by an LLM judge (gpt-oss-120b). "
           "Gold set, question review and judge validation are model-made (see the report's limitations).")
summary_path = RESULTS / "benchmark_summary.json"
if not summary_path.exists():
    st.warning("No benchmark summary yet (make report).")
    st.stop()
s = json.loads(summary_path.read_text())
rows = []
for k, v in s["by_system_version"].items():
    sy, ver = k.split("|")
    rows.append({"system": NAMES[sy], "graph": ver, "correctness (1-5)": v["correctness"], "entity recall": v["entity_recall"],
                 "latency p50 (s)": v["latency_median_s"], "LLM calls/q": v["llm_calls_per_q"],
                 "cost / 1,000 q ($)": (v["cost_usd_per_q"] or 0) * 1000})
st.subheader("By system and graph version")
df = pd.DataFrame(rows)
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
st.dataframe(df.style.format(precision=2, na_rep="–"), hide_index=True, use_container_width=True)
for name, caption in (("phase6_correctness_by_category.png", "Correctness and entity recall by category (resolved)"),
                      ("phase6_raw_vs_resolved.png", "Raw vs resolved"),
                      ("phase6_quality_vs_cost.png", "Quality vs paid-tier cost"),
                      ("phase6_latency_by_strategy.png", "Latency per answer"),
                      ("phase6_rq1_extraction.png", "RQ1 extraction P/R/F1")):
    if (FIGURES / name).exists():
        st.image(str(FIGURES / name), caption=caption)
