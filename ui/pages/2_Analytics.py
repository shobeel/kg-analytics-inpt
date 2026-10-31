"""Page 2: graph analytics (Phase 4 results written into Neo4j by GDS)."""

import altair as alt
import pandas as pd
import streamlit as st

import client

st.set_page_config(page_title="AI-ecosystem KG · Analytics", layout="wide")
st.title("Graph analytics")
kg = st.radio("Graph version", ["resolved", "raw"], horizontal=True)
try:
    top = pd.DataFrame(client.get("/analytics/top", kg_version=kg, limit=15))
    comm = pd.DataFrame(client.get("/analytics/communities", kg_version=kg, limit=10))
    quality = pd.DataFrame(client.get("/analytics/quality"))
except Exception as e:
    st.error(f"API error {client.error_text(e)}")
    st.stop()

left, right = st.columns([3, 2])
with left:
    st.subheader("Top 15 entities by PageRank")
    st.altair_chart(alt.Chart(top).mark_bar(color="#2a78d6", cornerRadiusEnd=3).encode(
        x=alt.X("pagerank:Q", title="PageRank"), y=alt.Y("name:N", sort="-x", title=None),
        tooltip=["name", "label", "pagerank", "community"]).properties(height=460), use_container_width=True)
with right:
    st.dataframe(top[["name", "label", "pagerank", "community"]], hide_index=True, height=460)
st.subheader("Largest Louvain communities")
comm["top_members"] = comm["top_members"].apply(lambda m: ", ".join(m))
st.dataframe(comm, hide_index=True, use_container_width=True)
st.subheader("Raw vs resolved graph (RQ2)")
for col in ("raw", "resolved", "relative_change"):
    if col in quality:
        quality[col] = pd.to_numeric(quality[col], errors="coerce")
st.dataframe(quality.style.format({"raw": "{:.4g}", "resolved": "{:.4g}", "relative_change": "{:+.1%}"}, na_rep="–"),
             hide_index=True, use_container_width=True)
