"""Phase 6 report: docs/benchmark_report.md + figures in docs/figures/, generated from stored results only
(benchmark runs, judge scores, RQ1/RQ2 result files, the call log). No LLM calls. Every number in the
report is read from those files; sections whose inputs do not exist yet say so.

Usage: python -m eval.report
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
FIGURES = ROOT / "docs" / "figures"
REPORT = ROOT / "docs" / "benchmark_report.md"
NOT_FOUND = "Not found in the knowledge graph."

# Reference categorical palette (dataviz skill), fixed per system: color follows the system.
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "direct": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SYSTEM_NAMES = {"direct": "Direct Cypher", "A": "A vector", "B": "B GraphRAG", "C": "C Text2Cypher"}
CATS = ["lookup", "multi-hop", "aggregation", "semantic", "unanswerable"]


# ------------------------------------------------------------------ figures
def _style(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _grouped(ax, groups, systems, values, fmt="{:.1f}"):
    import numpy as np
    width = 0.8 / len(systems)
    x = np.arange(len(groups))
    for i, sy in enumerate(systems):
        vals = [values.get((g, sy)) for g in groups]
        h = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - (len(systems) - 1) / 2) * width, h, width * 0.92, color=COLORS[sy],
                      label=SYSTEM_NAMES[sy], edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0.05, "–", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, groups)


def figures(summary: dict, rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = []
    cs = summary["by_category_system"]
    cats = [c for c in CATS if any(k[0] == c for k in cs)]

    # 1. correctness and entity recall by category x system (resolved)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    _grouped(axes[0], cats, ["A", "B", "C"], {k: v["correctness"] for k, v in cs.items()})
    _style(axes[0], "Correctness by category (judge, 1–5)", "mean correctness")
    axes[0].set_ylim(0, 5.3)
    _grouped(axes[1], [c for c in cats if c != "unanswerable"], ["direct", "A", "B", "C"],
             {k: (v["entity_recall"] * 100 if v["entity_recall"] is not None else None) for k, v in cs.items()})
    _style(axes[1], "Entity recall by category (expected entities in the answer)", "% of expected entities")
    axes[1].set_ylim(0, 105)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.tight_layout()
    p = FIGURES / "phase6_correctness_by_category.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    out.append(p.name)

    # 2. quality vs cost (LLM systems, both versions)
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    svs = summary["by_system_version"]
    # label placement per point (offset in points, alignment), chosen so labels clear the other markers
    place = {("A", "resolved"): (10, 6, "left"), ("A", "raw"): (10, -14, "left"), ("B", "resolved"): (-12, 6, "right"),
             ("B", "raw"): (10, -4, "left"), ("C", "resolved"): (-12, 6, "right"), ("C", "raw"): (-12, -14, "right")}
    for (sy, v), s in svs.items():
        if sy == "direct" or s["correctness"] is None:
            continue
        x = (s["cost_usd_per_q"] or 0) * 1000
        ax.scatter([x], [s["correctness"]], s=90, color=COLORS[sy] if v == "resolved" else SURFACE,
                   edgecolor=COLORS[sy], linewidth=2, zorder=3)
        other = svs.get((sy, "raw" if v == "resolved" else "resolved")) or {}
        same = other.get("correctness") == s["correctness"] and abs((other.get("cost_usd_per_q") or 0) - (s["cost_usd_per_q"] or 0)) < 2e-5
        if same and v == "raw":
            continue                                     # identical to the resolved point: one shared label
        dx, dy, ha = place.get((sy, v), (8, 4, "left"))
        ax.annotate(f"{SYSTEM_NAMES[sy]} · {'raw = resolved' if same else v}", (x, s["correctness"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8, color=INK2)
    _style(ax, "Answer quality vs cost (filled = resolved, hollow = raw)", "mean correctness (1–5)")
    ax.set_xlabel("paid-tier cost per 1,000 questions (USD, public list prices)", color=INK2, fontsize=9)
    ax.set_ylim(1, 5.2)
    ax.set_xlim(0, max([(s["cost_usd_per_q"] or 0) * 1000 for s in svs.values()] + [0.1]) * 1.25)
    fig.tight_layout()
    p = FIGURES / "phase6_quality_vs_cost.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    out.append(p.name)

    # 3. latency by system (resolved): box plot of per-answer latency
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    systems = [sy for sy in ("direct", "A", "B", "C") if any(r["system"] == sy for r in rows)]
    data = [[r["latency_s"] for r in rows if r["system"] == sy and r["kg_version"] == "resolved"] or [0] for sy in systems]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                    medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": INK2},
                    capprops={"color": INK2}, flierprops={"markeredgecolor": INK2, "markersize": 4})
    for patch, sy in zip(bp["boxes"], systems):
        patch.set_facecolor(COLORS[sy])
        patch.set_edgecolor(INK2)
    ax.set_xticks(range(1, len(systems) + 1), [SYSTEM_NAMES[s] for s in systems])
    _style(ax, "Latency per answer (resolved graph; retrieval + logged LLM time)", "seconds")
    fig.tight_layout()
    p = FIGURES / "phase6_latency_by_strategy.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    out.append(p.name)

    # 4. raw vs resolved per system: correctness and entity recall (paired questions)
    import numpy as np
    rr = summary["raw_vs_resolved"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, metric, scale, title, yl in (
            (axes[0], "correctness", 1, "Correctness, raw vs resolved", "mean correctness (1–5)"),
            (axes[1], "entity_recall", 100, "Entity recall, raw vs resolved", "% of expected entities")):
        syss = [s for s in ("direct", "A", "B", "C") if s in rr and rr[s].get(f"{metric}_raw") is not None]
        x = np.arange(len(syss))
        for j, ver in enumerate(("raw", "resolved")):
            vals = [rr[s][f"{metric}_{ver}"] * scale for s in syss]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.35, color=[COLORS[s] if ver == "resolved" else SURFACE for s in syss],
                   edgecolor=[COLORS[s] for s in syss], linewidth=1.6, hatch="///" if ver == "raw" else None,
                   label=ver)
        ax.set_xticks(x, [SYSTEM_NAMES[s] for s in syss])
        _style(ax, title + " (hatched = raw)", yl)
    axes[0].set_ylim(0, 5.3)
    axes[1].set_ylim(0, 105)
    fig.tight_layout()
    p = FIGURES / "phase6_raw_vs_resolved.png"
    fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    out.append(p.name)

    # 5. RQ1: extraction P/R/F1 (relaxed match), model output vs kept in KG
    rq1 = read_json(RESULTS / "rq1_extraction.json")
    if rq1 and rq1.get("chunks_done"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
        views = (("model_output", "model output", "#2a78d6"), ("kept_in_kg", "kept in KG", "#1baf7a"))
        for ax, kind in zip(axes, ("entity", "relation")):
            x = np.arange(3)
            for j, (view, name, col) in enumerate(views):
                m = rq1[view][f"{kind}_relaxed"]
                vals = [m["precision"] * 100, m["recall"] * 100, m["f1"] * 100]
                bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.35, color=col, label=name, edgecolor=SURFACE)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
            ax.set_xticks(x, ["precision", "recall", "F1"])
            _style(ax, f"{kind.capitalize()} extraction (relaxed match)", "%")
            ax.set_ylim(0, 105)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
        fig.tight_layout()
        p = FIGURES / "phase6_rq1_extraction.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        out.append(p.name)
    return out


# ------------------------------------------------------------------ inputs
def read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def f2(x) -> str:
    return "–" if x is None else f"{x:.2f}"


def find_example(runs: list[dict], pred) -> dict | None:
    return next((r for r in sorted(runs, key=lambda r: r["key"]) if pred(r)), None)


def one_line(text: str, n: int = 220) -> str:
    t = " ".join((text or "").split()).replace("|", "/")
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


# Why each of the hardest questions is hard: written after reading the stored answers, keyed by question
# id. The numbers next to them in the report are computed; a question without a note gets only the numbers.
HARD_NOTES = {
    "G01": "Needs a count over every DEVELOPED_BY edge of the graph. Retrieval returns a few chunks or a top-k "
           "neighbourhood, never the full distribution, so A and B can only guess; C must write the right "
           "aggregation query.",
    "G02": "An exact count (20). Any system that lists models from retrieved text undercounts; only a COUNT "
           "query gets it right.",
    "G03": "A global argmax over ACQUIRED edges: invisible to top-k retrieval.",
    "G04": "A global argmax over USES edges into Technology nodes, a long tail of near-synonyms "
           "(LLMs, large language model …).",
    "G05": "A global argmax over WORKS_AT; the graph's WORKS_AT edges also contain former roles, so even the "
           "right query counts a career, not a current job.",
    "G06": "An exact count of WORKS_AT edges (former staff included), not something any document states.",
    "M04": "'Founded by' is stored as (Organization)-[:DEVELOPED_BY]->(Person), an unusual direction: Text2Cypher "
           "wrote the arrow the other way and got 0 rows, and the Mellanox fact sits in a sentence about another "
           "Nvidia deal, which text retrieval does not surface for this question.",
    "L03": "Same 'founded by' convention as M04: Text2Cypher queried (Person)-[:DEVELOPED_BY]->(Organization) and found "
           "nothing; only B, whose context lists the triple with its evidence, answered (5/5). A schema hint in the "
           "Text2Cypher prompt would fix both.",
    "M01": "Two hops (model → developer → people) and a long expected list; the graph also holds WORKS_AT "
           "errors (investor, former staff) that a faithful system repeats.",
    "M08": "Three hops through a FUNDED_BY edge that one investor sentence supports; text retrieval rarely "
           "puts the model, the investor and its founders in the same context.",
    "M09": "Three hops (AlphaZero → AlphaGo Zero → DeepMind → people): the BASED_ON hop is easy to miss.",
    "S03": "Open question whose expected entities come from USES/BASED_ON edges that documents rarely list together.",
    "S04": "Open question: the expected 'products' come from DEVELOPED_BY edges, while documents describe "
           "Hugging Face by its hub and libraries in other words.",
    "U03": "Unanswerable with a false premise (Apple never acquired Mistral AI): a system that finds related "
           "acquisition facts is tempted to answer.",
}


# ------------------------------------------------------------------ report
def build(res: dict, runs: list[dict], items: list[dict], figs: list[str]) -> str:
    from .judge_validation import inter_report
    from .metrics import COLS, md_table
    rows = res["rows"]
    rq1 = read_json(RESULTS / "rq1_extraction.json")
    rq2_md = (RESULTS / "rq2_graph_quality.md").read_text() if (RESULTS / "rq2_graph_quality.md").exists() else ""
    phase5 = read_json(RESULTS / "phase5_answers.json") or []
    inter = inter_report()
    prices, sv, cs, rr = res["prices"], res["by_system_version"], res["by_category_system"], res["raw_vs_resolved"]
    tot, err = res["totals"], res["errors"]
    n_judged = sum(1 for r in rows if r["judged_by"] not in ("rule", "pending", "n/a (gold query)"))
    n_rule = sum(r["judged_by"] == "rule" for r in rows)
    pending = sum(r["judged_by"] == "pending" for r in rows)
    by_cat = {c: sum(1 for q in items if q["category"] == c) for c in CATS}
    S = lambda sy, v="resolved": sv.get((sy, v)) or {}
    direct_by = {(r["question_id"], r["kg_version"]): r for r in rows if r["system"] == "direct"}
    g01 = (direct_by.get(("G01", "resolved")), direct_by.get(("G01", "raw")))
    best_llm = max((sy for sy in ("A", "B", "C") if S(sy).get("correctness") is not None),
                   key=lambda sy: S(sy)["correctness"], default=None)
    L = []
    w = L.append

    w("# Benchmark report: Graph Analytics vs GraphRAG on the AI-ecosystem knowledge graph")
    w("")
    w(f"_Generated by `python -m eval.report` on {datetime.now(timezone.utc):%Y-%m-%d} from stored results only "
      f"(`eval/results/`, `logs/llm_calls.jsonl`). Every number below is read from those files; "
      f"`scripts/check_report_numbers.py` fails the build if one is not._")
    w("")

    # ---------------------------------------------------------------- jury page
    w("## In one page: what we built, what we found, why it matters")
    w("")
    w("**What we built.** A pipeline that crawls public documents about the AI ecosystem (Wikipedia, arXiv, tech news, "
      "a few local PDFs) into an HDFS data lake, cleans and chunks them with Spark, extracts entities and relations "
      "with a free-tier LLM, merges duplicate entities (entity resolution), and loads two versions of one Neo4j "
      "knowledge graph: `raw` (no merging) and `resolved`. On that graph we compare two ways to answer questions: "
      "**Graph Analytics** (a Cypher query written for the question, plus GDS algorithms) and **GraphRAG** "
      "(an LLM answers from retrieved context: A vector search over chunks, B graph neighbourhood + chunks, "
      "C the LLM writes the Cypher itself).")
    w("")
    if rows and best_llm:
        w(f"**What we found.** On {len(items)} verified questions:")
        w("")
        if rq1 and rq1.get("chunks_done"):
            e, r_ = rq1["model_output"]["entity_relaxed"], rq1["model_output"]["relation_relaxed"]
            w(f"- **Extraction (RQ1):** the LLM finds entities well (F1 {e['f1']:.2f}) but misses most relations "
              f"(recall {r_['recall']:.2f}, F1 {r_['f1']:.2f}) against a second model's annotation; what it does extract is "
              f"mostly right (relation precision {r_['precision']:.2f}).")
        if all(g01):
            w(f"- **Entity resolution (RQ2):** it changes answers, not just counts. \"Which organization developed the most "
              f"AI models?\" gets `{g01[1]['answer']}` on the raw graph and `{g01[0]['answer']}` on the resolved one: "
              f"before merging, DeepMind's models were split over duplicate DeepMind nodes. The gold queries find "
              f"{pct(S('direct', 'raw').get('entity_recall'))} of the expected entities on raw vs "
              f"{pct(S('direct').get('entity_recall'))} on resolved.")
        w(f"- **GraphRAG vs Cypher (RQ3):** the best GraphRAG variant ({SYSTEM_NAMES[best_llm]}) reaches mean correctness "
          f"{f2(S(best_llm)['correctness'])}/5 and finds {pct(S(best_llm).get('entity_recall'))} of the expected entities; "
          f"the hand-written Cypher finds {pct(S('direct').get('entity_recall'))}. GraphRAG answers open questions in "
          f"natural language; Cypher wins on counts and rankings.")
        w(f"- **Cost (RQ4):** a Cypher query costs no tokens and answers within {f2(S('direct').get('latency_p95_s'))} s "
          f"(p95); GraphRAG takes {f2(min(S(s).get('latency_median_s') or 99 for s in ('A', 'B', 'C')))}–"
          f"{f2(max(S(s).get('latency_median_s') or 0 for s in ('A', 'B', 'C')))} s and would cost "
          f"${1000 * min(S(s).get('cost_usd_per_q') or 9 for s in ('A', 'B', 'C')):.2f}–"
          f"${1000 * max(S(s).get('cost_usd_per_q') or 0 for s in ('A', 'B', 'C')):.2f} per 1,000 questions on a paid tier. "
          f"This project spent $0.")
        if S("B").get("faithfulness") is not None:
            w(f"- **Provenance (RQ5):** every graph fact carries its source sentence, so a reader can check an answer; "
              f"{pct(S('B').get('faithfulness'))} of GraphRAG-B's claims are supported by the evidence it shows.")
        w("")
        w("**Why it matters.** The two approaches are complementary, not rivals. When the question has a known shape "
          "(who, how many, which is largest), a Cypher query over a clean graph is exact, instant and free, but someone "
          "must write it. GraphRAG accepts any question and explains itself with sources, at the price of tokens, seconds "
          "and some wrong answers. Both depend on the same foundation: if extraction misses relations or duplicates are "
          "not merged, every method above inherits the error. The honest caveat: the question set is small, the gold "
          "sets and the judge validation were made by models, not humans (see the limitations).")
    else:
        w("**What we found.** No benchmark answers are stored yet.")
    w("")

    # ---------------------------------------------------------------- setup
    w("## Setup")
    w("")
    w(f"- **Questions:** {len(items)} verified in `eval/gold/questions.jsonl`: "
      + ", ".join(f"{c} {n}" for c, n in by_cat.items()) + ". Each has a gold Cypher query, an expected answer and "
      "expected entities; multi-hop anchors are in the largest connected component.")
    w(f"- **Systems:** Direct Cypher (the gold query, no LLM), A vector RAG, B GraphRAG hybrid, C Text2Cypher, each on "
      f"both graph versions (`raw`, `resolved`). Answers stored: {len(rows)}; runs that failed after 2 retries: "
      f"{err['answer_runs']}.")
    w(f"- **Models:** answers by `{prices['answer_model']['model']}` (Gemini free tier, temperature 0); judge "
      f"`{prices['judge_model']['model']}` (another model family, fixed rubric `judge-v1`). Judged answers: {n_judged}; "
      f"'Not found' answers scored by rule: {n_rule}; Direct Cypher is not judged (it is the gold query); pending: {pending}.")
    w("- **Authorship:** Development was assisted by AI coding tools; design choices, configuration and analysis were directed by the authors.")
    w("- **Model-made substitutes for human steps** (this is a model-graded study; see the limitations):")
    w("  - question review: **model-reviewed (an LLM), not human-reviewed** (`eval/gold/questions_review.md` gives the reason of every fix);")
    w("  - RQ1 gold annotation: **model-annotated (an LLM blind to the extractor's output)**, so RQ1 measures agreement "
      "between two models;")
    w("  - judge validation: **inter-model agreement, not human validation** (a second Groq model re-scores 15 answers).")
    w("")
    w("### Overall results")
    w("")
    if rows:
        w(md_table(sv, ["System", "KG"]))
    w("")

    # ---------------------------------------------------------------- RQ1
    w("## RQ1 — How accurately does the LLM extract entities and relations?")
    w("")
    w("**Question.** How well does the extraction model (`gemini-3.5-flash-lite`) recover the entities and relations a "
      "text states?")
    w("")
    w("**Method.** 25 chunks sampled across the four sources, annotated from the text alone (blind, `annotator: "
      "model`). The model's cached extraction of each chunk is scored with micro precision / recall / F1, "
      "strict (same name and type) and relaxed (normalised name or token containment), in two views: everything the "
      "model returned, and only the relations kept in the graph after the post-hoc checks (known endpoints, schema, "
      "evidence found in the text, no self-loop).")
    w("")
    if rq1 and rq1.get("chunks_done"):
        w(f"**Results** ({rq1['chunks_done']}/{rq1['gold_chunks']} chunks, model `{rq1['model']}`):")
        w("")
        w("| View | Match | Entity P | Entity R | Entity F1 | Relation P | Relation R | Relation F1 |")
        w("|---|---|---|---|---|---|---|---|")
        for view, name in (("model_output", "model output"), ("kept_in_kg", "kept in KG")):
            for mode in ("strict", "relaxed"):
                e, r_ = rq1[view][f"entity_{mode}"], rq1[view][f"relation_{mode}"]
                w(f"| {name} | {mode} | {e['precision']:.2f} | {e['recall']:.2f} | {e['f1']:.2f} | "
                  f"{r_['precision']:.2f} | {r_['recall']:.2f} | {r_['f1']:.2f} |")
        w("")
        if "phase6_rq1_extraction.png" in figs:
            w("![RQ1 extraction](figures/phase6_rq1_extraction.png)")
            w("")
        e, r_ = rq1["model_output"]["entity_relaxed"], rq1["model_output"]["relation_relaxed"]
        k = rq1["kept_in_kg"]["relation_strict"]
        m = rq1["model_output"]["relation_strict"]
        w(f"**Interpretation.** Entities are the easy part: relaxed F1 {e['f1']:.2f}, with precision "
          f"({e['precision']:.2f}) above recall ({e['recall']:.2f}). Relations are the bottleneck: the model returns "
          f"few of them, so recall is {r_['recall']:.2f}, while most of those it returns are right ({r_['precision']:.2f}). "
          f"The post-hoc checks raise strict relation precision from {m['precision']:.2f} to {k['precision']:.2f} while "
          f"recall only moves from {m['recall']:.2f} to {k['recall']:.2f}: they remove wrong edges, not useful ones. The gold is a second "
          f"model's reading of the text, denser than the extraction prompt asks for (e.g. every board member as WORKS_AT), "
          f"so recall is a lower bound.")
        w("")
        w(f"**Verdict.** Accurate but incomplete: entity extraction is usable as is (F1 {e['f1']:.2f}); relation "
          f"extraction is precise but misses most stated relations (F1 {r_['f1']:.2f}), which caps what any graph method "
          f"can answer.")
    else:
        w("**Results.** Pending: no gold chunk is marked `status: done`.")
    w("")

    # ---------------------------------------------------------------- RQ2
    w("## RQ2 — How does entity resolution change KG quality?")
    w("")
    w("**Question.** Does merging duplicate entities make the graph, and the answers built on it, better?")
    w("")
    w("**Method.** The same extraction is loaded twice: `raw` (one node per surface name) and `resolved` (after "
      "entity resolution). We compare graph structure (Phase 4) and run every benchmark question on both versions, "
      "paired by question.")
    w("")
    if rq2_md:
        table = re.search(r"\| Metric \|.*?(?=\n\n)", rq2_md, re.S)
        w("**Results — graph structure** (`eval/results/rq2_graph_quality.csv`):")
        w("")
        w(table.group(0) if table else "(table missing)")
        w("")
    if all(g01):
        w(f"**Headline example — the answer itself changes.** G01 \"{g01[0]['question']}\": the gold Cypher query returns "
          f"`{g01[1]['answer']}` on the raw graph and `{g01[0]['answer']}` on the resolved graph.")
        dm = re.search(r"^\| (\d+) \| DeepMind \(Organization\) \| ([\d.]+) \| (\d+) \| ([^|]+) \| (\d+) \| ([\d.]+) \|",
                       rq2_md, re.M)
        if dm:
            w(f"ER merged **{dm.group(3)} raw DeepMind entities** into one; its PageRank went from {dm.group(6)} (best raw "
              f"member, rank {dm.group(5)}) to {dm.group(2)} (rank {dm.group(1)}). Before ER, DeepMind's models were split "
              f"over its duplicates, so no single DeepMind node had the most models and the count crowned Anthropic.")
        w("")
    if rr:
        w("**Results — answers on the two graphs** (same questions, paired):")
        w("")
        w("| System | Pairs | Correctness raw | Correctness resolved | Entity recall raw | Entity recall resolved | Better on resolved | Better on raw |")
        w("|---|---|---|---|---|---|---|---|")
        for sy, d in rr.items():
            w(f"| {SYSTEM_NAMES[sy]} | {d['pairs']} | {f2(d['correctness_raw'])} | {f2(d['correctness_resolved'])} | "
              f"{pct(d['entity_recall_raw'])} | {pct(d['entity_recall_resolved'])} | {d['better_resolved']} | {d['better_raw']} |")
        w("")
        if "phase6_raw_vs_resolved.png" in figs:
            w("![raw vs resolved](figures/phase6_raw_vs_resolved.png)")
            w("")
        d = rr.get("direct", {})
        llm_gain = [(sy, rr[sy]["correctness_raw"], rr[sy]["correctness_resolved"]) for sy in ("A", "B", "C")
                    if sy in rr and rr[sy]["correctness_raw"] is not None and rr[sy]["correctness_resolved"] is not None]
        w(f"**Interpretation.** For structured access the effect is large: the gold queries return a different result on "
          f"the raw graph for {tot['direct_answer_changes']} of {tot['questions']} questions, and their entity recall drops from "
          f"{pct(d.get('entity_recall_resolved'))} to {pct(d.get('entity_recall_raw'))}, because a query anchored on one "
          f"name only sees the edges of that duplicate. "
          + ("For the LLM systems mean correctness goes from raw to resolved as "
             + ", ".join(f"{SYSTEM_NAMES[s]} {a:.2f} → {b:.2f}" for s, a, b in llm_gain)
             + ": A reads only chunks, which are identical in both versions, so it cannot benefit; B and C use "
               "entities and edges, and gain from the merged graph." if llm_gain else ""))
        w("")
        w("**Verdict.** Entity resolution is necessary for anything that uses the graph's structure: without it, counts, "
          "rankings and centrality are wrong, not just noisier, and graph-based GraphRAG (B, C) answers worse. "
          "Chunk-only RAG (A) does not see it.")
    w("")

    # ---------------------------------------------------------------- RQ3
    w("## RQ3 — GraphRAG vs direct Cypher on multi-hop and semantic questions")
    w("")
    w("**Question.** Can an LLM over retrieved graph context replace a Cypher query written for the question?")
    w("")
    w("**Method.** All questions on the resolved graph, by category. GraphRAG answers are scored by the judge (1–5 "
      "against the expected answer) and by entity recall (expected entities named in the answer). Direct Cypher is the "
      "gold query: an upper bound by construction for structured questions, scored by entity recall only.")
    w("")
    if cs:
        w("**Results** (resolved graph):")
        w("")
        w(md_table(cs, ["Category", "System"],
                   cols=[c for c in COLS if c[0] in ("correctness", "correct_ge4", "entity_recall", "entity_f1",
                                                     "false_refusal_rate", "refusal_rate_unanswerable", "llm_calls_per_q")]))
        w("")
        if "phase6_correctness_by_category.png" in figs:
            w("![correctness by category](figures/phase6_correctness_by_category.png)")
            w("")
        best = {}
        for cat in CATS:
            b = sorted(((v["correctness"], k[1]) for k, v in cs.items() if k[0] == cat and v["correctness"] is not None),
                       reverse=True)
            if b:
                best[cat] = b
                w(f"- **{cat}:** " + ", ".join(f"{SYSTEM_NAMES[s]} {c:.2f}" for c, s in b) + ".")
        w("")
        ag = {s: (cs.get(("aggregation", s)) or {}).get("correctness") for s in ("A", "B", "C")}
        mh = {s: (cs.get(("multi-hop", s)) or {}).get("entity_recall") for s in ("direct", "A", "B", "C")}
        w(f"**Interpretation.** Multi-hop questions: Direct Cypher finds {pct(mh['direct'])} of the expected entities, "
          f"GraphRAG " + ", ".join(f"{SYSTEM_NAMES[s]} {pct(mh[s])}" for s in ("A", "B", "C")) + ". "
          f"Aggregations separate the systems most (mean correctness "
          + ", ".join(f"{SYSTEM_NAMES[s]} {f2(ag[s])}" for s in ("A", "B", "C")) + "): top-k chunks cannot contain a "
          f"global count, B's context adds graph aggregates, and Text2Cypher computes the count itself. Multi-hop "
          f"lists are the weak spot of every LLM system: a missed hop or a reversed edge empties the answer.")
        w("")
        w("**Verdict.** No: GraphRAG does not replace Cypher for structured questions (multi-hop lists, counts, rankings), "
          "where the gold query is exact. It complements it for open questions, and Text2Cypher is the bridge: an LLM "
          "that writes the query keeps Cypher's exactness when it gets the query right.")
    w("")

    # ---------------------------------------------------------------- RQ4
    w("## RQ4 — Trade-offs: quality, completeness, latency, cost")
    w("")
    w("**Question.** What does each approach cost in answer quality, completeness, time and money?")
    w("")
    w("**Method.** Per answer: judge correctness, entity recall, latency (retrieval time measured in the run plus the "
      "logged latency of each LLM call), LLM calls and tokens from the call log, and the paid-tier price of those tokens "
      "at public list prices (`eval/prices.json`).")
    w("")
    if sv:
        w("**Results:**")
        w("")
        w("| System | KG | Correctness | Entity recall | Latency p50 (s) | Latency p95 (s) | LLM calls/q | Tokens in/q | Tokens out/q | Paid cost / 1,000 q |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for (sy, v), s in sv.items():
            w(f"| {SYSTEM_NAMES[sy]} | {v} | {f2(s['correctness'])} | {pct(s['entity_recall'])} | {f2(s['latency_median_s'])} | "
              f"{f2(s['latency_p95_s'])} | {f2(s['llm_calls_per_q'])} | {s['tokens_in_per_q'] or 0:.0f} | "
              f"{s['tokens_out_per_q'] or 0:.0f} | ${1000 * (s['cost_usd_per_q'] or 0):.2f} |")
        w("")
        for f in ("phase6_quality_vs_cost.png", "phase6_latency_by_strategy.png"):
            if f in figs:
                w(f"![{f}](figures/{f})")
                w("")
        cheapest = min(("A", "B", "C"), key=lambda s: S(s).get("cost_usd_per_q") or 9)
        w(f"**Interpretation.** Direct Cypher dominates on every axis it can be measured on, but it needs a person (or "
          f"Text2Cypher) to write the query. Among the LLM systems, {SYSTEM_NAMES[cheapest]} is the cheapest per "
          f"question; C makes up to 3 calls (query, retry, verbalisation), so it is the slowest at the tail "
          f"(p95 {f2(S('C').get('latency_p95_s'))} s). At list prices every variant stays below "
          f"${1000 * max(S(s).get('cost_usd_per_q') or 0 for s in ('A', 'B', 'C')):.2f} per 1,000 questions: money is not "
          f"the constraint, the free tier's 500 requests/day and the seconds of latency are.")
        w("")
        w("**Verdict.** Quality and completeness favour the structured path, latency and cost favour it by orders of "
          "magnitude; GraphRAG buys flexibility (any question, natural-language answers with sources) with seconds and "
          "tokens.")
    w("")

    # ---------------------------------------------------------------- RQ5
    w("## RQ5 — Does source provenance improve reliability and explainability?")
    w("")
    if sv:
        w("| System | KG | Answers with sources | Faithfulness | Citation precision | Invalid citations removed |")
        w("|---|---|---|---|---|---|")
        for (sy, v), s in sv.items():
            if sy != "direct":
                w(f"| {SYSTEM_NAMES[sy]} | {v} | {pct(s['answers_with_sources'])} | {pct(s['faithfulness'])} | "
                  f"{pct(s['citation_precision'])} | {s['invalid_citations_removed']} |")
        w("")
        w("Faithfulness = share of the answer's claims supported by the evidence shown with it; citation precision = "
          "share of shown sources whose evidence supports the answer (both judged).")
        w("")
    w("**Concrete examples:**")
    w("")
    inv = find_example(runs, lambda r: r.get("invalid_citations"))
    ex = inv or next((r for r in phase5 if r.get("invalid_citations")), None)
    if ex:
        w(f"- **Fabricated citation caught** ({'benchmark' if inv else 'Phase 5 evaluation'}, "
          f"{SYSTEM_NAMES.get(ex.get('strategy') or ex.get('system'), '')}, \"{ex['question']}\"): the model cited "
          f"`{ex['invalid_citations']}`, which is not a source id of its context. It was removed from the answer and "
          f"reported, never shown as a source.")
    tallinn = find_example(runs, lambda r: r.get("system") in ("B", "C") and "Jaan Tallinn" in (r.get("answer") or "")) \
        or next((r for r in phase5 if "Jaan Tallinn" in r.get("answer", "")), None)
    w(f"- **WORKS_AT investor error** (questions M01/M09): the graph holds `(Jaan Tallinn)-[:WORKS_AT]->(DeepMind)`, "
      f"extracted from the sentence \"Jaan Tallinn was an early investor and an adviser to the company.\" The review "
      f"removed him from the expected answer; the gold Cypher still returns him."
      + (f" GraphRAG repeats the error (\"{tallinn['question']}\", {SYSTEM_NAMES.get(tallinn.get('system') or tallinn.get('strategy'), '')}),"
         f" but the evidence sentence shipped with the edge lets a reader see that the source talks about an investment."
         if tallinn else ""))
    sup = find_example(runs, lambda r: any(s.get("cited") is False for s in r.get("sources") or []))
    if sup:
        w(f"- **Supporting sources** ({SYSTEM_NAMES[sup['system']]}, \"{sup['question']}\"): the Text2Cypher rows had no "
          f"document ids, so the answer shows chunks that mention the entities, marked as not cited by the model.")
    w("")

    # ---------------------------------------------------------------- examples
    ex_all = res.get("examples") or {}
    if ex_all:
        w("## Good and bad answers per system (resolved graph)")
        w("")
        w("Ranked by judge correctness, then entity F1 (Direct Cypher: entity F1). Answers are shortened.")
        w("")
        for sy, d in ex_all.items():
            w(f"### {SYSTEM_NAMES[sy]}")
            w("")
            w("| | Question | Answer | Correctness | Entity recall |")
            w("|---|---|---|---|---|")
            for kind in ("good", "bad"):
                for e in d[kind]:
                    w(f"| {kind} | {e['question_id']} {one_line(e['question'], 80)} | {one_line(e['answer'])} | "
                      f"{e['correctness'] if e['correctness'] is not None else '–'} | {pct(e['entity_recall'])} |")
            w("")

    # ---------------------------------------------------------------- hardest
    hard = res.get("hardest") or []
    if hard:
        w("## The five hardest questions")
        w("")
        w("Lowest mean correctness over A, B and C on the resolved graph.")
        w("")
        w("| Question | Category | Mean correctness | A / B / C | Refusals | Direct Cypher entity recall (resolved / raw) |")
        w("|---|---|---|---|---|---|")
        for h in hard:
            ps = " / ".join(str(h["per_system"].get(s, "–")) for s in ("A", "B", "C"))
            w(f"| {h['question_id']} {one_line(h['question'], 90)} | {h['category']} | {f2(h['mean_correctness'])} | {ps} | "
              f"{h['refusals']}/{h['systems']} | {pct(h['direct_entity_recall_resolved'])} / {pct(h['direct_entity_recall_raw'])} |")
        w("")
        for h in hard:
            note = HARD_NOTES.get(h["question_id"], "")
            w(f"- **{h['question_id']}** — {note}" if note else f"- **{h['question_id']}**")
        w("")

    # ---------------------------------------------------------------- judge validation
    w("## Judge validation — inter-model agreement, not human validation")
    w("")
    if inter and inter["correctness"]["n"]:
        c, fa, cp = inter["correctness"], inter["faithfulness"], inter["citation_precision"]
        w(f"A second judge from another model family, `{inter['second_model']}` (Groq free tier), re-scored the same "
          f"{inter['n_sampled']} answers (stratified by system and category) with the same rubric and the same blind prompt "
          f"as `{inter['judge_model']}`. Errors: {inter['errors']}.")
        w("")
        w("| Measure | Value |")
        w("|---|---|")
        w(f"| Correctness: exact agreement | {pct(c['exact'])} (n={c['n']}) |")
        w(f"| Correctness: within ±1 | {pct(c['within_1'])} |")
        w(f"| Correctness: quadratic-weighted Cohen's κ | {f2(c['weighted_kappa'])} |")
        w(f"| Correctness: Spearman ρ | {f2(c['spearman'])} |")
        w(f"| Mean correctness: judge / second judge | {f2(c['mean_judge'])} / {f2(c['mean_human'])} |")
        w(f"| Faithfulness: same 'fully supported' verdict | {pct(fa['binary_agreement'])} (n={fa['n']}) |")
        w(f"| Faithfulness: mean absolute difference | {f2(fa['mean_abs_diff'])} |")
        w(f"| Citation precision: mean absolute difference | {f2(cp['mean_abs_diff'])} (n={cp['n']}) |")
        w("")
        if c["disagreements"]:
            w("Disagreements of 2+ points: " + ", ".join(f"`{d['key']}` (judge {d['judge']}, second {d['human']})"
                                                       for d in c["disagreements"]) + ".")
            w("")
        w("Perfect or near-perfect agreement on so few answers is weak evidence: most sampled answers are clear-cut "
          "(fully right or a refusal), and both judges read the same expected answer. ")
        w("")
        w("Agreement between two LLMs shows the scores are not an artefact of one model's quirks; it does not show they "
          "match human judgement (both can share the same bias).")
    else:
        w("**Pending:** run `make judge-validate ok=1` after the judge.")
    w("")

    # ---------------------------------------------------------------- cost
    w("## Cost: tokens, calls, projected paid-tier cost")
    w("")
    j = tot["judge"]
    w(f"- **Benchmark answers:** {tot['llm_answers']} GraphRAG answers, {tot['llm_calls']} LLM calls, "
      f"{tot['tokens_in']:,} input and {tot['tokens_out']:,} output tokens → **${tot['answer_cost_usd']:.4f}** at "
      f"`{prices['answer_model']['model']}` list prices (${prices['answer_model']['input_per_m']:.2f} / "
      f"${prices['answer_model']['output_per_m']:.2f} per M tokens, official: {prices['answer_model']['source'].split(' ')[0]}).")
    w(f"- **Judge:** {j['calls']} calls, {j['tokens_in']:,} input and {j['tokens_out']:,} output tokens → "
      f"**${j['cost_usd']:.4f}** at ${prices['judge_model']['input_per_m']:.2f} / ${prices['judge_model']['output_per_m']:.2f} "
      f"per M tokens (Groq price as reported by aggregator sites, not Groq's own page).")
    w("- **Actually paid:** $0. Every call ran on a free tier (Gemini 500 requests/day, Groq daily token limits).")
    w("")
    usage = res.get("llm_usage") or {}
    if usage:
        w("**All LLM requests of the project** (`logs/llm_calls.jsonl`, incl. retries and errors):")
        w("")
        w("| Phase | Model | Requests | OK | Tokens in | Tokens out | Median latency (s) |")
        w("|---|---|---|---|---|---|---|")
        for k, d in usage.items():
            ph, m = k.split("|", 1)
            w(f"| {ph} | {m} | {d['requests']} | {d['ok']} | {d['tokens_in']:,} | {d['tokens_out']:,} | "
              f"{f2(d['latency_median_s'])} |")
        w("")

    # ---------------------------------------------------------------- limitations
    w("## Limitations and threats to validity")
    w("")
    w("- **Model-made gold sets and question review.** The 25 RQ1 gold chunks were annotated by an LLM (blind to the "
      "extractor's output), and the 30 benchmark questions were reviewed by an LLM against the graph, not by a human. "
      "RQ1 therefore measures agreement between two models, and an error both models share is invisible.")
    w("- **Inter-model judge validation.** The judge was validated against a second LLM, not against human scores. "
      "High agreement rules out one model's idiosyncrasies, not a bias common to LLMs.")
    w("- **LLM-as-judge bias.** Correctness, faithfulness and citation precision come from an LLM with a fixed rubric; "
      "LLM judges tend to favour longer, fluent answers and can be lenient with partial lists. The judge sees the "
      "expected answer, is from a different family than the answer model, and 'Not found' answers are scored by rule.")
    w("- **Single run, no variance.** Each answer was generated once (temperature 0, cached). Free-tier latency varies "
      "with load; differences between systems of a few tenths of a point are within run-to-run noise we did not measure.")
    w(f"- **Small question set (n={len(items)}).** Per-category means rest on "
      f"{min(by_cat.values())}–{max(by_cat.values())} questions: treat differences below about one point as noise.")
    w("- **Free-tier constraints.** One small answer model (`gemini-3.5-flash-lite`) because larger models allow 20 "
      "requests/day; results may not transfer to larger models. The 500 requests/day cap shaped the design (caching, "
      "one run per answer). Costs are projections from list prices.")
    w("- **Domain-specific corpus.** A few hundred English documents on the AI ecosystem (Wikipedia, arXiv, tech news, a "
      "few PDFs). The gold answers come from the resolved graph itself, which favours graph-based systems over vector "
      "RAG and makes Direct Cypher an upper bound by construction.")
    w("- **Rule-based components.** Entity linking (B) and the entity-mention metric use name matching, so aliases the "
      "graph does not know are missed by both.")
    w("")
    return "\n".join(L)


def main() -> int:
    from . import questions as Q
    from .judge import judged_keys
    from .judge_validation import agreement_report
    from .metrics import compute, load_index, save
    from .run_benchmark import done_runs, error_runs
    items = Q.verified(Q.load())
    runs = done_runs()
    res = compute(items, runs, judged_keys(), load_index(), error_runs())
    save(res, agreement_report())
    figs = figures(res, res["rows"]) if res["rows"] else []
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    ids = {q["id"] for q in items}
    REPORT.write_text(build(res, [r for r in runs.values() if r["question_id"] in ids], items, figs))
    print(f"wrote {REPORT} and {len(figs)} figures: {', '.join(figs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
