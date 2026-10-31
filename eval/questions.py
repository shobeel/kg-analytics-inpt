"""Phase 6 question set: 30 drafts with hand-checkable gold Cypher, generated from the resolved graph
and verified by hand before they count.

  draft         run every gold query on kg_version=resolved, fill expected entities/answer from the rows,
                check multi-hop anchors are in the largest connected component, write
                eval/gold/questions.jsonl (verified=false) and eval/gold/questions_review.md.
                Existing files are kept (a review in progress is never overwritten) unless --force.
  apply-review  read the ticks and edits in questions_review.md back into questions.jsonl
                ([x] OK or [x] FIX -> verified=true, [x] DROP -> dropped, unticked -> verified=false)
  status        counts by category and verification state

Every gold query takes $kg_version and filters entities AND relations on it, so the same query is the
"Direct Cypher" system on both graph versions. Column `entity` holds the answer entities.
No LLM calls here.
"""

import argparse
import json
import re
import sys
from pathlib import Path

GOLD_DIR = Path(__file__).parent / "gold"
QUESTIONS = GOLD_DIR / "questions.jsonl"
REVIEW = GOLD_DIR / "questions_review.md"
CATEGORIES = {"lookup": 6, "multi-hop": 9, "aggregation": 6, "semantic": 6, "unanswerable": 3}
MAX_EXPECTED = 15

V = "$kg_version"


def rel(*names: str) -> str:
    return " AND ".join(f"{n}.kg_version = {V}" for n in names)


# (id, category, question, gold query, expected-answer template, extra)
#   template fields: {entities} (all expected entities), {first}, {n} (column n of the first row)
#   extra: anchor = question entity that must be in the LCC (multi-hop), hops, entities = explicit list
DRAFTS = [
    # ---------------------------------------------------------------- lookup (1 hop)
    ("L01", "lookup", "Who developed Claude?", f"""
MATCH (m:Entity {{kg_version: {V}, name: 'Claude'}})-[r:DEVELOPED_BY]->(o:Organization)
WHERE {rel('r')}
RETURN DISTINCT o.name AS entity""", "{entities}", {}),
    ("L02", "lookup", "Where is Cohere headquartered?", f"""
MATCH (o:Organization {{kg_version: {V}, name: 'Cohere'}})-[r:LOCATED_IN]->(l:Location)
WHERE {rel('r')}
RETURN DISTINCT l.name AS entity""", "{entities}", {}),
    ("L03", "lookup", "Who founded Hugging Face?", f"""
MATCH (o:Organization {{kg_version: {V}, name: 'Hugging Face'}})-[r:DEVELOPED_BY]-(p:Person)
WHERE {rel('r')}
RETURN DISTINCT p.name AS entity""", "Hugging Face was founded by {entities}.", {}),
    ("L04", "lookup", "Who created the ImageNet dataset?", f"""
MATCH (d:Dataset {{kg_version: {V}, name: 'ImageNet'}})-[r:DEVELOPED_BY]->(p)
WHERE {rel('r')}
RETURN DISTINCT p.name AS entity""", "{entities}", {}),
    ("L05", "lookup", "Which company developed GPT-4?", f"""
MATCH (m:AIModel {{kg_version: {V}, name: 'GPT-4'}})-[r:DEVELOPED_BY]->(o:Organization)
WHERE {rel('r')}
RETURN DISTINCT o.name AS entity""", "{entities}", {}),
    ("L06", "lookup", "Which companies has Mistral AI acquired?", f"""
MATCH (o:Organization {{kg_version: {V}, name: 'Mistral AI'}})-[r:ACQUIRED]->(x:Organization)
WHERE {rel('r')}
RETURN DISTINCT x.name AS entity""", "Mistral AI acquired {entities}.", {}),

    # ---------------------------------------------------------------- multi-hop (2-3 hops, LCC)
    ("M01", "multi-hop", "Which people work at the organization that developed AlphaFold?", f"""
MATCH (m:Entity {{kg_version: {V}, name: 'AlphaFold'}})-[r1:DEVELOPED_BY]->(o:Organization)<-[r2:WORKS_AT]-(p:Person)
WHERE {rel('r1', 'r2')}
RETURN o.name AS via, p.name AS entity ORDER BY r2.mention_count DESC, entity""",
     "AlphaFold was developed by DeepMind; people working there include {entities}.",
     {"anchor": "AlphaFold", "hops": 2}),
    ("M02", "multi-hop", "In which city is the company that developed Mixtral 8x7B headquartered?", f"""
MATCH (m:AIModel {{kg_version: {V}, name: 'Mixtral 8x7B'}})-[r1:DEVELOPED_BY]->(o:Organization)-[r2:LOCATED_IN]->(l:Location)
WHERE {rel('r1', 'r2')}
RETURN o.name AS via, l.name AS entity""", "Mixtral 8x7B was developed by Mistral AI, located in {entities}.",
     {"anchor": "Mixtral 8x7B", "hops": 2}),
    ("M03", "multi-hop", "Which companies were acquired by the developer of Claude?", f"""
MATCH (m:Entity {{kg_version: {V}, name: 'Claude'}})-[r1:DEVELOPED_BY]->(o:Organization)-[r2:ACQUIRED]->(x:Organization)
WHERE {rel('r1', 'r2')}
RETURN DISTINCT o.name AS via, x.name AS entity""", "Claude's developer Anthropic acquired {entities}.",
     {"anchor": "Claude", "hops": 2}),
    ("M04", "multi-hop", "Who founded the company that acquired Mellanox?", f"""
MATCH (x:Organization {{kg_version: {V}, name: 'Mellanox'}})<-[r1:ACQUIRED]-(o:Organization)-[r2:DEVELOPED_BY]-(p:Person)
WHERE {rel('r1', 'r2')}
RETURN DISTINCT o.name AS via, p.name AS entity""", "Mellanox was acquired by Nvidia, founded by {entities}.",
     {"anchor": "Mellanox", "hops": 2}),
    ("M05", "multi-hop", "Which AI models were developed by the company that acquired Aleph Alpha?", f"""
MATCH (x:Organization {{kg_version: {V}, name: 'Aleph Alpha'}})<-[r1:ACQUIRED]-(o:Organization)<-[r2:DEVELOPED_BY]-(m:AIModel)
WHERE {rel('r1', 'r2')}
RETURN DISTINCT o.name AS via, m.name AS entity""", "Aleph Alpha was acquired by Cohere, which developed {entities}.",
     {"anchor": "Aleph Alpha", "hops": 2}),
    ("M06", "multi-hop", "Where is the organization that developed ChatGPT located?", f"""
MATCH (m:Entity {{kg_version: {V}, name: 'ChatGPT'}})-[r1:DEVELOPED_BY]->(o:Organization)-[r2:LOCATED_IN]->(l:Location)
WHERE {rel('r1', 'r2')}
RETURN DISTINCT o.name AS via, l.name AS entity""", "ChatGPT was developed by OpenAI, located in {entities}.",
     {"anchor": "ChatGPT", "hops": 2}),
    ("M07", "multi-hop", "Which people work at the company that developed Aya Vision?", f"""
MATCH (m:Entity {{kg_version: {V}, name: 'Aya Vision'}})-[r1:DEVELOPED_BY]->(o:Organization)<-[r2:WORKS_AT]-(p:Person)
WHERE {rel('r1', 'r2')}
RETURN DISTINCT o.name AS via, p.name AS entity""", "Aya Vision was developed by Cohere; people working there: {entities}.",
     {"anchor": "Aya Vision", "hops": 2}),
    ("M08", "multi-hop", "Who founded the company that invested in the developer of the Jamba model?", f"""
MATCH (m:AIModel {{kg_version: {V}, name: 'Jamba'}})-[r1:DEVELOPED_BY]->(o:Organization)-[r2:FUNDED_BY]->(f:Organization)-[r3:DEVELOPED_BY]-(p:Person)
WHERE {rel('r1', 'r2', 'r3')}
RETURN DISTINCT o.name AS developer, f.name AS investor, p.name AS entity""",
     "Jamba was developed by AI21 Labs, which is funded by Nvidia, founded by {entities}.",
     {"anchor": "Jamba", "hops": 3}),
    ("M09", "multi-hop", "Which people work at the organization that developed the model AlphaZero is based on?", f"""
MATCH (z:AIModel {{kg_version: {V}, name: 'AlphaZero'}})-[r1:BASED_ON]->(b:Entity)-[r2:DEVELOPED_BY]->(o:Organization)<-[r3:WORKS_AT]-(p:Person)
WHERE {rel('r1', 'r2', 'r3')}
RETURN DISTINCT b.name AS base_model, o.name AS via, p.name AS entity""",
     "AlphaZero is based on AlphaGo Zero, developed by DeepMind; people working there include {entities}.",
     {"anchor": "AlphaZero", "hops": 3}),

    # ---------------------------------------------------------------- aggregation
    ("G01", "aggregation", "Which organization developed the most AI models?", f"""
MATCH (m:AIModel {{kg_version: {V}}})-[r:DEVELOPED_BY]->(o:Organization)
WHERE {rel('r')}
RETURN o.name AS entity, count(DISTINCT m) AS n ORDER BY n DESC, entity LIMIT 1""", "{first} ({n} AI models)", {}),
    ("G02", "aggregation", "How many AI models did Mistral AI develop?", f"""
MATCH (m:AIModel {{kg_version: {V}}})-[r:DEVELOPED_BY]->(o:Organization {{name: 'Mistral AI'}})
WHERE {rel('r', 'o')}
RETURN o.name AS entity, count(DISTINCT m) AS n""", "{n} AI models", {}),
    ("G03", "aggregation", "Which organization has acquired the most companies?", f"""
MATCH (o:Organization {{kg_version: {V}}})-[r:ACQUIRED]->(x:Organization)
WHERE {rel('r')}
RETURN o.name AS entity, count(DISTINCT x) AS n ORDER BY n DESC, entity LIMIT 1""", "{first} ({n} acquisitions)", {}),
    ("G04", "aggregation", "Which technology is used by the most entities in the knowledge graph?", f"""
MATCH (x:Entity {{kg_version: {V}}})-[r:USES]->(t:Technology)
WHERE {rel('r')}
RETURN t.name AS entity, count(DISTINCT x) AS n ORDER BY n DESC, entity LIMIT 1""", "{first} (used by {n} entities)", {}),
    ("G05", "aggregation", "Which person has worked at the most organizations?", f"""
MATCH (p:Person {{kg_version: {V}}})-[r:WORKS_AT]->(o:Organization)
WHERE {rel('r')}
RETURN p.name AS entity, count(DISTINCT o) AS n ORDER BY n DESC, entity LIMIT 1""", "{first} ({n} organizations)", {}),
    ("G06", "aggregation", "How many people in the knowledge graph work at OpenAI?", f"""
MATCH (p:Person {{kg_version: {V}}})-[r:WORKS_AT]->(o:Organization {{name: 'OpenAI'}})
WHERE {rel('r', 'o')}
RETURN o.name AS entity, count(DISTINCT p) AS n""", "{n} people", {}),

    # ---------------------------------------------------------------- semantic / open
    ("S01", "semantic", "What is DeepMind known for?", f"""
MATCH (o:Organization {{kg_version: {V}, name: 'DeepMind'}})<-[r:DEVELOPED_BY]-(m:AIModel)
WHERE {rel('r')}
RETURN o.description AS description, m.name AS entity ORDER BY m.mention_count DESC, entity LIMIT 5""",
     "DeepMind is an AI research laboratory known for models such as {entities}.", {}),
    ("S02", "semantic", "What is Constitutional AI?", f"""
MATCH (t:Entity {{kg_version: {V}, name: 'Constitutional AI'}})
OPTIONAL MATCH (t)-[r:DEVELOPED_BY]->(o:Organization) WHERE {rel('r')}
RETURN t.description AS description, o.name AS entity""",
     "A method for training a harmless AI assistant through self-improvement, developed by {entities}.", {}),
    ("S03", "semantic", "What is the Transformer architecture?", f"""
MATCH (t:Technology {{kg_version: {V}, name: 'Transformer'}})
OPTIONAL MATCH (m:AIModel)-[r:USES|BASED_ON]->(t) WHERE {rel('r', 'm')}
RETURN t.description AS description, m.name AS entity ORDER BY m.mention_count DESC LIMIT 5""",
     "A family of neural network architectures based on multi-head attention, used by models such as {entities}.", {}),
    ("S04", "semantic", "What is Hugging Face?", f"""
MATCH (o:Organization {{kg_version: {V}, name: 'Hugging Face'}})
OPTIONAL MATCH (o)<-[r:DEVELOPED_BY]-(x:Entity) WHERE {rel('r')}
RETURN o.description AS description, x.name AS entity ORDER BY x.mention_count DESC LIMIT 5""",
     "An American AI company based in New York City that hosts models and datasets and develops tools such as {entities}.", {}),
    ("S05", "semantic", "What is ImageNet used for?", f"""
MATCH (d:Dataset {{kg_version: {V}, name: 'ImageNet'}})
OPTIONAL MATCH (x:Entity)-[r:TRAINED_ON|USES]->(d) WHERE {rel('r')}
OPTIONAL MATCH (d)-[r2:DEVELOPED_BY]->(p:Person) WHERE {rel('r2')}
RETURN d.description AS description, collect(DISTINCT x.name)[..5] AS used_by, p.name AS entity""",
     "A large visual database for visual object recognition research, created by {entities}.", {}),
    ("S06", "semantic", "What is Geoffrey Hinton known for?", f"""
MATCH (p:Person {{kg_version: {V}, name: 'Geoffrey Hinton'}})<-[r:DEVELOPED_BY]-(x:Entity)
WHERE {rel('r')}
RETURN p.description AS description, x.name AS entity ORDER BY x.mention_count DESC, entity""",
     "A deep learning pioneer known for {entities}.", {}),

    # ---------------------------------------------------------------- unanswerable (gold query must return 0 rows)
    ("U01", "unanswerable", "Who is the CEO of INPT?", f"""
MATCH (o:Entity {{kg_version: {V}}})<-[r:WORKS_AT]-(p:Person)
WHERE {rel('r')} AND (toLower(o.name) CONTAINS 'inpt' OR toLower(o.name) CONTAINS 'institut national des postes')
RETURN p.name AS entity""", "Not found in the knowledge graph.", {}),
    ("U02", "unanswerable", "Which company developed the AI model Zephyrion-9?", f"""
MATCH (m:Entity {{kg_version: {V}}})-[r:DEVELOPED_BY]->(o)
WHERE {rel('r')} AND toLower(m.name) CONTAINS 'zephyrion'
RETURN o.name AS entity""", "Not found in the knowledge graph.", {}),
    ("U03", "unanswerable", "How much did Apple pay to acquire Mistral AI?", f"""
MATCH (a:Organization {{kg_version: {V}, name: 'Apple'}})-[r:ACQUIRED]->(m:Organization {{name: 'Mistral AI'}})
WHERE {rel('r', 'm')}
RETURN m.name AS entity""", "Not found in the knowledge graph.", {}),
]


def run_query(driver, cypher: str, kg_version: str, **params) -> list[dict]:
    """Read transaction with a 10 s timeout (the Direct Cypher system runs gold queries this way)."""
    from neo4j import unit_of_work

    @unit_of_work(timeout=10)
    def work(tx):
        return [r.data() for r in tx.run(cypher, kg_version=kg_version, **params)]

    with driver.session() as s:
        return s.execute_read(work)


def expected_from_rows(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(str(r["entity"]) for r in rows if r.get("entity") is not None))[:MAX_EXPECTED]


def fill_template(template: str, rows: list[dict], entities: list[str]) -> str:
    first = rows[0] if rows else {}
    return template.format(entities=", ".join(entities) or "(none)", first=first.get("entity", ""),
                           n=first.get("n", ""))


def in_lcc(driver, name: str) -> bool:
    rows = run_query(driver, "MATCH (e:Entity {kg_version: $kg_version, name: $name}) "
                             "RETURN coalesce(e.in_lcc_resolved, false) AS lcc", "resolved", name=name)
    return any(r["lcc"] for r in rows)


def make_drafts(driver) -> list[dict]:
    out = []
    for qid, cat, q, cypher, template, extra in DRAFTS:
        rows = run_query(driver, cypher.strip(), "resolved")
        ents = extra.get("entities") or expected_from_rows(rows)
        item = {"id": qid, "category": cat, "question": q, "gold_query": cypher.strip(),
                "expected_answer": fill_template(template, rows, ents), "expected_entities": ents,
                "verified": False, "gold_rows_resolved": len(rows), "gold_preview": rows[:8]}
        if cat == "unanswerable":
            item["expected_entities"] = []
        if "anchor" in extra:
            item.update(anchor=extra["anchor"], hops=extra["hops"], anchor_in_lcc=in_lcc(driver, extra["anchor"]))
        out.append(item)
    return out


def check_drafts(items: list[dict]) -> list[str]:
    """Structural problems that would make a draft unusable (shown in the review file)."""
    problems = []
    counts = {c: sum(1 for i in items if i["category"] == c) for c in CATEGORIES}
    for c, n in CATEGORIES.items():
        if counts[c] != n:
            problems.append(f"category {c}: {counts[c]} questions, expected {n}")
    for i in items:
        if "$kg_version" not in i["gold_query"]:
            problems.append(f"{i['id']}: gold query does not take $kg_version")
        if i["category"] == "unanswerable" and i["gold_rows_resolved"]:
            problems.append(f"{i['id']}: unanswerable but the gold query returns rows")
        if i["category"] != "unanswerable" and not i["gold_rows_resolved"]:
            problems.append(f"{i['id']}: the gold query returns no rows")
        if i["category"] == "multi-hop" and not i.get("anchor_in_lcc"):
            problems.append(f"{i['id']}: anchor {i.get('anchor')!r} is not in the largest connected component")
    return problems


# ------------------------------------------------------------------ review file
REVIEW_HEAD = """# Phase 6 question review

Tick ONE box per question: `[x] OK` (correct as is), `[x] FIX` (you edited the lines below it) or
`[x] DROP` (remove it). Edit `question:`, `expected_answer:` and `expected_entities:` (separated by
` ; `) in place, and the Cypher block if the gold query is wrong. Unticked questions stay unverified
and do not count. Then run `make questions-apply`.

Check each answer against the source documents, not only against the graph: the gold query shows
what the graph says; if the graph is wrong (an extraction error), fix the expected answer and say so
in `notes:`.

Multi-hop anchors were checked to be in the largest connected component of the resolved graph.
"""


def render_review(items: list[dict], problems: list[str]) -> str:
    out = [REVIEW_HEAD]
    if problems:
        out += ["## Generator warnings", ""] + [f"- {p}" for p in problems] + [""]
    for i in items:
        mark = lambda k: "x" if i.get("decision") == k else " "
        out += [f"## {i['id']} · {i['category']}" + (f" · {i['hops']} hops" if i.get("hops") else ""), "",
                f"[{mark('OK')}] OK   [{mark('FIX')}] FIX   [{mark('DROP')}] DROP", "",
                f"question: {i['question']}",
                f"expected_answer: {i['expected_answer']}",
                f"expected_entities: {' ; '.join(i['expected_entities'])}",
                f"notes: {i.get('notes', '')}", "",
                f"Gold query result on resolved: {i['gold_rows_resolved']} rows"
                + (f"; first {len(i['gold_preview'])}:" if i["gold_preview"] else "")]
        for r in i["gold_preview"]:
            out.append("    " + json.dumps(r, ensure_ascii=False, default=str)[:220])
        out += ["", "```cypher", i["gold_query"], "```", ""]
    return "\n".join(out)


FIELD = re.compile(r"^(question|expected_answer|expected_entities|notes):[ \t]*(.*)$", re.M)
DECISION = re.compile(r"\[(x|X)\]\s*(OK|FIX|DROP)")


def parse_review(text: str) -> dict:
    """{id: {decision, question, expected_answer, expected_entities, notes, gold_query}}."""
    out = {}
    for block in re.split(r"^## (?=[A-Z]\d\d )", text, flags=re.M)[1:]:
        qid = block.split()[0]
        d = {"decision": None}
        head = block.split("question:")[0]
        ticks = DECISION.findall(head)
        if len(ticks) == 1:
            d["decision"] = ticks[0][1]
        elif len(ticks) > 1:
            d["decision"] = "AMBIGUOUS"
        for k, v in FIELD.findall(block):
            d[k] = v.strip()
        if "expected_entities" in d:
            d["expected_entities"] = [e.strip() for e in d["expected_entities"].split(";") if e.strip()]
        m = re.search(r"```cypher\n(.*?)\n```", block, re.S)
        if m:
            d["gold_query"] = m.group(1).strip()
        out[qid] = d
    return out


def apply_review(items: list[dict], review: dict) -> tuple[list[dict], dict]:
    summary = {"OK": 0, "FIX": 0, "DROP": 0, "unticked": 0, "ambiguous": []}
    for i in items:
        r = review.get(i["id"])
        if not r:
            summary["unticked"] += 1
            i["verified"] = False
            continue
        for k in ("question", "expected_answer", "expected_entities", "notes", "gold_query"):
            if k in r:
                i[k] = r[k]
        dec = r["decision"]
        i["decision"] = dec if dec in ("OK", "FIX", "DROP") else None
        i["verified"] = dec in ("OK", "FIX")
        i["dropped"] = dec == "DROP"
        if dec == "AMBIGUOUS":
            summary["ambiguous"].append(i["id"])
        else:
            summary[dec or "unticked"] += 1
    return items, summary


def load(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def save(items: list[dict], path: Path = QUESTIONS) -> None:
    path.write_text("".join(json.dumps(i, ensure_ascii=False, default=str) + "\n" for i in items))


def verified(items: list[dict]) -> list[dict]:
    return [i for i in items if i.get("verified") and not i.get("dropped")]


def status(items: list[dict]) -> str:
    lines = [f"{'category':<14}{'drafts':>7}{'verified':>9}{'dropped':>8}"]
    for c in CATEGORIES:
        g = [i for i in items if i["category"] == c]
        lines.append(f"{c:<14}{len(g):>7}{len(verified(g)):>9}{sum(bool(i.get('dropped')) for i in g):>8}")
    lines.append(f"{'total':<14}{len(items):>7}{len(verified(items)):>9}{sum(bool(i.get('dropped')) for i in items):>8}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="eval.questions")
    ap.add_argument("cmd", choices=("draft", "apply-review", "status"))
    ap.add_argument("--force", action="store_true", help="draft: overwrite existing files")
    a = ap.parse_args()
    if a.cmd == "draft":
        if QUESTIONS.exists() and not a.force:
            print(f"{QUESTIONS} exists (review in progress?); use --force to regenerate")
            return 1
        from analytics.db import get_driver
        items = make_drafts(get_driver())
        problems = check_drafts(items)
        save(items)
        REVIEW.write_text(render_review(items, problems))
        print(status(items))
        print("\n".join(f"WARNING {p}" for p in problems) or "no generator warnings")
        print(f"wrote {QUESTIONS} and {REVIEW}")
        return 0
    items = load()
    if a.cmd == "apply-review":
        items, summary = apply_review(items, parse_review(REVIEW.read_text()))
        save(items)
        print(f"review applied: {summary}")
    print(status(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
