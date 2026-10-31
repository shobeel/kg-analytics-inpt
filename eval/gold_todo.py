"""RQ1 gold set as one editable text file per chunk, pre-filled by ANSWER_MODEL.

  make  annotation_batch.jsonl + prefill.jsonl -> todo/NN_<chunk_id>.txt  (existing files are kept;
        chunks without a pre-fill wait for the next `make gold-prefill`, or start empty with --empty)
  convert  todo/*.txt -> gold.jsonl (scorer input) + changes.jsonl (what the annotator changed
           relative to the pre-fill, per chunk) and a printed summary

File format (lines starting with # are ignored; one item per line; " | " separates fields):
  status: todo|done          annotator: initials          notes: free text
  === ENTITIES ===           Type | Name
  === RELATIONS ===          Source name | RELATION | Target name

Usage: python -m eval.gold_todo make|convert [--dir eval/gold/extraction]
"""

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

ENTITY_TYPES = ("Organization", "Person", "AIModel", "Paper", "Technology", "Dataset", "Product", "Location")
RELATION_TYPES = ("DEVELOPED_BY", "WORKS_AT", "AUTHORED", "CITES", "USES", "TRAINED_ON",
                  "COMPETES_WITH", "PARTNERS_WITH", "FUNDED_BY", "ACQUIRED", "LOCATED_IN", "BASED_ON")
SYMMETRIC = {"COMPETES_WITH", "PARTNERS_WITH"}
TEXT, ENTS, RELS, END = "=== TEXT (read only) ===", "=== ENTITIES ===", "=== RELATIONS ===", "=== END ==="

HELP = f"""\
# Correct the pre-filled ENTITIES and RELATIONS so they match what the TEXT states.
# Delete wrong lines, fix names/types, add missing items. One item per line, fields separated by " | ".
#   entity:    Type | Name
#   relation:  Source name | RELATION | Target name      (names must be listed under ENTITIES)
# Entity types: {", ".join(ENTITY_TYPES)}
# Relations:    {", ".join(RELATION_TYPES)}
# Direction rules and examples: README.md in this folder. When finished set "status: done".
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def file_name(i: int, chunk_id: str) -> str:
    return f"{i:02d}_{chunk_id}.txt"


def render(i: int, n: int, chunk: dict, pre: dict) -> str:
    out = [f"# Gold chunk {i}/{n}   chunk_id: {chunk['chunk_id']}   source: {chunk['source']}",
           f"# title: {chunk['title']}", f"# url: {chunk['url']}",
           f"# pre-filled by {pre.get('model', '?')}: check every line, it is a draft, not the truth", HELP,
           "status: todo", "annotator: ", "notes: ", "", TEXT]
    for para in chunk["text"].split("\n"):
        out += textwrap.wrap(para, 100) or [""]
    out += ["", ENTS]
    out += [f"{e['type']} | {e['name']}" for e in pre.get("entities", [])]
    out += ["", RELS]
    for r in pre.get("relations", []):
        flags = [] if r.get("schema_valid", True) else ["types outside schema"]
        flags += [] if r.get("evidence_verified", True) else ["evidence not in text"]
        out.append(f"#   evidence: \"{r.get('evidence_text', '')[:200]}\"" +
                   (f"   [model check: {', '.join(flags)}]" if flags else ""))
        out.append(f"{r['source']} | {r['relation']} | {r['target']}")
    out += ["", END, ""]
    return "\n".join(out)


def make(d: Path, empty: bool = False) -> None:
    chunks = read_jsonl(d / "annotation_batch.jsonl")
    pre = {p["chunk_id"]: p for p in read_jsonl(d / "prefill.jsonl")}
    todo = d / "todo"
    todo.mkdir(exist_ok=True)
    written, kept, waiting = 0, 0, []
    for i, c in enumerate(chunks, 1):
        path = todo / file_name(i, c["chunk_id"])
        if path.exists():                      # may already hold corrections
            kept += 1
            continue
        if c["chunk_id"] not in pre and not empty:
            waiting.append(c["chunk_id"])      # re-run after the next `make gold-prefill`
            continue
        path.write_text(render(i, len(chunks), c, pre.get(c["chunk_id"], {})))
        written += 1
    print(json.dumps({"dir": str(todo), "written": written, "kept_existing": kept,
                      "waiting_for_prefill": len(waiting)}))


class FormatError(ValueError):
    pass


def parse(text: str, where: str) -> dict:
    """One todo file -> {status, annotator, notes, entities, relations}. Raises FormatError
    with file:line on anything the scorer could not use."""
    meta, ents, rels, section = {}, [], [], "meta"
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line in (TEXT, ENTS, RELS, END):
            section = {TEXT: "text", ENTS: "ents", RELS: "rels", END: "end"}[line]
            continue
        if section in ("text", "end") or not line or line.startswith("#"):
            continue
        err = lambda msg: FormatError(f"{where}:{no}: {msg}: {raw!r}")
        if section == "meta":
            key, sep, val = line.partition(":")
            if not sep or key.strip() not in ("status", "annotator", "notes"):
                raise err("expected 'status:', 'annotator:' or 'notes:'")
            meta[key.strip()] = val.strip()
        elif section == "ents":
            parts = [p.strip() for p in line.split(" | ")]
            if len(parts) != 2 or not parts[1]:
                raise err("entity must be 'Type | Name'")
            if parts[0] not in ENTITY_TYPES:
                raise err(f"unknown entity type {parts[0]!r}")
            ents.append({"name": parts[1], "type": parts[0]})
        elif section == "rels":
            parts = [p.strip() for p in line.split(" | ")]
            if len(parts) != 3 or not parts[0] or not parts[2]:
                raise err("relation must be 'Source | RELATION | Target'")
            if parts[1] not in RELATION_TYPES:
                raise err(f"unknown relation {parts[1]!r}")
            rels.append({"source": parts[0], "relation": parts[1], "target": parts[2]})
    if meta.get("status", "todo") not in ("todo", "done"):
        raise FormatError(f"{where}: status must be 'todo' or 'done'")
    names = {e["name"].lower() for e in ents}
    for r in rels:
        for end in (r["source"], r["target"]):
            if end.lower() not in names:
                raise FormatError(f"{where}: relation endpoint {end!r} is not listed under ENTITIES")
    return {"status": meta.get("status", "todo"), "annotator": meta.get("annotator", ""),
            "notes": meta.get("notes", ""), "entities": ents, "relations": rels}


def ent_key(e: dict) -> tuple:
    return (" ".join(e["name"].casefold().split()), e["type"])


def rel_key(r: dict) -> tuple:
    s, t = (" ".join(x.casefold().split()) for x in (r["source"], r["target"]))
    if r["relation"] in SYMMETRIC:
        s, t = sorted((s, t))
    return (s, r["relation"], t)


def diff(pre: dict, gold: dict) -> dict:
    """What the annotator changed relative to the pre-fill (exact, case-insensitive keys).
    An entity whose name stayed but whose type changed is reported as retyped."""
    pe = {ent_key(e): e for e in pre.get("entities", [])}
    ge = {ent_key(e): e for e in gold["entities"]}
    removed = [pe[k] for k in pe.keys() - ge.keys()]
    added = [ge[k] for k in ge.keys() - pe.keys()]
    retyped = []
    for a in list(added):
        for rm in list(removed):
            if ent_key(rm)[0] == ent_key(a)[0]:
                retyped.append({"name": a["name"], "from": rm["type"], "to": a["type"]})
                added.remove(a)
                removed.remove(rm)
                break
    pr = {rel_key(r): r for r in pre.get("relations", [])}
    gr = {rel_key(r): r for r in gold["relations"]}
    strip = lambda r: {k: r[k] for k in ("source", "relation", "target")}
    return {"entities": {"kept": len(pe.keys() & ge.keys()), "removed": removed, "added": added,
                         "retyped": retyped},
            "relations": {"kept": len(pr.keys() & gr.keys()),
                          "removed": [strip(pr[k]) for k in pr.keys() - gr.keys()],
                          "added": [strip(gr[k]) for k in gr.keys() - pr.keys()]}}


def convert(d: Path) -> int:
    chunks = {c["chunk_id"]: c for c in read_jsonl(d / "annotation_batch.jsonl")}
    pre_path = d / "prefill.jsonl"
    pre = {p["chunk_id"]: p for p in read_jsonl(pre_path)} if pre_path.exists() else {}
    files = sorted((d / "todo").glob("*.txt"))
    gold_rows, change_rows, errors = [], [], []
    for f in files:
        chunk_id = f.stem.split("_", 1)[1]
        if chunk_id not in chunks:
            errors.append(f"{f}: chunk_id {chunk_id} is not in annotation_batch.jsonl")
            continue
        try:
            g = parse(f.read_text(), str(f))
        except FormatError as e:
            errors.append(str(e))
            continue
        c = chunks[chunk_id]
        gold_rows.append({k: c[k] for k in ("chunk_id", "doc_id", "source", "title", "url", "text")} | g)
        change_rows.append({"chunk_id": chunk_id, "file": f.name, "status": g["status"],
                            "prefill_model": pre.get(chunk_id, {}).get("model"),
                            **diff(pre.get(chunk_id, {}), g)})
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"NOT WRITTEN: {len(errors)} problem(s); fix the files and re-run.", file=sys.stderr)
        return 1
    (d / "gold.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold_rows))
    (d / "changes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in change_rows))
    done = [c for c in change_rows if c["status"] == "done"]
    tot = Counter()
    for c in done:
        for kind in ("entities", "relations"):
            tot[f"{kind}_kept"] += c[kind]["kept"]
            tot[f"{kind}_removed"] += len(c[kind]["removed"])
            tot[f"{kind}_added"] += len(c[kind]["added"])
        tot["entities_retyped"] += len(c["entities"]["retyped"])
    print(json.dumps({"gold": str(d / "gold.jsonl"), "changes": str(d / "changes.jsonl"),
                      "files": len(files), "done": len(done), "todo": len(files) - len(done),
                      "changes_in_done_chunks": dict(tot)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("make", "convert"))
    ap.add_argument("--dir", default="eval/gold/extraction")
    ap.add_argument("--empty", action="store_true", help="make: also write files for chunks with no pre-fill")
    a = ap.parse_args()
    if a.cmd == "make":
        make(Path(a.dir), a.empty)
        return 0
    return convert(Path(a.dir))


if __name__ == "__main__":
    sys.exit(main())
