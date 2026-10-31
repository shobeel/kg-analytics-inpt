"""RQ1 scorer: entity and relation precision / recall / F1 of predictions vs a hand gold set.

Files are JSONL, one chunk per line:
  gold: {"chunk_id", "status": "done", "entities": [{"name","type"}], "relations": [{"source","relation","target"}]}
  pred: {"chunk_id", "entities": [...], "relations": [...]}
Gold lines whose status is not "done" are skipped (still being annotated).

Matching is one-to-one (greedy) inside each chunk; scores are micro-averaged over chunks.
  strict   entity: same name (case/whitespace-insensitive) AND same type
           relation: same relation type AND both endpoint names equal (as above)
  relaxed  entity: names match after normalization (case, punctuation, "the", legal suffixes like
           Inc/Ltd), or one name's tokens are all contained in the other ("Altman" ~ "Sam Altman"),
           or character similarity >= 0.85; the type is ignored
           relation: same relation type AND both endpoints match relaxed
Symmetric relations (COMPETES_WITH, PARTNERS_WITH) match in either direction in both modes.

Usage: python -m eval.extraction_scorer --gold G.jsonl --pred P.jsonl [--json-out R.json]
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

SYMMETRIC = {"COMPETES_WITH", "PARTNERS_WITH"}
LEGAL = {"inc", "incorporated", "ltd", "limited", "llc", "corp", "corporation", "co", "company",
         "gmbh", "plc", "sa", "ag", "lp", "llp", "pbc"}


def strict_name(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def relaxed_name(s: str) -> str:
    s = re.sub(r"\([^)]*\)", " ", unicodedata.normalize("NFKC", s).casefold())
    toks = re.sub(r"[^\w\s]", " ", s).split()
    if toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and toks[-1] in LEGAL:
        toks = toks[:-1]
    return " ".join(toks)


def names_match(a: str, b: str, mode: str) -> bool:
    if mode == "strict":
        return strict_name(a) == strict_name(b)
    ra, rb = relaxed_name(a), relaxed_name(b)
    if not ra or not rb:
        return False
    if ra == rb:
        return True
    ta, tb = set(ra.split()), set(rb.split())
    if ta <= tb or tb <= ta:
        return True
    return SequenceMatcher(None, ra, rb).ratio() >= 0.85


def entity_match(g: dict, p: dict, mode: str) -> bool:
    return names_match(g["name"], p["name"], mode) and (mode == "relaxed" or g["type"] == p["type"])


def relation_match(g: dict, p: dict, mode: str) -> bool:
    if g["relation"] != p["relation"]:
        return False
    fwd = names_match(g["source"], p["source"], mode) and names_match(g["target"], p["target"], mode)
    if fwd or g["relation"] not in SYMMETRIC:
        return fwd
    return names_match(g["source"], p["target"], mode) and names_match(g["target"], p["source"], mode)


def greedy(gold: list, pred: list, match) -> tuple[int, int, int, list]:
    used, tp_pairs = set(), []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if pi not in used and match(g, p):
                used.add(pi)
                tp_pairs.append((gi, pi))
                break
    tp = len(tp_pairs)
    return tp, len(pred) - tp, len(gold) - tp, tp_pairs


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def load(path: str) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[row["chunk_id"]] = row
    return out


def score(gold_path: str, pred_path: str) -> dict:
    gold, pred = load(gold_path), load(pred_path)
    done = {cid: g for cid, g in gold.items() if g.get("status", "done") == "done"}
    totals = {k: Counter() for k in ("entity_strict", "entity_relaxed", "relation_strict", "relation_relaxed")}
    per_type = {}
    for cid, g in done.items():
        p = pred.get(cid, {"entities": [], "relations": []})
        for mode in ("strict", "relaxed"):
            tp, fp, fn, _ = greedy(g["entities"], p["entities"], lambda a, b: entity_match(a, b, mode))
            totals[f"entity_{mode}"].update(tp=tp, fp=fp, fn=fn)
            tp, fp, fn, _ = greedy(g["relations"], p["relations"], lambda a, b: relation_match(a, b, mode))
            totals[f"relation_{mode}"].update(tp=tp, fp=fp, fn=fn)
        for etype in {e["type"] for e in g["entities"] + p["entities"]}:
            ge = [e for e in g["entities"] if e["type"] == etype]
            pe = [e for e in p["entities"] if e["type"] == etype]
            tp, fp, fn, _ = greedy(ge, pe, lambda a, b: entity_match(a, b, "strict"))
            per_type.setdefault(etype, Counter()).update(tp=tp, fp=fp, fn=fn)
    return {
        "chunks_scored": len(done), "chunks_skipped_not_done": len(gold) - len(done),
        "chunks_without_prediction": sorted(set(done) - set(pred)),
        **{k: prf(v["tp"], v["fp"], v["fn"]) for k, v in totals.items()},
        "entity_strict_by_type": {t: prf(v["tp"], v["fp"], v["fn"]) for t, v in sorted(per_type.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    res = score(a.gold, a.pred)
    print(f"chunks scored: {res['chunks_scored']} (skipped, not done: {res['chunks_skipped_not_done']})")
    print(f"{'':18}{'P':>8}{'R':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    for k in ("entity_strict", "entity_relaxed", "relation_strict", "relation_relaxed"):
        m = res[k]
        print(f"{k:18}{m['precision']:8.3f}{m['recall']:8.3f}{m['f1']:8.3f}{m['tp']:6}{m['fp']:6}{m['fn']:6}")
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(res, fh, indent=2)
    print("SCORER_OK")


if __name__ == "__main__":
    main()
