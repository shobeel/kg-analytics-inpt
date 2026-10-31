"""RQ1: extraction P/R/F1 of EXTRACTION_MODEL on the hand-corrected gold chunks (eval/gold/extraction/gold.jsonl,
from `make gold-convert`). Only chunks with status: done are scored; the count is reported.

Predictions are the cached extractions of those chunks (no LLM call), scored two ways:
  model output   every entity / relation the model returned
  kept in KG     relations that pass the post-hoc checks (known endpoints, schema, verified evidence,
                 no self-loop), i.e. what entered the knowledge graph
Runs in the extraction container (it needs extraction.extract.recheck). Writes eval/results/rq1_extraction.json.
"""

import json
import sys
import tempfile
from pathlib import Path

from .extraction_scorer import score

GOLD = Path(__file__).parent / "gold" / "extraction" / "gold.jsonl"
OUT = Path(__file__).parent / "results" / "rq1_extraction.json"


def kept(r: dict) -> bool:
    return bool(r.get("endpoints_known", True) and r.get("schema_valid") and r.get("evidence_verified")
                and not r.get("self_loop"))


def predictions(gold: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    from extraction import PROMPT_VERSION
    from extraction.cache import ResultCache, chunk_hash
    from extraction.config import settings
    from extraction.extract import recheck
    s = settings()
    cache = ResultCache(s.cache_dir, s.extraction_model, PROMPT_VERSION)
    full, kg, missing = [], [], []
    for g in gold:
        rec = cache.get(chunk_hash(g["text"]))
        if rec is None:
            missing.append(g["chunk_id"])
            continue
        rec = recheck(rec, g["text"])
        ents = [{"name": e["name"], "type": e["type"]} for e in rec["entities"]]
        rels = [{"source": r["source"], "relation": r["relation"], "target": r["target"], "_kept": kept(r)}
                for r in rec["relations"]]
        full.append({"chunk_id": g["chunk_id"], "entities": ents,
                     "relations": [{k: v for k, v in r.items() if k != "_kept"} for r in rels]})
        kg.append({"chunk_id": g["chunk_id"], "entities": ents,
                   "relations": [{k: v for k, v in r.items() if k != "_kept"} for r in rels if r["_kept"]]})
    return full, kg, missing


def write_jsonl(rows: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    f.close()
    return f.name


def main() -> int:
    if not GOLD.exists():
        print(f"{GOLD} missing: run make gold-convert")
        return 1
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    from extraction.config import settings
    full, kg, missing = predictions(gold)
    res = {"model": settings().extraction_model, "gold_chunks": len(gold),
           "chunks_done": sum(g.get("status") == "done" for g in gold), "missing_predictions": missing,
           "model_output": score(str(GOLD), write_jsonl(full)), "kept_in_kg": score(str(GOLD), write_jsonl(kg))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"RQ1: {res['chunks_done']}/{res['gold_chunks']} gold chunks annotated (status: done); "
          f"{len(missing)} without a cached prediction")
    if res["chunks_done"]:
        for view in ("model_output", "kept_in_kg"):
            for k in ("entity_strict", "entity_relaxed", "relation_strict", "relation_relaxed"):
                m = res[view][k]
                print(f"  {view:<13}{k:<18} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    else:
        print("  no annotated chunk yet: P/R/F1 pending (correct eval/gold/extraction/todo/*.txt, set status: done)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
