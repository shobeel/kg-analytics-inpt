"""RQ1 gold set: sample chunks for manual annotation, export model predictions for scoring."""

import json
import random
from collections import Counter
from pathlib import Path

from . import PROMPT_VERSION
from .cache import ResultCache, chunk_hash
from .spark_io import get_spark, read_chunks


def sample_gold(s, n: int, out: str, seed: int = 2026) -> dict:
    """Stratified by source (proportional, at least 2 per source), fixed seed. Never overwrites
    an existing file: it may already hold hand annotations."""
    path = Path(out)
    if path.exists():
        raise SystemExit(f"{out} already exists (annotations?). Move it away to resample.")
    spark = get_spark("gold-sample")
    chunks = read_chunks(spark, s.hdfs)
    spark.stop()
    rnd = random.Random(seed)
    by_src: dict[str, list] = {}
    for c in chunks:
        by_src.setdefault(c["source"], []).append(c)
    quota = {src: max(2, round(n * len(v) / len(chunks))) for src, v in by_src.items()}
    while sum(quota.values()) > n:                              # trim the largest strata
        quota[max(quota, key=quota.get)] -= 1
    picked = []
    for src, v in sorted(by_src.items()):
        picked += rnd.sample(v, min(quota[src], len(v)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in sorted(picked, key=lambda c: c["chunk_id"]):
            f.write(json.dumps({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "source": c["source"],
                "title": c["title"], "url": c["url"], "text": c["text"],
                "status": "todo", "annotator": "", "entities": [], "relations": [], "notes": ""},
                ensure_ascii=False) + "\n")
    return {"written": str(path), "chunks": len(picked),
            "per_source": {k: sum(1 for c in picked if c["source"] == k) for k in sorted(by_src)}}


def export_predictions(s, gold: str, out: str) -> dict:
    """Write the cached extraction for every chunk in a gold file, in the scorer's format."""
    cache = ResultCache(s.cache_dir, s.extraction_model, PROMPT_VERSION)
    rows, missing = [], []
    for line in Path(gold).read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        rec = cache.get(chunk_hash(g["text"]))
        if rec is None:
            missing.append(g["chunk_id"])
            continue
        rows.append({"chunk_id": g["chunk_id"],
                     "entities": [{"name": e["name"], "type": e["type"]} for e in rec["entities"]],
                     "relations": [{"source": r["source"], "relation": r["relation"], "target": r["target"]}
                                   for r in rec["relations"]]})
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return {"written": out, "chunks": len(rows), "missing_predictions": missing,
            "model": s.extraction_model, "prompt_version": PROMPT_VERSION}


def prefill_gold(s, gold: str, out: str) -> dict:
    """Draft annotations for the gold chunks with PREFILL_MODEL on Groq (another provider and
    model family than the extractor being scored), cached like extraction. Chunks already drafted
    by an earlier pre-fill model (cache under that model's name) are kept as they are.
    The annotator only corrects the drafts."""
    from .extract import extract_chunk, recheck
    from .llm import LLMCallFailed, LLMClient, QuotaExhausted

    if not s.prefill_model or not s.prefill_api_key:
        raise SystemExit("PREFILL_MODEL and GROQ_API_KEY must be set in .env")
    if s.extraction_model in (s.prefill_model, s.prefill_fallback_model):
        raise SystemExit("PREFILL_MODEL must differ from EXTRACTION_MODEL (anchoring bias)")
    llm = LLMClient(s, base_url=s.prefill_base_url, api_key=s.prefill_api_key,
                    min_interval_s=s.prefill_min_interval_s)
    models = [m for m in (s.prefill_model, s.prefill_fallback_model) if m]
    caches = {m: ResultCache(s.cache_dir, m, PROMPT_VERSION) for m in models}
    earlier = [ResultCache(s.cache_dir, m, PROMPT_VERSION)
               for m in (s.answer_model,) if m and m not in (*models, s.extraction_model)]
    rows, calls, stopped, failed = [], 0, None, []
    for line in Path(gold).read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        key = chunk_hash(g["text"])
        rec = next((c.get(key) for c in (*caches.values(), *earlier) if c.get(key)), None)
        if rec is None and stopped:
            continue                                # quota gone: keep collecting cached drafts only
        if rec is None:
            for m in models:                         # fallback model only if the first one fails
                try:
                    rec = {"chunk_id": g["chunk_id"], "model": m, "prompt_version": PROMPT_VERSION,
                           **extract_chunk(llm, g, model=m, purpose="gold-prefill")}
                except LLMCallFailed:               # e.g. Groq json_validate_failed every time
                    continue
                except QuotaExhausted as e:
                    stopped = str(e)[:300]
                    break
                caches[m].put(key, rec)
                calls += len(rec["usage"])
                break
            if rec is None:
                if not stopped:
                    failed.append(g["chunk_id"])
                continue
        rec = recheck(rec, g["text"])                # flags under the current validation rules
        rows.append({"chunk_id": g["chunk_id"], "model": rec["model"], "status": rec["status"],
                     "entities": [{"name": e["name"], "type": e["type"]} for e in rec["entities"]],
                     "relations": [{k: r[k] for k in ("source", "relation", "target", "evidence_text",
                                                      "confidence", "schema_valid", "evidence_verified")}
                                   for r in rec["relations"]]})
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return {"written": out, "chunks": len(rows), "llm_calls": calls, "model": s.prefill_model,
            "models_used": dict(Counter(r["model"] for r in rows)),
            "entities": sum(len(r["entities"]) for r in rows),
            "relations": sum(len(r["relations"]) for r in rows), "failed": failed,
            "quota_exhausted": stopped}
