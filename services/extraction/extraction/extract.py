"""Chunk -> entities + relations (validated, provenance-carrying), cached and logged."""

import json
import logging
import random
import re
import time
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

from pydantic import ValidationError
from pyspark.sql import types as T

from . import PROMPT_VERSION
from .cache import ResultCache, chunk_hash
from .config import Settings
from .llm import LLMCallFailed, LLMClient, QuotaExhausted
from .prompts import EXTRACTION_SYSTEM, extraction_user
from .schema import RESPONSE_SCHEMA, SYMMETRIC, ExtractionResult, type_valid
from .spark_io import get_spark, read_chunks, silver, write_table

log = logging.getLogger(__name__)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def evidence_in_chunk(evidence: str, chunk_text: str) -> bool:
    """Evidence must be copied from the chunk (whitespace/case-insensitive, quotes unified)."""
    unify = lambda s: _norm_ws(s).replace("’", "'").replace("“", '"').replace("”", '"')
    return unify(evidence).strip(" .\"'") in unify(chunk_text)


def postprocess(result: ExtractionResult, chunk_text: str) -> dict:
    """Dedupe entities, attach checks to relations. Nothing is silently dropped: relations
    keep flags so the scorer / Phase 3 can decide what to load."""
    entities, seen = [], set()
    for e in result.entities:
        key = (e.name.lower(), e.type)
        if key not in seen:
            seen.add(key)
            entities.append(e.model_dump())
    return {"entities": entities,
            "relations": check_relations(entities, [r.model_dump() for r in result.relations], chunk_text)}


def check_relations(entities: list[dict], relations: list[dict], chunk_text: str) -> list[dict]:
    """Attach checks to each relation. Pure function of (entities, relations, text), so the
    flags of cached extractions are recomputed whenever the validation rules change.
    A relation whose types only fit the schema reversed is swapped and marked direction_repaired."""
    types = {}
    for e in entities:
        types.setdefault(e["name"].lower(), e["type"])
    out = []
    for r in relations:
        r = {k: r[k] for k in ("source", "relation", "target", "evidence_text", "confidence")} | \
            {"direction_repaired": r.get("direction_repaired", False)}
        st, tt = types.get(r["source"].lower()), types.get(r["target"].lower())
        if (st and tt and r["relation"] not in SYMMETRIC and not type_valid(st, r["relation"], tt)
                and type_valid(tt, r["relation"], st)):
            r.update(source=r["target"], target=r["source"], direction_repaired=True)
            st, tt = tt, st
        out.append({**r, "source_type": st, "target_type": tt,
                    "endpoints_known": st is not None and tt is not None,
                    "schema_valid": type_valid(st, r["relation"], tt),
                    "self_loop": r["source"].strip().lower() == r["target"].strip().lower(),
                    "evidence_verified": evidence_in_chunk(r["evidence_text"], chunk_text)})
    return out


def recheck(rec: dict, chunk_text: str) -> dict:
    """A cached record with its relation flags recomputed under the current rules."""
    return {**rec, "relations": check_relations(rec["entities"], rec["relations"], chunk_text)}


def extract_chunk(llm: LLMClient, chunk: dict, model: Optional[str] = None,
                  purpose: str = "extraction") -> dict:
    """Call the LLM; validate with pydantic; on invalid output retry once with the error."""
    user = extraction_user(chunk["text"], chunk.get("title") or "", chunk.get("source", ""))
    usages, error = [], None
    for attempt in (1, 2):
        raw, usage = llm.chat_json(system=EXTRACTION_SYSTEM, user=user, schema=RESPONSE_SCHEMA,
                                   schema_name="kg_extraction", purpose=purpose, ref=chunk["chunk_id"],
                                   model=model)
        usages.append(usage)
        try:
            result = ExtractionResult.model_validate(json.loads(raw))
            return {"status": "ok", "attempts": attempt, "usage": usages,
                    **postprocess(result, chunk["text"])}
        except (json.JSONDecodeError, ValidationError) as e:
            error = f"{type(e).__name__}: {str(e)[:400]}"
            log.warning("invalid output for %s (attempt %d): %s", chunk["chunk_id"], attempt, error[:150])
            user = (extraction_user(chunk["text"], chunk.get("title") or "", chunk.get("source", "")) +
                    f"\n\nYour previous answer was invalid ({error[:300]}). Return valid JSON matching the schema.")
    return {"status": "invalid", "attempts": 2, "usage": usages, "error": error,
            "entities": [], "relations": []}


# ---------------------------------------------------------------------- tables

ENTITY_SCHEMA = T.StructType([
    T.StructField("chunk_id", T.StringType()), T.StructField("doc_id", T.StringType()),
    T.StructField("source", T.StringType()), T.StructField("name", T.StringType()),
    T.StructField("type", T.StringType()), T.StructField("description", T.StringType()),
    T.StructField("extractor_model", T.StringType()), T.StructField("prompt_version", T.StringType()),
    T.StructField("extracted_at", T.StringType())])

RELATION_SCHEMA = T.StructType([
    T.StructField("chunk_id", T.StringType()), T.StructField("doc_id", T.StringType()),
    T.StructField("source", T.StringType()),
    T.StructField("source_name", T.StringType()), T.StructField("source_type", T.StringType()),
    T.StructField("relation", T.StringType()),
    T.StructField("target_name", T.StringType()), T.StructField("target_type", T.StringType()),
    T.StructField("evidence_text", T.StringType()), T.StructField("confidence", T.DoubleType()),
    T.StructField("evidence_verified", T.BooleanType()), T.StructField("schema_valid", T.BooleanType()),
    T.StructField("endpoints_known", T.BooleanType()), T.StructField("self_loop", T.BooleanType()),
    T.StructField("direction_repaired", T.BooleanType()),
    T.StructField("extractor_model", T.StringType()), T.StructField("prompt_version", T.StringType()),
    T.StructField("extracted_at", T.StringType())])


def to_rows(rec: dict) -> tuple[list, list]:
    base = {"chunk_id": rec["chunk_id"], "doc_id": rec["doc_id"], "source": rec["source"],
            "extractor_model": rec["model"], "prompt_version": rec["prompt_version"],
            "extracted_at": rec["extracted_at"]}
    ents = [{**base, "name": e["name"], "type": e["type"], "description": e.get("description", "")}
            for e in rec["entities"]]
    rels = [{**base, "source_name": r["source"], "source_type": r["source_type"], "relation": r["relation"],
             "target_name": r["target"], "target_type": r["target_type"], "evidence_text": r["evidence_text"],
             "confidence": float(r["confidence"]), "evidence_verified": r["evidence_verified"],
             "schema_valid": r["schema_valid"], "endpoints_known": r["endpoints_known"],
             "self_loop": r["self_loop"], "direction_repaired": r["direction_repaired"]}
            for r in rec["relations"]]
    return ents, rels


# ---------------------------------------------------------------------- run

def pick_pilot(chunks: list[dict], n: int, seed: int = 42) -> list[dict]:
    """n chunks spread over sources (round-robin, seeded) for a representative pilot."""
    rnd = random.Random(seed)
    by_src: dict[str, list] = {}
    for c in chunks:
        by_src.setdefault(c["source"], []).append(c)
    for v in by_src.values():
        rnd.shuffle(v)
    picked, i = [], 0
    while len(picked) < n and any(by_src.values()):
        src = sorted(by_src)[i % len(by_src)]
        if by_src[src]:
            picked.append(by_src[src].pop())
        i += 1
    return picked


def run(s: Settings, limit: Optional[int] = None, show_json: bool = False, write: bool = True,
        cached_only: bool = False) -> dict:
    spark = get_spark("extraction")
    chunks = read_chunks(spark, s.hdfs)
    total_chunks = len(chunks)
    todo = [] if cached_only else pick_pilot(chunks, limit) if limit else chunks
    llm = LLMClient(s)
    cache = ResultCache(s.cache_dir, s.extraction_model, PROMPT_VERSION)

    stats = Counter()
    usage_rows: list[dict] = []
    quota_msg = None

    def work(chunk: dict) -> dict:
        key = chunk_hash(chunk["text"])
        hit = cache.get(key)
        if hit is not None:
            stats["cache_hits"] += 1
            return hit
        out = extract_chunk(llm, chunk)
        rec = {"chunk_hash": key, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
               "source": chunk["source"], "model": s.extraction_model, "prompt_version": PROMPT_VERSION,
               "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **out}
        cache.put(key, rec)
        stats["llm_extracted"] += 1
        usage_rows.extend(out["usage"])
        return rec

    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=s.concurrency) as pool:
        futures = {pool.submit(work, c): c for c in todo}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                results.append(rec)
                stats[f"status_{rec['status']}"] += 1
                done = len(results)
                if done % 25 == 0:
                    log.info("progress %d/%d (%.0fs)", done, len(todo), time.monotonic() - t0)
            except CancelledError:
                continue                            # not started: cancelled after the quota stop
            except QuotaExhausted as e:
                quota_msg = quota_msg or str(e)
                for f in futures:
                    f.cancel()
            except LLMCallFailed as e:
                stats["failed"] += 1
                log.error("chunk %s failed: %s", futures[fut]["chunk_id"], e)
    wall = time.monotonic() - t0

    if show_json:
        for rec in sorted(results, key=lambda r: r["chunk_id"]):
            print(json.dumps({k: rec[k] for k in ("chunk_id", "source", "status", "attempts", "usage",
                                                  "entities", "relations") if k in rec},
                             indent=1, ensure_ascii=False))

    # Silver tables are rebuilt from every cached result for the current chunks, so a resumed
    # run always writes the complete picture so far.
    text_by_hash = {chunk_hash(c["text"]): c["text"] for c in chunks}
    all_recs = [recheck(r, text_by_hash[r["chunk_hash"]]) for r in cache.all() if r["chunk_hash"] in text_by_hash]
    ent_rows, rel_rows = [], []
    for r in all_recs:
        e, rl = to_rows(r)
        ent_rows += e
        rel_rows += rl
    if write:
        write_table(spark, ent_rows, ENTITY_SCHEMA, silver(s.hdfs, "extracted_entities"))
        write_table(spark, rel_rows, RELATION_SCHEMA, silver(s.hdfs, "extracted_relations"))
    spark.stop()

    live = stats["llm_extracted"]
    calls = len(usage_rows)
    tin = sum(u["input_tokens"] for u in usage_rows)
    tout = sum(u["output_tokens"] for u in usage_rows)
    think = sum(u["thinking_tokens"] for u in usage_rows)
    remaining = total_chunks - len(all_recs)
    per_chunk = wall / live if live else None
    rels = [r for rec in all_recs for r in rec["relations"]]
    return {
        "model": s.extraction_model, "prompt_version": PROMPT_VERSION,
        "chunks_total": total_chunks, "chunks_this_run": len(todo), "chunks_extracted_total": len(all_recs),
        "chunks_remaining": remaining, "this_run": dict(stats),
        "calls": calls, "input_tokens": tin, "output_tokens": tout, "thinking_tokens": think,
        "avg_input_tokens_per_call": round(tin / calls) if calls else 0,
        "avg_output_tokens_per_call": round(tout / calls) if calls else 0,
        "avg_latency_s": round(sum(u["latency_s"] for u in usage_rows) / calls, 2) if calls else 0,
        "wall_time_s": round(wall, 1),
        "throughput_s_per_chunk": round(per_chunk, 2) if per_chunk else None,
        "projected_time_remaining_min": round(remaining * per_chunk / 60, 1) if per_chunk else None,
        "projected_calls_remaining": round(remaining * calls / live) if live else None,
        "settings": {"concurrency": s.concurrency, "min_interval_s": s.min_interval_s},
        "entities_total": len(ent_rows), "relations_total": len(rel_rows),
        "relation_flags": relation_flag_stats(rels),
        "quota_exhausted": quota_msg,
    }


def relation_flag_stats(rels: list[dict]) -> dict:
    """Counts and rates of relations failing each check (kept in silver, excluded from the KG)."""
    n = len(rels)
    checks = {"evidence_unverified": lambda r: not r["evidence_verified"],
              "schema_invalid": lambda r: not r["schema_valid"],
              "unknown_endpoint": lambda r: not r["endpoints_known"],
              "self_loop": lambda r: r["self_loop"]}
    out = {name: sum(map(f, rels)) for name, f in checks.items()}
    out["any_flag"] = sum(any(f(r) for f in checks.values()) for r in rels)
    out["direction_repaired"] = sum(r["direction_repaired"] for r in rels)     # repaired, not excluded
    return {k: {"count": v, "rate": round(v / n, 4) if n else 0.0} for k, v in out.items()}


def current_records(s: Settings) -> tuple[list[dict], list[dict]]:
    """Current silver chunks, and the cached extractions that belong to them."""
    spark = get_spark("extraction-status")
    chunks = read_chunks(spark, s.hdfs)
    spark.stop()
    cache = ResultCache(s.cache_dir, s.extraction_model, PROMPT_VERSION)
    text_by_hash = {chunk_hash(c["text"]): c["text"] for c in chunks}
    return chunks, [recheck(r, text_by_hash[r["chunk_hash"]]) for r in cache.all()
                    if r["chunk_hash"] in text_by_hash]


def status(s: Settings) -> dict:
    """How many current silver chunks have a cached extraction (no LLM calls)."""
    chunks, recs = current_records(s)
    rels = [r for rec in recs for r in rec["relations"]]
    return {"model": s.extraction_model, "prompt_version": PROMPT_VERSION, "chunks_total": len(chunks),
            "chunks_extracted": len(recs), "chunks_remaining": len(chunks) - len(recs),
            "status_counts": dict(Counter(r["status"] for r in recs)),
            "entities": sum(len(r["entities"]) for r in recs), "relations": len(rels),
            "relation_flags": relation_flag_stats(rels)}
