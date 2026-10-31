"""Entity resolution. Builds BOTH graphs for RQ2 from the same extracted mentions:

KG_raw       one node per exact (type, name) string. No resolution at all.
KG_resolved  1. normalization  (case, punctuation, "the", "(company)", legal suffixes)
             2. blocking by entity type (pairs never cross types)
             3. similarity: string (token-sort ratio) + embedding cosine (fastembed)
                - version guard: names with different version tokens never merge (GPT-3 / GPT-4)
                - near-identical strings merge automatically
                - ambiguous pairs (similar, or acronym <-> expansion) go to the LLM, batched
             4. union-find clusters -> canonical id, canonical name (most mentioned), aliases
Every decision is written to the merge log (silver/er_merge_log).
"""

import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
from pyspark.sql import types as T

from .normalize import is_acronym_pair, norm_key, version_signature

log = logging.getLogger(__name__)

ADJUDICATION_VERSION = "adjudicate-v1"


@dataclass(frozen=True)
class Thresholds:
    auto_string: float = 0.93       # token-sort ratio for an automatic merge
    ambiguous_string: float = 0.75  # ... or cosine >= ambiguous_cosine -> ask the LLM
    ambiguous_cosine: float = 0.80
    max_adjudications: int = 400    # cap LLM pair checks (free-tier quota)
    batch_size: int = 25            # pairs per LLM call


def raw_id(etype: str, name: str) -> str:
    return "raw_" + hashlib.sha1(f"{etype}|{name}".encode()).hexdigest()[:16]


def resolved_id(etype: str, key: str) -> str:
    return "ent_" + hashlib.sha1(f"{etype}|{key}".encode()).hexdigest()[:16]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


def string_sim_matrix(keys: list[str]) -> np.ndarray:
    from rapidfuzz import fuzz, process

    return process.cdist(keys, keys, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0


def resolve_type(groups: list[dict], emb: np.ndarray, th: Thresholds,
                 adjudicate: Optional[Callable[[list[dict]], dict]]) -> tuple[UnionFind, list[dict], list[dict]]:
    """Stage 3 for ONE entity type. groups: [{key, name, description, version}] (already
    merged by normalized key). emb: row-normalized embeddings of group names.
    Returns (union-find over groups, merge-log rows, ambiguous pairs not adjudicated)."""
    n = len(groups)
    uf, logrows, ambiguous = UnionFind(n), [], []
    if n < 2:
        return uf, logrows, ambiguous
    sim = string_sim_matrix([g["key"] for g in groups])
    cos = emb @ emb.T if emb.size else np.zeros((n, n), dtype=np.float32)
    nospace = [g["key"].replace(" ", "") for g in groups]

    for i in range(n):
        for j in range(i + 1, n):
            s, c = float(sim[i, j]), float(cos[i, j])
            acr = is_acronym_pair(groups[i]["key"], groups[j]["key"])
            near = s >= th.ambiguous_string or c >= th.ambiguous_cosine or acr
            if not near and nospace[i] != nospace[j]:
                continue
            pair = {"i": i, "j": j, "string_sim": round(s, 4), "cosine": round(c, 4), "acronym": acr}
            if groups[i]["version"] != groups[j]["version"]:
                logrows.append({**pair, "decision": "reject_version_mismatch"})
            elif nospace[i] == nospace[j] or s >= th.auto_string:
                uf.union(i, j)
                logrows.append({**pair, "decision": "merge_auto_string"})
            else:
                ambiguous.append(pair)

    ambiguous.sort(key=lambda p: max(p["string_sim"], p["cosine"]), reverse=True)
    to_ask, skipped = ambiguous[:th.max_adjudications], ambiguous[th.max_adjudications:]
    if adjudicate and to_ask:
        for k in range(0, len(to_ask), th.batch_size):
            batch = to_ask[k:k + th.batch_size]
            answers = adjudicate([{"a": groups[p["i"]], "b": groups[p["j"]]} for p in batch])
            for idx, p in enumerate(batch):
                same = answers.get(idx)
                if same is None:
                    skipped.append(p)
                    continue
                if same:
                    uf.union(p["i"], p["j"])
                logrows.append({**p, "decision": "merge_llm" if same else "keep_llm"})
    else:
        skipped = ambiguous
    for p in skipped:
        logrows.append({**p, "decision": "keep_not_adjudicated"})
    return uf, logrows, skipped


# --------------------------------------------------------------------------- LLM adjudication

def make_adjudicator(llm, cache, model: str) -> Callable[[list[dict]], dict]:
    from .prompts import ADJUDICATION_SCHEMA, ADJUDICATION_SYSTEM
    from .llm import QuotaExhausted

    state = {"quota": False}

    def describe(g: dict) -> str:
        return f'"{g["name"]}" ({g["type"]}){": " + g["description"][:160] if g["description"] else ""}'

    def adjudicate(pairs: list[dict]) -> dict:
        out, ask = {}, []
        for idx, p in enumerate(pairs):
            key = hashlib.sha256(json.dumps(sorted([describe(p["a"]), describe(p["b"])])).encode()).hexdigest()
            hit = cache.get(key)
            if hit is not None:
                out[idx] = hit["same"]
            else:
                ask.append((idx, key, p))
        if not ask or state["quota"]:
            return out
        lines = [f"{n}. A = {describe(p['a'])}\n   B = {describe(p['b'])}" for n, (_, _, p) in enumerate(ask)]
        try:
            raw, _ = llm.chat_json(system=ADJUDICATION_SYSTEM, user="Pairs:\n" + "\n".join(lines),
                                   schema=ADJUDICATION_SCHEMA, schema_name="er_adjudication",
                                   purpose="er_adjudication", ref=f"{len(ask)} pairs", model=model)
            decisions = {d["pair"]: bool(d["same"]) for d in json.loads(raw).get("decisions", [])}
        except QuotaExhausted:
            log.warning("daily quota reached during adjudication: remaining pairs kept separate")
            state["quota"] = True
            return out
        except Exception as e:                      # invalid output: leave this batch unmerged
            log.warning("adjudication batch failed: %s", e)
            return out
        for n, (idx, key, _) in enumerate(ask):
            if n in decisions:
                out[idx] = decisions[n]
                cache.put(key, {"same": decisions[n], "model": model})
        return out

    adjudicate.state = state
    return adjudicate


# --------------------------------------------------------------------------- full run

def run_resolution(s, th: Thresholds = Thresholds()) -> dict:
    from pyspark.sql import functions as F

    from .cache import ResultCache
    from .embed import Embedder
    from .llm import LLMClient
    from .spark_io import get_spark, silver, write_table

    spark = get_spark("entity_resolution")
    ments = [r.asDict() for r in spark.read.format("delta").load(silver(s.hdfs, "extracted_entities")).collect()]
    rels = [r.asDict() for r in spark.read.format("delta").load(silver(s.hdfs, "extracted_relations")).collect()]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- KG_raw: exact (type, name)
    raw: dict[str, dict] = {}
    desc_counts: dict[str, Counter] = defaultdict(Counter)
    for m in ments:
        rid = raw_id(m["type"], m["name"])
        r = raw.setdefault(rid, {"raw_id": rid, "name": m["name"], "type": m["type"], "mention_count": 0})
        r["mention_count"] += 1
        if m["description"]:
            desc_counts[rid][m["description"]] += 1
    for rid, r in raw.items():
        r["description"] = max(desc_counts[rid], key=lambda d: (desc_counts[rid][d], len(d)), default="")

    # ---- stage 1+2: normalized key within type
    groups_by_type: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for rid, r in raw.items():
        groups_by_type[r["type"]][norm_key(r["name"]) or r["name"].casefold()].append(rid)

    embedder = Embedder(s.embedding_model)
    llm = LLMClient(s)
    adj_cache = ResultCache(s.cache_dir, s.extraction_model, ADJUDICATION_VERSION)
    adjudicate = make_adjudicator(llm, adj_cache, s.extraction_model)

    merge_log, raw_to_resolved, resolved = [], {}, []
    stage_counts = Counter()
    for etype, by_key in sorted(groups_by_type.items()):
        groups = []
        for key, rids in sorted(by_key.items()):
            members = sorted(rids, key=lambda x: (-raw[x]["mention_count"], raw[x]["name"]))
            rep = raw[members[0]]
            for other in members[1:]:
                merge_log.append({"type": etype, "a": rep["name"], "b": raw[other]["name"],
                                  "decision": "merge_normalized", "string_sim": 1.0, "cosine": None, "acronym": False})
                stage_counts["merge_normalized"] += 1
            groups.append({"key": key, "name": rep["name"], "type": etype, "raw_ids": members,
                           "description": max((raw[x]["description"] for x in members), key=len),
                           "version": version_signature(rep["name"])})
        emb = embedder.embed([g["name"] for g in groups])
        uf, logrows, _ = resolve_type(groups, emb, th, adjudicate)
        for row in logrows:
            merge_log.append({"type": etype, "a": groups[row["i"]]["name"], "b": groups[row["j"]]["name"],
                              "decision": row["decision"], "string_sim": row["string_sim"],
                              "cosine": row["cosine"], "acronym": row["acronym"]})
            stage_counts[row["decision"]] += 1

        clusters = defaultdict(list)
        for gi in range(len(groups)):
            clusters[uf.find(gi)].append(gi)
        for members in clusters.values():
            rids = [rid for gi in members for rid in groups[gi]["raw_ids"]]
            names = Counter()
            for rid in rids:
                names[raw[rid]["name"]] += raw[rid]["mention_count"]
            canonical = max(names, key=lambda nm: (names[nm], len(nm)))
            eid = resolved_id(etype, norm_key(canonical) or canonical.casefold())
            desc = max((raw[rid]["description"] for rid in rids), key=len)
            resolved.append({"entity_id": eid, "name": canonical, "type": etype,
                             "aliases": sorted(n for n in names if n != canonical), "description": desc,
                             "mention_count": sum(names.values()), "raw_ids": sorted(rids)})
            for rid in rids:
                raw_to_resolved[rid] = eid

    # ---- node embeddings ("name: description") for both graphs (Neo4j vector index, Phase 3)
    def node_text(e):
        return f"{e['name']} ({e['type']}): {e['description']}" if e["description"] else f"{e['name']} ({e['type']})"
    raw_list = sorted(raw.values(), key=lambda r: r["raw_id"])
    for e, v in zip(raw_list, embedder.embed([node_text(e) for e in raw_list])):
        e["embedding"] = v.tolist()
    for e, v in zip(resolved, embedder.embed([node_text(e) for e in resolved])):
        e["embedding"] = v.tolist()

    # ---- mentions and relations with both id spaces
    mention_rows = [{"chunk_id": m["chunk_id"], "doc_id": m["doc_id"], "name": m["name"], "type": m["type"],
                     "raw_id": raw_id(m["type"], m["name"]),
                     "resolved_id": raw_to_resolved[raw_id(m["type"], m["name"])]} for m in ments]
    by_chunk_name = {(m["chunk_id"], m["name"].lower()): m for m in ments}
    # KG edges = relations that pass every check. Flagged relations stay in
    # silver/extracted_relations with their flags (hallucination analysis, RQ1/RQ5).
    rel_rows, flags = [], Counter()
    for k, r in enumerate(sorted(rels, key=lambda x: (x["chunk_id"], x["source_name"], x["relation"], x["target_name"]))):
        src = by_chunk_name.get((r["chunk_id"], r["source_name"].lower()))
        tgt = by_chunk_name.get((r["chunk_id"], r["target_name"].lower()))
        problems = [name for name, bad in (("unknown_endpoint", not src or not tgt),
                                           ("schema_invalid", not r["schema_valid"]),
                                           ("evidence_unverified", not r["evidence_verified"]),
                                           ("self_loop", r["self_loop"])) if bad]
        for p in problems:
            flags[p] += 1
        if problems:
            flags["any_flag"] += 1
            continue
        s_raw, t_raw = raw_id(src["type"], src["name"]), raw_id(tgt["type"], tgt["name"])
        rel_rows.append({
            "rel_id": "rel_" + hashlib.sha1(f"{r['chunk_id']}|{r['source_name']}|{r['relation']}|{r['target_name']}|{k}".encode()).hexdigest()[:16],
            "relation": r["relation"], "chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "source": r["source"],
            "source_raw_id": s_raw, "target_raw_id": t_raw,
            "source_resolved_id": raw_to_resolved[s_raw], "target_resolved_id": raw_to_resolved[t_raw],
            "evidence_text": r["evidence_text"], "confidence": r["confidence"],
            "evidence_verified": r["evidence_verified"], "schema_valid": r["schema_valid"],
            "direction_repaired": r["direction_repaired"], "extractor_model": r["extractor_model"]})

    # ---- write silver
    vec = T.ArrayType(T.FloatType())
    write_table(spark, raw_list, T.StructType([
        T.StructField("raw_id", T.StringType()), T.StructField("name", T.StringType()),
        T.StructField("type", T.StringType()), T.StructField("mention_count", T.IntegerType()),
        T.StructField("description", T.StringType()), T.StructField("embedding", vec)]),
        silver(s.hdfs, "entities_raw"))
    write_table(spark, resolved, T.StructType([
        T.StructField("entity_id", T.StringType()), T.StructField("name", T.StringType()),
        T.StructField("type", T.StringType()), T.StructField("aliases", T.ArrayType(T.StringType())),
        T.StructField("description", T.StringType()), T.StructField("mention_count", T.IntegerType()),
        T.StructField("raw_ids", T.ArrayType(T.StringType())), T.StructField("embedding", vec)]),
        silver(s.hdfs, "entities_resolved"))
    write_table(spark, mention_rows, T.StructType([T.StructField(c, T.StringType()) for c in
        ("chunk_id", "doc_id", "name", "type", "raw_id", "resolved_id")]), silver(s.hdfs, "entity_mentions"))
    write_table(spark, rel_rows, T.StructType(
        [T.StructField(c, T.StringType()) for c in ("rel_id", "relation", "chunk_id", "doc_id", "source",
         "source_raw_id", "target_raw_id", "source_resolved_id", "target_resolved_id", "evidence_text")]
        + [T.StructField("confidence", T.DoubleType()), T.StructField("evidence_verified", T.BooleanType()),
           T.StructField("schema_valid", T.BooleanType()), T.StructField("direction_repaired", T.BooleanType()),
           T.StructField("extractor_model", T.StringType())]),
        silver(s.hdfs, "relations"))
    for row in merge_log:
        row.update(adjudicator_model=s.extraction_model if row["decision"].endswith("_llm") else None, logged_at=now)
    write_table(spark, merge_log, T.StructType([
        T.StructField("type", T.StringType()), T.StructField("a", T.StringType()), T.StructField("b", T.StringType()),
        T.StructField("decision", T.StringType()), T.StructField("string_sim", T.DoubleType()),
        T.StructField("cosine", T.DoubleType()), T.StructField("acronym", T.BooleanType()),
        T.StructField("adjudicator_model", T.StringType()), T.StructField("logged_at", T.StringType())]),
        silver(s.hdfs, "er_merge_log"))
    spark.stop()

    per_type = {t: {"raw": sum(1 for r in raw_list if r["type"] == t),
                    "resolved": sum(1 for e in resolved if e["type"] == t)} for t in sorted(groups_by_type)}
    distinct_raw_edges = {(r["source_raw_id"], r["relation"], r["target_raw_id"]) for r in rel_rows}
    distinct_res_edges = {(r["source_resolved_id"], r["relation"], r["target_resolved_id"]) for r in rel_rows}
    return {
        "mentions": len(ments), "entities_raw": len(raw_list), "entities_resolved": len(resolved),
        "entities_merged_away": len(raw_list) - len(resolved),
        "reduction_pct": round(100 * (1 - len(resolved) / len(raw_list)), 1) if raw_list else 0,
        "clusters_with_merges": sum(1 for e in resolved if len(e["raw_ids"]) > 1),
        "decisions": dict(stage_counts), "per_type": per_type,
        "relations_extracted": len(rels), "relations_in_kg": len(rel_rows),
        "relations_flagged": {k: {"count": v, "rate": round(v / len(rels), 4) if rels else 0.0}
                              for k, v in sorted(flags.items())},
        "distinct_edges_raw": len(distinct_raw_edges), "distinct_edges_resolved": len(distinct_res_edges),
        "adjudication_stopped_by_quota": adjudicate.state["quota"],
        "embedding_model": s.embedding_model,
    }
