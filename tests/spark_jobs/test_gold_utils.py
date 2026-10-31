import pytest
from pyspark.sql import SparkSession

import gold_utils as gu


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("test_gold_utils")
         .config("spark.sql.shuffle.partitions", "1").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


EMB = [0.1, 0.2]


def frames(spark):
    raw = spark.createDataFrame([
        ("raw_openai", "OpenAI", "Organization", 3, "lab", EMB),
        ("raw_openai_inc", "OpenAI Inc.", "Organization", 1, "", EMB),
        ("raw_gpt4", "GPT-4", "AIModel", 2, "model", EMB),
        ("raw_ms", "Microsoft", "Organization", 1, "", EMB)],
        "raw_id string, name string, type string, mention_count int, description string, embedding array<double>")
    res = spark.createDataFrame([
        ("ent_openai", "OpenAI", "Organization", ["OpenAI", "OpenAI Inc."], "lab", 4, ["raw_openai", "raw_openai_inc"], EMB),
        ("ent_gpt4", "GPT-4", "AIModel", ["GPT-4"], "model", 2, ["raw_gpt4"], EMB),
        ("ent_ms", "Microsoft", "Organization", ["Microsoft"], "", 1, ["raw_ms"], EMB)],
        "entity_id string, name string, type string, aliases array<string>, description string, "
        "mention_count int, raw_ids array<string>, embedding array<double>")
    cols = ("relation string, chunk_id string, doc_id string, source_raw_id string, target_raw_id string, "
            "source_resolved_id string, target_resolved_id string, evidence_text string, confidence double, "
            "evidence_verified boolean, schema_valid boolean, direction_repaired boolean, extractor_model string")
    rels = spark.createDataFrame([
        # same edge in two chunks, through two raw aliases -> 2 raw edges, 1 resolved edge
        ("DEVELOPED_BY", "c1", "d1", "raw_gpt4", "raw_openai", "ent_gpt4", "ent_openai", "ev1", 0.8, True, True, False, "m"),
        ("DEVELOPED_BY", "c2", "d2", "raw_gpt4", "raw_openai_inc", "ent_gpt4", "ent_openai", "ev2", 0.9, True, True, True, "m"),
        # symmetric, both directions -> one edge
        ("COMPETES_WITH", "c1", "d1", "raw_openai", "raw_ms", "ent_openai", "ent_ms", "ev3", 0.7, True, True, False, "m"),
        ("COMPETES_WITH", "c2", "d2", "raw_ms", "raw_openai", "ent_ms", "ent_openai", "ev4", 0.6, True, True, False, "m"),
        # aliases of one cluster: raw edge, resolved self-loop (dropped)
        ("PARTNERS_WITH", "c1", "d1", "raw_openai", "raw_openai_inc", "ent_openai", "ent_openai", "ev5", 0.9, True, True, False, "m"),
        # flagged: never reaches gold
        ("USES", "c1", "d1", "raw_ms", "raw_gpt4", "ent_ms", "ent_gpt4", "ev6", 0.9, False, True, False, "m")], cols)
    meta = spark.createDataFrame([("c1", "m", "extract-v2"), ("c2", "m", "extract-v2")],
                                 "chunk_id string, extractor_model string, prompt_version string")
    return raw, res, rels, meta


def test_entity_nodes_both_versions(spark):
    raw, res, _, _ = frames(spark)
    n = {(r.id, r.kg_version): r for r in gu.entity_nodes(raw, res).collect()}
    assert len(n) == 7
    assert n[("raw_openai_inc", "raw")].resolved_id == "ent_openai"
    assert n[("ent_openai", "resolved")].raw_ids == ["raw_openai", "raw_openai_inc"]
    assert n[("raw_gpt4", "raw")].aliases == ["GPT-4"]


def test_edges_aggregate_provenance_symmetric_and_flags(spark):
    raw, res, rels, meta = frames(spark)
    nodes = gu.entity_nodes(raw, res)
    edges, loops = gu.aggregate_edges(gu.relation_mentions(rels, meta, nodes))
    e = {(r.kg_version, r.key): r for r in edges.collect()}
    assert not any("USES" in k for _, k in e)                                  # flagged excluded
    dev = e[("resolved", "ent_gpt4|DEVELOPED_BY|ent_openai")]
    assert dev.mention_count == 2 and dev.confidence == 0.9
    assert dev.chunk_ids == ["c2", "c1"] and dev.evidence_text == "ev2"        # best mention first
    assert dev.direction_repaired and dev.direction_repaired_count == 1
    assert dev.prompt_version == "extract-v2" and dev.source_doc_ids == ["d2", "d1"]
    assert sum(1 for v, k in e if v == "raw" and "DEVELOPED_BY" in k) == 2
    comp = [r for (v, k), r in e.items() if v == "resolved" and r.relation == "COMPETES_WITH"]
    assert len(comp) == 1 and comp[0].mention_count == 2                        # either direction
    assert ("raw", "raw_openai|PARTNERS_WITH|raw_openai_inc") in e
    assert [r.kg_version for r in loops.collect()] == ["resolved"]              # merged-alias self-loop


def test_mention_edges_per_version(spark):
    raw, res, _, _ = frames(spark)
    ments = spark.createDataFrame([("c1", "raw_openai", "ent_openai"), ("c1", "raw_openai_inc", "ent_openai")],
                                  "chunk_id string, raw_id string, resolved_id string")
    m = {(r.kg_version, r.entity_id): r.mention_count for r in gu.mention_edges(ments, gu.entity_nodes(raw, res)).collect()}
    assert m == {("raw", "raw_openai"): 1, ("raw", "raw_openai_inc"): 1, ("resolved", "ent_openai"): 2}
