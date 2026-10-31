"""Pure DataFrame transforms for silver -> gold (unit-tested in tests/spark_jobs).

Two graph versions share Documents and Chunks:
  kg_version = 'raw'       one entity per (type, exact name)     ids raw_<sha1>
  kg_version = 'resolved'  one entity per ER cluster             ids ent_<sha1>
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

ENTITY_LABELS = ("Organization", "Person", "AIModel", "Paper", "Technology", "Dataset", "Product", "Location")
RELATION_TYPES = ("DEVELOPED_BY", "WORKS_AT", "AUTHORED", "CITES", "USES", "TRAINED_ON",
                  "COMPETES_WITH", "PARTNERS_WITH", "FUNDED_BY", "ACQUIRED", "LOCATED_IN", "BASED_ON")
SYMMETRIC = ("COMPETES_WITH", "PARTNERS_WITH")
VERSIONS = ("raw", "resolved")

NODE_COLUMNS = ["id", "kg_version", "label", "name", "aliases", "description", "mention_count",
                "raw_ids", "resolved_id", "embedding"]


def entity_nodes(raw: DataFrame, resolved: DataFrame) -> DataFrame:
    """Both versions in one frame with the same columns. Raw nodes point to their cluster
    (resolved_id); resolved nodes list their members (raw_ids)."""
    raw_to_res = (resolved.select(F.col("entity_id").alias("resolved_id"), F.explode("raw_ids").alias("raw_id")))
    r = (raw.join(raw_to_res, "raw_id", "left")
         .select(F.col("raw_id").alias("id"), F.lit("raw").alias("kg_version"), F.col("type").alias("label"),
                 "name", F.array("name").alias("aliases"), F.coalesce("description", F.lit("")).alias("description"),
                 F.col("mention_count").cast("int").alias("mention_count"),
                 F.array("raw_id").alias("raw_ids"), "resolved_id", "embedding"))
    s = resolved.select(F.col("entity_id").alias("id"), F.lit("resolved").alias("kg_version"),
                        F.col("type").alias("label"), "name", "aliases",
                        F.coalesce("description", F.lit("")).alias("description"),
                        F.col("mention_count").cast("int").alias("mention_count"),
                        "raw_ids", F.col("entity_id").alias("resolved_id"), "embedding")
    return r.unionByName(s).select(*NODE_COLUMNS)


def relation_mentions(rels: DataFrame, chunk_meta: DataFrame, nodes: DataFrame) -> DataFrame:
    """One row per (relation mention, kg_version) with endpoint ids and labels of that version.
    Only relations passing every Phase 2 check; symmetric relations get a canonical direction."""
    ok = rels.where(F.col("evidence_verified") & F.col("schema_valid"))
    ok = ok.join(chunk_meta.select("chunk_id", "prompt_version"), "chunk_id", "left")
    labels = nodes.select("id", "kg_version", "label")
    out = []
    for v, src, tgt in (("raw", "source_raw_id", "target_raw_id"),
                        ("resolved", "source_resolved_id", "target_resolved_id")):
        m = ok.select("relation", "chunk_id", F.col("doc_id").alias("source_doc_id"), "evidence_text",
                      "confidence", "direction_repaired", "extractor_model", "prompt_version",
                      F.col(src).alias("src_id"), F.col(tgt).alias("tgt_id"), F.lit(v).alias("kg_version"))
        swap = F.col("relation").isin(*SYMMETRIC) & (F.col("src_id") > F.col("tgt_id"))
        m = m.select(*[c for c in m.columns if c not in ("src_id", "tgt_id")],
                     F.when(swap, F.col("tgt_id")).otherwise(F.col("src_id")).alias("src_id"),
                     F.when(swap, F.col("src_id")).otherwise(F.col("tgt_id")).alias("tgt_id"))
        out.append(m)
    m = out[0].unionByName(out[1])
    m = (m.join(labels.select(F.col("id").alias("src_id"), "kg_version", F.col("label").alias("src_label")),
                ["src_id", "kg_version"], "inner")
         .join(labels.select(F.col("id").alias("tgt_id"), "kg_version", F.col("label").alias("tgt_label")),
               ["tgt_id", "kg_version"], "inner"))
    return m


def aggregate_edges(mentions: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Duplicate edges (same version, endpoints, type) collapse into one with mention_count,
    max confidence and aligned provenance lists (best mention first). Returns (edges,
    dropped_self_loops): entities merged by ER can turn an edge into a self-loop."""
    loops = mentions.where(F.col("src_id") == F.col("tgt_id"))
    m = mentions.where(F.col("src_id") != F.col("tgt_id"))
    prov = F.struct((-F.col("confidence")).alias("neg_conf"), "chunk_id", "source_doc_id", "evidence_text",
                    "confidence", "extractor_model", "prompt_version", "direction_repaired")
    keys = ["kg_version", "relation", "src_id", "src_label", "tgt_id", "tgt_label"]
    g = m.groupBy(*keys).agg(F.sort_array(F.collect_list(prov)).alias("p"),
                             F.count(F.lit(1)).cast("int").alias("mention_count"),
                             F.max("confidence").alias("confidence"))
    best = F.col("p")[0]
    edges = g.select(
        F.concat_ws("|", "src_id", "relation", "tgt_id").alias("key"), *keys, "mention_count", "confidence",
        best["chunk_id"].alias("chunk_id"), best["source_doc_id"].alias("source_doc_id"),
        best["evidence_text"].alias("evidence_text"),
        best["extractor_model"].alias("extractor_model"), best["prompt_version"].alias("prompt_version"),
        F.expr("aggregate(p, false, (acc, x) -> acc OR x.direction_repaired)").alias("direction_repaired"),
        F.expr("size(filter(p, x -> x.direction_repaired))").cast("int").alias("direction_repaired_count"),
        F.col("p.chunk_id").alias("chunk_ids"), F.col("p.source_doc_id").alias("source_doc_ids"),
        F.col("p.evidence_text").alias("evidence_texts"), F.col("p.confidence").alias("confidences"),
        F.array_distinct(F.col("p.extractor_model")).alias("extractor_models"),
        F.array_distinct(F.col("p.prompt_version")).alias("prompt_versions"))
    return edges, loops


def mention_edges(mentions: DataFrame, nodes: DataFrame) -> DataFrame:
    """(Chunk)-[:MENTIONS]->(Entity) per version, with how often the chunk names the entity."""
    labels = nodes.select(F.col("id").alias("entity_id"), "kg_version", F.col("label").alias("entity_label"))
    out = None
    for v, col in (("raw", "raw_id"), ("resolved", "resolved_id")):
        m = (mentions.groupBy("chunk_id", F.col(col).alias("entity_id"))
             .agg(F.count(F.lit(1)).cast("int").alias("mention_count"))
             .withColumn("kg_version", F.lit(v)))
        out = m if out is None else out.unionByName(m)
    return out.join(labels, ["entity_id", "kg_version"], "inner").select(
        "chunk_id", "entity_id", "entity_label", "kg_version", "mention_count")


def chunk_provenance(extracted_relations: DataFrame, extracted_entities: DataFrame) -> DataFrame:
    """chunk_id -> extractor_model, prompt_version (each chunk is extracted once per version)."""
    cols = ["chunk_id", "extractor_model", "prompt_version"]
    both = extracted_relations.select(*cols).unionByName(extracted_entities.select(*cols))
    w = Window.partitionBy("chunk_id").orderBy(F.col("prompt_version").desc())
    return both.distinct().withColumn("_r", F.row_number().over(w)).where("_r = 1").drop("_r")
