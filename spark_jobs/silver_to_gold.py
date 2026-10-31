"""Silver -> gold: graph-ready node and edge tables for Neo4j, in two versions (RQ2).

Reads   /datalake/silver/{documents, chunks, entities_raw, entities_resolved, entity_mentions,
                          relations, extracted_relations, extracted_entities}
Writes  /datalake/gold/nodes_<Label>   Organization ... Location (both kg_versions), Document, Chunk
        /datalake/gold/edges_<TYPE>    one table per relation type (both kg_versions), aggregated
                                       duplicates with provenance; plus edges_MENTIONS, edges_PART_OF
        /datalake/gold/_counts         rows per table and kg_version (what Neo4j must match)

Only relations that passed every Phase 2 check reach gold; flagged ones stay in silver.
Full recompute (overwrite) on every run. No LLM calls. Run with `make gold`.
"""

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import gold_utils as gu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdfs", default="hdfs://namenode:9000")
    a = ap.parse_args()
    silver, gold = f"{a.hdfs}/datalake/silver", f"{a.hdfs}/datalake/gold"

    spark = SparkSession.builder.appName("silver_to_gold").getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    rd = lambda t: spark.read.format("delta").load(f"{silver}/{t}")

    nodes = gu.entity_nodes(rd("entities_raw"), rd("entities_resolved")).cache()
    meta = gu.chunk_provenance(rd("extracted_relations"), rd("extracted_entities"))
    mentions = gu.relation_mentions(rd("relations"), meta, nodes)
    edges, loops = gu.aggregate_edges(mentions)
    edges = edges.cache()
    chunks = rd("chunks")
    docs = rd("documents")

    counts = []

    def write(df, name, by_version=True):
        df.coalesce(1).write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
            .save(f"{gold}/{name}")
        if by_version:
            for r in df.groupBy("kg_version").count().collect():
                counts.append((name, r["kg_version"], r["count"]))
        else:
            counts.append((name, "shared", df.count()))

    for label in gu.ENTITY_LABELS:
        write(nodes.where(F.col("label") == label), f"nodes_{label}")
    write(docs.select(F.col("doc_id").alias("id"), "source", "url", "title", "published_at", "retrieved_at"),
          "nodes_Document", by_version=False)
    write(chunks.select(F.col("chunk_id").alias("id"), "doc_id", "source", "chunk_index", "text",
                        "char_start", "char_end", "n_tokens_est"), "nodes_Chunk", by_version=False)
    for rel in gu.RELATION_TYPES:
        write(edges.where(F.col("relation") == rel), f"edges_{rel}")
    write(gu.mention_edges(rd("entity_mentions"), nodes), "edges_MENTIONS")
    write(chunks.select("chunk_id", "doc_id"), "edges_PART_OF", by_version=False)

    spark.createDataFrame(counts, "table string, kg_version string, rows long").coalesce(1) \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(f"{gold}/_counts")
    # versions missing from a table (e.g. no CITES edges) count as 0
    summary = {"tables": {}, "resolved_self_loops_dropped": loops.where("kg_version = 'resolved'").count(),
               "relation_mentions": mentions.count()}
    for t, v, n in sorted(counts):
        summary["tables"].setdefault(t, {})[v] = n
    print("GOLD_SUMMARY " + json.dumps(summary, indent=1))
    spark.stop()


if __name__ == "__main__":
    main()
