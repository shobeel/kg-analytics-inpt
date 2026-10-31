import argparse
import json
import logging
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(prog="loader")
sub = parser.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("load", help="schema + gold -> Neo4j (both kg_versions); idempotent")
p.add_argument("--batch-size", type=int, default=500)
sub.add_parser("check", help="Neo4j vs gold counts, provenance, vector search, raw vs resolved")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("py4j", "neo4j", "urllib3", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from neo4j import GraphDatabase  # noqa: E402

from .embed import ChunkEmbedder  # noqa: E402
from .gold import get_spark  # noqa: E402

NEO4J_DIR = Path(os.environ.get("NEO4J_DIR", "/app/neo4j"))
driver = GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://neo4j:7687"),
                              auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))
embedder = ChunkEmbedder(os.environ.get("FASTEMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                         os.environ.get("EMBED_CACHE_DIR", "/data/cache/embeddings"))
spark = get_spark(f"loader-{args.cmd}")
spark.sparkContext.setLogLevel("ERROR")

try:
    if args.cmd == "load":
        from .load import load_all
        summary = load_all(driver, spark, embedder, NEO4J_DIR / "schema.cypher", args.batch_size)
        print("LOAD_SUMMARY " + json.dumps(summary))
        Path("/app/logs/neo4j_load_summary.json").write_text(json.dumps(summary, indent=2))
    elif args.cmd == "check":
        from .checks import report, run_checks
        r = run_checks(driver, spark, embedder, (NEO4J_DIR / "checks.cypher").read_text())
        print(report(r))
        Path("/app/logs/phase3_checks.json").write_text(json.dumps(r, indent=2, default=str))
        flags = {k: r[k] for k in ("counts_ok", "provenance_ok", "vector_ok")}
        print("CHECKS " + json.dumps(flags))
        if not all(flags.values()):
            sys.exit(1)
finally:
    spark.stop()
    driver.close()
