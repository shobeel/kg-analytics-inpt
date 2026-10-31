"""Idempotent batch load of the gold tables into Neo4j: UNWIND + MERGE on unique keys.

Nodes:   (:<Label>:Entity {id, kg_version})   (:Document {id})   (:Chunk {id})
Edges:   (Entity)-[:<TYPE> {kg_version, key}]->(Entity)   (Chunk)-[:MENTIONS {kg_version, key}]->(Entity)
         (Chunk)-[:PART_OF]->(Document)
Re-running MERGEs the same keys (no duplicates) and then deletes what gold no longer has
(stale nodes/edges), so Neo4j always mirrors gold exactly.
"""

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import ENTITY_LABELS, RELATION_TYPES, VERSIONS
from .cypher import statements

log = logging.getLogger(__name__)

EDGE_PROPS = ("key", "kg_version", "mention_count", "confidence", "chunk_id", "source_doc_id", "evidence_text",
              "extractor_model", "prompt_version", "direction_repaired", "direction_repaired_count",
              "chunk_ids", "source_doc_ids", "evidence_texts", "confidences", "extractor_models", "prompt_versions")


def edge_props(row: dict) -> dict:
    """Neo4j stores no nulls inside lists: drop None items (and None scalars, which SET += would
    turn into property removals)."""
    out = {}
    for k in EDGE_PROPS:
        v = row.get(k)
        if isinstance(v, list):
            out[k] = [x for x in v if x is not None]
        elif v is not None:
            out[k] = v
    return out


def batches(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


class Loader:
    def __init__(self, driver, batch_size: int = 500):
        self.driver = driver
        self.batch_size = batch_size
        self.counters = Counter()

    def run(self, query: str, **params):
        with self.driver.session() as s:
            res = s.run(query, **params)
            data = [r.data() for r in res]
            c = res.consume().counters
        for k in ("nodes_created", "nodes_deleted", "relationships_created", "relationships_deleted",
                  "labels_added", "labels_removed", "indexes_added", "constraints_added"):
            self.counters[k] += getattr(c, k)
        return data

    def run_batched(self, query: str, rows: list, size: int = None):
        for b in batches(rows, size or self.batch_size):
            self.run(query, rows=b)

    # ------------------------------------------------------------------ schema
    def apply_schema(self, path: Path) -> int:
        stmts = statements(path.read_text())
        for q in stmts:
            self.run(q)
        self.run("CALL db.awaitIndexes(300)")
        return len(stmts)

    # ------------------------------------------------------------------ nodes
    def load_documents(self, docs: list[dict]) -> None:
        self.run_batched("""
            UNWIND $rows AS r MERGE (d:Document {id: r.id})
            SET d.source = r.source, d.url = r.url, d.title = r.title,
                d.published_at = r.published_at, d.retrieved_at = r.retrieved_at""", docs)

    def load_chunks(self, chunks: list[dict]) -> None:
        self.run_batched("""
            UNWIND $rows AS r MERGE (c:Chunk {id: r.id})
            SET c.doc_id = r.doc_id, c.source = r.source, c.chunk_index = r.chunk_index, c.text = r.text,
                c.char_start = r.char_start, c.char_end = r.char_end, c.n_tokens_est = r.n_tokens_est,
                c.embedding = r.embedding""", chunks, size=100)

    def load_entities(self, label: str, nodes: list[dict]) -> None:
        assert label in ENTITY_LABELS
        self.run_batched(f"""
            UNWIND $rows AS r MERGE (n:{label} {{id: r.id, kg_version: r.kg_version}})
            SET n:Entity, n.name = r.name, n.aliases = r.aliases, n.description = r.description,
                n.mention_count = r.mention_count, n.raw_ids = r.raw_ids, n.resolved_id = r.resolved_id,
                n.embedding = r.embedding""", nodes)

    # ------------------------------------------------------------------ edges
    def load_part_of(self, rows: list[dict]) -> None:
        self.run_batched("""
            UNWIND $rows AS r MATCH (c:Chunk {id: r.chunk_id}) MATCH (d:Document {id: r.doc_id})
            MERGE (c)-[:PART_OF]->(d)""", rows)

    def load_mentions(self, rows: list[dict]) -> None:
        by_label = defaultdict(list)
        for r in rows:
            by_label[r["entity_label"]].append({**r, "key": f"{r['chunk_id']}|{r['entity_id']}"})
        for label, group in sorted(by_label.items()):
            assert label in ENTITY_LABELS
            self.run_batched(f"""
                UNWIND $rows AS r MATCH (c:Chunk {{id: r.chunk_id}})
                MATCH (e:{label} {{id: r.entity_id, kg_version: r.kg_version}})
                MERGE (c)-[m:MENTIONS {{kg_version: r.kg_version, key: r.key}}]->(e)
                SET m.mention_count = r.mention_count""", group)

    def load_relations(self, rel: str, rows: list[dict]) -> None:
        assert rel in RELATION_TYPES
        groups = defaultdict(list)
        for r in rows:
            groups[(r["src_label"], r["tgt_label"])].append({"src": r["src_id"], "tgt": r["tgt_id"],
                                                             "props": edge_props(r)})
        for (sl, tl), group in sorted(groups.items()):
            assert sl in ENTITY_LABELS and tl in ENTITY_LABELS
            self.run_batched(f"""
                UNWIND $rows AS r
                MATCH (s:{sl} {{id: r.src, kg_version: r.props.kg_version}})
                MATCH (t:{tl} {{id: r.tgt, kg_version: r.props.kg_version}})
                MERGE (s)-[e:{rel} {{kg_version: r.props.kg_version, key: r.props.key}}]->(t)
                SET e += r.props""", group)

    # ------------------------------------------------------------------ stale cleanup
    def delete_stale(self, label: str, version: str, keep_ids: list[str]) -> None:
        self.run(f"""MATCH (n:{label} {{kg_version: $v}}) WHERE NOT n.id IN $keep DETACH DELETE n""",
                 v=version, keep=keep_ids)

    def delete_stale_plain(self, label: str, keep_ids: list[str]) -> None:
        assert label in ("Document", "Chunk")
        self.run(f"MATCH (n:{label}) WHERE NOT n.id IN $keep DETACH DELETE n", keep=keep_ids)

    def delete_stale_edges(self, rel: str, version: str, keep_keys: list[str]) -> None:
        assert rel in RELATION_TYPES + ("MENTIONS",)
        self.run(f"MATCH ()-[e:{rel} {{kg_version: $v}}]->() WHERE NOT e.key IN $keep DELETE e",
                 v=version, keep=keep_keys)


def load_all(driver, spark, embedder, schema_path: Path, batch_size: int = 500) -> dict:
    from .gold import read

    t0 = time.monotonic()
    L = Loader(driver, batch_size)
    n_schema = L.apply_schema(schema_path)

    docs = read(spark, "nodes_Document")
    chunks = sorted(read(spark, "nodes_Chunk"), key=lambda c: c["id"])
    title = {d["id"]: d["title"] or "" for d in docs}
    for c, v in zip(chunks, embedder.embed_chunks([c["text"] for c in chunks], [title[c["doc_id"]] for c in chunks])):
        c["embedding"] = v
    embedder.release()
    L.load_documents(docs)
    L.load_chunks(chunks)
    L.delete_stale_plain("Document", [d["id"] for d in docs])
    L.delete_stale_plain("Chunk", [c["id"] for c in chunks])

    for label in ENTITY_LABELS:
        nodes = read(spark, f"nodes_{label}")
        for n in nodes:
            n["embedding"] = [float(x) for x in n["embedding"]]
        L.load_entities(label, nodes)
        for v in VERSIONS:
            L.delete_stale(label, v, [n["id"] for n in nodes if n["kg_version"] == v])
        log.info("loaded %s: %d nodes", label, len(nodes))
        del nodes

    L.load_part_of(read(spark, "edges_PART_OF"))
    mentions = read(spark, "edges_MENTIONS")
    L.load_mentions(mentions)
    for v in VERSIONS:
        L.delete_stale_edges("MENTIONS", v, [f"{m['chunk_id']}|{m['entity_id']}" for m in mentions
                                            if m["kg_version"] == v])
    for rel in RELATION_TYPES:
        rows = read(spark, f"edges_{rel}")
        L.load_relations(rel, rows)
        for v in VERSIONS:
            L.delete_stale_edges(rel, v, [r["key"] for r in rows if r["kg_version"] == v])
    return {"schema_statements": n_schema, "chunk_embeddings_computed": embedder.computed,
            "counters": dict(L.counters), "seconds": round(time.monotonic() - t0, 1)}
