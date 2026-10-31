import numpy as np
import pytest

from loader import ENTITY_LABELS, RELATION_TYPES
from loader.checks import compare_counts
from loader.cypher import named_queries, statements
from loader.embed import ChunkEmbedder, mean_pool, windows
from loader.load import Loader, batches, edge_props

NEO4J = "/app/neo4j"


def test_schema_file_has_constraints_indexes_and_vector_indexes():
    stmts = statements(open(f"{NEO4J}/schema.cypher").read())
    joined = "\n".join(stmts)
    for label in ENTITY_LABELS:
        assert f"FOR (n:{label}) REQUIRE (n.id, n.kg_version) IS UNIQUE" in joined
    assert sum("VECTOR INDEX" in s for s in stmts) == 2
    assert all(not s.endswith(";") for s in stmts)


def test_checks_file_queries_filter_on_kg_version():
    q = named_queries(open(f"{NEO4J}/checks.cypher").read())
    assert {"node_counts", "edge_counts", "provenance", "vector_search", "orphans", "top_degree",
            "gds_project", "gds_wcc"} <= set(q)
    for name in ("provenance", "relation_edges", "orphans", "top_degree", "gds_project"):
        assert "$kg_version" in q[name]
    with pytest.raises(ValueError):
        named_queries("// name: two\nRETURN 1;\nRETURN 2;")


def test_windows_and_mean_pool():
    assert len(windows("w " * 400, 180)) == 3 and windows("") == [""]
    v = mean_pool(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.isclose(np.linalg.norm(v), 1.0) and np.isclose(v[0], v[1])


def test_chunk_embedder_caches(tmp_path):
    calls = []
    fn = lambda texts: (calls.append(len(texts)), np.ones((len(texts), 4), dtype=np.float32))[1]
    e = ChunkEmbedder("m/x", str(tmp_path), embed_fn=fn)
    a = e.embed_chunks(["a b c", "d " * 300])
    assert calls == [1 + 2] and len(a[0]) == 4 and e.computed == 2
    e2 = ChunkEmbedder("m/x", str(tmp_path), embed_fn=fn)
    assert e2.embed_chunks(["a b c", "d " * 300]) == a and e2.computed == 0 and calls == [3]
    e2.embed_chunks(["a b c"], titles=["T"])            # the title is part of the embedded text
    assert e2.computed == 1 and calls == [3, 1]


def test_compare_counts_flags_missing_and_extra():
    gold = {("nodes_Person", "raw"): 3, ("edges_USES", "resolved"): 2}
    neo = {("Person", "raw"): 3, ("USES", "resolved"): 1, ("Paper", "raw"): 1}
    rows = {(r["name"], r["kg_version"]): r["ok"] for r in compare_counts(gold, neo)}
    assert rows == {("Person", "raw"): True, ("USES", "resolved"): False, ("Paper", "raw"): False}


def test_edge_props_drops_nulls_and_batches():
    p = edge_props({"key": "a|USES|b", "kg_version": "raw", "chunk_ids": ["c1", None], "prompt_version": None})
    assert p == {"key": "a|USES|b", "kg_version": "raw", "chunk_ids": ["c1"]}
    assert [len(b) for b in batches(list(range(7)), 3)] == [3, 3, 1]


class FakeDriver:
    def __init__(self):
        self.queries = []

    def session(self):
        drv = self

        class S:
            def __enter__(self): return self
            def __exit__(self, *a): return False

            def run(self, q, **p):
                drv.queries.append((q, p))

                class R:
                    def __iter__(self): return iter([])
                    def consume(self):
                        class C: pass
                        c = C()
                        for k in ("nodes_created", "nodes_deleted", "relationships_created", "relationships_deleted",
                                  "labels_added", "labels_removed", "indexes_added", "constraints_added"):
                            setattr(c, k, 0)
                        return type("Sum", (), {"counters": c})()
                return R()
        return S()


def test_relations_are_merged_per_label_pair_on_version_and_key():
    d = FakeDriver()
    rows = [{"src_id": f"s{i}", "tgt_id": "t", "src_label": "AIModel", "tgt_label": lab, "key": f"s{i}|DEVELOPED_BY|t",
             "kg_version": "raw", "mention_count": 1, "confidence": 0.9} for i, lab in enumerate(["Organization"] * 3 + ["Person"])]
    Loader(d, batch_size=2).load_relations("DEVELOPED_BY", rows)
    qs = [q for q, _ in d.queries]
    assert len(qs) == 3                                  # Organization: 2 batches, Person: 1
    assert all("MERGE (s)-[e:DEVELOPED_BY {kg_version: r.props.kg_version, key: r.props.key}]->(t)" in q for q in qs)
    assert "MATCH (t:Person" in qs[2]
    with pytest.raises(AssertionError):
        Loader(d).load_relations("DROP", rows)
    assert RELATION_TYPES[0] == "DEVELOPED_BY"
