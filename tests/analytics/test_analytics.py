import pytest

from analytics import algorithms as A
from analytics import quality as QL
from analytics import queries as Q
from analytics.db import RELATION_TYPES, check_version, load_queries, named_queries


def test_every_library_query_filters_on_kg_version():
    qs = load_queries("queries.cypher")
    assert len(qs) >= 15
    for name, text in qs.items():
        if name != "ensure_fulltext":
            assert "$kg_version" in text, name
    with pytest.raises(ValueError):
        named_queries("// name: x\nRETURN 1;\nRETURN 2;")


def test_relation_placeholders_are_whitelisted():
    t = Q.query_text("neighbours")
    assert "{REL_TYPES}" not in t and "DEVELOPED_BY|WORKS_AT" in t and "MENTIONS" not in t
    assert "[r:USES {kg_version" in Q.query_text("edge_provenance", "USES")
    with pytest.raises(ValueError):
        Q.query_text("edge_provenance", "MENTIONS]->() DETACH DELETE (")
    with pytest.raises(ValueError):
        check_version("both")
    with pytest.raises(ValueError):
        A.prop("pagerank; DROP", "raw")
    assert A.prop("pagerank", "resolved") == "pagerank_resolved"
    assert len(RELATION_TYPES) == 12


def test_lucene_query_escapes_and_fuzzes():
    assert Q.lucene_query("deepmnd") == "deepmnd~"
    assert Q.lucene_query("GPT-4 (OpenAI)") == "GPT\\-4~ AND \\(OpenAI\\)~"
    assert Q.lucene_query("AI lab") == "AI AND lab"


def test_norm_key_matches_phase2_rules():
    assert QL.norm_key("The Hugging Face, Inc.") == "hugging face"
    assert QL.norm_key("xAI (company)") == "xai"
    assert QL.norm_key("Nvidia Corporation") == "nvidia"
    assert QL.norm_key("GPT-4") == "gpt 4"


def test_duplicate_stats_same_type_and_cross_type():
    ents = [{"id": "1", "label": "Organization", "name": "Nvidia", "mentions": 5},
            {"id": "2", "label": "Organization", "name": "Nvidia Corporation", "mentions": 2},
            {"id": "3", "label": "AIModel", "name": "Gemini", "mentions": 4},
            {"id": "4", "label": "Product", "name": "Gemini", "mentions": 3},
            {"id": "5", "label": "Person", "name": "Ada", "mentions": 1}]
    s = QL.duplicate_stats(ents)
    assert s["duplicate_entities"] == 1 and s["duplicate_rate"] == 0.2
    assert s["cross_type_groups"] == 1 and s["cross_type_entities"] == 2
    assert s["cross_type"][0] == {"normalized_name": "gemini", "labels": ["AIModel", "Product"], "entities": 2,
                                  "names": ["Gemini"], "mentions": 7}


def test_graph_stats():
    g = QL.graph_stats(4, 2, [1, 2, 1, 0], {"count": 2, "largest": 3}, [3, 1])
    assert g["avg_degree"] == 1.0 and g["orphan_pct"] == 25.0 and g["largest_component_pct"] == 75.0
    assert g["density"] == 2 * 2 / (4 * 3) and g["louvain_communities_size_ge_2"] == 1


def test_pagerank_change_maps_resolved_to_best_raw_member():
    raw = [{"id": "r1", "name": "DeepMind", "pagerank": 5.0, "resolved_id": "e1"},
           {"id": "r2", "name": "Google DeepMind", "pagerank": 3.0, "resolved_id": "e1"},
           {"id": "r3", "name": "OpenAI", "pagerank": 9.0, "resolved_id": "e2"}]
    res = [{"id": "e1", "name": "DeepMind", "label": "Organization", "pagerank": 8.4},
           {"id": "e2", "name": "OpenAI", "label": "Organization", "pagerank": 9.1}]
    rows = QL.pagerank_change(raw, res, k=2)
    assert [r["name"] for r in rows] == ["OpenAI", "DeepMind"]
    assert rows[1]["raw_members"] == 2 and rows[1]["best_raw_member"] == "DeepMind" and rows[1]["rank_raw"] == 2


def test_quality_table_and_save(tmp_path):
    base = {"entities": 10, "relation_edges": 5, "duplicate_entities": 2, "duplicate_rate": 0.2,
            "cross_type_groups": 1, "cross_type_entities": 2, "connected_components": 4,
            "largest_component_pct": 50.0, "orphan_pct": 30.0, "density": 0.1, "avg_degree": 1.0,
            "louvain_communities": 4, "louvain_communities_size_ge_2": 2,
            "cross_type": [{"normalized_name": "gemini", "labels": ["AIModel", "Product"], "names": ["Gemini"],
                            "entities": 2, "mentions": 7}]}
    r = {"raw": base, "resolved": {**base, "entities": 8, "duplicate_entities": 0}, "pagerank_top10": [],
         "pagerank_top10_overlap": 0}
    rows = {x["key"]: x for x in QL.table_rows(r)}
    assert rows["entities"]["change_fmt"] == "-20.0%" and rows["duplicate_entities"]["resolved"] == 0
    csv_path, md_path = QL.save(r, str(tmp_path))
    assert "cross_type_groups" in csv_path.read_text() and "Cross-type duplicate names" in md_path.read_text()
