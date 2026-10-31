"""Graph analytics on the Neo4j KG (Phase 4). Pure Cypher + GDS, no LLM calls.

Every function takes kg_version ('raw' | 'resolved'). Typical use (Phase 5 GraphRAG, the API):

    from analytics import get_driver, queries, algorithms, quality
    d = get_driver()                                   # NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
    e = queries.find(d, "Google DeepMind", "resolved") # exact name/alias, then full-text
    queries.neighbours(d, e["id"], "resolved")
    queries.shortest_path(d, a_id, b_id, "resolved")
    queries.edge_provenance(d, src_id, "DEVELOPED_BY", tgt_id, "resolved")   # evidence + URLs
    algorithms.run_all(d, "resolved")                  # PageRank, degree, betweenness, Louvain, WCC, similarity
    algorithms.top(d, "pagerank", "resolved")
    quality.compare(d)                                 # RQ2 raw vs resolved
"""

from . import algorithms, quality, queries
from .db import ENTITY_LABELS, KG_VERSIONS, RELATION_TYPES, get_driver

__all__ = ["algorithms", "quality", "queries", "get_driver", "KG_VERSIONS", "ENTITY_LABELS", "RELATION_TYPES"]
