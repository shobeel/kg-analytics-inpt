"""Parameterized Cypher library (analytics/queries.cypher). Every function takes kg_version and
returns plain rows (list of dicts). Entities are addressed by id; resolve names with find()."""

import re
import time

from .db import RELATION_TYPES, check_version, load_queries, run

REL_TYPES = "|".join(RELATION_TYPES)
LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def query_text(query_name: str, rel: str = None) -> str:
    q = load_queries("queries.cypher")[query_name].replace("{REL_TYPES}", REL_TYPES)
    if "{REL}" in q:
        if rel not in RELATION_TYPES:
            raise ValueError(f"relation must be one of {RELATION_TYPES}, got {rel!r}")
        q = q.replace("{REL}", rel)
    return q


def q(driver, query_name: str, kg_version: str, rel: str = None, **params) -> list[dict]:
    return run(driver, query_text(query_name, rel), kg_version=check_version(kg_version), **params)


def timed(driver, query_name: str, kg_version: str, rel: str = None, **params) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    rows = q(driver, query_name, kg_version, rel, **params)
    return rows, time.perf_counter() - t0


def ensure_schema(driver) -> None:
    run(driver, load_queries("queries.cypher")["ensure_fulltext"])
    run(driver, "CALL db.awaitIndexes(300)")


def lucene_query(text: str) -> str:
    """Escape Lucene syntax and make every term fuzzy-tolerant ('deepmnd' still finds DeepMind)."""
    terms = [LUCENE_SPECIAL.sub(r"\\\1", t) for t in text.split()]
    return " AND ".join(f"{t}~" if len(t) > 3 else t for t in terms)


# ------------------------------------------------------------------ lookups
def entity_by_name(driver, name: str, kg_version: str, limit: int = 5) -> list[dict]:
    return q(driver, "entity_by_name", kg_version, name=name, limit=limit)


def entity_search(driver, text: str, kg_version: str, limit: int = 10) -> list[dict]:
    return q(driver, "entity_search", kg_version, query=lucene_query(text), limit=limit)


def find(driver, name: str, kg_version: str):
    """Best entity for a name: exact name/alias match first, then full-text. None if nothing."""
    rows = entity_by_name(driver, name, kg_version, 1) or entity_search(driver, name, kg_version, 1)
    return rows[0] if rows else None


# ------------------------------------------------------------------ neighbourhood and paths
def neighbours(driver, entity_id: str, kg_version: str, limit: int = 50) -> list[dict]:
    return q(driver, "neighbours", kg_version, id=entity_id, limit=limit)


def paths(driver, source_id: str, target_id: str, kg_version: str, hops: int = 2, limit: int = 25) -> list[dict]:
    if hops not in (2, 3):
        raise ValueError("hops must be 2 or 3 (use shortest_path for other lengths)")
    return q(driver, f"paths_{hops}hop", kg_version, source=source_id, target=target_id, limit=limit)


def shortest_path(driver, source_id: str, target_id: str, kg_version: str) -> list[dict]:
    return q(driver, "shortest_path", kg_version, source=source_id, target=target_id)


# ------------------------------------------------------------------ aggregations
def top_model_developers(driver, kg_version: str, limit: int = 10) -> list[dict]:
    return q(driver, "top_model_developers", kg_version, limit=limit)


def top_technologies(driver, kg_version: str, limit: int = 10) -> list[dict]:
    return q(driver, "top_technologies", kg_version, limit=limit)


def top_papers(driver, kg_version: str, limit: int = 10) -> list[dict]:
    return q(driver, "top_papers", kg_version, limit=limit)


def top_affiliations(driver, kg_version: str, limit: int = 10) -> list[dict]:
    return q(driver, "top_affiliations", kg_version, limit=limit)


def top_by_documents(driver, kg_version: str, limit: int = 10) -> list[dict]:
    return q(driver, "top_by_documents", kg_version, limit=limit)


# ------------------------------------------------------------------ co-occurrence and provenance
def cooccurrence(driver, kg_version: str, min_chunks: int = 2, limit: int = 20) -> list[dict]:
    return q(driver, "cooccurrence", kg_version, min_chunks=min_chunks, limit=limit)


def cooccurring_with(driver, entity_id: str, kg_version: str, limit: int = 20) -> list[dict]:
    return q(driver, "cooccurring_with", kg_version, id=entity_id, limit=limit)


def edge_provenance(driver, source_id: str, relation: str, target_id: str, kg_version: str) -> list[dict]:
    return q(driver, "edge_provenance", kg_version, rel=relation, source=source_id, target=target_id)


def entity_triples(driver, entity_ids: list[str], kg_version: str, prefer: list[str] = (),
                   per_entity: int = 30) -> list[dict]:
    """Relation edges around entities, with evidence and source doc ids (GraphRAG context)."""
    return q(driver, "entity_triples", kg_version, ids=list(entity_ids), prefer=list(prefer), per_entity=per_entity)


def sample_edges(driver, relation: str, kg_version: str, limit: int = 5) -> list[dict]:
    return q(driver, "sample_edge", kg_version, rel=relation, limit=limit)
