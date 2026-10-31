"""Neo4j connection and the named-query file format shared by the analytics module."""

import os
import re
from functools import lru_cache
from pathlib import Path

KG_VERSIONS = ("raw", "resolved")
ENTITY_LABELS = ("Organization", "Person", "AIModel", "Paper", "Technology", "Dataset", "Product", "Location")
RELATION_TYPES = ("DEVELOPED_BY", "WORKS_AT", "AUTHORED", "CITES", "USES", "TRAINED_ON",
                  "COMPETES_WITH", "PARTNERS_WITH", "FUNDED_BY", "ACQUIRED", "LOCATED_IN", "BASED_ON")


def check_version(kg_version: str) -> str:
    if kg_version not in KG_VERSIONS:
        raise ValueError(f"kg_version must be one of {KG_VERSIONS}, got {kg_version!r}")
    return kg_version


def get_driver(uri: str = None, user: str = None, password: str = None):
    from neo4j import GraphDatabase
    return GraphDatabase.driver(uri or os.environ.get("NEO4J_URI", "bolt://neo4j:7687"),
                                auth=(user or os.environ.get("NEO4J_USER", "neo4j"),
                                      password or os.environ["NEO4J_PASSWORD"]))


def named_queries(text: str) -> dict[str, str]:
    """'// name: <id>' blocks -> {id: query}; comment lines dropped, one statement per block."""
    out = {}
    for block in re.split(r"^//\s*name:\s*", text, flags=re.M)[1:]:
        name, _, body = block.partition("\n")
        body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("//"))
        stmts = [s.strip() for s in re.split(r";\s*$", body, flags=re.M) if s.strip()]
        if len(stmts) != 1:
            raise ValueError(f"query {name.strip()!r} must hold exactly one statement")
        out[name.strip()] = stmts[0]
    return out


@lru_cache(maxsize=None)
def load_queries(filename: str) -> dict[str, str]:
    return named_queries((Path(__file__).parent / filename).read_text())


def run(driver, cypher: str, **params) -> list[dict]:
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, parameters=params)]
