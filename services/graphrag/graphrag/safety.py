"""Text2Cypher safety filter. LLM-written Cypher runs only if it is a single read-only statement
that filters on kg_version; a LIMIT is forced. Execution is additionally a read transaction with
a 10 s timeout (defence in depth: the database refuses writes there too)."""

import re

MAX_LIMIT = 50
WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|USE|GRANT|DENY|REVOKE|"
                   r"ALTER|RENAME|START|STOP|TERMINATE|SHOW|IN\s+TRANSACTIONS)\b", re.I)
CALL = re.compile(r"\bCALL\b\s*([\w.]+|\{)?", re.I)
ALLOWED_CALLS = {"db.index.vector.querynodes", "db.index.fulltext.querynodes"}


class UnsafeCypher(ValueError):
    pass


def strip_literals(q: str) -> str:
    """Remove comments and string literals so names like 'Set' or 'Create ML' cannot trigger or
    hide keywords."""
    q = re.sub(r"/\*.*?\*/", " ", q, flags=re.S)
    q = re.sub(r"//[^\n]*", " ", q)
    return re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "''", q)


def check_cypher(query: str) -> str:
    """Return the query with a safe LIMIT, or raise UnsafeCypher with the reason."""
    q = query.strip()
    if q.startswith("```"):                                    # markdown code fence around the query
        q = re.sub(r"^```\w*\s*|\s*```$", "", q).strip()
    q = q.rstrip(";").strip()
    bare = strip_literals(q)
    if not q:
        raise UnsafeCypher("empty query")
    if ";" in bare:
        raise UnsafeCypher("only one statement is allowed")
    m = WRITE.search(bare)
    if m:
        raise UnsafeCypher(f"write or admin clause not allowed: {m.group(1).upper()}")
    for c in CALL.finditer(bare):
        target = (c.group(1) or "").lower()
        if target not in ALLOWED_CALLS:
            raise UnsafeCypher(f"CALL {c.group(1) or ''} not allowed (only vector/full-text index queries)")
    if "kg_version" not in bare:
        raise UnsafeCypher("the query must filter on kg_version")
    if not re.search(r"\bRETURN\b", bare, re.I):
        raise UnsafeCypher("the query must RETURN rows")
    lim = re.search(r"\bLIMIT\s+(\$?\w+)\s*$", q, re.I)            # LIMIT of the final RETURN
    if lim is None:
        return f"{q} LIMIT {MAX_LIMIT}"
    n = lim.group(1)
    if not n.isdigit() or int(n) > MAX_LIMIT:
        return q[:lim.start()] + f"LIMIT {MAX_LIMIT}"
    return q
