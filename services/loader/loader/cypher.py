"""Read the .cypher files in neo4j/: plain statement lists and named query blocks."""

import re


def _strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))


def statements(text: str) -> list[str]:
    """schema.cypher -> statements (split on ';' at the end of a line, comments removed)."""
    parts = re.split(r";\s*$", _strip_comments(text), flags=re.M)
    return [p.strip() for p in parts if p.strip()]


def named_queries(text: str) -> dict[str, str]:
    """checks.cypher -> {name: query}. Each block starts with '// name: <id>'."""
    out = {}
    for block in re.split(r"^//\s*name:\s*", text, flags=re.M)[1:]:
        name, _, body = block.partition("\n")
        stmts = statements(body)
        if len(stmts) != 1:
            raise ValueError(f"query {name.strip()!r} must hold exactly one statement")
        out[name.strip()] = stmts[0]
    return out
