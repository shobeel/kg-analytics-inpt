"""Entity linking without an LLM: candidate n-grams from the question -> exact name/alias match
(filtered by kg_version), then a fuzzy full-text fallback for unmatched salient words."""

import re
from difflib import SequenceMatcher

STOP = {"who", "whom", "whose", "what", "which", "where", "when", "why", "how", "is", "are", "was", "were",
        "do", "does", "did", "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "and", "or",
        "that", "this", "these", "those", "it", "its", "be", "been", "has", "have", "had", "most", "many",
        "much", "known", "about", "from", "into", "than", "between", "work", "works", "worked", "developed",
        "develop", "people", "person", "organization", "organizations", "company", "companies", "model",
        "models", "ai", "ceo", "name", "names", "list", "tell", "me", "there", "any", "all", "some",
        "use", "uses", "used", "built", "created", "made"}
TOKEN = re.compile(r"[A-Za-z0-9][\w.\-+&']*[\w+]|[A-Za-z0-9]")
MAX_NGRAM = 5


def tokens(question: str) -> list[str]:
    return TOKEN.findall(question)


def salient(tok: str, first: bool) -> bool:
    """A token that can start a name: capitalized (not just the sentence's first word), or with a digit."""
    if tok.lower() in STOP:
        return False
    return any(ch.isdigit() for ch in tok) or (tok[0].isupper() and not first) or \
        (first and sum(ch.isupper() for ch in tok) > 1)


def candidates(question: str) -> list[str]:
    """N-grams (longest first) that contain a salient token and do not start or end with a stopword."""
    toks = tokens(question)
    out = []
    for n in range(min(MAX_NGRAM, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            gram = toks[i:i + n]
            if gram[0].lower() in STOP or gram[-1].lower() in STOP:
                continue
            if any(salient(t, i + j == 0) for j, t in enumerate(gram)):
                out.append(" ".join(gram))
    return list(dict.fromkeys(out))


def keywords(question: str) -> list[str]:
    """Content words for full-text chunk search."""
    return [t for t in tokens(question) if t.lower() not in STOP and len(t) > 2]


def link(driver, run, question: str, kg_version: str, search=None, max_per_name: int = 3) -> list[dict]:
    """Longest-match-first exact linking; overlapping shorter matches are dropped. `run` executes a
    named graphrag query; `search` is the fuzzy full-text fallback (analytics.queries.entity_search)."""
    cands = candidates(question)
    if not cands:
        return []
    rows = run("link_exact", kg_version, candidates=[c.lower() for c in cands])
    by_key = {}
    for r in rows:
        for k in [r["name"].lower()] + [a.lower() for a in (r["aliases"] or [])]:
            by_key.setdefault(k, []).append(r)
    linked, used = [], []
    for c in cands:                                          # longest n-grams first
        span = set(c.lower().split())
        if any(span <= u for u in used):
            continue
        hits = sorted({r["id"]: r for r in by_key.get(c.lower(), [])}.values(),
                      key=lambda r: -(r["mentions"] or 0))[:max_per_name]
        if not hits and search and " " not in c and len(c) >= 5:
            fuzzy = search(c)
            hits = [h for h in fuzzy[:1] if SequenceMatcher(None, h["name"].lower(), c.lower()).ratio() >= 0.8]
        if hits:
            used.append(span)
            linked += [{**h, "matched": c} for h in hits]
    return list({e["id"]: e for e in linked}.values())
