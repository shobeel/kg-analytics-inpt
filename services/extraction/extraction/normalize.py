"""Name normalization and pair features for entity resolution (pure Python, unit-tested)."""

import re
import unicodedata

LEGAL_SUFFIXES = {"inc", "incorporated", "ltd", "limited", "llc", "corp", "corporation", "co",
                  "company", "gmbh", "plc", "sa", "ag", "lp", "llp", "pbc"}


def norm_key(name: str) -> str:
    """'The Hugging Face, Inc.' -> 'hugging face'; 'xAI (company)' -> 'xai'.
    Case, punctuation, a leading 'the', parenthetical disambiguators and legal suffixes removed."""
    s = unicodedata.normalize("NFKC", name).casefold()
    s = re.sub(r"\([^)]*\)", " ", s)                      # Wikipedia-style "(company)"
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)                         # punctuation -> space (GPT-4 -> gpt 4)
    tokens = s.split()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def version_signature(name: str) -> frozenset:
    """Tokens carrying digits ('gpt-4o' -> {'4o'}, 'Llama 2 70B' -> {'2', '70b'}). Two names
    with different signatures are different entities (GPT-3 vs GPT-4, Llama 2 vs Llama 3)."""
    s = unicodedata.normalize("NFKC", name).casefold()
    return frozenset(re.findall(r"\d+(?:\.\d+)*[a-z]*", s))


def acronym(name_key: str) -> str:
    """Initials of a multi-word normalized name ('large language model' -> 'llm')."""
    tokens = [t for t in name_key.split() if t not in {"of", "and", "for", "the", "in", "on"}]
    return "".join(t[0] for t in tokens) if len(tokens) >= 2 else ""


def is_acronym_pair(a_key: str, b_key: str) -> bool:
    """'llm' / 'llms' vs 'large language model(s)'."""
    for short, long in ((a_key, b_key), (b_key, a_key)):
        if " " not in short and len(short) >= 2:
            acr = acronym(long)
            if acr and short.rstrip("s") == acr.rstrip("s"):
                return True
    return False
