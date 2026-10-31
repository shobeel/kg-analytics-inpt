"""GraphRAG assistant (Phase 5): three strategies behind one interface.

    from graphrag import answer
    answer("Who developed Claude?", strategy="B", kg_version="resolved")

  A  vector RAG        top-k chunks by embedding similarity -> LLM answer
  B  GraphRAG hybrid   entity linking (no LLM) -> k-hop triples with evidence + MENTIONS chunks +
                       vector/full-text chunks (+ graph aggregates) -> structured context -> LLM answer
  C  Text2Cypher       LLM writes read-only Cypher (safety-filtered) -> rows -> LLM verbalizes
"""

PROMPT_VERSION = "graphrag-v1"
STRATEGIES = {"A": "vector", "B": "graphrag", "C": "text2cypher"}
NOT_FOUND = "Not found in the knowledge graph."


def answer(question: str, strategy: str = "B", kg_version: str = "resolved") -> dict:
    from .engine import default_engine
    return default_engine().answer(question, strategy, kg_version)
