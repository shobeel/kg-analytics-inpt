"""Extraction and adjudication prompts. Bump PROMPT_VERSION (extraction/__init__.py) when
changing the extraction prompt: it is part of the cache key."""

from .schema import ENTITY_TYPES, PROMPT_DOMAIN_RANGE

_relations = "\n".join(
    f"- {r}: ({' | '.join(sorted(src))}) -> ({' | '.join(sorted(tgt))})" for r, (src, tgt) in PROMPT_DOMAIN_RANGE.items())

EXTRACTION_SYSTEM = f"""You extract a knowledge graph about the AI ecosystem from one text chunk.

Entity types (use exactly these): {", ".join(ENTITY_TYPES)}.
- Organization: companies, labs, universities, agencies. Person: named people.
- AIModel: named trained models (GPT-4, BERT, Llama 2). Technology: methods, architectures,
  techniques (transformer, RLHF, retrieval-augmented generation). Paper: named research papers.
- Dataset: named datasets/benchmarks. Product: named products/services (ChatGPT, Hugging Face Hub).
- Location: cities, countries, regions.

Relation types, direction source -> target, with allowed (source types) -> (target types):
{_relations}

Rules:
1. Only extract what the text states. No outside knowledge, no guesses.
2. Entities must be specific and named. Skip generic phrases ("the model", "researchers", "our method").
3. Use the most complete name that appears in the text (e.g. "Google DeepMind", not "it").
4. Every relation's source and target must exactly match the name of an entity in your entities list.
5. evidence_text must be copied verbatim from the chunk: the shortest sentence or clause that states
   the relation. Never paraphrase it.
6. confidence in [0,1]: 0.9+ explicitly stated, 0.6-0.9 clearly implied by the text, below 0.6 do not output.
7. description: one short phrase from the text about the entity (may be empty).
8. If nothing relevant is present, return empty lists.
Return only JSON matching the schema."""


ABSTRACT_HINT = ("This chunk is the title and abstract of a research paper. Include the paper itself "
                 "as a Paper entity named with its exact title (the first line), and relate it to "
                 "what the abstract states (e.g. the paper USES a technology or dataset).")


def extraction_user(chunk_text: str, title: str, source: str = "") -> str:
    hint = f"\n{ABSTRACT_HINT}\n" if source == "arxiv" else ""
    return f"Document title: {title or '(none)'}\n{hint}\nChunk:\n\"\"\"\n{chunk_text}\n\"\"\""


ADJUDICATION_SYSTEM = """You decide whether pairs of entity mentions from a knowledge graph about the
AI ecosystem refer to the SAME real-world entity. Different versions or sizes of a model
(GPT-3 vs GPT-4, Llama 2 vs Llama 3) are DIFFERENT entities. A product and the company that makes
it are DIFFERENT. An acronym and its expansion are the SAME when they clearly denote the same thing.
Answer for every pair. Return only JSON: {"decisions": [{"pair": <id>, "same": true|false}]}."""

ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {"decisions": {"type": "array", "items": {
        "type": "object",
        "properties": {"pair": {"type": "integer"}, "same": {"type": "boolean"}},
        "required": ["pair", "same"]}}},
    "required": ["decisions"],
}
