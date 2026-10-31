"""Ontology (graph schema of the project) and the pydantic schema every LLM extraction must validate against."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ENTITY_TYPES = ("Organization", "Person", "AIModel", "Paper", "Technology", "Dataset", "Product", "Location")
RELATION_TYPES = ("DEVELOPED_BY", "WORKS_AT", "AUTHORED", "CITES", "USES", "TRAINED_ON",
                  "COMPETES_WITH", "PARTNERS_WITH", "FUNDED_BY", "ACQUIRED", "LOCATED_IN", "BASED_ON")
SYMMETRIC = {"COMPETES_WITH", "PARTNERS_WITH"}

# Allowed (source types, target types) as written in the extract-v2 prompt. Frozen: changing it
# changes the prompt text, so all chunks of one PROMPT_VERSION must see the same table.
PROMPT_DOMAIN_RANGE = {
    "DEVELOPED_BY": ({"AIModel", "Product", "Technology", "Dataset"}, {"Organization", "Person"}),
    "WORKS_AT": ({"Person"}, {"Organization"}),
    "AUTHORED": ({"Person"}, {"Paper"}),
    "CITES": ({"Paper"}, {"Paper"}),
    "USES": ({"AIModel", "Product", "Organization", "Paper", "Technology"}, {"Technology", "Dataset", "AIModel"}),
    "TRAINED_ON": ({"AIModel"}, {"Dataset"}),
    "COMPETES_WITH": ({"Organization", "Product", "AIModel"}, {"Organization", "Product", "AIModel"}),
    "PARTNERS_WITH": ({"Organization"}, {"Organization"}),
    "FUNDED_BY": ({"Organization", "Person", "Paper"}, {"Organization", "Person"}),
    "ACQUIRED": ({"Organization"}, {"Organization", "Product"}),
    "LOCATED_IN": ({"Organization", "Person"}, {"Location"}),
    "BASED_ON": ({"AIModel", "Product", "Technology"}, {"AIModel", "Technology", "Paper"}),
}

# Post-hoc validation rules = the prompt table widened by the tweaks justified in
# eval/results/relation_stats_450_chunks.md (applied 2026-10-28; no new LLM calls needed).
SCHEMA_TWEAKS = {
    "USES": (set(), {"Product"}),              # Common Crawl USES Amazon S3, Ollama USES Hugging Face Hub
    "LOCATED_IN": ({"Location"}, set()),       # Memphis LOCATED_IN Tennessee
    "FUNDED_BY": ({"AIModel"}, set()),         # DeepZenGo, developed with support from Dwango
}
DOMAIN_RANGE = {rel: (src | SCHEMA_TWEAKS.get(rel, (set(), set()))[0],
                      tgt | SCHEMA_TWEAKS.get(rel, (set(), set()))[1])
                for rel, (src, tgt) in PROMPT_DOMAIN_RANGE.items()}
# Single triples allowed without widening a whole domain.
ALLOWED_TRIPLES = frozenset({("Organization", "DEVELOPED_BY", "Person")})   # "founded by"


def type_valid(source_type, relation: str, target_type) -> bool:
    src_ok, tgt_ok = DOMAIN_RANGE[relation]
    return (source_type in src_ok and target_type in tgt_ok) or \
        (source_type, relation, target_type) in ALLOWED_TRIPLES


EntityType = Literal[ENTITY_TYPES]
RelationType = Literal[RELATION_TYPES]


class Entity(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    type: EntityType
    description: str = Field(default="", max_length=400)

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        return " ".join(v.split())


class Relation(BaseModel):
    source: str = Field(min_length=1)
    relation: RelationType
    target: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# JSON schema sent as response_format (OpenAI-compatible structured output). Written by hand
# with only the keywords Gemini supports (no $ref / additionalProperties).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                "description": {"type": "string"},
            },
            "required": ["name", "type", "description"],
        }},
        "relations": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "relation": {"type": "string", "enum": list(RELATION_TYPES)},
                "target": {"type": "string"},
                "evidence_text": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["source", "relation", "target", "evidence_text", "confidence"],
        }},
    },
    "required": ["entities", "relations"],
}
