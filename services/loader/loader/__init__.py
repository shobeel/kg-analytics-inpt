"""Phase 3: gold Delta tables -> Neo4j (both kg_versions), embeddings, verification. No LLM calls."""

ENTITY_LABELS = ("Organization", "Person", "AIModel", "Paper", "Technology", "Dataset", "Product", "Location")
RELATION_TYPES = ("DEVELOPED_BY", "WORKS_AT", "AUTHORED", "CITES", "USES", "TRAINED_ON",
                  "COMPETES_WITH", "PARTNERS_WITH", "FUNDED_BY", "ACQUIRED", "LOCATED_IN", "BASED_ON")
VERSIONS = ("raw", "resolved")
