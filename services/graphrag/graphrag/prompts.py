"""Prompts and response schemas. Bump PROMPT_VERSION (graphrag/__init__.py) on any change."""

from . import NOT_FOUND

ANSWER_SYSTEM = f"""You answer questions about the AI ecosystem using ONLY the context provided.
Rules:
- Use only facts stated in the context. Do not use outside knowledge, even if you know the answer.
- Cite every claim with the source ids from the context in square brackets, e.g. [3f2a9c1b].
  Only use ids that appear in the context.
- If the context does not contain the information needed, set found=false and answer exactly:
  "{NOT_FOUND}"
- Be concise: 1-4 sentences, or a short list."""

ANSWER_SCHEMA = {"type": "object", "properties": {
    "answer": {"type": "string"}, "found": {"type": "boolean"},
    "cited_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["answer", "found", "cited_ids"]}


def answer_user(question: str, context: str) -> str:
    return f"CONTEXT\n{context}\n\nQUESTION\n{question}"


SCHEMA_DESCRIPTION = """Neo4j graph about the AI ecosystem. Two versions of the entity graph live side by side:
EVERY entity node and EVERY relation has a property kg_version; always filter both on $kg_version.

Entity nodes: (:Organization|:Person|:AIModel|:Paper|:Technology|:Dataset|:Product|:Location), every
one also labelled :Entity. Properties: id, name, aliases (list), description, mention_count, kg_version.
Relations between entities (direction as written), properties: kg_version, mention_count, confidence,
evidence_text, source_doc_id, source_doc_ids (list), chunk_ids (list):
  (AIModel|Product|Technology|Dataset)-[:DEVELOPED_BY]->(Organization|Person)
  (Organization)-[:DEVELOPED_BY]->(Person)      // "founded by"
  (Person)-[:WORKS_AT]->(Organization)          // includes founders, CEOs, researchers
  (Person)-[:AUTHORED]->(Paper)   (Paper)-[:CITES]->(Paper)
  (AIModel|Product|Organization|Paper|Technology)-[:USES]->(Technology|Dataset|AIModel|Product)
  (AIModel)-[:TRAINED_ON]->(Dataset)
  (Organization|Product|AIModel)-[:COMPETES_WITH]-(...)   (Organization)-[:PARTNERS_WITH]-(Organization)
  (Organization|Person|Paper|AIModel)-[:FUNDED_BY]->(Organization|Person)
  (Organization)-[:ACQUIRED]->(Organization|Product)
  (Organization|Person|Location)-[:LOCATED_IN]->(Location)
  (AIModel|Product|Technology)-[:BASED_ON]->(AIModel|Technology|Paper)
Text: (:Chunk {id, text})-[:PART_OF]->(:Document {id, title, url, source});
      (:Chunk)-[:MENTIONS {kg_version}]->(:Entity).
Match names case-insensitively: toLower(e.name) = toLower('X') OR toLower('X') IN [a IN e.aliases | toLower(a)]."""

FEW_SHOT = """Example 1. Question: Where is Mistral AI located?
MATCH (o:Organization {kg_version: $kg_version})-[r:LOCATED_IN]->(l:Location)
WHERE r.kg_version = $kg_version AND (toLower(o.name) = 'mistral ai' OR 'mistral ai' IN [a IN o.aliases | toLower(a)])
RETURN o.name AS organization, l.name AS location, r.evidence_text AS evidence, r.source_doc_id AS doc_id LIMIT 10

Example 2. Question: Which technologies are used by models developed by OpenAI?
MATCH (o:Organization {kg_version: $kg_version})<-[d:DEVELOPED_BY]-(m:AIModel)-[u:USES]->(t:Technology)
WHERE d.kg_version = $kg_version AND u.kg_version = $kg_version AND toLower(o.name) = 'openai'
RETURN m.name AS model, t.name AS technology, u.evidence_text AS evidence, u.source_doc_id AS doc_id LIMIT 25

Example 3. Question: Which people have the most affiliations?
MATCH (p:Person {kg_version: $kg_version})-[r:WORKS_AT]->(o:Organization)
WHERE r.kg_version = $kg_version
RETURN p.name AS person, count(DISTINCT o) AS affiliations, collect(DISTINCT o.name)[..5] AS organizations,
       collect(DISTINCT r.source_doc_id)[..3] AS doc_ids
ORDER BY affiliations DESC LIMIT 10"""

CYPHER_SYSTEM = f"""You translate questions into ONE read-only Cypher query for Neo4j 5.
{SCHEMA_DESCRIPTION}

Rules:
- Read-only: MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT only. No CALL, no writes.
- Filter every entity node and every relation on kg_version = $kg_version (a query parameter).
- When the answer comes from relations, also return r.evidence_text AS evidence and r.source_doc_id AS doc_id
  (or collect(DISTINCT r.source_doc_id)[..3] AS doc_ids when aggregating).
- End with LIMIT (at most 50).

{FEW_SHOT}"""

CYPHER_SCHEMA = {"type": "object", "properties": {"cypher": {"type": "string"}}, "required": ["cypher"]}


def cypher_user(question: str, error: str = "") -> str:
    fix = f"\n\nYour previous query failed: {error}\nReturn a corrected query." if error else ""
    return f"Question: {question}{fix}"


VERBALIZE_SYSTEM = f"""You turn database rows into a short answer to the question.
- Use only the rows. Cite sources with the ids in square brackets, e.g. [3f2a9c1b], taking them from the
  doc_id / doc_ids columns. Only use ids that appear in the rows.
- If the rows do not answer the question, set found=false and answer exactly: "{NOT_FOUND}"
- Be concise: 1-4 sentences, or a short list."""


def verbalize_user(question: str, cypher: str, rows_json: str) -> str:
    return f"QUESTION\n{question}\n\nCYPHER\n{cypher}\n\nROWS (JSON)\n{rows_json}"
