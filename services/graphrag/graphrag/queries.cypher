// GraphRAG retrieval queries (in addition to the analytics library). All take $kg_version.

// name: link_exact
// Entities whose name or alias equals one of the candidate strings (lower-cased).
MATCH (e:Entity {kg_version: $kg_version})
WHERE toLower(e.name) IN $candidates OR any(a IN e.aliases WHERE toLower(a) IN $candidates)
RETURN e.id AS id, e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label, e.aliases AS aliases,
       e.description AS description, e.mention_count AS mentions;

// name: mention_chunks
// Chunks mentioning the linked entities, most linked entities first.
MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
WHERE m.kg_version = $kg_version AND e.id IN $ids
WITH c, count(DISTINCT e) AS linked, sum(m.mention_count) AS mentions
RETURN c.id AS chunk_id, linked, mentions ORDER BY linked DESC, mentions DESC, chunk_id LIMIT $k;

// name: vector_chunks
CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding) YIELD node, score
RETURN node.id AS chunk_id, score;

// name: fulltext_chunks
CALL db.index.fulltext.queryNodes('chunk_text', $query) YIELD node, score
RETURN node.id AS chunk_id, score LIMIT $k;

// name: chunk_details
MATCH (c:Chunk)-[:PART_OF]->(d:Document) WHERE c.id IN $ids
RETURN c.id AS chunk_id, c.text AS text, d.id AS doc_id, d.title AS title, d.url AS url;

// name: documents
MATCH (d:Document) WHERE d.id IN $ids
RETURN d.id AS doc_id, d.title AS title, d.url AS url, d.source AS source;

// name: agg_model_developers
MATCH (m:AIModel {kg_version: $kg_version})-[r:DEVELOPED_BY]->(o:Organization)
WHERE r.kg_version = $kg_version
WITH o, count(DISTINCT m) AS n, collect(DISTINCT m.name)[..5] AS examples, collect(DISTINCT r.source_doc_id)[..3] AS doc_ids
RETURN o.name AS name, n, 'AI models developed' AS what, examples, doc_ids ORDER BY n DESC, name LIMIT $k;

// name: agg_technologies
MATCH (x:Entity {kg_version: $kg_version})-[r:USES]->(t:Technology)
WHERE r.kg_version = $kg_version
WITH t, count(DISTINCT x) AS n, collect(DISTINCT x.name)[..5] AS examples, collect(DISTINCT r.source_doc_id)[..3] AS doc_ids
RETURN t.name AS name, n, 'entities using it' AS what, examples, doc_ids ORDER BY n DESC, name LIMIT $k;

// name: agg_employers
MATCH (p:Person {kg_version: $kg_version})-[r:WORKS_AT]->(o:Organization)
WHERE r.kg_version = $kg_version
WITH o, count(DISTINCT p) AS n, collect(DISTINCT p.name)[..5] AS examples, collect(DISTINCT r.source_doc_id)[..3] AS doc_ids
RETURN o.name AS name, n, 'people working there' AS what, examples, doc_ids ORDER BY n DESC, name LIMIT $k;
