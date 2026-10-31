// Phase 4 Cypher library. Each query starts with "// name: <id>"; analytics/queries.py runs them.
// Every query takes $kg_version ('raw' | 'resolved') and filters entities AND relations on it.
// {REL_TYPES} is replaced by the 12 entity relation types (MENTIONS / PART_OF are never traversed
// as entity-entity edges); {REL} by one validated relation type.

// ------------------------------------------------------------------ schema
// name: ensure_fulltext
CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.aliases];

// ------------------------------------------------------------------ lookups
// name: entity_by_name
// Exact (case-insensitive) match on the name or any alias.
MATCH (e:Entity {kg_version: $kg_version})
WHERE toLower(e.name) = toLower($name) OR any(a IN e.aliases WHERE toLower(a) = toLower($name))
RETURN e.id AS id, e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label,
       e.aliases AS aliases, e.mention_count AS mentions, e.description AS description
ORDER BY e.mention_count DESC LIMIT $limit;

// name: entity_search
// Full-text (Lucene) search over names and aliases: fuzzy, partial, multi-word.
CALL db.index.fulltext.queryNodes('entity_names', $query) YIELD node, score
WHERE node.kg_version = $kg_version
RETURN node.id AS id, node.name AS name, [l IN labels(node) WHERE l <> 'Entity'][0] AS label,
       node.aliases AS aliases, node.mention_count AS mentions, round(score, 3) AS score
ORDER BY score DESC, mentions DESC LIMIT $limit;

// ------------------------------------------------------------------ neighbourhood and paths
// name: neighbours
MATCH (e:Entity {id: $id, kg_version: $kg_version})-[r:{REL_TYPES}]-(n:Entity)
WHERE r.kg_version = $kg_version
RETURN type(r) AS relation, CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END AS direction,
       n.id AS id, n.name AS name, [l IN labels(n) WHERE l <> 'Entity'][0] AS label,
       r.mention_count AS mentions, r.confidence AS confidence
ORDER BY mentions DESC, name LIMIT $limit;

// name: paths_2hop
MATCH p = (a:Entity {id: $source, kg_version: $kg_version})-[r1:{REL_TYPES}]-(m:Entity)-[r2:{REL_TYPES}]-(b:Entity {id: $target, kg_version: $kg_version})
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version AND m <> a AND m <> b
RETURN [n IN nodes(p) | n.name] AS nodes, [r IN relationships(p) | type(r)] AS relations,
       reduce(s = 0, r IN relationships(p) | s + r.mention_count) AS support
ORDER BY support DESC LIMIT $limit;

// name: paths_3hop
MATCH p = (a:Entity {id: $source, kg_version: $kg_version})-[r1:{REL_TYPES}]-(m1:Entity)-[r2:{REL_TYPES}]-(m2:Entity)-[r3:{REL_TYPES}]-(b:Entity {id: $target, kg_version: $kg_version})
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version AND r3.kg_version = $kg_version
  AND m1 <> b AND m2 <> a AND m1 <> m2
RETURN [n IN nodes(p) | n.name] AS nodes, [r IN relationships(p) | type(r)] AS relations,
       reduce(s = 0, r IN relationships(p) | s + r.mention_count) AS support
ORDER BY support DESC LIMIT $limit;

// name: shortest_path
MATCH (a:Entity {id: $source, kg_version: $kg_version}), (b:Entity {id: $target, kg_version: $kg_version})
MATCH p = SHORTEST 1 (a)(()-[r:{REL_TYPES} WHERE r.kg_version = $kg_version]-()){1,6}(b)
RETURN length(p) AS hops, [n IN nodes(p) | n.name] AS nodes, [r IN relationships(p) | type(r)] AS relations;

// ------------------------------------------------------------------ aggregations
// name: top_model_developers
MATCH (m:AIModel {kg_version: $kg_version})-[r:DEVELOPED_BY]->(o:Organization)
WHERE r.kg_version = $kg_version
RETURN o.name AS organization, count(DISTINCT m) AS models, collect(DISTINCT m.name)[..6] AS examples
ORDER BY models DESC, organization LIMIT $limit;

// name: top_technologies
MATCH (x:Entity {kg_version: $kg_version})-[r:USES]->(t:Technology)
WHERE r.kg_version = $kg_version
RETURN t.name AS technology, count(DISTINCT x) AS users, sum(r.mention_count) AS mentions,
       collect(DISTINCT x.name)[..5] AS examples
ORDER BY users DESC, mentions DESC, technology LIMIT $limit;

// name: top_papers
// "Cited" = incoming CITES or BASED_ON edges (CITES alone is near-empty: reference lists are cut),
// then the number of source documents mentioning the paper.
MATCH (p:Paper {kg_version: $kg_version})
OPTIONAL MATCH (x)-[r:CITES|BASED_ON]->(p) WHERE r.kg_version = $kg_version
WITH p, count(DISTINCT x) AS cited_by
OPTIONAL MATCH (d:Document)<-[:PART_OF]-(:Chunk)-[m:MENTIONS]->(p) WHERE m.kg_version = $kg_version
WITH p, cited_by, count(DISTINCT d) AS documents
RETURN p.name AS paper, cited_by, documents
ORDER BY cited_by DESC, documents DESC, paper LIMIT $limit;

// name: top_affiliations
MATCH (p:Person {kg_version: $kg_version})-[r:WORKS_AT]->(o:Organization)
WHERE r.kg_version = $kg_version
RETURN p.name AS person, count(DISTINCT o) AS affiliations, collect(DISTINCT o.name)[..6] AS organizations
ORDER BY affiliations DESC, person LIMIT $limit;

// name: top_by_documents
MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity {kg_version: $kg_version})
WHERE m.kg_version = $kg_version
MATCH (c)-[:PART_OF]->(d:Document)
RETURN e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label, count(DISTINCT d) AS documents,
       count(DISTINCT d.source) AS source_types
ORDER BY documents DESC, name LIMIT $limit;

// ------------------------------------------------------------------ co-occurrence
// name: cooccurrence
MATCH (c:Chunk)-[m1:MENTIONS]->(a:Entity), (c)-[m2:MENTIONS]->(b:Entity)
WHERE m1.kg_version = $kg_version AND m2.kg_version = $kg_version AND a.id < b.id
WITH a, b, count(DISTINCT c) AS chunks WHERE chunks >= $min_chunks
RETURN a.name AS entity_a, [l IN labels(a) WHERE l <> 'Entity'][0] AS label_a,
       b.name AS entity_b, [l IN labels(b) WHERE l <> 'Entity'][0] AS label_b, chunks
ORDER BY chunks DESC, entity_a, entity_b LIMIT $limit;

// name: cooccurring_with
MATCH (e:Entity {id: $id, kg_version: $kg_version})<-[m1:MENTIONS]-(c:Chunk)-[m2:MENTIONS]->(o:Entity)
WHERE m1.kg_version = $kg_version AND m2.kg_version = $kg_version AND o <> e
RETURN o.name AS name, [l IN labels(o) WHERE l <> 'Entity'][0] AS label, count(DISTINCT c) AS chunks,
       EXISTS { (e)-[r:{REL_TYPES}]-(o) WHERE r.kg_version = $kg_version } AS related
ORDER BY chunks DESC, name LIMIT $limit;

// ------------------------------------------------------------------ provenance
// name: edge_provenance
// Every mention of one edge: evidence, confidence, chunk and source document.
MATCH (a:Entity {id: $source, kg_version: $kg_version})-[r:{REL} {kg_version: $kg_version}]-(b:Entity {id: $target, kg_version: $kg_version})
UNWIND range(0, size(r.chunk_ids) - 1) AS i
MATCH (c:Chunk {id: r.chunk_ids[i]})-[:PART_OF]->(d:Document)
RETURN startNode(r).name AS source, type(r) AS relation, endNode(r).name AS target,
       r.evidence_texts[i] AS evidence, r.confidences[i] AS confidence, r.extractor_model AS extractor_model,
       r.direction_repaired AS direction_repaired, c.id AS chunk_id, d.title AS document, d.url AS url
ORDER BY confidence DESC, chunk_id;

// name: sample_edge
// Any edge of a type (highest support first): handy input for edge_provenance.
MATCH (a:Entity {kg_version: $kg_version})-[r:{REL}]->(b:Entity)
WHERE r.kg_version = $kg_version
RETURN a.id AS source, a.name AS source_name, type(r) AS relation, b.id AS target, b.name AS target_name,
       r.mention_count AS mentions
ORDER BY mentions DESC, r.key LIMIT $limit;

// name: entity_triples
// 1-hop relation edges of several entities with their provenance (GraphRAG context). Edges of a
// preferred type (from the question) rank first, then by support; at most $per_entity per entity.
UNWIND $ids AS eid
MATCH (e:Entity {id: eid, kg_version: $kg_version})-[r:{REL_TYPES}]-(n:Entity)
WHERE r.kg_version = $kg_version
WITH e, r, n ORDER BY (type(r) IN $prefer) DESC, r.mention_count DESC, r.confidence DESC, n.name
WITH e, collect(r)[..$per_entity] AS rels
UNWIND rels AS r
WITH DISTINCT r
RETURN startNode(r).id AS source_id, startNode(r).name AS source,
       [l IN labels(startNode(r)) WHERE l <> 'Entity'][0] AS source_label, type(r) AS relation,
       endNode(r).id AS target_id, endNode(r).name AS target,
       [l IN labels(endNode(r)) WHERE l <> 'Entity'][0] AS target_label,
       r.evidence_text AS evidence, r.source_doc_id AS doc_id, r.source_doc_ids AS doc_ids,
       r.mention_count AS mentions, r.confidence AS confidence, r.direction_repaired AS direction_repaired;
