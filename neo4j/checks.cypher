// Phase 3 verification queries, run by `make neo4j-check` (loader/checks.py).
// Each query starts with a "// name: <id>" line. Parameters: $kg_version ('raw' | 'resolved'),
// $embedding (query vector), $k. Every entity/relation query filters on kg_version.
// You can paste any of them into the Neo4j Browser (set params first, e.g. :param kg_version => 'resolved').

// name: node_counts
MATCH (n:Entity)
RETURN [l IN labels(n) WHERE l <> 'Entity'][0] AS label, n.kg_version AS kg_version, count(*) AS n
UNION ALL
MATCH (d:Document) RETURN 'Document' AS label, 'shared' AS kg_version, count(*) AS n
UNION ALL
MATCH (c:Chunk) RETURN 'Chunk' AS label, 'shared' AS kg_version, count(*) AS n;

// name: edge_counts
MATCH ()-[r]->()
RETURN type(r) AS type, coalesce(r.kg_version, 'shared') AS kg_version, count(*) AS n;

// name: provenance
// The most-mentioned DEVELOPED_BY edge, traced back to its evidence, chunk and source document.
MATCH (s:Entity {kg_version: $kg_version})-[r:DEVELOPED_BY {kg_version: $kg_version}]->(t:Entity)
WITH s, r, t ORDER BY r.mention_count DESC, r.key LIMIT 1
MATCH (c:Chunk {id: r.chunk_id})-[:PART_OF]->(d:Document)
RETURN s.name AS source, type(r) AS relation, t.name AS target, r.mention_count AS mentions,
       r.confidence AS confidence, r.evidence_text AS evidence, r.extractor_model AS extractor_model,
       r.prompt_version AS prompt_version, c.id AS chunk_id, d.title AS document, d.url AS url;

// name: vector_search
CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding) YIELD node, score
MATCH (node)-[:PART_OF]->(d:Document)
RETURN round(score, 4) AS score, node.id AS chunk_id, d.source AS source, d.title AS document,
       left(replace(node.text, '\n', ' '), 140) AS text;

// name: relation_edges
MATCH (:Entity {kg_version: $kg_version})-[r]->(:Entity)
WHERE r.kg_version = $kg_version
RETURN count(r) AS n;

// name: orphans
// Entities with no relation edge (they are still reachable through MENTIONS from their chunks).
MATCH (e:Entity {kg_version: $kg_version})
WHERE NOT EXISTS { (e)-[r]-(:Entity) WHERE r.kg_version = $kg_version }
RETURN count(e) AS n;

// name: top_degree
MATCH (e:Entity {kg_version: $kg_version})-[r]-(:Entity)
WHERE r.kg_version = $kg_version
RETURN e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label, count(r) AS degree
ORDER BY degree DESC, name LIMIT 10;

// name: gds_project
// Entity-relation graph of one version (relations only, no MENTIONS); isolated entities included.
MATCH (s:Entity {kg_version: $kg_version})
OPTIONAL MATCH (s)-[r]->(t:Entity {kg_version: $kg_version}) WHERE r.kg_version = $kg_version
WITH gds.graph.project($graph, s, t) AS g
RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships;

// name: gds_wcc
CALL gds.wcc.stats($graph) YIELD componentCount, componentDistribution
RETURN componentCount AS components, componentDistribution.max AS largest,
       componentDistribution.p50 AS median_size;

// name: gds_drop
CALL gds.graph.drop($graph, false) YIELD graphName RETURN graphName;
