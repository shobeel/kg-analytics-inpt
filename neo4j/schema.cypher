// Phase 3 schema: run before loading (idempotent: IF NOT EXISTS). Applied by `make neo4j-load`.
// Entities exist twice, once per kg_version ('raw' | 'resolved'), so uniqueness is on (id, kg_version).
// Every entity also carries the :Entity label (for the shared vector index and generic queries).

CREATE CONSTRAINT organization_id_version IF NOT EXISTS FOR (n:Organization) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX organization_name IF NOT EXISTS FOR (n:Organization) ON (n.name);
CREATE CONSTRAINT person_id_version IF NOT EXISTS FOR (n:Person) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX person_name IF NOT EXISTS FOR (n:Person) ON (n.name);
CREATE CONSTRAINT aimodel_id_version IF NOT EXISTS FOR (n:AIModel) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX aimodel_name IF NOT EXISTS FOR (n:AIModel) ON (n.name);
CREATE CONSTRAINT paper_id_version IF NOT EXISTS FOR (n:Paper) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX paper_name IF NOT EXISTS FOR (n:Paper) ON (n.name);
CREATE CONSTRAINT technology_id_version IF NOT EXISTS FOR (n:Technology) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX technology_name IF NOT EXISTS FOR (n:Technology) ON (n.name);
CREATE CONSTRAINT dataset_id_version IF NOT EXISTS FOR (n:Dataset) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX dataset_name IF NOT EXISTS FOR (n:Dataset) ON (n.name);
CREATE CONSTRAINT product_id_version IF NOT EXISTS FOR (n:Product) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX product_name IF NOT EXISTS FOR (n:Product) ON (n.name);
CREATE CONSTRAINT location_id_version IF NOT EXISTS FOR (n:Location) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX location_name IF NOT EXISTS FOR (n:Location) ON (n.name);
CREATE CONSTRAINT entity_id_version IF NOT EXISTS FOR (n:Entity) REQUIRE (n.id, n.kg_version) IS UNIQUE;
CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name);
CREATE INDEX entity_kg_version IF NOT EXISTS FOR (n:Entity) ON (n.kg_version);
// Full-text index on names and aliases (Phase 4 lookups: fuzzy / partial / alias search).
CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.aliases];
// Full-text index on chunk text (Phase 5 hybrid retrieval: keyword side of vector + full-text).
CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

// Relationship indexes: every relation query filters on kg_version; key drives the stale-edge cleanup.
CREATE INDEX rel_developed_by_version IF NOT EXISTS FOR ()-[r:DEVELOPED_BY]-() ON (r.kg_version, r.key);
CREATE INDEX rel_works_at_version IF NOT EXISTS FOR ()-[r:WORKS_AT]-() ON (r.kg_version, r.key);
CREATE INDEX rel_authored_version IF NOT EXISTS FOR ()-[r:AUTHORED]-() ON (r.kg_version, r.key);
CREATE INDEX rel_cites_version IF NOT EXISTS FOR ()-[r:CITES]-() ON (r.kg_version, r.key);
CREATE INDEX rel_uses_version IF NOT EXISTS FOR ()-[r:USES]-() ON (r.kg_version, r.key);
CREATE INDEX rel_trained_on_version IF NOT EXISTS FOR ()-[r:TRAINED_ON]-() ON (r.kg_version, r.key);
CREATE INDEX rel_competes_with_version IF NOT EXISTS FOR ()-[r:COMPETES_WITH]-() ON (r.kg_version, r.key);
CREATE INDEX rel_partners_with_version IF NOT EXISTS FOR ()-[r:PARTNERS_WITH]-() ON (r.kg_version, r.key);
CREATE INDEX rel_funded_by_version IF NOT EXISTS FOR ()-[r:FUNDED_BY]-() ON (r.kg_version, r.key);
CREATE INDEX rel_acquired_version IF NOT EXISTS FOR ()-[r:ACQUIRED]-() ON (r.kg_version, r.key);
CREATE INDEX rel_located_in_version IF NOT EXISTS FOR ()-[r:LOCATED_IN]-() ON (r.kg_version, r.key);
CREATE INDEX rel_based_on_version IF NOT EXISTS FOR ()-[r:BASED_ON]-() ON (r.kg_version, r.key);
CREATE INDEX rel_mentions_version IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.kg_version, r.key);

// Vector indexes (fastembed sentence-transformers/all-MiniLM-L6-v2, 384-d, cosine).
// entity_embedding holds both kg_versions: filter the results on kg_version.
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX entity_embedding IF NOT EXISTS FOR (e:Entity) ON e.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
