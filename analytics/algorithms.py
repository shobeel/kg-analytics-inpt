"""GDS algorithms on the entity graph of one kg_version (entities + the 12 relation types; no
chunks, no MENTIONS). Results are written back as node properties suffixed by the version
(pagerank_resolved, community_raw, ...). Projections are dropped after use (lite: Neo4j 2 GB).

  pagerank_<v>      PageRank on the directed graph (edges point model -> developer, person -> employer, ...)
  degree_<v>        number of relation edges (undirected)
  betweenness_<v>   exact betweenness centrality (undirected)
  community_<v>     Louvain community id (undirected, concurrency 1 for reproducible ids)
  component_<v>     weakly connected component id; in_lcc_<v> marks the largest one
  similar_<v>       ids of the top-3 most similar entities (Jaccard over neighbours, node similarity)
  similarity_<v>    their scores (aligned with similar_<v>)
"""

import time

from .db import check_version, run

PROPS = ("pagerank", "degree", "betweenness", "community", "component", "in_lcc", "similar", "similarity")

PROJECT = """
MATCH (s:Entity {kg_version: $kg_version})
OPTIONAL MATCH (s)-[r]->(t:Entity {kg_version: $kg_version})
WHERE r.kg_version = $kg_version AND type(r) <> 'MENTIONS'
WITH gds.graph.project($graph, s, t, {}, $config) AS g
RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships"""


def prop(name: str, kg_version: str) -> str:
    if name not in PROPS:
        raise ValueError(f"unknown algorithm property {name!r}")
    return f"{name}_{check_version(kg_version)}"


def _drop(driver, graph: str) -> None:
    run(driver, "CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", g=graph)


def run_all(driver, kg_version: str, similarity_top_k: int = 3) -> dict:
    """Project, run the five algorithms (+ WCC), write properties, drop projections."""
    v = check_version(kg_version)
    gd, gu = f"kg_{v}_directed", f"kg_{v}_undirected"
    out, t0 = {"kg_version": v}, time.monotonic()
    for g in (gd, gu):
        _drop(driver, g)
    try:
        out["projection_directed"] = run(driver, PROJECT, kg_version=v, graph=gd, config={})[0]
        out["projection_undirected"] = run(driver, PROJECT, kg_version=v, graph=gu,
                                           config={"undirectedRelationshipTypes": ["*"]})[0]

        out["pagerank"] = run(driver, """
            CALL gds.pageRank.write($g, {writeProperty: $p, maxIterations: 50, dampingFactor: 0.85})
            YIELD ranIterations, didConverge, nodePropertiesWritten
            RETURN ranIterations, didConverge, nodePropertiesWritten""", g=gd, p=prop("pagerank", v))[0]
        out["degree"] = run(driver, """
            CALL gds.degree.write($g, {writeProperty: $p}) YIELD nodePropertiesWritten, centralityDistribution
            RETURN nodePropertiesWritten, centralityDistribution.max AS max, centralityDistribution.mean AS mean""",
                            g=gu, p=prop("degree", v))[0]
        out["betweenness"] = run(driver, """
            CALL gds.betweenness.write($g, {writeProperty: $p}) YIELD nodePropertiesWritten, centralityDistribution
            RETURN nodePropertiesWritten, centralityDistribution.max AS max""", g=gu, p=prop("betweenness", v))[0]
        out["louvain"] = run(driver, """
            CALL gds.louvain.write($g, {writeProperty: $p, concurrency: 1})
            YIELD communityCount, modularity, ranLevels, nodePropertiesWritten
            RETURN communityCount, modularity, ranLevels, nodePropertiesWritten""", g=gu, p=prop("community", v))[0]
        out["wcc"] = run(driver, """
            CALL gds.wcc.write($g, {writeProperty: $p}) YIELD componentCount, componentDistribution
            RETURN componentCount, componentDistribution.max AS largest""", g=gu, p=prop("component", v))[0]

        # Largest connected component: flag its members (betweenness/paths are only meaningful there).
        comp, lcc = prop("component", v), prop("in_lcc", v)
        out["lcc"] = run(driver, f"""
            MATCH (e:Entity {{kg_version: $kg_version}})
            WITH e.{comp} AS c, count(*) AS n ORDER BY n DESC LIMIT 1
            MATCH (e:Entity {{kg_version: $kg_version}})
            SET e.{lcc} = (e.{comp} = c)
            RETURN c AS component, n AS size, count(e) AS entities""", kg_version=v)[0]
        out["lcc"]["pct"] = round(100 * out["lcc"]["size"] / out["lcc"]["entities"], 1)

        sims = run(driver, """
            CALL gds.nodeSimilarity.stream($g, {topK: $k, similarityCutoff: 0.1})
            YIELD node1, node2, similarity
            WITH gds.util.asNode(node1).id AS a, gds.util.asNode(node2).id AS b, similarity
            ORDER BY a, similarity DESC, b
            RETURN a, collect(b) AS similar, collect(round(similarity, 4)) AS scores""", g=gu, k=similarity_top_k)
        sp, ss = prop("similar", v), prop("similarity", v)
        run(driver, f"MATCH (e:Entity {{kg_version: $kg_version}}) REMOVE e.{sp}, e.{ss}", kg_version=v)
        for i in range(0, len(sims), 500):
            run(driver, f"""
                UNWIND $rows AS r MATCH (e:Entity {{id: r.a, kg_version: $kg_version}})
                SET e.{sp} = r.similar, e.{ss} = r.scores""", rows=sims[i:i + 500], kg_version=v)
        out["node_similarity"] = {"entities_with_similar": len(sims), "top_k": similarity_top_k}
    finally:
        for g in (gd, gu):
            _drop(driver, g)
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


def top(driver, name: str, kg_version: str, limit: int = 10) -> list[dict]:
    """Entities ranked by an algorithm property (pagerank, degree, betweenness)."""
    p = prop(name, kg_version)
    return run(driver, f"""
        MATCH (e:Entity {{kg_version: $kg_version}}) WHERE e.{p} IS NOT NULL
        RETURN e.id AS id, e.name AS name, [l IN labels(e) WHERE l <> 'Entity'][0] AS label,
               round(e.{p}, 5) AS {name}, e.{prop('degree', kg_version)} AS degree
        ORDER BY e.{p} DESC, name LIMIT $limit""", kg_version=kg_version, limit=limit)


def communities(driver, kg_version: str, limit: int = 15, members: int = 6) -> list[dict]:
    """Largest Louvain communities with their highest-PageRank members."""
    c, pr = prop("community", kg_version), prop("pagerank", kg_version)
    return run(driver, f"""
        MATCH (e:Entity {{kg_version: $kg_version}}) WHERE e.{c} IS NOT NULL
        WITH e.{c} AS community, e ORDER BY e.{pr} DESC
        WITH community, count(e) AS size, collect(e.name)[..$members] AS top_members
        RETURN community, size, top_members ORDER BY size DESC LIMIT $limit""",
               kg_version=kg_version, limit=limit, members=members)


def community_sizes(driver, kg_version: str) -> list[int]:
    c = prop("community", kg_version)
    return [r["n"] for r in run(driver, f"""
        MATCH (e:Entity {{kg_version: $kg_version}}) WHERE e.{c} IS NOT NULL
        RETURN e.{c} AS community, count(*) AS n ORDER BY n DESC""", kg_version=kg_version)]


def degrees(driver, kg_version: str) -> list[int]:
    d = prop("degree", kg_version)
    return [int(r["d"]) for r in run(driver, f"MATCH (e:Entity {{kg_version: $kg_version}}) RETURN e.{d} AS d",
                                     kg_version=kg_version) if r["d"] is not None]


def similar(driver, entity_id: str, kg_version: str) -> list[dict]:
    sp, ss = prop("similar", kg_version), prop("similarity", kg_version)
    return run(driver, f"""
        MATCH (e:Entity {{id: $id, kg_version: $kg_version}})
        UNWIND range(0, size(coalesce(e.{sp}, [])) - 1) AS i
        MATCH (o:Entity {{id: e.{sp}[i], kg_version: $kg_version}})
        RETURN o.name AS name, [l IN labels(o) WHERE l <> 'Entity'][0] AS label, e.{ss}[i] AS similarity
        ORDER BY similarity DESC""", id=entity_id, kg_version=kg_version)
