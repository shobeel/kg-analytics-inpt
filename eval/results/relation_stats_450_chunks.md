# Relation statistics: first 450 of 642 chunks

Extractor `gemini-3.5-flash-lite`, prompt `extract-v2`, snapshot of 2026-10-28 (quota stop, 192 chunks left). Reproduce with `make relation-stats`, which reads only the cache, makes no LLM calls and writes `logs/relation_stats.json`.

## Relations per type (1,878 total)

| Relation | Count | Share |
|---|---|---|
| DEVELOPED_BY | 528 | 28.1% |
| WORKS_AT | 367 | 19.5% |
| USES | 292 | 15.5% |
| COMPETES_WITH | 126 | 6.7% |
| BASED_ON | 121 | 6.4% |
| FUNDED_BY | 104 | 5.5% |
| PARTNERS_WITH | 84 | 4.5% |
| LOCATED_IN | 77 | 4.1% |
| AUTHORED | 76 | 4.0% |
| ACQUIRED | 69 | 3.7% |
| TRAINED_ON | 34 | 1.8% |
| CITES | 0 | 0% |

`CITES` never appears because reference lists are now cut from PDFs and arXiv chunks are abstracts only.

## Top 10 schema violations under the original prompt table (205 relations, 10.9%)

| # | Source type | Relation | Target type | Count | What it is |
|---|---|---|---|---|---|
| 1 | Organization | DEVELOPED_BY | AIModel | 23 | **Reversed**: "Baidu released ERNIE 4.5" |
| 2 | Organization | DEVELOPED_BY | Person | 22 | Founding: "Anthropic was founded by … Dario Amodei" |
| 3 | Organization | DEVELOPED_BY | Organization | 14 | Wrong: self-loops ("Baidu → Baidu") and ownership ("DeepMind, a subsidiary of Alphabet") |
| 4 | Organization | DEVELOPED_BY | Product | 12 | **Reversed**: "OpenAI's release of ChatGPT" |
| 5 | *(unknown)* | WORKS_AT | Organization | 10 | Source person missing from the entity list ("Jensen Huang (CEO of Nvidia)") |
| 6 | Organization | USES | Product | 8 | **Valid**: "Common Crawl … storage on Amazon S3" |
| 7 | AIModel | USES | Product | 7 | **Valid**: "DALL-E 3 … used ChatGPT to write prompts" |
| 8 | AIModel | FUNDED_BY | Organization | 7 | **Valid**: "DeepZenGo, developed with support from Dwango" |
| 9 | Product | USES | Product | 6 | **Valid**: "Ollama … models hosted on the Hugging Face Hub" |
| 10 | Person | DEVELOPED_BY | Technology | 6 | **Reversed**: "Mohri … proposed SA-DPO" |

Also seen, but left as violations:
- Organization USES Organization (4): "Google Brain used Nvidia GPUs". The real target is a product the text doesn't name.
- Person AUTHORED AIModel (4): award citations ("for AlphaGo").
- AIModel TRAINED_ON Technology (3): hardware (TPUs) or a training method, not a dataset.

## Applied rule changes (2026-10-28)

All changes are post-hoc. Flags are recomputed from the cache with `make rebuild-silver`, with no LLM calls. The extract-v2 prompt table is frozen (`schema.PROMPT_DOMAIN_RANGE`), so the remaining 192 chunks see exactly the same prompt as the first 450.

| Change | Relations affected | Why |
|---|---|---|
| USES target += Product | 30 | Tools and platforms (GitHub, Amazon S3, Hugging Face Hub) are Products in our ontology. |
| Allow the single triple Organization DEVELOPED_BY Person | 22 | Means "founded by". Allowed only with a Person target, so ownership (Org → Org) stays invalid. |
| FUNDED_BY source += AIModel | 7 | Research systems funded or supported by organizations. |
| LOCATED_IN source += Location | 3 | Place hierarchy (Memphis → Tennessee). |
| **Direction repair**: swap the endpoints when only the reversed triple is valid | 56 | "Baidu released ERNIE 4.5" means ERNIE 4.5 DEVELOPED_BY Baidu. Marked `direction_repaired=true` in silver and on KG edges. |
| **`self_loop` flag**: source = target | 16 | "Baidu DEVELOPED_BY Baidu". Excluded from the KG like the other flags. |

## Classes before and after

| Class | Original prompt table | Applied rules |
|---|---|---|
| valid | 1,673 (89.1%) | 1,733 (92.3%) |
| valid after direction repair | n/a | 56 (3.0%) |
| invalid | 205, including unknown endpoints and 14 self-loops | 52 (2.8%) |
| unknown endpoint | 21 (1.1%) | 21 (1.1%) |
| self-loop | not checked | 16 (0.9%) |

**Flag rates in silver after the rebuild:**

| Flag | Before | After |
|---|---|---|
| schema_invalid | 10.9% | 4.6% |
| evidence_unverified | 3.0% | 3.0% |
| unknown_endpoint | 1.1% | 1.1% |
| self_loop | n/a | 0.9% |
| **any flag (excluded from the KG)** | **13.8%** | **7.7% (145)** |

Remaining violations, kept as violations:
- Organization DEVELOPED_BY Organization (14, including self-loops and "subsidiary of").
- Organization USES Organization (4).
- Person AUTHORED AIModel (4).
- Person USES AIModel (4).
- AIModel TRAINED_ON Technology (3).

## Entities with zero relations

2,748 of 5,176 entity mentions (53.1%) are not an endpoint of any relation in their chunk.

| Type | Zero-relation rate |
|---|---|
| Paper | 29.4% |
| Dataset | 45.0% |
| Organization | 47.5% |
| AIModel | 49.6% |
| Technology | 51.7% |
| Location | 57.6% |
| Person | 62.7% |
| Product | 63.8% |

Isolated mentions still become `MENTIONS` edges from their chunk in Phase 3, so they remain reachable for GraphRAG. They add nothing to the relation graph that GDS runs on.
