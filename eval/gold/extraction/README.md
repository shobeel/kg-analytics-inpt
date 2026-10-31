# RQ1 gold set: how to annotate

`annotation_batch.jsonl` holds 25 silver chunks, stratified by source with a fixed seed. Create it with `make gold-sample`, which never overwrites an existing file. Each line is one chunk:

```json
{"chunk_id": "...", "doc_id": "...", "source": "wikipedia", "title": "...", "url": "...",
 "text": "... the chunk text ...",
 "status": "todo", "annotator": "", "entities": [], "relations": [], "notes": ""}
```

## Workflow: correct a pre-filled draft

The 25 chunks were pre-filled by models that are **never the extractor being scored** (`EXTRACTION_MODEL` = `gemini-3.5-flash-lite`; `make gold-prefill` refuses to run if they match). You correct the drafts instead of annotating from scratch.

| Pre-fill model | Provider | Chunks | Why |
|---|---|---|---|
| `openai/gpt-oss-120b` (`PREFILL_MODEL`) | Groq free tier | 21 | Main pre-fill model. Groq lists no Llama 70B model any more; this is the closest open-weight model of that class, and it is unrelated to Gemini. |
| `gemini-3.5-flash` (`ANSWER_MODEL`) | Gemini free tier | 3 (01, 02, 04) | First attempt, stopped by its 20 requests/day limit. These drafts are kept as they are. |
| `qwen/qwen3.8-27b` (`PREFILL_FALLBACK_MODEL`) | Groq free tier | 1 (07) | `gpt-oss-120b` failed Groq's JSON validation on every retry for this infobox-heavy chunk. |

The drafts are sparse: 215 entities and only 62 relations over 25 chunks. Expect to **add** many items, not just delete wrong ones; chunk 07 (Anthropic infobox) in particular has no relations. Each chunk's file header and `prefill.jsonl` name the model that drafted it.

| # | Source | Title | Pre-fill model | Entities | Relations |
|---|---|---|---|---|---|
| 01 | arxiv | Behavioral and Representational Evidence of Bino | `gemini-3.5-flash` | 3 | 2 |
| 02 | local | Constitutional AI 2212.08073 | `gemini-3.5-flash` | 3 | 0 |
| 03 | local | Constitutional AI 2212.08073 | `openai/gpt-oss-120b` | 13 | 2 |
| 04 | wikipedia | AlphaGo | `gemini-3.5-flash` | 16 | 4 |
| 05 | arxiv | The Philosopher's Stone: Trojaning Plugins of La | `openai/gpt-oss-120b` | 4 | 3 |
| 06 | rss | Legora is building the agentic operating system  | `openai/gpt-oss-120b` | 1 | 0 |
| 07 | wikipedia | Anthropic | `qwen/qwen3.8-27b` | 5 | 0 |
| 08 | wikipedia | Gemini (language model) | `openai/gpt-oss-120b` | 27 | 4 |
| 09 | arxiv | Scaling Laws for Neural Language Models | `openai/gpt-oss-120b` | 4 | 3 |
| 10 | arxiv | Exploring the Limits of Transfer Learning with a | `openai/gpt-oss-120b` | 3 | 2 |
| 11 | rss | Build more responsive apps with the Decisions AP | `openai/gpt-oss-120b` | 1 | 0 |
| 12 | rss | Cohere says Alibaba’s Qwen model tried to send a | `openai/gpt-oss-120b` | 9 | 2 |
| 13 | rss | How OpenAI turned its coding tools into dots, an | `openai/gpt-oss-120b` | 11 | 4 |
| 14 | rss | Giving AI a “crystal clear” view of the material | `openai/gpt-oss-120b` | 7 | 0 |
| 15 | wikipedia | Reinforcement learning from human feedback | `openai/gpt-oss-120b` | 3 | 2 |
| 16 | rss | Greptile takes bug detection beyond code review  | `openai/gpt-oss-120b` | 10 | 9 |
| 17 | arxiv | AdaShield: Safeguarding Multimodal Large Languag | `openai/gpt-oss-120b` | 3 | 2 |
| 18 | wikipedia | OpenAI | `openai/gpt-oss-120b` | 4 | 0 |
| 19 | wikipedia | OpenAI | `openai/gpt-oss-120b` | 11 | 1 |
| 20 | wikipedia | Large language model | `openai/gpt-oss-120b` | 25 | 4 |
| 21 | wikipedia | Large language model | `openai/gpt-oss-120b` | 10 | 2 |
| 22 | wikipedia | Large language model | `openai/gpt-oss-120b` | 17 | 7 |
| 23 | rss | Patch the Planet: global response coordination a | `openai/gpt-oss-120b` | 7 | 2 |
| 24 | rss | Welcome FLUX.3 Image Klein in Diffusers | `openai/gpt-oss-120b` | 7 | 6 |
| 25 | rss | Rapidly scaling online storage to serve over 1 b | `openai/gpt-oss-120b` | 11 | 1 |

```bash
make gold-prefill    # PREFILL_MODEL (Groq) drafts -> prefill.jsonl (cached; re-run fills only missing chunks)
make gold-todo       # one file per pre-filled chunk in todo/ (existing files are never overwritten;
                     # chunks still missing a draft are added after the next gold-prefill)
# ... edit todo/NN_<chunk_id>.txt, set "status: done" ...
make gold-convert    # todo/ -> gold.jsonl (scorer input) + changes.jsonl (what you changed vs the draft)
```

Each `todo/` file has the chunk TEXT, then one line per item:

```text
=== ENTITIES ===
Organization | OpenAI
AIModel | GPT-4
=== RELATIONS ===
#   evidence: "OpenAI developed GPT-4"
GPT-4 | DEVELOPED_BY | OpenAI
=== END ===
```

- Delete wrong lines, fix names or types, and add missing items. Lines starting with `#` are ignored.
- Every relation endpoint must be listed under ENTITIES.
- `make gold-convert` refuses to write anything if a file has an unknown type, a malformed line or a dangling endpoint, and prints `file:line` for each problem.
- `changes.jsonl` records, per chunk, the entities kept, removed, added and retyped, and the relations kept, removed and added. Its totals over the `done` chunks are printed too, so the size of the correction effort is documented.

**Anchoring bias:** correcting a draft is faster but can pull the gold toward model output. Mitigations: the draft comes from a different model than the one being scored, every relation shows its evidence quote, and `changes.jsonl` makes the effect measurable. Mention this in the report (RQ1 threats to validity).

## What to annotate in each chunk

The rules below apply whether you correct a draft or start from scratch. The steps describe the JSONL fields; in the `todo/` files, the same content goes in `Type | Name` and `Source | RELATION | Target` lines.

1. Read `text` only. Annotate what the text states, not what you know.
2. Fill `entities` with **every specific, named** entity, using exactly one of these types:
   `Organization, Person, AIModel, Paper, Technology, Dataset, Product, Location`
   ```json
   {"name": "Google DeepMind", "type": "Organization"}
   ```
   - Use the fullest name that appears in the text ("Google DeepMind", not "DeepMind", if both appear, list it once).
   - Skip generic mentions: "the model", "researchers", "our approach".
   - AIModel means a named trained model (GPT-4, BERT). Technology means a method or architecture (transformer, RLHF). Product means a named product or service (ChatGPT). Dataset includes benchmarks.
3. Fill `relations` with relations the text **explicitly** states, using the entity names from step 2:
   ```json
   {"source": "AlphaFold", "relation": "DEVELOPED_BY", "target": "Google DeepMind"}
   ```

   | Relation | Direction |
   |---|---|
   | DEVELOPED_BY | (AIModel, Product, Technology, Dataset) → (Organization, Person); also Organization → Person for "founded by" |
   | WORKS_AT | Person → Organization |
   | AUTHORED | Person → Paper |
   | CITES | Paper → Paper |
   | USES | (AIModel, Product, Organization, Paper, Technology) → (Technology, Dataset, AIModel, Product) |
   | TRAINED_ON | AIModel → Dataset |
   | COMPETES_WITH / PARTNERS_WITH | symmetric (either order) |
   | FUNDED_BY | (Organization, Person, Paper, AIModel) → (Organization, Person) |
   | ACQUIRED | Organization → (Organization, Product) |
   | LOCATED_IN | (Organization, Person, Location) → Location |
   | BASED_ON | (AIModel, Product, Technology) → (AIModel, Technology, Paper) |
4. Set `"status": "done"` and put your initials in `annotator`. Use `notes` for anything ambiguous. **Only `done` chunks are scored**, so you can annotate in several sittings.

Keep one entity or relation per line.

## Scoring

```bash
docker compose ... run --rm extraction python -m extraction export-predictions \
  --gold /app/eval/gold/extraction/gold.jsonl --out /app/eval/gold/extraction/predictions.jsonl
make score gold=eval/gold/extraction/gold.jsonl pred=eval/gold/extraction/predictions.jsonl
```

The scorer ([eval/extraction_scorer.py](../../extraction_scorer.py)) reports micro precision, recall and F1:

| | Entity | Relation |
|---|---|---|
| **Strict** | same name (case-insensitive) and same type | same relation and exact endpoint names |
| **Relaxed** | normalized name (Inc/Ltd/"the"/punctuation dropped), or token containment ("Altman" ~ "Sam Altman"), or ≥ 0.85 character similarity; type ignored | same relation and relaxed endpoints |

Symmetric relations match in either direction in both modes. `toy/` holds a 3-chunk example with hand-computed expected scores, used by the unit tests.
