# Phase 6 question review

> Review status: **model-reviewed (an LLM), not human-reviewed** (2026-10-31). Every answer was checked against the gold query and the relation evidence text in the resolved graph.

Tick ONE box per question: `[x] OK` (correct as is), `[x] FIX` (you edited the lines below it) or
`[x] DROP` (remove it). Edit `question:`, `expected_answer:` and `expected_entities:` (separated by
` ; `) in place, and the Cypher block if the gold query is wrong. Unticked questions stay unverified
and do not count. Then run `make questions-apply`.

Check each answer against the source documents, not only against the graph: the gold query shows
what the graph says; if the graph is wrong (an extraction error), fix the expected answer and say so
in `notes:`.

Multi-hop anchors were checked to be in the largest connected component of the resolved graph.

## L01 · lookup

[x] OK   [ ] FIX   [ ] DROP

question: Who developed Claude?
expected_answer: Anthropic
expected_entities: Anthropic
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 1 rows; first 1:
    {"entity": "Anthropic"}

```cypher
MATCH (m:Entity {kg_version: $kg_version, name: 'Claude'})-[r:DEVELOPED_BY]->(o:Organization)
WHERE r.kg_version = $kg_version
RETURN DISTINCT o.name AS entity
```

## L02 · lookup

[x] OK   [ ] FIX   [ ] DROP

question: Where is Cohere headquartered?
expected_answer: Toronto
expected_entities: Toronto
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 1 rows; first 1:
    {"entity": "Toronto"}

```cypher
MATCH (o:Organization {kg_version: $kg_version, name: 'Cohere'})-[r:LOCATED_IN]->(l:Location)
WHERE r.kg_version = $kg_version
RETURN DISTINCT l.name AS entity
```

## L03 · lookup

[x] OK   [ ] FIX   [ ] DROP

question: Who founded Hugging Face?
expected_answer: Hugging Face was founded by Clément Delangue, Thomas Wolf, Julien Chaumond.
expected_entities: Clément Delangue ; Thomas Wolf ; Julien Chaumond
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 3 rows; first 3:
    {"entity": "Clément Delangue"}
    {"entity": "Thomas Wolf"}
    {"entity": "Julien Chaumond"}

```cypher
MATCH (o:Organization {kg_version: $kg_version, name: 'Hugging Face'})-[r:DEVELOPED_BY]-(p:Person)
WHERE r.kg_version = $kg_version
RETURN DISTINCT p.name AS entity
```

## L04 · lookup

[ ] OK   [x] FIX   [ ] DROP

question: Who created the ImageNet dataset?
expected_answer: Fei-Fei Li
expected_entities: Fei-Fei Li
notes: model-reviewed: dropped stray 'Li' (unresolved duplicate of Fei-Fei Li); gold query still returns both

Gold query result on resolved: 2 rows; first 2:
    {"entity": "Li"}
    {"entity": "Fei-Fei Li"}

```cypher
MATCH (d:Dataset {kg_version: $kg_version, name: 'ImageNet'})-[r:DEVELOPED_BY]->(p)
WHERE r.kg_version = $kg_version
RETURN DISTINCT p.name AS entity
```

## L05 · lookup

[ ] OK   [x] FIX   [ ] DROP

question: Which company developed GPT-4?
expected_answer: OpenAI
expected_entities: OpenAI
notes: model-reviewed: Microsoft edge is an extraction error (evidence pairs Microsoft TRELLIS with GPT-4)

Gold query result on resolved: 2 rows; first 2:
    {"entity": "OpenAI"}
    {"entity": "Microsoft"}

```cypher
MATCH (m:AIModel {kg_version: $kg_version, name: 'GPT-4'})-[r:DEVELOPED_BY]->(o:Organization)
WHERE r.kg_version = $kg_version
RETURN DISTINCT o.name AS entity
```

## L06 · lookup

[x] OK   [ ] FIX   [ ] DROP

question: Which companies has Mistral AI acquired?
expected_answer: Mistral AI acquired Emmi AI, Koyeb.
expected_entities: Emmi AI ; Koyeb
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 2 rows; first 2:
    {"entity": "Emmi AI"}
    {"entity": "Koyeb"}

```cypher
MATCH (o:Organization {kg_version: $kg_version, name: 'Mistral AI'})-[r:ACQUIRED]->(x:Organization)
WHERE r.kg_version = $kg_version
RETURN DISTINCT x.name AS entity
```

## M01 · multi-hop · 2 hops

[ ] OK   [x] FIX   [ ] DROP

question: Which people work at the organization that developed AlphaFold?
expected_answer: AlphaFold was developed by DeepMind; people working there include Demis Hassabis, David Silver, Aja Huang, Diego Rivas, Koray Kavukcuoglu, Lila Ibrahim, Murray Shanahan, Naina Raisinghani.
expected_entities: Demis Hassabis ; David Silver ; Aja Huang ; Diego Rivas ; Koray Kavukcuoglu ; Lila Ibrahim ; Murray Shanahan ; Naina Raisinghani
notes: model-reviewed: removed Jaan Tallinn (investor), Arthur Mensch and Igor Babuschkin (former staff), Fan Hui (adviser) - WORKS_AT extraction errors

Gold query result on resolved: 12 rows; first 8:
    {"via": "DeepMind", "entity": "Demis Hassabis"}
    {"via": "DeepMind", "entity": "David Silver"}
    {"via": "DeepMind", "entity": "Aja Huang"}
    {"via": "DeepMind", "entity": "Arthur Mensch"}
    {"via": "DeepMind", "entity": "Diego Rivas"}
    {"via": "DeepMind", "entity": "Fan Hui"}
    {"via": "DeepMind", "entity": "Igor Babuschkin"}
    {"via": "DeepMind", "entity": "Jaan Tallinn"}

```cypher
MATCH (m:Entity {kg_version: $kg_version, name: 'AlphaFold'})-[r1:DEVELOPED_BY]->(o:Organization)<-[r2:WORKS_AT]-(p:Person)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version
RETURN o.name AS via, p.name AS entity ORDER BY r2.mention_count DESC, entity
```

## M02 · multi-hop · 2 hops

[ ] OK   [x] FIX   [ ] DROP

question: In which city is the company that developed Mixtral 8x7B headquartered?
expected_answer: Mistral AI is headquartered in Paris, France.
expected_entities: Paris ; France
notes: model-reviewed: city is Paris; France kept as acceptable context

Gold query result on resolved: 2 rows; first 2:
    {"via": "Mistral AI", "entity": "France"}
    {"via": "Mistral AI", "entity": "Paris"}

```cypher
MATCH (m:AIModel {kg_version: $kg_version, name: 'Mixtral 8x7B'})-[r1:DEVELOPED_BY]->(o:Organization)-[r2:LOCATED_IN]->(l:Location)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version
RETURN o.name AS via, l.name AS entity
```

## M03 · multi-hop · 2 hops

[x] OK   [ ] FIX   [ ] DROP

question: Which companies were acquired by the developer of Claude?
expected_answer: Claude's developer Anthropic acquired Stainless, Bun.
expected_entities: Stainless ; Bun
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 2 rows; first 2:
    {"via": "Anthropic", "entity": "Stainless"}
    {"via": "Anthropic", "entity": "Bun"}

```cypher
MATCH (m:Entity {kg_version: $kg_version, name: 'Claude'})-[r1:DEVELOPED_BY]->(o:Organization)-[r2:ACQUIRED]->(x:Organization)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version
RETURN DISTINCT o.name AS via, x.name AS entity
```

## M04 · multi-hop · 2 hops

[x] OK   [ ] FIX   [ ] DROP

question: Who founded the company that acquired Mellanox?
expected_answer: Mellanox was acquired by Nvidia, founded by Chris Malachowsky, Jensen Huang, Curtis Priem.
expected_entities: Chris Malachowsky ; Jensen Huang ; Curtis Priem
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 3 rows; first 3:
    {"via": "Nvidia", "entity": "Chris Malachowsky"}
    {"via": "Nvidia", "entity": "Jensen Huang"}
    {"via": "Nvidia", "entity": "Curtis Priem"}

```cypher
MATCH (x:Organization {kg_version: $kg_version, name: 'Mellanox'})<-[r1:ACQUIRED]-(o:Organization)-[r2:DEVELOPED_BY]-(p:Person)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version
RETURN DISTINCT o.name AS via, p.name AS entity
```

## M05 · multi-hop · 2 hops

[x] OK   [ ] FIX   [ ] DROP

question: Which AI models were developed by the company that acquired Aleph Alpha?
expected_answer: Aleph Alpha was acquired by Cohere, which developed Takane, Aya Vision.
expected_entities: Takane ; Aya Vision
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 2 rows; first 2:
    {"via": "Cohere", "entity": "Takane"}
    {"via": "Cohere", "entity": "Aya Vision"}

```cypher
MATCH (x:Organization {kg_version: $kg_version, name: 'Aleph Alpha'})<-[r1:ACQUIRED]-(o:Organization)<-[r2:DEVELOPED_BY]-(m:AIModel)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version
RETURN DISTINCT o.name AS via, m.name AS entity
```

## M06 · multi-hop · 2 hops

[ ] OK   [x] FIX   [ ] DROP

question: Where is the organization that developed ChatGPT located?
expected_answer: ChatGPT was developed by OpenAI, headquartered in San Francisco, California.
expected_entities: San Francisco ; California
notes: model-reviewed: removed Pioneer Building (a former office building, not a location answer)

Gold query result on resolved: 3 rows; first 3:
    {"via": "OpenAI", "entity": "San Francisco"}
    {"via": "OpenAI", "entity": "California"}
    {"via": "OpenAI", "entity": "Pioneer Building"}

```cypher
MATCH (m:Entity {kg_version: $kg_version, name: 'ChatGPT'})-[r1:DEVELOPED_BY]->(o:Organization)-[r2:LOCATED_IN]->(l:Location)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version
RETURN DISTINCT o.name AS via, l.name AS entity
```

## M07 · multi-hop · 2 hops

[x] OK   [ ] FIX   [ ] DROP

question: Which people work at the company that developed Aya Vision?
expected_answer: Aya Vision was developed by Cohere; people working there: Ivan Zhang, Martin Kon, Aidan N. Gomez, Joëlle Pineau.
expected_entities: Ivan Zhang ; Martin Kon ; Aidan N. Gomez ; Joëlle Pineau
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 4 rows; first 4:
    {"via": "Cohere", "entity": "Ivan Zhang"}
    {"via": "Cohere", "entity": "Martin Kon"}
    {"via": "Cohere", "entity": "Aidan N. Gomez"}
    {"via": "Cohere", "entity": "Joëlle Pineau"}

```cypher
MATCH (m:Entity {kg_version: $kg_version, name: 'Aya Vision'})-[r1:DEVELOPED_BY]->(o:Organization)<-[r2:WORKS_AT]-(p:Person)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version
RETURN DISTINCT o.name AS via, p.name AS entity
```

## M08 · multi-hop · 3 hops

[x] OK   [ ] FIX   [ ] DROP

question: Who founded the company that invested in the developer of the Jamba model?
expected_answer: Jamba was developed by AI21 Labs, which is funded by Nvidia, founded by Chris Malachowsky, Jensen Huang, Curtis Priem.
expected_entities: Chris Malachowsky ; Jensen Huang ; Curtis Priem
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 3 rows; first 3:
    {"developer": "AI21 Labs", "investor": "Nvidia", "entity": "Chris Malachowsky"}
    {"developer": "AI21 Labs", "investor": "Nvidia", "entity": "Jensen Huang"}
    {"developer": "AI21 Labs", "investor": "Nvidia", "entity": "Curtis Priem"}

```cypher
MATCH (m:AIModel {kg_version: $kg_version, name: 'Jamba'})-[r1:DEVELOPED_BY]->(o:Organization)-[r2:FUNDED_BY]->(f:Organization)-[r3:DEVELOPED_BY]-(p:Person)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version AND r3.kg_version = $kg_version
RETURN DISTINCT o.name AS developer, f.name AS investor, p.name AS entity
```

## M09 · multi-hop · 3 hops

[ ] OK   [x] FIX   [ ] DROP

question: Which people work at the organization that developed the model AlphaZero is based on?
expected_answer: AlphaZero is based on AlphaGo Zero, developed by DeepMind; people working there include Demis Hassabis, David Silver, Aja Huang, Diego Rivas, Koray Kavukcuoglu, Lila Ibrahim, Murray Shanahan, Naina Raisinghani.
expected_entities: Demis Hassabis ; David Silver ; Aja Huang ; Diego Rivas ; Koray Kavukcuoglu ; Lila Ibrahim ; Murray Shanahan ; Naina Raisinghani
notes: model-reviewed: same WORKS_AT corrections as M01

Gold query result on resolved: 12 rows; first 8:
    {"base_model": "AlphaGo Zero", "via": "DeepMind", "entity": "Jaan Tallinn"}
    {"base_model": "AlphaGo Zero", "via": "DeepMind", "entity": "Igor Babuschkin"}
    {"base_model": "AlphaGo Zero", "via": "DeepMind", "entity": "Diego Rivas"}
    {"base_model": "AlphaGo Zero", "via": "DeepMind", "entity": "Naina Raisinghani"}
    {"base_model": "AlphaGo Zero", "via": "DeepMind", "entity": "Aja Huang"}
    {"base_model": "AlphaGo Zero", "via": "DeepMind", "entity": "David Silver"}
    {"base_model": "AlphaGo Zero", "via": "DeepMind", "entity": "Fan Hui"}
    {"base_model": "AlphaGo Zero", "via": "DeepMind", "entity": "Koray Kavukcuoglu"}

```cypher
MATCH (z:AIModel {kg_version: $kg_version, name: 'AlphaZero'})-[r1:BASED_ON]->(b:Entity)-[r2:DEVELOPED_BY]->(o:Organization)<-[r3:WORKS_AT]-(p:Person)
WHERE r1.kg_version = $kg_version AND r2.kg_version = $kg_version AND r3.kg_version = $kg_version
RETURN DISTINCT b.name AS base_model, o.name AS via, p.name AS entity
```

## G01 · aggregation

[x] OK   [ ] FIX   [ ] DROP

question: Which organization developed the most AI models?
expected_answer: DeepMind (34 AI models)
expected_entities: DeepMind
notes: model-reviewed: graph-relative; raw graph gives Anthropic (33) - RQ2 example

Gold query result on resolved: 1 rows; first 1:
    {"entity": "DeepMind", "n": 34}

```cypher
MATCH (m:AIModel {kg_version: $kg_version})-[r:DEVELOPED_BY]->(o:Organization)
WHERE r.kg_version = $kg_version
RETURN o.name AS entity, count(DISTINCT m) AS n ORDER BY n DESC, entity LIMIT 1
```

## G02 · aggregation

[x] OK   [ ] FIX   [ ] DROP

question: How many AI models did Mistral AI develop?
expected_answer: 20 AI models
expected_entities: Mistral AI
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 1 rows; first 1:
    {"entity": "Mistral AI", "n": 20}

```cypher
MATCH (m:AIModel {kg_version: $kg_version})-[r:DEVELOPED_BY]->(o:Organization {name: 'Mistral AI'})
WHERE r.kg_version = $kg_version AND o.kg_version = $kg_version
RETURN o.name AS entity, count(DISTINCT m) AS n
```

## G03 · aggregation

[x] OK   [ ] FIX   [ ] DROP

question: Which organization has acquired the most companies?
expected_answer: Nvidia (18 acquisitions)
expected_entities: Nvidia
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 1 rows; first 1:
    {"entity": "Nvidia", "n": 18}

```cypher
MATCH (o:Organization {kg_version: $kg_version})-[r:ACQUIRED]->(x:Organization)
WHERE r.kg_version = $kg_version
RETURN o.name AS entity, count(DISTINCT x) AS n ORDER BY n DESC, entity LIMIT 1
```

## G04 · aggregation

[x] OK   [ ] FIX   [ ] DROP

question: Which technology is used by the most entities in the knowledge graph?
expected_answer: Large Language Models (used by 15 entities)
expected_entities: Large Language Models
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 1 rows; first 1:
    {"entity": "Large Language Models", "n": 15}

```cypher
MATCH (x:Entity {kg_version: $kg_version})-[r:USES]->(t:Technology)
WHERE r.kg_version = $kg_version
RETURN t.name AS entity, count(DISTINCT x) AS n ORDER BY n DESC, entity LIMIT 1
```

## G05 · aggregation

[x] OK   [ ] FIX   [ ] DROP

question: Which person has worked at the most organizations?
expected_answer: Geoffrey Hinton (10 organizations)
expected_entities: Geoffrey Hinton
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 1 rows; first 1:
    {"entity": "Geoffrey Hinton", "n": 10}

```cypher
MATCH (p:Person {kg_version: $kg_version})-[r:WORKS_AT]->(o:Organization)
WHERE r.kg_version = $kg_version
RETURN p.name AS entity, count(DISTINCT o) AS n ORDER BY n DESC, entity LIMIT 1
```

## G06 · aggregation

[x] OK   [ ] FIX   [ ] DROP

question: How many people in the knowledge graph work at OpenAI?
expected_answer: 23 people
expected_entities: OpenAI
notes: model-reviewed: graph-relative count (WORKS_AT may include former staff)

Gold query result on resolved: 1 rows; first 1:
    {"entity": "OpenAI", "n": 23}

```cypher
MATCH (p:Person {kg_version: $kg_version})-[r:WORKS_AT]->(o:Organization {name: 'OpenAI'})
WHERE r.kg_version = $kg_version AND o.kg_version = $kg_version
RETURN o.name AS entity, count(DISTINCT p) AS n
```

## S01 · semantic

[x] OK   [ ] FIX   [ ] DROP

question: What is DeepMind known for?
expected_answer: DeepMind is an AI research laboratory known for models such as AlphaGo, Gemini, AlphaFold, AlphaFold 2, AlphaFold 3.
expected_entities: AlphaGo ; Gemini ; AlphaFold ; AlphaFold 2 ; AlphaFold 3
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 5 rows; first 5:
    {"description": "British-American artificial intelligence research laboratory", "entity": "AlphaGo"}
    {"description": "British-American artificial intelligence research laboratory", "entity": "Gemini"}
    {"description": "British-American artificial intelligence research laboratory", "entity": "AlphaFold"}
    {"description": "British-American artificial intelligence research laboratory", "entity": "AlphaFold 2"}
    {"description": "British-American artificial intelligence research laboratory", "entity": "AlphaFold 3"}

```cypher
MATCH (o:Organization {kg_version: $kg_version, name: 'DeepMind'})<-[r:DEVELOPED_BY]-(m:AIModel)
WHERE r.kg_version = $kg_version
RETURN o.description AS description, m.name AS entity ORDER BY m.mention_count DESC, entity LIMIT 5
```

## S02 · semantic

[x] OK   [ ] FIX   [ ] DROP

question: What is Constitutional AI?
expected_answer: A method for training a harmless AI assistant through self-improvement, developed by Anthropic.
expected_entities: Anthropic
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 1 rows; first 1:
    {"description": "method for training a harmless AI assistant through self-improvement", "entity": "Anthropic"}

```cypher
MATCH (t:Entity {kg_version: $kg_version, name: 'Constitutional AI'})
OPTIONAL MATCH (t)-[r:DEVELOPED_BY]->(o:Organization) WHERE r.kg_version = $kg_version
RETURN t.description AS description, o.name AS entity
```

## S03 · semantic

[x] OK   [ ] FIX   [ ] DROP

question: What is the Transformer architecture?
expected_answer: A family of neural network architectures based on multi-head attention, used by models such as GPT-4, GPT-3, BERT, GPT-2, DALL-E.
expected_entities: GPT-4 ; GPT-3 ; BERT ; GPT-2 ; DALL-E
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 5 rows; first 5:
    {"description": "a family of artificial neural network architectures based on the multi-head attention mechanism", "entity": "GPT-4"}
    {"description": "a family of artificial neural network architectures based on the multi-head attention mechanism", "entity": "GPT-3"}
    {"description": "a family of artificial neural network architectures based on the multi-head attention mechanism", "entity": "BERT"}
    {"description": "a family of artificial neural network architectures based on the multi-head attention mechanism", "entity": "GPT-2"}
    {"description": "a family of artificial neural network architectures based on the multi-head attention mechanism", "entity": "DALL-E"}

```cypher
MATCH (t:Technology {kg_version: $kg_version, name: 'Transformer'})
OPTIONAL MATCH (m:AIModel)-[r:USES|BASED_ON]->(t) WHERE r.kg_version = $kg_version AND m.kg_version = $kg_version
RETURN t.description AS description, m.name AS entity ORDER BY m.mention_count DESC LIMIT 5
```

## S04 · semantic

[ ] OK   [x] FIX   [ ] DROP

question: What is Hugging Face?
expected_answer: An American AI company based in New York City that hosts models and datasets and develops tools such as the transformers library, llama.cpp (via GGML joining), Optimum TPU and ML Intern.
expected_entities: llama.cpp ; transformers ; Optimum TPU ; ML Intern
notes: model-reviewed: removed 'Transformer' (the architecture, mislinked from the Transformers library)

Gold query result on resolved: 5 rows; first 5:
    {"description": "an American company based in New York City", "entity": "Transformer"}
    {"description": "an American company based in New York City", "entity": "llama.cpp"}
    {"description": "an American company based in New York City", "entity": "transformers"}
    {"description": "an American company based in New York City", "entity": "Optimum TPU"}
    {"description": "an American company based in New York City", "entity": "ML Intern"}

```cypher
MATCH (o:Organization {kg_version: $kg_version, name: 'Hugging Face'})
OPTIONAL MATCH (o)<-[r:DEVELOPED_BY]-(x:Entity) WHERE r.kg_version = $kg_version
RETURN o.description AS description, x.name AS entity ORDER BY x.mention_count DESC LIMIT 5
```

## S05 · semantic

[ ] OK   [x] FIX   [ ] DROP

question: What is ImageNet used for?
expected_answer: A large visual database for visual object recognition research (used e.g. to train AlexNet), created by Fei-Fei Li.
expected_entities: Fei-Fei Li ; AlexNet
notes: model-reviewed: dropped stray 'Li'; added AlexNet as a use (gold used_by column)

Gold query result on resolved: 2 rows; first 2:
    {"description": "large visual database designed for use in visual object recognition software research", "used_by": ["AlexNet", "Chinchilla", "SIFT", "PyTorch", "Amazon Mechanical Turk"], "entity": "Li"}
    {"description": "large visual database designed for use in visual object recognition software research", "used_by": ["AlexNet", "Chinchilla", "SIFT", "PyTorch", "Amazon Mechanical Turk"], "entity": "Fei-Fei Li"}

```cypher
MATCH (d:Dataset {kg_version: $kg_version, name: 'ImageNet'})
OPTIONAL MATCH (x:Entity)-[r:TRAINED_ON|USES]->(d) WHERE r.kg_version = $kg_version
OPTIONAL MATCH (d)-[r2:DEVELOPED_BY]->(p:Person) WHERE r2.kg_version = $kg_version
RETURN d.description AS description, collect(DISTINCT x.name)[..5] AS used_by, p.name AS entity
```

## S06 · semantic

[x] OK   [ ] FIX   [ ] DROP

question: What is Geoffrey Hinton known for?
expected_answer: A deep learning pioneer known for AlexNet, Boltzmann machines, DNNresearch Inc., t-SNE.
expected_entities: AlexNet ; Boltzmann machines ; DNNresearch Inc. ; t-SNE
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 4 rows; first 4:
    {"description": "Geoffrey Hinton", "entity": "AlexNet"}
    {"description": "Geoffrey Hinton", "entity": "Boltzmann machines"}
    {"description": "Geoffrey Hinton", "entity": "DNNresearch Inc."}
    {"description": "Geoffrey Hinton", "entity": "t-SNE"}

```cypher
MATCH (p:Person {kg_version: $kg_version, name: 'Geoffrey Hinton'})<-[r:DEVELOPED_BY]-(x:Entity)
WHERE r.kg_version = $kg_version
RETURN p.description AS description, x.name AS entity ORDER BY x.mention_count DESC, entity
```

## U01 · unanswerable

[x] OK   [ ] FIX   [ ] DROP

question: Who is the CEO of INPT?
expected_answer: Not found in the knowledge graph.
expected_entities: 
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 0 rows

```cypher
MATCH (o:Entity {kg_version: $kg_version})<-[r:WORKS_AT]-(p:Person)
WHERE r.kg_version = $kg_version AND (toLower(o.name) CONTAINS 'inpt' OR toLower(o.name) CONTAINS 'institut national des postes')
RETURN p.name AS entity
```

## U02 · unanswerable

[x] OK   [ ] FIX   [ ] DROP

question: Which company developed the AI model Zephyrion-9?
expected_answer: Not found in the knowledge graph.
expected_entities: 
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 0 rows

```cypher
MATCH (m:Entity {kg_version: $kg_version})-[r:DEVELOPED_BY]->(o)
WHERE r.kg_version = $kg_version AND toLower(m.name) CONTAINS 'zephyrion'
RETURN o.name AS entity
```

## U03 · unanswerable

[x] OK   [ ] FIX   [ ] DROP

question: How much did Apple pay to acquire Mistral AI?
expected_answer: Not found in the knowledge graph.
expected_entities: 
notes: model-reviewed: gold query, answer and category checked

Gold query result on resolved: 0 rows

```cypher
MATCH (a:Organization {kg_version: $kg_version, name: 'Apple'})-[r:ACQUIRED]->(m:Organization {name: 'Mistral AI'})
WHERE r.kg_version = $kg_version AND m.kg_version = $kg_version
RETURN m.name AS entity
```
