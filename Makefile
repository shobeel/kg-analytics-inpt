# STACK=lite (default): 1 datanode + 1 Spark worker, ~5.9 GB of limits (8 GB Mac, Docker at 6 GB)
# STACK=full: 2 datanodes + 2 Spark workers, ~9.8 GB of limits (needs 10 GB for Docker)
STACK ?= lite
ifeq ($(STACK),lite)
COMPOSE := docker compose -f infra/docker-compose.yml -f infra/docker-compose.lite.yml --env-file .env
else ifeq ($(STACK),full)
COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env
else
$(error STACK must be lite or full)
endif

# scripts/ (acceptance tests, number check, document builders) is not published: targets that need it say so
need = test -f $(1) || { echo "skipped: $(1) is not in this repository"; exit 0; };

.PHONY: rapport all dry-run presentation judge-validate api-keys demo demo-down test-api help env build up up-apps down logs status reset test test-phase0 fetch-samples crawl silver test-unit test-phase1 llm-test extract extract-status rebuild-silver resolve relation-stats gold-prefill gold-todo gold-convert gold-sample score test-phase2 gold neo4j-load neo4j-check test-phase3 analytics-queries analytics analytics-notebook test-phase4 ask graphrag-eval test-phase5 questions-draft questions-apply benchmark-plan benchmark judge judge-sample judge-validate judge-agreement rq1 report test-phase6

help:            ## List targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-13s %s\n",$$1,$$2}'

env:             ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "Created .env - set NEO4J_PASSWORD, GEMINI_API_KEY, GROQ_API_KEY")

build: env       ## Build the custom images
	$(COMPOSE) build

up: env          ## Start the core stack [STACK=lite|full] (HDFS, Spark, Neo4j, Jupyter) and wait for health
	$(COMPOSE) up -d --build --wait

up-apps: env     ## Start the core stack plus the app services
	$(COMPOSE) --profile apps up -d --build --wait

down:            ## Stop containers (keeps data volumes)
	$(COMPOSE) --profile apps down

logs:            ## Follow logs (make logs s=namenode for one service)
	$(COMPOSE) logs -f --tail=100 $(s)

status:          ## Container health + memory use
	@$(COMPOSE) --profile apps ps --format 'table {{.Name}}\t{{.Status}}\t{{.Ports}}'
	@docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' $$($(COMPOSE) ps -q) 2>/dev/null || true

reset:           ## DELETE all data volumes (HDFS, Neo4j) after confirmation
	@read -p "This deletes HDFS and Neo4j data volumes. Type 'yes' to continue: " a && [ "$$a" = yes ]
	$(COMPOSE) --profile apps down -v --remove-orphans

test-phase0:     ## Phase 0 acceptance test [STACK=lite|full]
	@$(call need,scripts/acceptance_phase0.sh) STACK=$(STACK) ./scripts/acceptance_phase0.sh

# ---------------- Phase 1: ingestion ----------------
fetch-samples:   ## Download the local sample PDFs/HTML into data/local
	@$(call need,scripts/fetch_local_samples.sh) ./scripts/fetch_local_samples.sh

crawl: env       ## Crawl all enabled sources into HDFS bronze (src="arxiv rss" to limit)
	$(COMPOSE) run --rm crawler python -m crawler $(if $(src),--sources $(src))

silver:          ## Spark job: bronze -> silver documents + chunks (Delta)
	docker exec jupyter spark-submit --py-files work/spark_jobs/text_utils.py \
	  work/spark_jobs/bronze_to_silver_text.py $(args)

# ---------------- Phase 3: silver -> gold -> Neo4j (no LLM calls) ----------------
gold:            ## Spark job: silver -> gold node/edge tables, raw + resolved (Delta)
	docker exec jupyter spark-submit --py-files work/spark_jobs/gold_utils.py work/spark_jobs/silver_to_gold.py $(args)

LOADER := $(COMPOSE) run --rm loader python -m loader

neo4j-load:      ## Constraints/indexes + gold -> Neo4j (raw + resolved), chunk embeddings; idempotent
	$(LOADER) load

neo4j-check:     ## Neo4j vs gold counts, provenance, vector search, raw vs resolved (neo4j/checks.cypher)
	$(LOADER) check

# ---------------- Phase 4: graph analytics (Cypher + GDS; no LLM calls) ----------------
ANALYTICS := docker exec -w /home/jovyan/work jupyter python -m analytics

analytics-queries: ## Run every Cypher library query on both kg_versions (rows + timing)
	$(ANALYTICS) queries

analytics:       ## GDS algorithms on both versions (written back) + RQ2 raw vs resolved table
	$(ANALYTICS) algorithms
	$(ANALYTICS) quality

analytics-notebook: ## Execute notebooks/04_graph_analytics.ipynb headless (figures -> docs/figures)
	docker exec -w /home/jovyan/work/notebooks jupyter jupyter nbconvert --to notebook --execute --inplace 04_graph_analytics.ipynb

test-phase4:     ## Phase 4 acceptance test
	@$(call need,scripts/acceptance_phase4.sh) STACK=$(STACK) ./scripts/acceptance_phase4.sh

# ---------------- Phase 5: GraphRAG assistant (ANSWER_MODEL; cached) ----------------
GRAPHRAG := $(COMPOSE) run --rm graphrag python -m graphrag

ask:             ## Ask a question: make ask q="Who developed Claude?" s=B kg=resolved
	$(GRAPHRAG) ask "$(q)" --strategy $(or $(s),B) --kg $(or $(kg),resolved)

graphrag-eval:   ## 5 questions x 3 strategies on resolved (cached LLM calls) -> eval/results/phase5_answers.json
	$(GRAPHRAG) eval

test-phase5:     ## Phase 5 acceptance test
	@$(call need,scripts/acceptance_phase5.sh) STACK=$(STACK) ./scripts/acceptance_phase5.sh

# ---------------- Phase 6: ground truth + benchmark (eval/; resumable, every LLM call cached) ----------------
EVAL := $(COMPOSE) run --rm graphrag python -m eval
# benchmark + judge: 1 call + at most 2 retries, then the run is recorded as an error
EVAL_LLM := $(COMPOSE) run --rm -e LLM_MAX_RETRIES=3 graphrag python -m eval

questions-draft: ## Draft 30 questions + gold Cypher from the resolved graph -> eval/gold/questions{.jsonl,_review.md}
	$(EVAL).questions draft $(if $(force),--force)

questions-apply: ## Apply the ticks/edits of eval/gold/questions_review.md (verified questions)
	$(EVAL).questions apply-review

benchmark-plan:  ## Planned Gemini + Groq calls for what is left (no LLM call)
	$(EVAL).run_benchmark plan

benchmark:       ## Answers: verified questions x {Direct, A, B, C} x {raw, resolved}; plan only unless ok=1
	$(EVAL_LLM).run_benchmark run $(if $(ok),--confirm)

judge:           ## Groq judge on every answer not judged yet; plan only unless ok=1
	$(EVAL_LLM).run_benchmark judge $(if $(ok),--confirm)

judge-sample:    ## 15 judged answers -> eval/gold/judge_validation.md for blind hand scoring
	$(EVAL).judge_validation make

judge-validate:  ## Inter-model judge validation: a second Groq model re-scores the 15 samples; plan only unless ok=1
	$(EVAL_LLM).judge_validation inter $(if $(ok),--confirm)

judge-agreement: ## Agreement of your hand scores with the judge
	$(EVAL).judge_validation score

rq1:             ## RQ1 extraction P/R/F1 on the gold chunks marked done (no LLM call)
	$(COMPOSE) run --rm --no-deps -w /app extraction python -m eval.rq1

report:          ## Metrics, figures and docs/benchmark_report.md from stored results (no LLM call)
	$(EVAL).report

test-phase6:     ## Phase 6 acceptance test
	@$(call need,scripts/acceptance_phase6.sh) STACK=$(STACK) ./scripts/acceptance_phase6.sh

test-phase3:     ## Phase 3 acceptance test
	@$(call need,scripts/acceptance_phase3.sh) STACK=$(STACK) ./scripts/acceptance_phase3.sh

test: test-unit  ## All unit tests (alias of test-unit)

test-unit:       ## Unit tests (crawler, extraction, scorer, loader, spark_jobs) inside the containers
	$(COMPOSE) run --rm --no-deps crawler pytest -q /app/tests/crawler
	$(COMPOSE) run --rm --no-deps extraction pytest -q -p no:cacheprovider /app/tests/extraction
	$(COMPOSE) run --rm --no-deps loader pytest -q -p no:cacheprovider /app/tests/loader
	$(COMPOSE) run --rm --no-deps graphrag pytest -q -p no:cacheprovider /app/tests/graphrag /app/tests/eval
	docker exec -w /home/jovyan/work jupyter bash -c 'PYTHONPATH=spark_jobs:$$PYTHONPATH python -m pytest -q -p no:cacheprovider tests/spark_jobs'
	docker exec -w /home/jovyan/work jupyter python -m pytest -q -p no:cacheprovider tests/analytics
	$(MAKE) test-api

test-api:        ## Phase 7 API + auth unit tests (no Neo4j, no LLM)
	$(COMPOSE) build -q api
	$(COMPOSE) run --rm --no-deps -v $(CURDIR)/tests:/app/tests:ro -e API_LOG=/tmp/api.jsonl api pytest -q -p no:cacheprovider /app/tests/api

test-phase1:     ## Phase 1 acceptance test [STACK=lite|full]
	@$(call need,scripts/acceptance_phase1.sh) STACK=$(STACK) ./scripts/acceptance_phase1.sh

# ---------------- Phase 2: extraction + entity resolution (Gemini free tier) ----------------
EXTRACT := $(COMPOSE) run --rm extraction python -m extraction

llm-test: env    ## One test call to EXTRACTION_MODEL (Gemini)
	$(EXTRACT) test-call

extract: env     ## LLM extraction over silver chunks (n=10 for a pilot); resumes from cache
	$(EXTRACT) extract $(if $(n),--limit $(n) --show-json)

rebuild-silver:  ## Rewrite silver extracted_* from cache with the current validation rules (no LLM)
	$(EXTRACT) extract --cached-only

extract-status:  ## Extraction coverage + relation flag rates (cache vs silver; no LLM calls)
	$(EXTRACT) status

resolve: env     ## Entity resolution -> KG_raw + KG_resolved + merge log (silver)
	$(EXTRACT) resolve

relation-stats:  ## Relation types, top schema violations, proposed tweaks (cache only; no LLM)
	$(EXTRACT) relation-stats

gold-prefill: env ## Draft the 25 gold chunks with ANSWER_MODEL -> eval/gold/extraction/prefill.jsonl
	$(EXTRACT) prefill-gold

gold-todo:       ## One editable file per gold chunk in eval/gold/extraction/todo/ (never overwrites)
	$(COMPOSE) run --rm --no-deps -w /app extraction python -m eval.gold_todo make

gold-convert:    ## Corrected todo/ files -> gold.jsonl (scorer input) + changes log
	$(COMPOSE) run --rm --no-deps -w /app extraction python -m eval.gold_todo convert

gold-sample:     ## Sample 25 chunks into eval/gold/extraction/annotation_batch.jsonl
	$(EXTRACT) sample-gold

score:           ## Score predictions vs gold (default: toy files)
	$(COMPOSE) run --rm --no-deps -w /app extraction python -m eval.extraction_scorer \
	  --gold $(or $(gold),eval/gold/extraction/toy/gold.jsonl) --pred $(or $(pred),eval/gold/extraction/toy/pred.jsonl)

test-phase2:     ## Phase 2 acceptance test
	@$(call need,scripts/acceptance_phase2.sh) STACK=$(STACK) ./scripts/acceptance_phase2.sh

# ---------------- One-command reproduction ----------------
all: env         ## Whole pipeline, resumable: no-LLM steps always; LLM steps (extraction, answers, judge) only with ok=1
	$(MAKE) up
	$(MAKE) crawl
	$(MAKE) silver
	$(if $(ok),$(MAKE) extract,@echo "SKIP extract (LLM; make all ok=1)")
	$(if $(ok),$(MAKE) resolve,@echo "SKIP resolve (LLM adjudication; make all ok=1)")
	$(MAKE) gold
	$(MAKE) neo4j-load
	$(MAKE) analytics
	$(MAKE) rq1
	$(if $(ok),$(MAKE) benchmark ok=1,$(MAKE) benchmark-plan || true)
	$(if $(ok),$(MAKE) judge ok=1,@echo "SKIP judge (LLM; make all ok=1)")
	$(if $(ok),$(MAKE) judge-validate ok=1,@echo "SKIP judge-validate (LLM; make all ok=1)")
	$(MAKE) report
	@$(call need,scripts/check_report_numbers.py) python3 scripts/check_report_numbers.py

# no-deps: talks to the running stack (make up) without recreating it, e.g. from a second checkout
EVAL_ND := $(COMPOSE) run --rm --no-deps graphrag python -m eval

dry-run:         ## No-LLM check of a checkout against the running stack: unit tests, plan, report + number check
	$(MAKE) test
	$(EVAL_ND).run_benchmark plan || true
	$(EVAL_ND).report
	@$(call need,scripts/check_report_numbers.py) python3 scripts/check_report_numbers.py

presentation:    ## docs/presentation.pptx from the stored results and figures (python-pptx in .venv-docs)
	test -x .venv-docs/bin/python || (python3 -m venv .venv-docs && .venv-docs/bin/pip install -q python-pptx==1.0.2)
	@$(call need,scripts/make_presentation.py) .venv-docs/bin/python scripts/make_presentation.py

# ---------------- Phase 7: application (API + UI) ----------------
api-keys: env    ## Add random admin/viewer API keys to .env if missing (never printed)
	@grep -qE '^API_KEYS_ADMIN=.{16,}' .env || { k=$$(python3 -c "import secrets;print(secrets.token_urlsafe(24))"); \
	  sed -i.bak '/^API_KEYS_ADMIN=/d' .env; echo "API_KEYS_ADMIN=$$k" >> .env; }
	@grep -qE '^API_KEYS_VIEWER=.{16,}' .env || { k=$$(python3 -c "import secrets;print(secrets.token_urlsafe(24))"); \
	  sed -i.bak -e '/^API_KEYS_VIEWER=/d' -e '/^UI_API_KEY=/d' .env; echo "API_KEYS_VIEWER=$$k" >> .env; echo "UI_API_KEY=$$k" >> .env; }
	@rm -f .env.bak; echo "API keys present in .env (admin + viewer; the UI uses the viewer key)"

demo: api-keys   ## Start the stack + API + UI and print the UI URL (answers use cached LLM calls when available)
	$(MAKE) up
	$(COMPOSE) build graphrag
	$(COMPOSE) --profile apps up -d --build --wait api ui
	@echo ""; echo "UI:  http://localhost:8501   (Ask / Analytics / Results)"; \
	 echo "API: http://localhost:8000/docs  (X-API-Key from .env)"

demo-down:       ## Stop the API and UI (the core stack keeps running)
	$(COMPOSE) --profile apps stop api ui

rapport:         ## docs/rapport.md (French report, from the result files) -> docs/rapport.pdf (pandoc + headless Chrome)
	@$(call need,scripts/make_rapport.py) python3 scripts/make_rapport.py
	@$(call need,scripts/check_report_numbers.py) python3 scripts/check_report_numbers.py docs/rapport.md --fr
	cd docs && pandoc rapport.md -s --metadata lang=fr -c rapport.css --embed-resources -o rapport.html
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
	  --print-to-pdf="$(CURDIR)/docs/rapport.pdf" "file://$(CURDIR)/docs/rapport.html" 2>/dev/null
	rm -f docs/rapport.html
	@echo "wrote docs/rapport.pdf"
