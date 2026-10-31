# Phase 7 API: FastAPI over the GraphRAG engine (Phase 5) and the analytics module (Phase 4).
# Built on the graphrag image (same engine, prompts, cache, LLM client); `make demo` builds that first.
ARG BASE=kg/graphrag:latest
FROM ${BASE}
WORKDIR /app
COPY api/ /app/api/
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
