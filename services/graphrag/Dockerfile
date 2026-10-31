# GraphRAG assistant (Phase 5). Built on the shared PySpark image (project convention) (Python, neo4j
# driver identical everywhere). Reuses two packages from the repo via compose additional_contexts:
#   analytics/            Phase 4 Cypher library (entity lookup, neighbours, provenance)
#   extraction/{llm,config}.py  Phase 2 Gemini client: throttle, retries, quota stop, JSONL call log
FROM kg/pyspark:3.5.0

ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=5
COPY requirements.txt /tmp/graphrag-requirements.txt
RUN --mount=type=cache,target=/home/jovyan/.cache/pip,uid=1000,gid=100 \
    pip install -r /tmp/graphrag-requirements.txt \
 && ! python -c "import torch" 2>/dev/null

ENV FASTEMBED_CACHE_PATH=/home/jovyan/.cache/fastembed
RUN mkdir -p ${FASTEMBED_CACHE_PATH}

WORKDIR /app
COPY --from=analytics . /app/analytics/
COPY --from=extraction __init__.py config.py llm.py /app/extraction/
COPY graphrag/ /app/graphrag/
ENV PYTHONPATH=/app:${PYTHONPATH}

CMD ["uvicorn", "graphrag.app:app", "--host", "0.0.0.0", "--port", "8000"]
