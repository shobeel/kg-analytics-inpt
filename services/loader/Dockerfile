# Gold -> Neo4j loader (Phase 3). Built on the shared PySpark image (project convention): it reads the
# gold Delta tables with a local SparkSession, embeds chunks with fastembed (ONNX, no torch) and
# writes to Neo4j with the official driver (already in the base image). No LLM calls.
FROM kg/pyspark:3.5.0

ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=5
COPY requirements.txt /tmp/loader-requirements.txt
RUN --mount=type=cache,target=/home/jovyan/.cache/pip,uid=1000,gid=100 \
    pip install -r /tmp/loader-requirements.txt \
 && ! python -c "import torch" 2>/dev/null

ENV FASTEMBED_CACHE_PATH=/home/jovyan/.cache/fastembed
RUN mkdir -p ${FASTEMBED_CACHE_PATH}

WORKDIR /app
COPY loader/ /app/loader/
ENV PYTHONPATH=/app:${PYTHONPATH}

CMD ["python", "-m", "loader", "--help"]
