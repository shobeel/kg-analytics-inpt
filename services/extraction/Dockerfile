# Extraction + entity resolution service. Built on the shared PySpark image (project convention)
# (it reads/writes silver Delta tables with a local SparkSession).
FROM kg/pyspark:3.5.0

ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=5
COPY requirements.txt /tmp/extraction-requirements.txt
RUN --mount=type=cache,target=/home/jovyan/.cache/pip,uid=1000,gid=100 \
    pip install -r /tmp/extraction-requirements.txt \
 && ! python -c "import torch" 2>/dev/null

# fastembed model files live in a named volume mounted here (downloaded once).
ENV FASTEMBED_CACHE_PATH=/home/jovyan/.cache/fastembed
RUN mkdir -p ${FASTEMBED_CACHE_PATH}

WORKDIR /app
COPY extraction/ /app/extraction/
ENV PYTHONPATH=/app:${PYTHONPATH}

CMD ["uvicorn", "extraction.app:app", "--host", "0.0.0.0", "--port", "8000"]
