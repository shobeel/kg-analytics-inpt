# Crawler service. Python services build on the shared PySpark image (project convention).
FROM kg/pyspark:3.5.0

ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=5
COPY requirements.txt /tmp/crawler-requirements.txt
RUN --mount=type=cache,target=/home/jovyan/.cache/pip,uid=1000,gid=100 \
    pip install -r /tmp/crawler-requirements.txt

WORKDIR /app
COPY crawler/ /app/crawler/
COPY config/ /app/config/
ENV PYTHONPATH=/app:${PYTHONPATH}

CMD ["uvicorn", "crawler.app:app", "--host", "0.0.0.0", "--port", "8000"]
