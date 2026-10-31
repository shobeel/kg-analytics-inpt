# Phase 7 Streamlit UI. Built on the shared PySpark image (project convention); only streamlit + pyvis on top.
# It talks to the API over HTTP with a viewer key and reads figures/results from read-only mounts.
FROM kg/pyspark:3.5.0
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=5
COPY requirements.txt /tmp/ui-requirements.txt
RUN --mount=type=cache,target=/home/jovyan/.cache/pip,uid=1000,gid=100 \
    pip install -r /tmp/ui-requirements.txt \
 && ! python -c "import torch" 2>/dev/null
WORKDIR /app
COPY . /app/
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "Ask.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
