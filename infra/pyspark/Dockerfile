# One image for the Jupyter driver AND the Spark master/workers: Spark 3.5.0 + Python 3.11,
# plus Neo4j / GDS clients. Sharing it guarantees identical Python libs on driver and executors.
FROM jupyter/pyspark-notebook:spark-3.5.0

# The base image adds pyspark to PYTHONPATH only in a start-up hook, which `docker exec`,
# headless nbconvert and other non-notebook entry points skip. Set it for every process.
ENV PYTHONPATH=/usr/local/spark/python:/usr/local/spark/python/lib/py4j-0.10.9.7-src.zip

# Long timeout: large PyPI index pages (pyarrow) exceed pip's 15 s default on slow links.
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=5
COPY requirements.txt /tmp/requirements.txt
# BuildKit cache mount: a failed/retried build reuses wheels already downloaded.
RUN --mount=type=cache,target=/home/jovyan/.cache/pip,uid=1000,gid=100 \
    pip install -r /tmp/requirements.txt

# spark-defaults.conf is generated at startup from env vars (lite vs full stack).
USER root

# Delta Lake 3.2.0 (Spark 3.5) for the silver/gold layers. Jars are baked in so the driver
# and every executor have them without a runtime Maven download; SHA1 from Maven Central.
ARG DELTA_VERSION=3.2.0
RUN cd /usr/local/spark/jars \
 && curl -fsSLO https://repo1.maven.org/maven2/io/delta/delta-spark_2.12/${DELTA_VERSION}/delta-spark_2.12-${DELTA_VERSION}.jar \
 && curl -fsSLO https://repo1.maven.org/maven2/io/delta/delta-storage/${DELTA_VERSION}/delta-storage-${DELTA_VERSION}.jar \
 && echo "0e648d893d3e6cda312d0a6dc4c8f64f3dbe0fb5  delta-spark_2.12-${DELTA_VERSION}.jar" | sha1sum -c - \
 && echo "66d3ddd36fc062707109cb8023fcfab89293fcea  delta-storage-${DELTA_VERSION}.jar" | sha1sum -c -

COPY before-notebook.d/ /usr/local/bin/before-notebook.d/
RUN touch /usr/local/spark/conf/spark-defaults.conf \
 && chown ${NB_UID}:${NB_GID} /usr/local/spark/conf/spark-defaults.conf
USER ${NB_UID}
