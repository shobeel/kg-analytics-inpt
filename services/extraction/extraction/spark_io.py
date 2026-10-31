"""Read/write silver Delta tables with a local SparkSession (the data is small; this keeps the
extraction container independent of the cluster's executor slots)."""

import os

from pyspark.sql import SparkSession


def get_spark(app: str) -> SparkSession:
    return (SparkSession.builder.appName(app)
            .master(os.environ.get("EXTRACTION_SPARK_MASTER", "local[2]"))
            .config("spark.driver.memory", os.environ.get("EXTRACTION_SPARK_DRIVER_MEMORY", "640m"))
            .config("spark.hadoop.fs.defaultFS", os.environ.get("HDFS_URL", "hdfs://namenode:9000"))
            .config("spark.hadoop.dfs.replication", os.environ.get("DFS_REPLICATION", "1"))
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def silver(hdfs: str, name: str) -> str:
    return f"{hdfs}/datalake/silver/{name}"


def read_chunks(spark: SparkSession, hdfs: str) -> list[dict]:
    chunks = spark.read.format("delta").load(silver(hdfs, "chunks"))
    docs = spark.read.format("delta").load(silver(hdfs, "documents")).select("doc_id", "title", "url")
    rows = (chunks.join(docs, "doc_id", "left")
            .select("chunk_id", "doc_id", "source", "chunk_index", "text", "title", "url")
            .orderBy("chunk_id").collect())
    return [r.asDict() for r in rows]


def write_table(spark: SparkSession, rows: list[dict], schema, path: str) -> int:
    df = spark.createDataFrame(rows, schema=schema)
    (df.coalesce(1).write.format("delta").mode("overwrite")
     .option("overwriteSchema", "true").save(path))
    return len(rows)
