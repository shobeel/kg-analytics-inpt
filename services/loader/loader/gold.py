"""Read the gold Delta tables with a local SparkSession (the graph is small)."""

import os

from pyspark.sql import SparkSession


def get_spark(app: str) -> SparkSession:
    return (SparkSession.builder.appName(app).master("local[2]")
            .config("spark.driver.memory", os.environ.get("LOADER_SPARK_DRIVER_MEMORY", "1g"))
            .config("spark.hadoop.fs.defaultFS", os.environ.get("HDFS_URL", "hdfs://namenode:9000"))
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def gold_path(name: str) -> str:
    return f"{os.environ.get('HDFS_URL', 'hdfs://namenode:9000')}/datalake/gold/{name}"


def read(spark: SparkSession, name: str) -> list[dict]:
    return [r.asDict(recursive=True) for r in spark.read.format("delta").load(gold_path(name)).collect()]


def counts(spark: SparkSession, names: list[str]) -> dict[tuple[str, str], int]:
    """(table, kg_version) -> rows, counted from the tables themselves (shared tables: 'shared')."""
    out = {}
    for name in names:
        df = spark.read.format("delta").load(gold_path(name))
        if "kg_version" in df.columns:
            for r in df.groupBy("kg_version").count().collect():
                out[(name, r["kg_version"])] = r["count"]
        else:
            out[(name, "shared")] = df.count()
    return out
