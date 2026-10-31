#!/bin/bash
# Runs at container start (docker-stacks hook). Writes Spark driver settings that differ
# between the lite and full stacks from env vars set in docker-compose*.yml.
cat > "${SPARK_HOME}/conf/spark-defaults.conf" <<CONF
spark.master                      ${SPARK_MASTER_URL:-spark://spark-master:7077}
spark.hadoop.fs.defaultFS         ${HDFS_URL:-hdfs://namenode:9000}
spark.hadoop.dfs.replication      ${DFS_REPLICATION:-2}
spark.driver.host                 jupyter
spark.driver.bindAddress          0.0.0.0
spark.driver.memory               ${SPARK_DRIVER_MEMORY:-1g}
spark.executor.memory             ${SPARK_EXECUTOR_MEMORY:-768m}
spark.executor.cores              2
spark.cores.max                   ${SPARK_CORES_MAX:-4}
spark.sql.shuffle.partitions      ${SPARK_SHUFFLE_PARTITIONS:-8}
spark.sql.extensions              io.delta.sql.DeltaSparkSessionExtension
spark.sql.catalog.spark_catalog   org.apache.spark.sql.delta.catalog.DeltaCatalog
CONF
