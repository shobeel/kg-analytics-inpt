"""Bronze -> silver: raw documents to clean text, language and chunks.

Reads   /datalake/bronze/<source>/<date>/<id>.json   (never modified)
Writes  /datalake/silver/documents   one row per document (clean text, language, stats)
        /datalake/silver/chunks      ~800-token chunks with 100-token overlap (English only)

Full recompute on every run (overwrite): bronze is the source of truth, so re-runs are
idempotent. Run with `make silver` (spark-submit in the Jupyter container, cluster mode
on spark://spark-master:7077).
"""

import argparse
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

BRONZE_SCHEMA = T.StructType([
    T.StructField("id", T.StringType()),
    T.StructField("source", T.StringType()),
    T.StructField("url", T.StringType()),
    T.StructField("title", T.StringType()),
    T.StructField("published_at", T.StringType()),
    T.StructField("retrieved_at", T.StringType()),
    T.StructField("raw_content", T.StringType()),
    T.StructField("content_type", T.StringType()),
    T.StructField("content_encoding", T.StringType()),
    T.StructField("metadata", T.StringType()),   # nested object kept as its JSON text
])

EXTRACT_SCHEMA = T.StructType([
    T.StructField("text", T.StringType()),
    T.StructField("extract_error", T.StringType()),
    T.StructField("extraction_method", T.StringType()),
    T.StructField("language", T.StringType()),
    T.StructField("language_prob", T.DoubleType()),
])

CHUNK_SCHEMA = T.ArrayType(T.StructType([
    T.StructField("chunk_index", T.IntegerType()),
    T.StructField("text", T.StringType()),
    T.StructField("char_start", T.IntegerType()),
    T.StructField("char_end", T.IntegerType()),
    T.StructField("n_words", T.IntegerType()),
    T.StructField("n_tokens_est", T.IntegerType()),
]))


@F.udf(EXTRACT_SCHEMA)
def extract_udf(content_type, encoding, raw):
    import text_utils as tu

    try:
        (text, method), error = tu.extract_text_with_method(content_type, encoding, raw), None
    except Exception as e:  # keep the document, record why it has no text
        text, method, error = "", "failed", f"{type(e).__name__}: {e}"[:500]
    lang, prob = tu.detect_language(text)
    return text, error, method, lang, float(prob)


def make_chunk_udf(chunk_tokens: int, overlap_tokens: int):
    @F.udf(CHUNK_SCHEMA)
    def chunk_udf(text):
        import text_utils as tu

        return tu.chunk_text(text or "", chunk_tokens, overlap_tokens)
    return chunk_udf


@F.udf(T.IntegerType())
def words_udf(text):
    import text_utils as tu

    return tu.count_words(text or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdfs", default=os.environ.get("HDFS_URL", "hdfs://namenode:9000"))
    ap.add_argument("--format", default="delta", choices=["delta", "parquet"])
    ap.add_argument("--chunk-tokens", type=int, default=800)
    ap.add_argument("--overlap-tokens", type=int, default=100)
    ap.add_argument("--min-words", type=int, default=20, help="docs shorter than this are not chunked")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("bronze_to_silver_text").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    bronze_glob = f"{args.hdfs}/datalake/bronze/*/*/*.json"
    silver = f"{args.hdfs}/datalake/silver"

    bronze = (spark.read.schema(BRONZE_SCHEMA).json(bronze_glob)
              .withColumn("bronze_path", F.input_file_name()))
    # Defensive dedupe by id (the crawler already skips known ids): keep the latest copy.
    latest = Window.partitionBy("id").orderBy(F.col("retrieved_at").desc())
    bronze = bronze.withColumn("_rn", F.row_number().over(latest)).filter("_rn = 1").drop("_rn")

    docs = (bronze
            .withColumn("x", extract_udf("content_type", "content_encoding", "raw_content"))
            .select(
                F.col("id").alias("doc_id"), "source", "url", "title", "published_at",
                "retrieved_at", "content_type", "x.text", "x.extract_error", "x.extraction_method", "x.language",
                "x.language_prob", F.col("metadata").alias("metadata_json"), "bronze_path")
            .withColumn("n_chars", F.length("text"))
            .withColumn("n_words", words_udf("text"))
            .withColumn("n_tokens_est", F.ceil(F.col("n_words") / F.lit(0.75)).cast("int"))
            .withColumn("is_english", F.col("language") == "en")
            .withColumn("processed_at", F.current_timestamp())
            .persist())

    chunk_udf = make_chunk_udf(args.chunk_tokens, args.overlap_tokens)
    chunks = (docs.filter(F.col("is_english") & (F.col("n_words") >= args.min_words))
              .select("doc_id", "source", F.explode(chunk_udf("text")).alias("c"))
              .select(
                  F.concat_ws("_", "doc_id", F.lpad(F.col("c.chunk_index").cast("string"), 4, "0")).alias("chunk_id"),
                  "doc_id", "source", "c.chunk_index", "c.text", "c.char_start", "c.char_end",
                  "c.n_words", "c.n_tokens_est")
              .persist())

    def write(df, name):
        (df.coalesce(2).write.format(args.format).mode("overwrite")
         .option("overwriteSchema", "true").save(f"{silver}/{name}"))

    write(docs, "documents")
    write(chunks, "chunks")

    # ---- summary (read back from silver so the numbers are what was written)
    d = spark.read.format(args.format).load(f"{silver}/documents")
    c = spark.read.format(args.format).load(f"{silver}/chunks")
    print("\n=== silver documents per source ===")
    (d.groupBy("source")
     .agg(F.count("*").alias("docs"), F.sum(F.col("is_english").cast("int")).alias("english"),
          F.round(F.avg("n_words")).cast("int").alias("avg_words"),
          F.sum(F.col("extract_error").isNotNull().cast("int")).alias("extract_errors"))
     .orderBy("source").show(truncate=False))
    print("=== extraction method ===")
    d.groupBy("source", "extraction_method").count().orderBy("source").show(truncate=False)
    print("=== languages ===")
    d.groupBy("language").count().orderBy(F.desc("count")).show(10)
    print("=== silver chunks per source ===")
    (c.groupBy("source").agg(F.count("*").alias("chunks"),
                             F.round(F.avg("n_tokens_est")).cast("int").alias("avg_tokens_est"))
     .orderBy("source").show(truncate=False))
    print(f"SILVER_DOCUMENTS={d.count()}")
    print(f"SILVER_CHUNKS={c.count()}")
    print(f"FORMAT={args.format}")
    spark.stop()


if __name__ == "__main__":
    main()
