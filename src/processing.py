"""
=============================================================
  Scalable AI Data Pipeline - Sentiment Analysis
  Architecture: Medallion (Bronze -> Gold) on AWS S3
  Engine: PySpark + Pandas UDFs + HuggingFace Transformers
=============================================================
"""

import os
import logging

# ── 1. Standard library & AWS ────────────────────────────────────────────────
import boto3

# ── 2. PySpark core ──────────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, FloatType, StructType, StructField

# ── 3. Pandas UDF support ────────────────────────────────────────────────────
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType

# ── 4. HuggingFace ───────────────────────────────────────────────────────────
from transformers import pipeline

# =============================================================================
# CONFIGURATION  (all secrets come from environment variables — never hardcode!)
# =============================================================================

AWS_ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION       = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET        = os.getenv("S3_BUCKET_NAME", "my-ai-pipeline-bucket")

# Medallion layer paths
BRONZE_PATH = f"s3a://{S3_BUCKET}/bronze/reviews/"   # raw input data
GOLD_PATH   = f"s3a://{S3_BUCKET}/gold/sentiment/"   # AI-enriched output

# HuggingFace model ID
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# STEP 1 — Build the SparkSession (the "engine" of our pipeline)
# =============================================================================

def create_spark_session() -> SparkSession:
    """
    Creates a SparkSession pre-configured for:
      - Delta Lake  (ACID transactions on top of S3 Parquet)
      - S3 access   (hadoop-aws + AWS credentials)
    """
    log.info("🔧  Building SparkSession …")

    spark = (
        SparkSession.builder
        .appName("AI-Sentiment-Pipeline")
        .config("spark.jars.packages",
                "io.delta:delta-spark_2.12:3.2.0,"
                "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # ── Pass HuggingFace env vars to workers ────────────────────────────
        .config("spark.executorEnv.HF_HOME", os.getenv("HF_HOME", "/model_cache"))
        .config("spark.executorEnv.TRANSFORMERS_OFFLINE", "1")
        .config("spark.executorEnv.HF_DATASETS_OFFLINE", "1")
        # ── S3 credentials ──────────────────────────────────────────────────
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY or "")
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY or "")
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_REGION}.amazonaws.com")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.master", "local[1]")  # ← only 1 worker process
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")   # quieten Spark's own verbose logs
    log.info("✅  SparkSession ready.")
    return spark


# =============================================================================
# STEP 2 — The Pandas UDF  (the "Pro" magic)
# =============================================================================
#
#  Why Pandas UDF?
#  ───────────────
#  A regular Python UDF processes ONE row at a time → slow (row-by-row overhead).
#  A Pandas UDF receives a whole *batch* (pandas.Series) at once → fast (vectorised).
#  The AI model then scores the whole batch in a single forward pass → GPU/CPU friendly.
#
#  The @pandas_udf decorator tells Spark:
#    "This function eats a pd.Series and returns a pd.Series of StructType rows."

# Return schema for our UDF: each row gets a label + a confidence score
SENTIMENT_SCHEMA = StructType([
    StructField("label", StringType(),  nullable=False),
    StructField("score", FloatType(),   nullable=False),
])

@pandas_udf(SENTIMENT_SCHEMA)
def sentiment_udf(texts: pd.Series) -> pd.DataFrame:
    import os
    from transformers import pipeline as hf_pipeline

    global _sentiment_pipeline

    if "_sentiment_pipeline" not in globals():
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

        _sentiment_pipeline = hf_pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
            batch_size=8,   # ← reduced from 64 to save memory
        )

    results = _sentiment_pipeline(texts.tolist())
    return pd.DataFrame({
        "label": [r["label"] for r in results],
        "score": [float(r["score"]) for r in results],
    })


# =============================================================================
# STEP 3 — Bronze layer  (ingest raw data)
# =============================================================================

def read_bronze(spark: SparkSession) -> "pyspark.sql.DataFrame":
    """
    Reads raw CSV/JSON reviews from the Bronze S3 path.
    Expects at least a column called `review_text`.
    """
    log.info(f"📥  Reading Bronze data from: {BRONZE_PATH}")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(BRONZE_PATH)               # swap for .json() if your source is JSON
    )

    # Light cleaning: drop nulls and empty strings in the text column
    df = (
        df.filter(F.col("review_text").isNotNull())
          .filter(F.length(F.trim(F.col("review_text"))) > 0)
    )

    log.info(f"📊  Bronze row count: {df.count():,}")
    return df


# =============================================================================
# STEP 4 — Gold layer  (apply AI + write enriched data)
# =============================================================================

def write_gold(df: "pyspark.sql.DataFrame") -> None:
    log.info("🤖  Running sentiment inference on driver …")

    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    from transformers import pipeline as hf_pipeline

    rows = df.collect()
    texts = [row["review_text"] for row in rows]

    p = hf_pipeline(
        task="sentiment-analysis",
        model=MODEL_ID,
        truncation=True,
        max_length=512,
        batch_size=8
    )
    results = p(texts)

    # Build pandas DataFrame — no Spark workers involved at all
    pandas_df = pd.DataFrame({
        "review_text":     texts,
        "sentiment_label": [r["label"] for r in results],
        "sentiment_score": [float(r["score"]) for r in results],
        "processed_at":    pd.Timestamp.now(),
    })

    # Add any remaining original columns
    for field in df.schema.fields:
        if field.name not in pandas_df.columns:
            pandas_df[field.name] = [row[field.name] for row in rows]

    # Write directly with pandas — zero Spark workers
    os.makedirs(GOLD_PATH, exist_ok=True)
    out_path = os.path.join(GOLD_PATH, "output.parquet")
    pandas_df.to_parquet(out_path, index=False)

    log.info("✅  Gold layer written successfully.")
    log.info(f"\n{pandas_df.to_string()}")
# =============================================================================
# STEP 5 — Entry point
# =============================================================================

def main():
    log.info("🚀  Pipeline starting …")

    spark = create_spark_session()

    try:
        bronze_df = read_bronze(spark)
        write_gold(bronze_df)
        log.info("🎉  Pipeline complete!")
    except Exception as exc:
        log.error(f"💥  Pipeline failed: {exc}", exc_info=True)
        raise
    finally:
        spark.stop()
        log.info("🔌  SparkSession stopped.")


if __name__ == "__main__":
    main()
