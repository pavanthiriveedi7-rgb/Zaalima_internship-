"""
Week 1 - Day 1-3: Distributed Data Processing & Feature Engineering
Project 3: Context-Aware Neural Recommendation Engine
Zaalima Development
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType
from pyspark.sql.window import Window
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "HMRecommendationEngine") -> SparkSession:
    """Initialize a production-grade Spark session."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.memory.fraction", "0.8")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully.")
    return spark


def load_hm_datasets(spark: SparkSession, data_path: str):
    """
    Load H&M datasets: articles, customers, transactions.
    Expected files: articles.csv, customers.csv, transactions_train.csv
    """
    logger.info(f"Loading datasets from: {data_path}")

    # --- Articles (Items) ---
    articles_schema = StructType([
        StructField("article_id", StringType(), False),
        StructField("product_code", StringType(), True),
        StructField("prod_name", StringType(), True),
        StructField("product_type_no", IntegerType(), True),
        StructField("product_type_name", StringType(), True),
        StructField("product_group_name", StringType(), True),
        StructField("graphical_appearance_no", IntegerType(), True),
        StructField("graphical_appearance_name", StringType(), True),
        StructField("colour_group_code", StringType(), True),
        StructField("colour_group_name", StringType(), True),
        StructField("perceived_colour_value_id", IntegerType(), True),
        StructField("perceived_colour_value_name", StringType(), True),
        StructField("perceived_colour_master_id", IntegerType(), True),
        StructField("perceived_colour_master_name", StringType(), True),
        StructField("department_no", IntegerType(), True),
        StructField("department_name", StringType(), True),
        StructField("index_code", StringType(), True),
        StructField("index_name", StringType(), True),
        StructField("index_group_no", IntegerType(), True),
        StructField("index_group_name", StringType(), True),
        StructField("section_no", IntegerType(), True),
        StructField("section_name", StringType(), True),
        StructField("garment_group_no", IntegerType(), True),
        StructField("garment_group_name", StringType(), True),
        StructField("detail_desc", StringType(), True),
    ])

    articles_df = spark.read.csv(
        os.path.join(data_path, "articles.csv"),
        header=True, schema=articles_schema
    )

    # --- Customers (Users) ---
    customers_df = spark.read.csv(
        os.path.join(data_path, "customers.csv"),
        header=True, inferSchema=True
    )

    # --- Transactions ---
    transactions_df = spark.read.csv(
        os.path.join(data_path, "transactions_train.csv"),
        header=True, inferSchema=True
    )
    transactions_df = transactions_df.withColumn("t_dat", F.to_timestamp("t_dat", "yyyy-MM-dd"))

    logger.info(f"Articles: {articles_df.count()} | Customers: {customers_df.count()} | Transactions: {transactions_df.count()}")
    return articles_df, customers_df, transactions_df


def handle_missing_values(customers_df, transactions_df):
    """Handle missing values and cold-start scenarios."""
    logger.info("Handling missing values...")

    # Customers: fill missing age with median, fill missing club_member_status
    median_age = customers_df.approxQuantile("age", [0.5], 0.01)[0]
    customers_df = customers_df.fillna({
        "age": median_age,
        "club_member_status": "UNKNOWN",
        "fashion_news_frequency": "NONE",
        "FN": 0,
        "Active": 0,
    })

    # Cold-start: tag customers with < 2 transactions (new users)
    user_tx_counts = transactions_df.groupBy("customer_id").agg(
        F.count("article_id").alias("tx_count")
    )
    customers_df = customers_df.join(user_tx_counts, on="customer_id", how="left")
    customers_df = customers_df.fillna({"tx_count": 0})
    customers_df = customers_df.withColumn(
        "is_cold_start", F.when(F.col("tx_count") < 2, 1).otherwise(0)
    )

    logger.info(f"Cold-start users: {customers_df.filter(F.col('is_cold_start') == 1).count()}")
    return customers_df, transactions_df


def engineer_contextual_features(transactions_df, articles_df):
    """
    Day 4-6: Engineer contextual features.
    - time_of_purchase (hour/day)
    - recency (days since last purchase per user)
    - product popularity over time
    """
    logger.info("Engineering contextual features...")

    # --- Time of purchase features ---
    transactions_df = transactions_df.withColumn("purchase_hour", F.hour("t_dat"))
    transactions_df = transactions_df.withColumn("purchase_dayofweek", F.dayofweek("t_dat"))
    transactions_df = transactions_df.withColumn("purchase_month", F.month("t_dat"))
    transactions_df = transactions_df.withColumn("purchase_year", F.year("t_dat"))

    # --- Recency: days since this purchase vs user's last purchase ---
    user_window = Window.partitionBy("customer_id").orderBy("t_dat")
    transactions_df = transactions_df.withColumn(
        "prev_purchase_date", F.lag("t_dat", 1).over(user_window)
    )
    transactions_df = transactions_df.withColumn(
        "days_since_last_purchase",
        F.when(
            F.col("prev_purchase_date").isNotNull(),
            F.datediff(F.col("t_dat"), F.col("prev_purchase_date"))
        ).otherwise(0)
    )

    # --- Product Popularity: rolling 30-day purchase count per article ---
    article_popularity = transactions_df.groupBy("article_id", "purchase_month", "purchase_year").agg(
        F.count("customer_id").alias("monthly_purchase_count")
    )
    transactions_df = transactions_df.join(
        article_popularity,
        on=["article_id", "purchase_month", "purchase_year"],
        how="left"
    ).fillna({"monthly_purchase_count": 0})

    # --- Join article metadata onto transactions ---
    transactions_df = transactions_df.join(
        articles_df.select("article_id", "product_group_name", "index_group_name", "colour_group_name"),
        on="article_id",
        how="left"
    )

    logger.info("Contextual features engineered.")
    return transactions_df


def define_vocabularies(transactions_df, articles_df, customers_df):
    """
    Day 7: Define user and item feature vocabularies for embedding generation.
    Returns dicts of unique values for each categorical feature.
    """
    logger.info("Defining feature vocabularies...")

    user_vocab = {
        "customer_id": [row["customer_id"] for row in customers_df.select("customer_id").distinct().collect()],
        "club_member_status": [row["club_member_status"] for row in customers_df.select("club_member_status").distinct().collect()],
        "fashion_news_frequency": [row["fashion_news_frequency"] for row in customers_df.select("fashion_news_frequency").distinct().collect()],
    }

    item_vocab = {
        "article_id": [row["article_id"] for row in articles_df.select("article_id").distinct().collect()],
        "product_group_name": [row["product_group_name"] for row in articles_df.select("product_group_name").distinct().collect()],
        "colour_group_name": [row["colour_group_name"] for row in articles_df.select("colour_group_name").distinct().collect()],
        "index_group_name": [row["index_group_name"] for row in articles_df.select("index_group_name").distinct().collect()],
    }

    logger.info(f"Users in vocab: {len(user_vocab['customer_id'])} | Items in vocab: {len(item_vocab['article_id'])}")
    return user_vocab, item_vocab


def save_processed_data(transactions_df, customers_df, articles_df, output_path: str):
    """Save processed data as parquet for downstream use."""
    logger.info(f"Saving processed data to {output_path}")
    transactions_df.write.mode("overwrite").parquet(os.path.join(output_path, "transactions_processed"))
    customers_df.write.mode("overwrite").parquet(os.path.join(output_path, "customers_processed"))
    articles_df.write.mode("overwrite").parquet(os.path.join(output_path, "articles_processed"))
    logger.info("Data saved successfully.")


if __name__ == "__main__":
    DATA_PATH = "./data/raw"
    OUTPUT_PATH = "./data/processed"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    spark = create_spark_session()
    articles_df, customers_df, transactions_df = load_hm_datasets(spark, DATA_PATH)
    customers_df, transactions_df = handle_missing_values(customers_df, transactions_df)
    transactions_df = engineer_contextual_features(transactions_df, articles_df)
    user_vocab, item_vocab = define_vocabularies(transactions_df, articles_df, customers_df)
    save_processed_data(transactions_df, customers_df, articles_df, OUTPUT_PATH)

    # Save vocabularies
    import json
    os.makedirs("./data/vocab", exist_ok=True)
    with open("./data/vocab/user_vocab.json", "w") as f:
        json.dump({k: list(v) for k, v in user_vocab.items()}, f)
    with open("./data/vocab/item_vocab.json", "w") as f:
        json.dump({k: list(v) for k, v in item_vocab.items()}, f)

    logger.info("Week 1 pipeline complete.")
    spark.stop()
