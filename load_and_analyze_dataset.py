from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, avg, max as spark_max, min as spark_min
import os

def load_and_analyze_paysim_dataset():
    """Load PaySim dataset with PySpark and perform basic analysis"""
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("PaySim Fraud Detection") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    print("Spark session created successfully!")
    print(f"Spark version: {spark.version}")
    
    # Load the dataset
    dataset_path = "data/PS_20174392719_1491204439457_log.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return None
    
    print(f"\nLoading dataset from: {dataset_path}")
    df = spark.read.csv(dataset_path, header=True, inferSchema=True)
    
    # Show schema
    print("\n=== DATASET SCHEMA ===")
    df.printSchema()
    
    # Show basic info
    print(f"\n=== BASIC INFO ===")
    print(f"Number of rows: {df.count():,}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Columns: {df.columns}")
    
    # Show first few rows
    print("\n=== FIRST 10 ROWS ===")
    df.show(10, truncate=False)
    
    
    # Fraud analysis
    print("\n=== FRAUD ANALYSIS ===")
    fraud_stats = df.groupBy("isFraud").agg(
        count("*").alias("transaction_count"),
        (count("*") / df.count() * 100).alias("percentage")
    ).orderBy("isFraud")
    
    fraud_stats.show()
    
    # Transaction type analysis
    print("\n=== TRANSACTION TYPE ANALYSIS ===")
    type_stats = df.groupBy("type").agg(
        count("*").alias("transaction_count"),
        avg("amount").alias("avg_amount"),
        spark_sum("amount").alias("total_amount"),
        spark_sum("isFraud").alias("fraud_count")
    ).orderBy("transaction_count", ascending=False)
    
    type_stats.show(truncate=False)
    
    # Amount statistics by fraud status
    print("\n=== AMOUNT STATISTICS BY FRAUD STATUS ===")
    amount_stats = df.groupBy("isFraud").agg(
        avg("amount").alias("avg_amount"),
        spark_max("amount").alias("max_amount"),
        spark_min("amount").alias("min_amount"),
        spark_sum("amount").alias("total_amount")
    ).orderBy("isFraud")
    
    amount_stats.show(truncate=False)
    
    # Flagged fraud analysis
    print("\n=== FLAGGED FRAUD ANALYSIS ===")
    flagged_stats = df.groupBy("isFlaggedFraud").agg(
        count("*").alias("transaction_count"),
        (count("*") / df.count() * 100).alias("percentage")
    ).orderBy("isFlaggedFraud")
    
    flagged_stats.show()
    
    print("\n=== ANALYSIS COMPLETE ===")
    return df, spark

if __name__ == "__main__":
    try:
        df, spark = load_and_analyze_paysim_dataset()
        if spark:
            spark.stop()
            print("Spark session stopped.")
    except Exception as e:
        print(f"Error during analysis: {e}")
