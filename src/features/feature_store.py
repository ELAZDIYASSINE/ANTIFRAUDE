#!/usr/bin/env python3
"""
Feature Store with Delta Lake for Anti-Fraud Detection
Features: Versioned feature tables, window functions, reproducible pipelines
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, std, max, min, 
    row_number, lag, when, lit, desc, asc
)
from pyspark.sql.types import FloatType, IntegerType, StringType, TimestampType
from pyspark.sql.utils import AnalysisException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for feature store"""
    feature_name: str
    feature_type: str
    description: str
    created_at: str
    version: int


class FeatureStore:
    """Feature Store with Delta Lake versioning"""
    
    def __init__(self, app_name: str = "FeatureStore"):
        """Initialize Feature Store"""
        self.spark = None
        self.app_name = app_name
        self.feature_tables: Dict[str, DataFrame] = {}
        self.feature_metadata: List[FeatureMetadata] = []
        
    def create_spark_session(self) -> SparkSession:
        """Create Spark session with Delta Lake support"""
        try:
            logger.info("Creating Spark session with Delta Lake...")
            
            self.spark = SparkSession.builder \
                .appName(self.app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("✅ Spark session created")
            return self.spark
            
        except Exception as e:
            logger.error(f"❌ Failed to create Spark session: {e}")
            raise
    
    def load_transaction_data(self, path: str) -> DataFrame:
        """Load transaction data"""
        logger.info(f"Loading transaction data from: {path}")
        
        df = self.spark.read.csv(path, header=True, inferSchema=True, mode="DROPMALFORMED")
        logger.info(f"✅ Loaded {df.count():,} transactions")
        
        return df
    
    def create_temporal_features(self, df: DataFrame) -> DataFrame:
        """Create temporal features using window functions"""
        logger.info("Creating temporal features with window functions...")
        start_time = time.time()
        
        # Window specification by account
        account_window = Window.partitionBy("nameOrig").orderBy("step")
        
        # Window specification by account with 10-step rolling window
        rolling_window = Window.partitionBy("nameOrig").orderBy("step").rowsBetween(-10, 0)
        
        # 1. Transaction count per account (last 10 steps)
        df = df.withColumn(
            "tx_count_last_10",
            count("*").over(rolling_window)
        )
        
        # 2. Total amount per account (last 10 steps)
        df = df.withColumn(
            "amount_sum_last_10",
            spark_sum("amount").over(rolling_window)
        )
        
        # 3. Average amount per account (last 10 steps)
        df = df.withColumn(
            "amount_avg_last_10",
            avg("amount").over(rolling_window)
        )
        
        # 4. Previous transaction amount
        df = df.withColumn(
            "prev_amount",
            lag("amount", 1).over(account_window)
        )
        
        # 5. Time since last transaction (in steps)
        df = df.withColumn(
            "time_since_last_tx",
            col("step") - lag("step", 1).over(account_window)
        )
        
        # 6. Amount change from previous transaction
        df = df.withColumn(
            "amount_change",
            when(col("prev_amount").isNull(), 0.0)
            .otherwise(col("amount") - col("prev_amount"))
        )
        
        # 7. Transaction frequency (transactions per step)
        df = df.withColumn(
            "tx_frequency",
            when(col("time_since_last_tx") == 0, 0.0)
            .otherwise(1.0 / col("time_since_last_tx"))
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Temporal features created in {processing_time:.2f}s")
        
        return df
    
    def create_aggregation_features(self, df: DataFrame) -> DataFrame:
        """Create aggregation features by transaction type"""
        logger.info("Creating aggregation features...")
        start_time = time.time()
        
        # Aggregate by transaction type and account
        type_account_window = Window.partitionBy("nameOrig", "type").orderBy("step")
        
        # Amount stats per transaction type
        df = df.withColumn(
            "type_amount_avg",
            avg("amount").over(type_account_window)
        )
        
        df = df.withColumn(
            "type_amount_max",
            max("amount").over(type_account_window)
        )
        
        df = df.withColumn(
            "type_amount_std",
            std("amount").over(type_account_window)
        )
        
        # Ratio to type average
        df = df.withColumn(
            "amount_to_type_avg_ratio",
            when(col("type_amount_avg") == 0, 0.0)
            .otherwise(col("amount") / col("type_amount_avg"))
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Aggregation features created in {processing_time:.2f}s")
        
        return df
    
    def create_risk_features(self, df: DataFrame) -> DataFrame:
        """Create risk-based features"""
        logger.info("Creating risk features...")
        start_time = time.time()
        
        # High-risk transaction types
        high_risk_types = ["TRANSFER", "CASH_OUT"]
        
        df = df.withColumn(
            "is_high_risk_type",
            when(col("type").isin(high_risk_types), 1).otherwise(0)
        )
        
        # Amount risk score
        df = df.withColumn(
            "amount_risk_score",
            when(col("amount") > 1000000, 1.0)
            .when(col("amount") > 500000, 0.8)
            .when(col("amount") > 100000, 0.6)
            .when(col("amount") > 50000, 0.4)
            .otherwise(0.2)
        )
        
        # Balance change risk
        df = df.withColumn(
            "balance_change_risk",
            when(abs(col("newbalanceOrig") - col("oldbalanceOrg")) > col("oldbalanceOrg") * 0.5, 1.0)
            .when(abs(col("newbalanceOrig") - col("oldbalanceOrg")) > col("oldbalanceOrg") * 0.3, 0.6)
            .otherwise(0.2)
        )
        
        # Combined risk score
        df = df.withColumn(
            "combined_risk_score",
            (col("is_high_risk_type") * 0.4 + 
             col("amount_risk_score") * 0.4 + 
             col("balance_change_risk") * 0.2)
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Risk features created in {processing_time:.2f}s")
        
        return df
    
    def create_feature_table(self, df: DataFrame, table_name: str, 
                           partition_cols: List[str] = None) -> None:
        """Create versioned feature table in Delta Lake"""
        logger.info(f"Creating feature table: {table_name}")
        start_time = time.time()
        
        if partition_cols is None:
            partition_cols = ["type"]
        
        # Write to Delta Lake with versioning
        df.write.format("delta") \
            .mode("overwrite") \
            .partitionBy(*partition_cols) \
            .option("overwriteSchema", "true") \
            .save(f"data/feature_store/{table_name}")
        
        # Register metadata
        metadata = FeatureMetadata(
            feature_name=table_name,
            feature_type="transaction_features",
            description=f"Feature table for {table_name}",
            created_at=datetime.now().isoformat(),
            version=1
        )
        self.feature_metadata.append(metadata)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Feature table {table_name} created in {processing_time:.2f}s")
        
        # Store in memory
        self.feature_tables[table_name] = df
    
    def load_feature_table(self, table_name: str) -> DataFrame:
        """Load feature table from Delta Lake"""
        logger.info(f"Loading feature table: {table_name}")
        
        path = f"data/feature_store/{table_name}"
        df = self.spark.read.format("delta").load(path)
        
        self.feature_tables[table_name] = df
        logger.info(f"✅ Feature table {table_name} loaded: {df.count():,} rows")
        
        return df
    
    def get_feature_version(self, table_name: str) -> int:
        """Get current version of feature table"""
        for metadata in self.feature_metadata:
            if metadata.feature_name == table_name:
                return metadata.version
        return 0
    
    def build_feature_pipeline(self, input_path: str, feature_table_name: str) -> DataFrame:
        """Build complete feature pipeline"""
        logger.info("Building complete feature pipeline...")
        
        # Load data
        df = self.load_transaction_data(input_path)
        
        # Create features
        df = self.create_temporal_features(df)
        df = self.create_aggregation_features(df)
        df = self.create_risk_features(df)
        
        # Save to feature store
        self.create_feature_table(df, feature_table_name)
        
        return df
    
    def get_feature_statistics(self, table_name: str) -> DataFrame:
        """Get feature statistics for monitoring"""
        logger.info(f"Computing feature statistics for {table_name}")
        
        df = self.feature_tables.get(table_name)
        if df is None:
            df = self.load_feature_table(table_name)
        
        # Compute statistics
        stats = df.describe()
        
        logger.info("✅ Feature statistics computed")
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        if self.spark:
            self.spark.stop()
            logger.info("✅ Spark session stopped")


def main():
    """Main execution"""
    INPUT_PATH = "data/PS_20174392719_1491204439457_log.csv"
    FEATURE_TABLE = "transaction_features"
    
    try:
        feature_store = FeatureStore()
        feature_store.create_spark_session()
        
        # Build feature pipeline
        df = feature_store.build_feature_pipeline(INPUT_PATH, FEATURE_TABLE)
        
        # Get statistics
        stats = feature_store.get_feature_statistics(FEATURE_TABLE)
        stats.show(truncate=False)
        
        print("\n" + "=" * 60)
        print("✅ FEATURE STORE PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Total features: {len(df.columns)}")
        print(f"Total records: {df.count():,}")
        print(f"Feature table: {FEATURE_TABLE}")
        print("=" * 60)
        
        feature_store.cleanup()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Feature pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
