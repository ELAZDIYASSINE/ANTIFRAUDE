#!/usr/bin/env python3
"""
Robust ETL Pipeline for Anti-Fraud Detection System
Features: Delta Lake, Error Handling, Optimized UDFs, Distributed Transformations
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, sum as spark_sum, avg, when, lit, udf, broadcast
from pyspark.sql.types import FloatType, IntegerType, StringType
from pyspark.sql.utils import AnalysisException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ETLMetrics:
    """Metrics for ETL pipeline"""
    total_records: int = 0
    processed_records: int = 0
    error_records: int = 0
    fraud_records: int = 0
    processing_time: float = 0.0
    stages_completed: List[str] = None
    
    def __post_init__(self):
        if self.stages_completed is None:
            self.stages_completed = []


class ETLException(Exception):
    """Custom exception for ETL pipeline errors"""
    pass


class ETLPipeline:
    """Robust ETL Pipeline with Delta Lake and Error Handling"""
    
    def __init__(self, app_name: str = "AntiFraudETL"):
        """Initialize ETL Pipeline"""
        self.spark = None
        self.app_name = app_name
        self.metrics = ETLMetrics()
        self.raw_data: Optional[DataFrame] = None
        self.processed_data: Optional[DataFrame] = None
        
    def create_spark_session(self) -> SparkSession:
        """Create optimized Spark session with Delta Lake support"""
        try:
            logger.info("Creating Spark session with Delta Lake support...")
            
            self.spark = SparkSession.builder \
                .appName(self.app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.shuffle.partitions", "200") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.S3SingleDriverLogStore") \
                .getOrCreate()
            
            # Set log level
            self.spark.sparkContext.setLogLevel("WARN")
            
            logger.info("✅ Spark session created successfully")
            return self.spark
            
        except Exception as e:
            logger.error(f"❌ Failed to create Spark session: {e}")
            raise ETLException(f"Spark session creation failed: {e}")
    
    def load_raw_data(self, path: str) -> DataFrame:
        """Load raw data with error handling"""
        try:
            logger.info(f"Loading raw data from: {path}")
            start_time = time.time()
            
            if not os.path.exists(path):
                raise ETLException(f"Data file not found: {path}")
            
            # Load data with schema inference
            self.raw_data = self.spark.read.csv(
                path,
                header=True,
                inferSchema=True,
                mode="DROPMALFORMED"  # Drop malformed rows
            )
            
            load_time = time.time() - start_time
            self.metrics.total_records = self.raw_data.count()
            
            logger.info(f"✅ Data loaded: {self.metrics.total_records:,} records in {load_time:.2f}s")
            logger.info(f"Schema: {len(self.raw_data.columns)} columns")
            
            return self.raw_data
            
        except AnalysisException as e:
            logger.error(f"❌ Analysis error loading data: {e}")
            raise ETLException(f"Data loading analysis error: {e}")
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise ETLException(f"Data loading failed: {e}")
    
    def validate_data(self, df: DataFrame) -> bool:
        """Validate data quality"""
        try:
            logger.info("Validating data quality...")
            
            # Check for null values in critical columns
            critical_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'nameOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
            
            for col_name in critical_columns:
                null_count = df.filter(col(col_name).isNull()).count()
                if null_count > 0:
                    logger.warning(f"⚠️  Column '{col_name}' has {null_count:,} null values")
            
            # Check for negative amounts (should not exist)
            negative_amounts = df.filter(col('amount') < 0).count()
            if negative_amounts > 0:
                logger.warning(f"⚠️  Found {negative_amounts:,} records with negative amounts")
            
            # Check balance consistency
            inconsistent = df.filter(
                (col('type') == 'CASH_OUT') & 
                (col('newbalanceOrig') != col('oldbalanceOrg') - col('amount'))
            ).count()
            
            if inconsistent > 0:
                logger.warning(f"⚠️  Found {inconsistent:,} inconsistent CASH_OUT transactions")
            
            logger.info("✅ Data validation complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            return False
    
    # Optimized UDFs
    @staticmethod
    @udf(FloatType())
    def calculate_amount_ratio(amount: float, old_balance: float) -> float:
        """Calculate amount ratio with null safety"""
        if amount is None or old_balance is None or old_balance == 0:
            return 0.0
        return amount / old_balance
    
    @staticmethod
    @udf(IntegerType())
    def is_large_amount(amount: float, threshold: float = 100000) -> int:
        """Flag large transactions"""
        if amount is None:
            return 0
        return 1 if amount > threshold else 0
    
    @staticmethod
    @udf(StringType())
    def categorize_risk(amount: float, transaction_type: str) -> str:
        """Categorize transaction risk level"""
        if amount is None or transaction_type is None:
            return "LOW"
        
        # High risk indicators
        if amount > 100000 and transaction_type in ['TRANSFER', 'CASH_OUT']:
            return "HIGH"
        elif amount > 50000 and transaction_type in ['TRANSFER', 'CASH_OUT']:
            return "MEDIUM"
        else:
            return "LOW"
    
    def apply_feature_engineering(self, df: DataFrame) -> DataFrame:
        """Apply feature engineering with optimized UDFs"""
        try:
            logger.info("Applying feature engineering...")
            start_time = time.time()
            
            # Apply optimized UDFs
            df = df.withColumn(
                "amount_ratio",
                self.calculate_amount_ratio(col("amount"), col("oldbalanceOrg"))
            )
            
            df = df.withColumn(
                "is_large_amount",
                self.is_large_amount(col("amount"))
            )
            
            df = df.withColumn(
                "risk_category",
                self.categorize_risk(col("amount"), col("type"))
            )
            
            # Additional calculated features
            df = df.withColumn(
                "balance_change_orig",
                col("newbalanceOrig") - col("oldbalanceOrg")
            )
            
            df = df.withColumn(
                "balance_change_dest",
                col("newbalanceDest") - col("oldbalanceDest")
            )
            
            # One-hot encoding for transaction types
            transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
            for t in transaction_types:
                df = df.withColumn(f"type_{t}", when(col("type") == t, 1).otherwise(0))
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Feature engineering complete in {processing_time:.2f}s")
            logger.info(f"New columns: {len(df.columns) - 11} features added")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            raise ETLException(f"Feature engineering failed: {e}")
    
    def filter_fraud_transactions(self, df: DataFrame) -> DataFrame:
        """Filter and analyze fraud transactions"""
        try:
            logger.info("Filtering fraud transactions...")
            
            fraud_df = df.filter(col("isFraud") == 1)
            self.metrics.fraud_records = fraud_df.count()
            
            logger.info(f"✅ Found {self.metrics.fraud_records:,} fraud records")
            
            return fraud_df
            
        except Exception as e:
            logger.error(f"❌ Fraud filtering failed: {e}")
            raise ETLException(f"Fraud filtering failed: {e}")
    
    def save_to_delta_lake(self, df: DataFrame, path: str, mode: str = "overwrite") -> None:
        """Save DataFrame to Delta Lake with ACID properties"""
        try:
            logger.info(f"Saving to Delta Lake: {path}")
            start_time = time.time()
            
            # Write to Delta Lake with ACID properties
            df.write.format("delta") \
                .mode(mode) \
                .option("overwriteSchema", "true") \
                .partitionBy("type") \
                .save(path)
            
            save_time = time.time() - start_time
            logger.info(f"✅ Saved to Delta Lake in {save_time:.2f}s")
            
        except AnalysisException as e:
            logger.error(f"❌ Delta Lake save failed: {e}")
            raise ETLException(f"Delta Lake save failed: {e}")
        except Exception as e:
            logger.error(f"❌ Save operation failed: {e}")
            raise ETLException(f"Save operation failed: {e}")
    
    def run_pipeline(self, input_path: str, output_path: str) -> ETLMetrics:
        """Run complete ETL pipeline"""
        pipeline_start = time.time()
        
        try:
            logger.info("=" * 60)
            logger.info("🚀 STARTING ETL PIPELINE")
            logger.info("=" * 60)
            
            # Stage 1: Initialize Spark
            self.create_spark_session()
            self.metrics.stages_completed.append("Spark Initialization")
            
            # Stage 2: Load Data
            self.load_raw_data(input_path)
            self.metrics.stages_completed.append("Data Loading")
            
            # Stage 3: Validate Data
            if self.validate_data(self.raw_data):
                self.metrics.stages_completed.append("Data Validation")
            else:
                raise ETLException("Data validation failed")
            
            # Stage 4: Feature Engineering
            self.processed_data = self.apply_feature_engineering(self.raw_data)
            self.metrics.processed_records = self.processed_data.count()
            self.metrics.stages_completed.append("Feature Engineering")
            
            # Stage 5: Filter Fraud
            fraud_df = self.filter_fraud_transactions(self.processed_data)
            self.metrics.stages_completed.append("Fraud Filtering")
            
            # Stage 6: Save to Delta Lake
            self.save_to_delta_lake(self.processed_data, output_path)
            self.metrics.stages_completed.append("Delta Lake Save")
            
            # Calculate total processing time
            self.metrics.processing_time = time.time() - pipeline_start
            
            logger.info("=" * 60)
            logger.info("✅ ETL PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total Records: {self.metrics.total_records:,}")
            logger.info(f"Processed Records: {self.metrics.processed_records:,}")
            logger.info(f"Fraud Records: {self.metrics.fraud_records:,}")
            logger.info(f"Processing Time: {self.metrics.processing_time:.2f}s")
            logger.info(f"Stages Completed: {len(self.metrics.stages_completed)}/5")
            logger.info("=" * 60)
            
            return self.metrics
            
        except ETLException as e:
            logger.error(f"❌ ETL Pipeline Failed: {e}")
            self.metrics.error_records = -1  # Indicate failure
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected Error: {e}")
            raise ETLException(f"Unexpected error: {e}")
        finally:
            # Cleanup
            if self.spark:
                self.spark.stop()
                logger.info("✅ Spark session stopped")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.spark:
            try:
                self.spark.stop()
                logger.info("✅ Spark session stopped")
            except:
                pass


def main():
    """Main execution function"""
    # Configuration
    INPUT_PATH = "data/PS_20174392719_1491204439457_log.csv"
    OUTPUT_PATH = "data/processed/transactions_delta"
    
    try:
        # Initialize pipeline
        pipeline = ETLPipeline(app_name="AntiFraudETL")
        
        # Run pipeline
        metrics = pipeline.run_pipeline(INPUT_PATH, OUTPUT_PATH)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 ETL PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total Records: {metrics.total_records:,}")
        print(f"Processed Records: {metrics.processed_records:,}")
        print(f"Fraud Records: {metrics.fraud_records:,}")
        print(f"Processing Time: {metrics.processing_time:.2f}s")
        print(f"Stages Completed: {len(metrics.stages_completed)}/5")
        print(f"Success Rate: {(metrics.processed_records / metrics.total_records * 100):.1f}%")
        print("=" * 60)
        
        return 0
        
    except ETLException as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
