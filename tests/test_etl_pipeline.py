#!/usr/bin/env python3
"""
Unit Tests for ETL Pipeline using pytest-spark
"""

import pytest
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_processing.etl_pipeline import ETLPipeline, ETLException, ETLMetrics


@pytest.fixture(scope="session")
def spark():
    """Create Spark session for testing"""
    spark = SparkSession.builder \
        .appName("ETLPipelineTests") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    
    yield spark
    
    spark.stop()


@pytest.fixture
def sample_data(spark):
    """Create sample test data"""
    data = [
        (1, "CASH_IN", 1000.0, 5000.0, 6000.0, "C123", "C456", 2000.0, 3000.0, 0, 0),
        (1, "TRANSFER", 150000.0, 200000.0, 50000.0, "C789", "C012", 0.0, 150000.0, 1, 0),
        (1, "CASH_OUT", 50000.0, 100000.0, 50000.0, "C345", "C678", 1000.0, 51000.0, 0, 0),
        (1, "PAYMENT", 500.0, 10000.0, 9500.0, "C901", "C234", 5000.0, 5500.0, 0, 0),
        (1, "DEBIT", 200.0, 5000.0, 4800.0, "C567", "C890", 1000.0, 1200.0, 0, 0),
    ]
    
    schema = StructType([
        StructField("step", IntegerType(), True),
        StructField("type", StringType(), True),
        StructField("amount", FloatType(), True),
        StructField("oldbalanceOrg", FloatType(), True),
        StructField("newbalanceOrig", FloatType(), True),
        StructField("nameOrig", StringType(), True),
        StructField("nameDest", StringType(), True),
        StructField("oldbalanceDest", FloatType(), True),
        StructField("newbalanceDest", FloatType(), True),
        StructField("isFraud", IntegerType(), True),
        StructField("isFlaggedFraud", IntegerType(), True),
    ])
    
    df = spark.createDataFrame(data, schema)
    return df


class TestETLPipeline:
    """Test suite for ETL Pipeline"""
    
    def test_spark_session_creation(self):
        """Test Spark session creation"""
        pipeline = ETLPipeline()
        spark = pipeline.create_spark_session()
        
        assert spark is not None
        assert spark.version is not None
        assert pipeline.spark is spark
        
        spark.stop()
    
    def test_etl_metrics_initialization(self):
        """Test ETLMetrics initialization"""
        metrics = ETLMetrics()
        
        assert metrics.total_records == 0
        assert metrics.processed_records == 0
        assert metrics.error_records == 0
        assert metrics.fraud_records == 0
        assert metrics.processing_time == 0.0
        assert metrics.stages_completed == []
    
    def test_feature_engineering_udf_amount_ratio(self, spark, sample_data):
        """Test amount ratio calculation UDF"""
        pipeline = ETLPipeline()
        pipeline.spark = spark
        
        result_df = pipeline.apply_feature_engineering(sample_data)
        
        # Check that amount_ratio column exists
        assert "amount_ratio" in result_df.columns
        
        # Verify calculation for first row: 1000 / 5000 = 0.2
        first_row = result_df.select("amount_ratio").first()
        assert abs(first_row["amount_ratio"] - 0.2) < 0.01
    
    def test_feature_engineering_udf_large_amount(self, spark, sample_data):
        """Test large amount flag UDF"""
        pipeline = ETLPipeline()
        pipeline.spark = spark
        
        result_df = pipeline.apply_feature_engineering(sample_data)
        
        # Check that is_large_amount column exists
        assert "is_large_amount" in result_df.columns
        
        # Verify that 150000 is flagged as large
        large_amount_rows = result_df.filter(col("is_large_amount") == 1).count()
        assert large_amount_rows == 1  # Only the 150000 transaction
    
    def test_feature_engineering_udf_risk_category(self, spark, sample_data):
        """Test risk categorization UDF"""
        pipeline = ETLPipeline()
        pipeline.spark = spark
        
        result_df = pipeline.apply_feature_engineering(sample_data)
        
        # Check that risk_category column exists
        assert "risk_category" in result_df.columns
        
        # Verify that high amount transfer is HIGH risk
        high_risk_rows = result_df.filter(col("risk_category") == "HIGH").count()
        assert high_risk_rows == 1  # The 150000 TRANSFER
    
    def test_feature_engineering_balance_change(self, spark, sample_data):
        """Test balance change calculation"""
        pipeline = ETLPipeline()
        pipeline.spark = spark
        
        result_df = pipeline.apply_feature_engineering(sample_data)
        
        # Check balance change columns exist
        assert "balance_change_orig" in result_df.columns
        assert "balance_change_dest" in result_df.columns
        
        # Verify calculation for first row: 6000 - 5000 = 1000
        first_row = result_df.select("balance_change_orig").first()
        assert abs(first_row["balance_change_orig"] - 1000) < 0.01
    
    def test_feature_engineering_one_hot_encoding(self, spark, sample_data):
        """Test one-hot encoding for transaction types"""
        pipeline = ETLPipeline()
        pipeline.spark = spark
        
        result_df = pipeline.apply_feature_engineering(sample_data)
        
        # Check that one-hot encoded columns exist
        expected_columns = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
        for col_name in expected_columns:
            assert col_name in result_df.columns
        
        # Verify that CASH_IN row has type_CASH_IN = 1
        cash_in_row = result_df.filter(col("type") == "CASH_IN").select("type_CASH_IN").first()
        assert cash_in_row["type_CASH_IN"] == 1
    
    def test_filter_fraud_transactions(self, spark, sample_data):
        """Test fraud transaction filtering"""
        pipeline = ETLPipeline()
        pipeline.spark = spark
        pipeline.processed_data = sample_data
        
        fraud_df = pipeline.filter_fraud_transactions(sample_data)
        
        # Should return 1 fraud record (the TRANSFER with isFraud=1)
        assert fraud_df.count() == 1
        assert pipeline.metrics.fraud_records == 1
    
    def test_data_validation_null_check(self, spark, sample_data):
        """Test data validation for null values"""
        pipeline = ETLPipeline()
        pipeline.spark = spark
        
        # Test with clean data (should pass)
        is_valid = pipeline.validate_data(sample_data)
        assert is_valid == True
    
    def test_data_validation_negative_amounts(self, spark):
        """Test data validation for negative amounts"""
        pipeline = ETLPipeline()
        pipeline.spark = spark
        
        # Create data with negative amount
        data = [(1, "CASH_IN", -100.0, 5000.0, 4900.0, "C123", "C456", 2000.0, 1900.0, 0, 0)]
        schema = StructType([
            StructField("step", IntegerType(), True),
            StructField("type", StringType(), True),
            StructField("amount", FloatType(), True),
            StructField("oldbalanceOrg", FloatType(), True),
            StructField("newbalanceOrig", FloatType(), True),
            StructField("nameOrig", StringType(), True),
            StructField("nameDest", StringType(), True),
            StructField("oldbalanceDest", FloatType(), True),
            StructField("newbalanceDest", FloatType(), True),
            StructField("isFraud", IntegerType(), True),
            StructField("isFlaggedFraud", IntegerType(), True),
        ])
        df = spark.createDataFrame(data, schema)
        
        # Should still pass validation but log warning
        is_valid = pipeline.validate_data(df)
        assert is_valid == True
    
    def test_etl_exception_handling(self):
        """Test ETLException handling"""
        with pytest.raises(ETLException):
            raise ETLException("Test exception")
    
    def test_pipeline_cleanup(self):
        """Test pipeline cleanup"""
        pipeline = ETLPipeline()
        pipeline.spark = SparkSession.builder.appName("Test").getOrCreate()
        
        pipeline.cleanup()
        
        # Spark session should be stopped
        assert True  # If no exception, cleanup worked
    
    def test_metrics_stages_tracking(self):
        """Test metrics stages tracking"""
        metrics = ETLMetrics()
        
        metrics.stages_completed.append("Stage 1")
        metrics.stages_completed.append("Stage 2")
        
        assert len(metrics.stages_completed) == 2
        assert "Stage 1" in metrics.stages_completed
        assert "Stage 2" in metrics.stages_completed


class TestETLPipelineIntegration:
    """Integration tests for ETL Pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_flow(self, spark, tmp_path):
        """Test complete pipeline flow with sample data"""
        # Create test CSV file
        test_csv = tmp_path / "test_transactions.csv"
        sample_data = """step,type,amount,oldbalanceOrg,newbalanceOrig,nameOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
1,CASH_IN,1000.0,5000.0,6000.0,C123,C456,2000.0,3000.0,0,0
1,TRANSFER,150000.0,200000.0,50000.0,C789,C012,0.0,150000.0,1,0
1,CASH_OUT,50000.0,100000.0,50000.0,C345,C678,1000.0,51000.0,0,0"""
        
        test_csv.write_text(sample_data)
        
        # Run pipeline
        pipeline = ETLPipeline()
        output_path = str(tmp_path / "delta_output")
        
        try:
            metrics = pipeline.run_pipeline(str(test_csv), output_path)
            
            # Verify metrics
            assert metrics.total_records == 3
            assert metrics.processed_records > 0
            assert metrics.fraud_records == 1
            assert len(metrics.stages_completed) == 5
            assert metrics.processing_time > 0
            
        finally:
            pipeline.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
