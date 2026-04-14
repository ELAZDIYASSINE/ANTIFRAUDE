#!/usr/bin/env python3
"""
PySpark Structured Streaming with Kafka for Real-time Fraud Detection
Features: Real-time transaction processing, streaming predictions, Kafka integration
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, to_json, struct, when, lit
from pyspark.sql.types import StructType, StringType, DoubleType, IntegerType, TimestampType
from pyspark.ml import PipelineModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KafkaStreamingPipeline:
    """Kafka Structured Streaming Pipeline for Real-time Fraud Detection"""
    
    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092"):
        """Initialize Kafka streaming pipeline"""
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.spark = None
        self.model = None
        self.query = None
        
    def create_spark_session(self) -> SparkSession:
        """Create Spark session with Kafka support"""
        logger.info("Creating Spark session with Kafka support...")
        
        self.spark = SparkSession.builder \
            .appName("KafkaFraudDetection") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.streaming.checkpointLocation", "data/checkpoint") \
            .config("spark.jars.packages", 
                   "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("✅ Spark session with Kafka support created")
        
        return self.spark
    
    def define_transaction_schema(self) -> StructType:
        """Define schema for transaction data"""
        return StructType() \
            .add("step", IntegerType()) \
            .add("type", StringType()) \
            .add("amount", DoubleType()) \
            .add("oldbalanceOrg", DoubleType()) \
            .add("newbalanceOrig", DoubleType()) \
            .add("nameOrig", StringType()) \
            .add("nameDest", StringType()) \
            .add("oldbalanceDest", DoubleType()) \
            .add("newbalanceDest", DoubleType()) \
            .add("isFraud", IntegerType()) \
            .add("timestamp", TimestampType())
    
    def read_from_kafka(self, topic: str = "transactions") -> DataFrame:
        """Read transaction stream from Kafka"""
        logger.info(f"Reading from Kafka topic: {topic}")
        
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON value
        schema = self.define_transaction_schema()
        
        df_parsed = df \
            .select(from_json(col("value").cast("string"), schema).alias("data")) \
            .select("data.*")
        
        logger.info("✅ Kafka stream reader created")
        
        return df_parsed
    
    def preprocess_streaming_data(self, df: DataFrame) -> DataFrame:
        """Preprocess streaming data for prediction"""
        logger.info("Preprocessing streaming data...")
        
        # Add features
        df = df.withColumn('amount_ratio', 
                         when(col('oldbalanceOrg') > 0,
                              col('amount') / col('oldbalanceOrg')).otherwise(0.0))
        
        df = df.withColumn('balance_change_orig',
                         col('oldbalanceOrg') - col('newbalanceOrig'))
        
        df = df.withColumn('balance_change_dest',
                         col('newbalanceDest') - col('oldbalanceDest'))
        
        df = df.withColumn('is_large_amount',
                         when(col('amount') > 100000, 1).otherwise(0))
        
        return df
    
    def load_model(self, model_path: str = "models/fraud_detection_rf"):
        """Load trained model for streaming predictions"""
        logger.info(f"Loading model from: {model_path}")
        
        try:
            self.model = PipelineModel.load(model_path)
            logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Model not found, using mock predictions: {e}")
            return False
    
    def predict_streaming(self, df: DataFrame) -> DataFrame:
        """Make predictions on streaming data"""
        logger.info("Setting up streaming predictions...")
        
        if self.model:
            # Use loaded model
            predictions = self.model.transform(df)
        else:
            # Mock predictions
            predictions = df.withColumn(
                "fraud_probability",
                when(col('amount') > 100000, 0.8)
                .when(col('amount') > 50000, 0.5)
                .otherwise(0.1)
            ).withColumn(
                "prediction",
                when(col('fraud_probability') > 0.5, 1).otherwise(0)
            )
        
        # Add timestamp
        predictions = predictions.withColumn(
            "prediction_timestamp",
            lit(datetime.now().isoformat())
        )
        
        return predictions
    
    def write_to_kafka(self, df: DataFrame, topic: str = "fraud_alerts"):
        """Write fraud alerts to Kafka"""
        logger.info(f"Writing fraud alerts to Kafka topic: {topic}")
        
        # Select and format output
        output_df = df.select(
            to_json(struct(
                col("nameOrig").alias("transaction_id"),
                col("amount"),
                col("type"),
                col("fraud_probability"),
                col("prediction"),
                col("prediction_timestamp")
            )).alias("value")
        )
        
        query = output_df \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("topic", topic) \
            .option("checkpointLocation", "data/checkpoint/kafka_output") \
            .start()
        
        return query
    
    def write_to_console(self, df: DataFrame):
        """Write predictions to console for debugging"""
        logger.info("Writing predictions to console...")
        
        query = df \
            .writeStream \
            .format("console") \
            .option("truncate", "false") \
            .option("numRows", 10) \
            .start()
        
        return query
    
    def write_to_memory(self, df: DataFrame):
        """Write predictions to memory for dashboard"""
        logger.info("Writing predictions to memory...")
        
        query = df \
            .writeStream \
            .format("memory") \
            .queryName("fraud_predictions") \
            .outputMode("append") \
            .start()
        
        return query
    
    def start_streaming_pipeline(self, kafka_topic: str = "transactions"):
        """Start complete streaming pipeline"""
        logger.info("Starting streaming pipeline...")
        
        # Read from Kafka
        stream_df = self.read_from_kafka(kafka_topic)
        
        # Preprocess
        preprocessed_df = self.preprocess_streaming_data(stream_df)
        
        # Predict
        predictions_df = self.predict_streaming(preprocessed_df)
        
        # Filter fraud alerts
        fraud_alerts = predictions_df.filter(col("prediction") == 1)
        
        # Write fraud alerts to Kafka
        self.write_to_kafka(fraud_alerts, "fraud_alerts")
        
        # Write all predictions to memory for dashboard
        self.write_to_memory(predictions_df)
        
        # Also write to console for monitoring
        self.write_to_console(fraud_alerts)
        
        logger.info("✅ Streaming pipeline started")
        
        return predictions_df
    
    def await_termination(self, timeout: Optional[int] = None):
        """Await streaming query termination"""
        if self.query:
            self.query.awaitTermination(timeout)
    
    def stop_streaming(self):
        """Stop streaming pipeline"""
        logger.info("Stopping streaming pipeline...")
        
        if self.query:
            self.query.stop()
        
        if self.spark:
            self.spark.stop()
        
        logger.info("✅ Streaming pipeline stopped")


def main():
    """Main execution for Kafka streaming"""
    try:
        pipeline = KafkaStreamingPipeline()
        pipeline.create_spark_session()
        
        # Start streaming (will wait for Kafka messages)
        # Note: This requires Kafka to be running
        pipeline.start_streaming_pipeline("transactions")
        
        # For testing, you can use mock data instead
        logger.info("Note: Kafka streaming requires Kafka to be running")
        logger.info("For testing, use mock data generator instead")
        
        # Keep streaming running
        pipeline.await_termination()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Kafka streaming failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
