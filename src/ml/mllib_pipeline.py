#!/usr/bin/env python3
"""
Distributed ML Pipeline with MLlib for Anti-Fraud Detection
Features: ML pipelines, hyperparameter tuning, cross-validation, model selection
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import PipelineModel
import pyspark.sql.functions as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MLModelMetrics:
    """Metrics for ML model evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    training_time: float
    model_name: str


class MLPipeline:
    """Distributed ML Pipeline with MLlib"""
    
    def __init__(self, spark: SparkSession):
        """Initialize ML pipeline"""
        self.spark = spark
        self.pipeline_model: Optional[PipelineModel] = None
        self.best_model = None
        self.metrics: Optional[MLModelMetrics] = None
        
    def load_features(self, path: str) -> DataFrame:
        """Load features from Delta Lake"""
        logger.info(f"Loading features from: {path}")
        
        try:
            df = self.spark.read.format("delta").load(path)
            logger.info(f"✅ Features loaded: {df.count():,} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load features: {e}")
            # Fallback to CSV
            df = self.spark.read.csv("data/PS_20174392719_1491204439457_log.csv", 
                                     header=True, inferSchema=True)
            logger.info(f"✅ Loaded from CSV: {df.count():,} rows")
            return df
    
    def prepare_data(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Prepare data for ML training"""
        logger.info("Preparing data for ML...")
        
        # Create basic features if not already present
        if 'amount_ratio' not in df.columns:
            df = df.withColumn('amount_ratio', 
                             F.when(F.col('oldbalanceOrg') > 0, 
                                   F.col('amount') / F.col('oldbalanceOrg')).otherwise(0.0))
        
        if 'is_large_amount' not in df.columns:
            df = df.withColumn('is_large_amount',
                             F.when(F.col('amount') > 100000, 1).otherwise(0))
        
        # Select numeric features
        numeric_features = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'amount_ratio', 'is_large_amount'
        ]
        
        # Filter to only existing columns
        available_features = [f for f in numeric_features if f in df.columns]
        
        # Handle categorical features
        categorical_features = ['type']
        available_categorical = [f for f in categorical_features if f in df.columns]
        
        # Create feature vector
        assembler = VectorAssembler(
            inputCols=available_features,
            outputCol="features_raw"
        )
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        # Index label
        label_indexer = StringIndexer(
            inputCol="isFraud",
            outputCol="label"
        )
        
        # Build preprocessing pipeline
        preprocessor = Pipeline(stages=[assembler, scaler, label_indexer])
        
        # Fit and transform
        preprocessor_model = preprocessor.fit(df)
        df_processed = preprocessor_model.transform(df)
        
        # Split data
        train_df, test_df = df_processed.randomSplit([0.8, 0.2], seed=42)
        
        logger.info(f"✅ Data prepared: Train={train_df.count():,}, Test={test_df.count():,}")
        
        return train_df, test_df
    
    def create_random_forest_pipeline(self) -> Pipeline:
        """Create Random Forest pipeline"""
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=10,
            seed=42
        )
        
        pipeline = Pipeline(stages=[rf])
        return pipeline
    
    def create_gbt_pipeline(self) -> Pipeline:
        """Create Gradient Boosted Trees pipeline"""
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            maxDepth=5,
            seed=42
        )
        
        pipeline = Pipeline(stages=[gbt])
        return pipeline
    
    def create_logistic_regression_pipeline(self) -> Pipeline:
        """Create Logistic Regression pipeline"""
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=0.01
        )
        
        pipeline = Pipeline(stages=[lr])
        return pipeline
    
    def train_model(self, train_df: DataFrame, model_type: str = "random_forest") -> PipelineModel:
        """Train ML model"""
        logger.info(f"Training {model_type} model...")
        start_time = datetime.now()
        
        if model_type == "random_forest":
            pipeline = self.create_random_forest_pipeline()
        elif model_type == "gbt":
            pipeline = self.create_gbt_pipeline()
        elif model_type == "logistic_regression":
            pipeline = self.create_logistic_regression_pipeline()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        model = pipeline.fit(train_df)
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Model trained in {training_time:.2f}s")
        
        self.pipeline_model = model
        return model
    
    def evaluate_model(self, model: PipelineModel, test_df: DataFrame) -> MLModelMetrics:
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        # Make predictions
        predictions = model.transform(test_df)
        
        # Calculate metrics
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        
        # AUC-ROC
        auc_roc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        
        # AUC-PR
        auc_pr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
        
        # Calculate accuracy, precision, recall, F1
        tp = predictions.filter((F.col("prediction") == 1.0) & (F.col("label") == 1.0)).count()
        tn = predictions.filter((F.col("prediction") == 0.0) & (F.col("label") == 0.0)).count()
        fp = predictions.filter((F.col("prediction") == 1.0) & (F.col("label") == 0.0)).count()
        fn = predictions.filter((F.col("prediction") == 0.0) & (F.col("label") == 1.0)).count()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = MLModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            training_time=0.0,
            model_name="trained_model"
        )
        
        logger.info(f"✅ Model Evaluation:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  AUC-ROC: {auc_roc:.4f}")
        logger.info(f"  AUC-PR: {auc_pr:.4f}")
        
        self.metrics = metrics
        return metrics
    
    def hyperparameter_tuning(self, train_df: DataFrame, model_type: str = "random_forest") -> PipelineModel:
        """Perform hyperparameter tuning with cross-validation"""
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        if model_type == "random_forest":
            pipeline = self.create_random_forest_pipeline()
            param_grid = ParamGridBuilder() \
                .addGrid(pipeline.stages[0].numTrees, [50, 100, 200]) \
                .addGrid(pipeline.stages[0].maxDepth, [5, 10, 15]) \
                .build()
        elif model_type == "gbt":
            pipeline = self.create_gbt_pipeline()
            param_grid = ParamGridBuilder() \
                .addGrid(pipeline.stages[0].maxIter, [50, 100, 200]) \
                .addGrid(pipeline.stages[0].maxDepth, [3, 5, 7]) \
                .build()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Cross-validation
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            seed=42
        )
        
        # Run CV
        cv_model = crossval.fit(train_df)
        
        logger.info(f"✅ Hyperparameter tuning complete")
        logger.info(f"  Best AUC-ROC: {cv_model.avgMetrics[0]:.4f}")
        
        self.best_model = cv_model.bestModel
        return cv_model.bestModel
    
    def save_model(self, model: PipelineModel, path: str):
        """Save trained model"""
        logger.info(f"Saving model to: {path}")
        
        model.write().overwrite().save(path)
        logger.info("✅ Model saved")
    
    def load_model(self, path: str) -> PipelineModel:
        """Load trained model"""
        logger.info(f"Loading model from: {path}")
        
        model = PipelineModel.load(path)
        logger.info("✅ Model loaded")
        
        self.pipeline_model = model
        return model
    
    def predict(self, model: PipelineModel, data: DataFrame) -> DataFrame:
        """Make predictions with trained model"""
        logger.info("Making predictions...")
        
        predictions = model.transform(data)
        logger.info(f"✅ Predictions made: {predictions.count():,} rows")
        
        return predictions


def main():
    """Main execution"""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .appName("MLPipeline") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        ml_pipeline = MLPipeline(spark)
        
        # Load data
        df = ml_pipeline.load_features("data/feature_store/transaction_features")
        
        # Prepare data
        train_df, test_df = ml_pipeline.prepare_data(df)
        
        # Train model
        model = ml_pipeline.train_model(train_df, "random_forest")
        
        # Evaluate
        metrics = ml_pipeline.evaluate_model(model, test_df)
        
        # Save model
        ml_pipeline.save_model(model, "models/fraud_detection_rf")
        
        print("\n" + "=" * 60)
        print("✅ ML PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall: {metrics.recall:.4f}")
        print(f"F1-Score: {metrics.f1:.4f}")
        print(f"AUC-ROC: {metrics.auc_roc:.4f}")
        print("=" * 60)
        
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ ML pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
