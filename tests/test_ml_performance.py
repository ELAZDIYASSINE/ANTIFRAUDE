#!/usr/bin/env python3
"""
Performance Tests and Cross-Validation for ML Models
Features: Performance benchmarks, cross-validation, model comparison
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for model"""
    training_time: float
    inference_time: float
    memory_usage: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float


class MLPerformanceTester:
    """Performance Tester for ML Models"""
    
    def __init__(self, spark: SparkSession):
        """Initialize performance tester"""
        self.spark = spark
        self.results: List[PerformanceMetrics] = []
        
    def measure_training_time(self, pipeline: Pipeline, train_df: DataFrame) -> float:
        """Measure training time"""
        start_time = time.time()
        model = pipeline.fit(train_df)
        training_time = time.time() - start_time
        return training_time
    
    def measure_inference_time(self, model, test_df: DataFrame, sample_size: int = 1000) -> float:
        """Measure inference time"""
        sample_df = test_df.limit(sample_size)
        
        start_time = time.time()
        predictions = model.transform(sample_df)
        inference_time = time.time() - start_time
        
        return inference_time / sample_size  # Time per prediction
    
    def calculate_metrics(self, model, test_df: DataFrame) -> Dict:
        """Calculate model metrics"""
        predictions = model.transform(test_df)
        
        # Confusion matrix
        tp = predictions.filter((F.col("prediction") == 1.0) & (F.col("label") == 1.0)).count()
        tn = predictions.filter((F.col("prediction") == 0.0) & (F.col("label") == 0.0)).count()
        fp = predictions.filter((F.col("prediction") == 1.0) & (F.col("label") == 0.0)).count()
        fn = predictions.filter((F.col("prediction") == 0.0) & (F.col("label") == 1.0)).count()
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC-ROC
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        auc_roc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    def run_cross_validation(self, pipeline: Pipeline, train_df: DataFrame, num_folds: int = 5) -> Dict:
        """Run cross-validation"""
        logger.info(f"Running {num_folds}-fold cross-validation...")
        
        param_grid = ParamGridBuilder().build()
        
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42
        )
        
        start_time = time.time()
        cv_model = crossval.fit(train_df)
        cv_time = time.time() - start_time
        
        metrics = cv_model.avgMetrics
        
        return {
            'cv_time': cv_time,
            'avg_auc_roc': metrics[0] if metrics else 0.0,
            'std_auc_roc': 0.0  # Would need to calculate from fold results
        }
    
    def benchmark_model(self, model_name: str, pipeline: Pipeline, train_df: DataFrame, test_df: DataFrame) -> PerformanceMetrics:
        """Benchmark a model"""
        logger.info(f"Benchmarking {model_name}...")
        
        # Training time
        training_time = self.measure_training_time(pipeline, train_df)
        
        # Train model for inference test
        model = pipeline.fit(train_df)
        
        # Inference time
        inference_time = self.measure_inference_time(model, test_df)
        
        # Calculate metrics
        metrics = self.calculate_metrics(model, test_df)
        
        # Memory usage (simplified)
        memory_usage = 0.0  # Would need actual memory measurement
        
        performance = PerformanceMetrics(
            training_time=training_time,
            inference_time=inference_time,
            memory_usage=memory_usage,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            auc_roc=metrics['auc_roc']
        )
        
        self.results.append(performance)
        
        logger.info(f"✅ {model_name} benchmark complete")
        logger.info(f"  Training Time: {training_time:.2f}s")
        logger.info(f"  Inference Time: {inference_time*1000:.2f}ms")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return performance
    
    def generate_performance_report(self) -> str:
        """Generate performance report"""
        report = "\n" + "=" * 60
        report += "\n📊 ML PERFORMANCE REPORT"
        report += "\n" + "=" * 60
        
        for i, result in enumerate(self.results):
            report += f"\n🤖 Model {i+1}"
            report += f"\n   Training Time: {result.training_time:.2f}s"
            report += f"\n   Inference Time: {result.inference_time*1000:.2f}ms"
            report += f"\n   Accuracy: {result.accuracy:.4f}"
            report += f"\n   Precision: {result.precision:.4f}"
            report += f"\n   Recall: {result.recall:.4f}"
            report += f"\n   F1-Score: {result.f1:.4f}"
            report += f"\n   AUC-ROC: {result.auc_roc:.4f}"
            report += "\n"
        
        report += "=" * 60
        
        return report


def main():
    """Main execution"""
    try:
        from pyspark.sql import SparkSession
        from src.ml.mllib_pipeline import MLPipeline
        
        spark = SparkSession.builder \
            .appName("MLPerformanceTests") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Prepare data
        ml_pipeline = MLPipeline(spark)
        df = ml_pipeline.load_features("data/PS_20174392719_1491204439457_log.csv")
        train_df, test_df = ml_pipeline.prepare_data(df)
        
        # Performance tester
        tester = MLPerformanceTester(spark)
        
        # Benchmark Random Forest
        rf_pipeline = ml_pipeline.create_random_forest_pipeline()
        rf_performance = tester.benchmark_model("Random Forest", rf_pipeline, train_df, test_df)
        
        # Cross-validation
        cv_results = tester.run_cross_validation(rf_pipeline, train_df)
        
        # Generate report
        report = tester.generate_performance_report()
        print(report)
        
        print(f"\n📊 Cross-Validation Results:")
        print(f"  CV Time: {cv_results['cv_time']:.2f}s")
        print(f"  Avg AUC-ROC: {cv_results['avg_auc_roc']:.4f}")
        
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Performance tests failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
