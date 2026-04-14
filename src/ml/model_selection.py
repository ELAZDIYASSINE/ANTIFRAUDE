#!/usr/bin/env python3
"""
Model Selection and Optimization for Anti-Fraud Detection
Features: Model comparison, performance optimization, best model selection
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Model comparison result"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    training_time: float
    parameters: Dict


class ModelSelector:
    """Model Selection and Optimization"""
    
    def __init__(self, spark: SparkSession):
        """Initialize model selector"""
        self.spark = spark
        self.model_results: List[ModelComparison] = []
        self.best_model = None
        self.best_model_name = None
        
    def compare_models(self, train_df: DataFrame, test_df: DataFrame) -> List[ModelComparison]:
        """Compare multiple models"""
        logger.info("Starting model comparison...")
        
        models_to_test = [
            ("Random Forest", self.create_random_forest()),
            ("Gradient Boosted Trees", self.create_gbt()),
            ("Logistic Regression", self.create_logistic_regression()),
            ("Decision Tree", self.create_decision_tree())
        ]
        
        results = []
        
        for model_name, pipeline in models_to_test:
            logger.info(f"Training {model_name}...")
            start_time = datetime.now()
            
            try:
                # Train
                model = pipeline.fit(train_df)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate
                metrics = self.evaluate_model(model, test_df)
                
                comparison = ModelComparison(
                    model_name=model_name,
                    accuracy=metrics['accuracy'],
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    f1=metrics['f1'],
                    auc_roc=metrics['auc_roc'],
                    training_time=training_time,
                    parameters={}
                )
                
                results.append(comparison)
                logger.info(f"  {model_name}: AUC-ROC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"  Error training {model_name}: {e}")
        
        self.model_results = results
        return results
    
    def create_random_forest(self) -> Pipeline:
        """Create Random Forest pipeline"""
        from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
        
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=100,
            maxDepth=10,
            seed=42
        )
        return Pipeline(stages=[rf])
    
    def create_gbt(self) -> Pipeline:
        """Create Gradient Boosted Trees pipeline"""
        from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
        
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            maxDepth=5,
            seed=42
        )
        return Pipeline(stages=[gbt])
    
    def create_logistic_regression(self) -> Pipeline:
        """Create Logistic Regression pipeline"""
        from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
        
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=0.01
        )
        return Pipeline(stages=[lr])
    
    def create_decision_tree(self) -> Pipeline:
        """Create Decision Tree pipeline"""
        from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
        
        dt = DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=10,
            seed=42
        )
        return Pipeline(stages=[dt])
    
    def evaluate_model(self, model, test_df: DataFrame) -> Dict:
        """Evaluate model and return metrics"""
        predictions = model.transform(test_df)
        
        # Calculate confusion matrix
        tp = predictions.filter((F.col("prediction") == 1.0) & (F.col("label") == 1.0)).count()
        tn = predictions.filter((F.col("prediction") == 0.0) & (F.col("label") == 0.0)).count()
        fp = predictions.filter((F.col("prediction") == 1.0) & (F.col("label") == 0.0)).count()
        fn = predictions.filter((F.col("prediction") == 0.0) & (F.col("label") == 1.0)).count()
        
        # Calculate metrics
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
    
    def select_best_model(self, metric: str = "f1") -> str:
        """Select best model based on metric"""
        logger.info(f"Selecting best model based on {metric}...")
        
        if not self.model_results:
            raise ValueError("No model results available")
        
        # Sort by metric
        sorted_results = sorted(self.model_results, key=lambda x: getattr(x, metric), reverse=True)
        best = sorted_results[0]
        
        self.best_model_name = best.model_name
        logger.info(f"✅ Best model: {best.model_name} ({metric}={getattr(best, metric):.4f})")
        
        return best.model_name
    
    def optimize_hyperparameters(self, train_df: DataFrame, test_df: DataFrame, 
                               model_name: str = "random_forest") -> Dict:
        """Optimize hyperparameters for specific model"""
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        if model_name == "random_forest":
            pipeline = self.create_random_forest()
            param_grid = ParamGridBuilder() \
                .addGrid(pipeline.stages[0].numTrees, [50, 100, 200, 300]) \
                .addGrid(pipeline.stages[0].maxDepth, [5, 10, 15, 20]) \
                .addGrid(pipeline.stages[0].minInstancesPerNode, [1, 2, 4]) \
                .build()
        elif model_name == "gbt":
            pipeline = self.create_gbt()
            param_grid = ParamGridBuilder() \
                .addGrid(pipeline.stages[0].maxIter, [50, 100, 200]) \
                .addGrid(pipeline.stages[0].maxDepth, [3, 5, 7, 10]) \
                .addGrid(pipeline.stages[0].stepSize, [0.01, 0.1, 0.2]) \
                .build()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
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
        
        # Get best parameters
        best_model = cv_model.bestModel
        best_params = best_model.stages[0].extractParamMap()
        
        # Evaluate
        metrics = self.evaluate_model(best_model, test_df)
        
        logger.info(f"✅ Hyperparameter optimization complete")
        logger.info(f"  Best AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'metrics': metrics
        }
    
    def generate_comparison_report(self) -> str:
        """Generate model comparison report"""
        report = "\n" + "=" * 60
        report += "\n📊 MODEL COMPARISON REPORT"
        report += "\n" + "=" * 60
        
        for result in self.model_results:
            report += f"\n🤖 {result.model_name}"
            report += f"\n   Accuracy: {result.accuracy:.4f}"
            report += f"\n   Precision: {result.precision:.4f}"
            report += f"\n   Recall: {result.recall:.4f}"
            report += f"\n   F1-Score: {result.f1:.4f}"
            report += f"\n   AUC-ROC: {result.auc_roc:.4f}"
            report += f"\n   Training Time: {result.training_time:.2f}s"
            report += "\n"
        
        report += "\n" + "=" * 60
        report += f"\n🏆 BEST MODEL: {self.best_model_name}"
        report += "\n" + "=" * 60
        
        return report


def main():
    """Main execution"""
    try:
        from pyspark.sql import SparkSession
        from src.ml.mllib_pipeline import MLPipeline
        
        spark = SparkSession.builder \
            .appName("ModelSelection") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Prepare data
        ml_pipeline = MLPipeline(spark)
        df = ml_pipeline.load_features("data/PS_20174392719_1491204439457_log.csv")
        train_df, test_df = ml_pipeline.prepare_data(df)
        
        # Model selection
        selector = ModelSelector(spark)
        results = selector.compare_models(train_df, test_df)
        
        # Select best model
        best_model_name = selector.select_best_model("f1")
        
        # Generate report
        report = selector.generate_comparison_report()
        print(report)
        
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Model selection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
