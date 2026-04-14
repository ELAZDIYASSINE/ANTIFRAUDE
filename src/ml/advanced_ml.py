#!/usr/bin/env python3
"""
Advanced ML Models for Anti-Fraud Detection
Features: Isolation Forest, XGBoost, Ensemble methods
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pyspark.sql.functions as F

# Try to import sklearn for Isolation Forest
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.ensemble import GradientBoostingClassifier as SklearnGBM
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn or xgboost not available - using PySpark only")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedMLPipeline:
    """Advanced ML Pipeline with Isolation Forest and XGBoost"""
    
    def __init__(self, spark: SparkSession):
        """Initialize advanced ML pipeline"""
        self.spark = spark
        self.models = {}
        self.metrics = {}
        
    def prepare_features(self, df: DataFrame) -> DataFrame:
        """Prepare features for ML"""
        logger.info("Preparing features...")
        
        # Create additional features
        df = df.withColumn('amount_ratio', 
                         F.when(F.col('oldbalanceOrg') > 0,
                               F.col('amount') / F.col('oldbalanceOrg')).otherwise(0.0))
        
        df = df.withColumn('balance_change_orig',
                         F.col('oldbalanceOrg') - F.col('newbalanceOrig'))
        
        df = df.withColumn('balance_change_dest',
                         F.col('newbalanceDest') - F.col('oldbalanceDest'))
        
        df = df.withColumn('is_large_amount',
                         F.when(F.col('amount') > 100000, 1).otherwise(0))
        
        # Select numeric features
        feature_cols = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'amount_ratio',
            'balance_change_orig', 'balance_change_dest', 'is_large_amount'
        ]
        
        available_features = [f for f in feature_cols if f in df.columns]
        
        # Create feature vector
        assembler = VectorAssembler(
            inputCols=available_features,
            outputCol="features_raw"
        )
        
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )
        
        pipeline = Pipeline(stages=[assembler, scaler])
        pipeline_model = pipeline.fit(df)
        df_prepared = pipeline_model.transform(df)
        
        logger.info(f"✅ Features prepared: {len(available_features)} features")
        
        return df_prepared
    
    def train_isolation_forest(self, df: DataFrame) -> Dict:
        """Train Isolation Forest for anomaly detection"""
        logger.info("Training Isolation Forest...")
        
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available for Isolation Forest")
            return None
        
        try:
            # Convert to pandas for sklearn
            pdf = df.select(['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                           'oldbalanceDest', 'newbalanceDest']).toPandas()
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                n_estimators=100,
                contamination=0.01,
                random_state=42,
                n_jobs=-1
            )
            
            iso_forest.fit(pdf)
            
            # Predict anomalies
            predictions = iso_forest.predict(pdf)
            anomaly_scores = iso_forest.score_samples(pdf)
            
            logger.info(f"✅ Isolation Forest trained: {sum(predictions == -1)} anomalies detected")
            
            return {
                'model': iso_forest,
                'anomaly_count': sum(predictions == -1),
                'anomaly_scores': anomaly_scores
            }
            
        except Exception as e:
            logger.error(f"Error training Isolation Forest: {e}")
            return None
    
    def train_xgboost(self, train_df: DataFrame, test_df: DataFrame) -> Dict:
        """Train XGBoost classifier"""
        logger.info("Training XGBoost...")
        
        if not SKLEARN_AVAILABLE:
            logger.error("xgboost not available")
            return None
        
        try:
            # Convert to pandas for XGBoost
            feature_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                          'oldbalanceDest', 'newbalanceDest']
            
            X_train = train_df.select(feature_cols).toPandas()
            y_train = train_df.select('isFraud').toPandas().values.ravel()
            
            X_test = test_df.select(feature_cols).toPandas()
            y_test = test_df.select('isFraud').toPandas().values.ravel()
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            xgb_model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = xgb_model.predict(X_test_scaled)
            y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"✅ XGBoost trained:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  AUC-ROC: {auc_roc:.4f}")
            
            return {
                'model': xgb_model,
                'scaler': scaler,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc_roc': auc_roc
                }
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            return None
    
    def train_ensemble(self, train_df: DataFrame, test_df: DataFrame) -> Dict:
        """Train ensemble of models"""
        logger.info("Training ensemble models...")
        
        results = {}
        
        # Train PySpark Random Forest
        rf_pipeline = self.create_rf_pipeline()
        rf_model = rf_pipeline.fit(train_df)
        rf_metrics = self.evaluate_model(rf_model, test_df)
        results['random_forest'] = rf_metrics
        self.models['random_forest'] = rf_model
        
        # Train PySpark GBT
        gbt_pipeline = self.create_gbt_pipeline()
        gbt_model = gbt_pipeline.fit(train_df)
        gbt_metrics = self.evaluate_model(gbt_model, test_df)
        results['gradient_boosted_trees'] = gbt_metrics
        self.models['gradient_boosted_trees'] = gbt_model
        
        # Train XGBoost if available
        if SKLEARN_AVAILABLE:
            xgb_results = self.train_xgboost(train_df, test_df)
            if xgb_results:
                results['xgboost'] = xgb_results['metrics']
                self.models['xgboost'] = xgb_results
        
        # Train Isolation Forest if available
        if SKLEARN_AVAILABLE:
            iso_results = self.train_isolation_forest(train_df)
            if iso_results:
                results['isolation_forest'] = iso_results
                self.models['isolation_forest'] = iso_results
        
        self.metrics = results
        return results
    
    def create_rf_pipeline(self) -> Pipeline:
        """Create Random Forest pipeline"""
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="isFraud",
            numTrees=200,
            maxDepth=15,
            minInstancesPerNode=2,
            seed=42
        )
        return Pipeline(stages=[rf])
    
    def create_gbt_pipeline(self) -> Pipeline:
        """Create GBT pipeline"""
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="isFraud",
            maxIter=200,
            maxDepth=8,
            stepSize=0.1,
            seed=42
        )
        return Pipeline(stages=[gbt])
    
    def evaluate_model(self, model, test_df: DataFrame) -> Dict:
        """Evaluate model performance"""
        predictions = model.transform(test_df)
        
        # Confusion matrix
        tp = predictions.filter((F.col("prediction") == 1.0) & (F.col("isFraud") == 1.0)).count()
        tn = predictions.filter((F.col("prediction") == 0.0) & (F.col("isFraud") == 0.0)).count()
        fp = predictions.filter((F.col("prediction") == 1.0) & (F.col("isFraud") == 0.0)).count()
        fn = predictions.filter((F.col("prediction") == 0.0) & (F.col("isFraud") == 1.0)).count()
        
        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC-ROC
        evaluator = BinaryClassificationEvaluator(labelCol="isFraud", rawPredictionCol="rawPrediction")
        auc_roc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    def validate_metrics(self, target_precision: float = 0.95, target_recall: float = 0.90) -> Dict:
        """Validate metrics against targets"""
        logger.info(f"Validating metrics (Precision >{target_precision}, Recall >{target_recall})...")
        
        validation_results = {}
        
        for model_name, metrics in self.metrics.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                precision = metrics['precision']
                recall = metrics['recall']
                
                precision_pass = precision >= target_precision
                recall_pass = recall >= target_recall
                
                validation_results[model_name] = {
                    'precision': precision,
                    'recall': recall,
                    'precision_pass': precision_pass,
                    'recall_pass': recall_pass,
                    'overall_pass': precision_pass and recall_pass
                }
                
                status = "✅" if precision_pass and recall_pass else "❌"
                logger.info(f"{status} {model_name}: Precision={precision:.4f}, Recall={recall:.4f}")
        
        return validation_results


def main():
    """Main execution"""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .appName("AdvancedML") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Load data
        df = spark.read.csv("data/PS_20174392719_1491204439457_log.csv", 
                           header=True, inferSchema=True)
        
        # Prepare features
        pipeline = AdvancedMLPipeline(spark)
        df_prepared = pipeline.prepare_features(df)
        
        # Split data
        train_df, test_df = df_prepared.randomSplit([0.8, 0.2], seed=42)
        
        # Train ensemble
        results = pipeline.train_ensemble(train_df, test_df)
        
        # Validate metrics
        validation = pipeline.validate_metrics(0.95, 0.90)
        
        print("\n" + "=" * 60)
        print("📊 ADVANCED ML RESULTS")
        print("=" * 60)
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                print(f"\n🤖 {model_name}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1']:.4f}")
                print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print("\n" + "=" * 60)
        print("🎯 VALIDATION RESULTS")
        print("=" * 60)
        
        for model_name, val in validation.items():
            status = "✅ PASS" if val['overall_pass'] else "❌ FAIL"
            print(f"{status} {model_name}")
            print(f"  Precision: {val['precision']:.4f} (target: 0.95)")
            print(f"  Recall: {val['recall']:.4f} (target: 0.90)")
        
        print("=" * 60)
        
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Advanced ML pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
