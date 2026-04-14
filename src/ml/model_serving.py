#!/usr/bin/env python3
"""
Model Serving and Prediction API Integration
Features: Model loading, batch prediction, real-time inference, API integration
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
import pyspark.sql.functions as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelServing:
    """Model Serving for Fraud Detection"""
    
    def __init__(self, model_path: str = "models/fraud_detection_rf"):
        """Initialize model serving"""
        self.model_path = model_path
        self.spark = None
        self.model = None
        self.is_loaded = False
        
    def initialize_spark(self):
        """Initialize Spark session for serving"""
        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName("ModelServing") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
            logger.info("✅ Spark session initialized for serving")
    
    def load_model(self):
        """Load trained model"""
        logger.info(f"Loading model from: {self.model_path}")
        
        self.initialize_spark()
        
        try:
            self.model = PipelineModel.load(self.model_path)
            self.is_loaded = True
            logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def preprocess_transaction(self, transaction: Dict) -> Dict:
        """Preprocess single transaction for prediction"""
        # Create basic features
        amount = transaction.get('amount', 0.0)
        old_balance = transaction.get('oldbalanceOrg', 0.0)
        
        amount_ratio = amount / old_balance if old_balance > 0 else 0.0
        is_large_amount = 1 if amount > 100000 else 0
        
        return {
            'step': transaction.get('step', 1),
            'amount': amount,
            'oldbalanceOrg': old_balance,
            'newbalanceOrig': transaction.get('newbalanceOrig', 0.0),
            'oldbalanceDest': transaction.get('oldbalanceDest', 0.0),
            'newbalanceDest': transaction.get('newbalanceDest', 0.0),
            'amount_ratio': amount_ratio,
            'is_large_amount': is_large_amount
        }
    
    def predict_single(self, transaction: Dict) -> Dict:
        """Make prediction for single transaction"""
        if not self.is_loaded:
            if not self.load_model():
                return {'error': 'Model not loaded'}
        
        try:
            # Preprocess
            features = self.preprocess_transaction(transaction)
            
            # Create DataFrame
            from pyspark.sql import Row
            row = Row(**features)
            df = self.spark.createDataFrame([row])
            
            # Create feature vector (simplified)
            assembler = VectorAssembler(
                inputCols=['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                          'oldbalanceDest', 'newbalanceDest', 'amount_ratio', 'is_large_amount'],
                outputCol="features_raw"
            )
            
            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True,
                withMean=True
            )
            
            # Transform
            df_features = assembler.transform(df)
            df_scaled = scaler.fit(df_features).transform(df_features)
            
            # Predict
            predictions = self.model.transform(df_scaled)
            
            # Get result
            result = predictions.select("prediction", "probability").collect()[0]
            
            return {
                'is_fraud': bool(result['prediction']),
                'fraud_probability': float(result['probability'][1]),
                'transaction_id': transaction.get('nameOrig', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Make predictions for batch of transactions"""
        logger.info(f"Batch prediction for {len(transactions)} transactions")
        
        results = []
        for transaction in transactions:
            result = self.predict_single(transaction)
            results.append(result)
        
        fraud_count = sum(1 for r in results if r.get('is_fraud', False))
        logger.info(f"✅ Batch prediction complete: {fraud_count} frauds detected")
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_loaded:
            return {'error': 'Model not loaded'}
        
        try:
            return {
                'model_path': self.model_path,
                'model_type': str(type(self.model.stages[0]).__name__),
                'loaded_at': datetime.now().isoformat(),
                'status': 'ready'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def health_check(self) -> Dict:
        """Health check for model serving"""
        return {
            'status': 'healthy' if self.is_loaded else 'unhealthy',
            'model_loaded': self.is_loaded,
            'spark_active': self.spark is not None,
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.spark:
            self.spark.stop()
            logger.info("✅ Spark session stopped")


class FastAPIModelServing:
    """Integration with FastAPI for model serving"""
    
    def __init__(self, model_path: str = "models/fraud_detection_rf"):
        """Initialize FastAPI model serving"""
        self.model_serving = ModelServing(model_path)
        
    def initialize(self):
        """Initialize model serving"""
        return self.model_serving.load_model()
    
    def predict(self, transaction: Dict) -> Dict:
        """Predict via API"""
        return self.model_serving.predict_single(transaction)
    
    def batch_predict(self, transactions: List[Dict]) -> List[Dict]:
        """Batch predict via API"""
        return self.model_serving.predict_batch(transactions)
    
    def get_info(self) -> Dict:
        """Get model info"""
        return self.model_serving.get_model_info()
    
    def health(self) -> Dict:
        """Health check"""
        return self.model_serving.health_check()
    
    def shutdown(self):
        """Shutdown model serving"""
        self.model_serving.cleanup()


def main():
    """Main execution for testing model serving"""
    try:
        # Initialize
        serving = ModelServing()
        
        # Load model (or use mock if not available)
        if not serving.load_model():
            logger.warning("Model not found, using mock prediction")
            # Create mock serving
            from src.api.main import mock_predict
            import pandas as pd
            
            # Test transaction
            test_transaction = {
                'step': 1,
                'type': 'TRANSFER',
                'amount': 150000.0,
                'oldbalanceOrg': 200000.0,
                'newbalanceOrig': 50000.0,
                'nameOrig': 'C123',
                'nameDest': 'C456',
                'oldbalanceDest': 0.0,
                'newbalanceDest': 150000.0
            }
            
            # Mock prediction
            features = pd.DataFrame([test_transaction])
            probability = mock_predict(features)
            
            result = {
                'is_fraud': probability >= 0.5,
                'fraud_probability': probability,
                'transaction_id': test_transaction['nameOrig'],
                'timestamp': datetime.now().isoformat()
            }
            
            print("✅ Mock Prediction Result:")
            print(json.dumps(result, indent=2))
            
            return 0
        
        # Test with real model
        test_transaction = {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 150000.0,
            'oldbalanceOrg': 200000.0,
            'newbalanceOrig': 50000.0,
            'nameOrig': 'C123',
            'nameDest': 'C456',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 150000.0
        }
        
        result = serving.predict_single(test_transaction)
        print("✅ Prediction Result:")
        print(json.dumps(result, indent=2))
        
        # Health check
        health = serving.health_check()
        print("✅ Health Check:")
        print(json.dumps(health, indent=2))
        
        serving.cleanup()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Model serving test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
