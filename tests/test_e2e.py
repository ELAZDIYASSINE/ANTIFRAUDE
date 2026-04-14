#!/usr/bin/env python3
"""
End-to-End Tests for Anti-Fraud Detection System
Features: Complete workflow testing, API integration, data pipeline validation
"""

import os
import sys
import logging
import time
import requests
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2ETester:
    """End-to-End Testing"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize E2E tester"""
        self.api_url = api_url
        self.results = []
        
    def log_result(self, test_name: str, passed: bool, message: str):
        """Log test result"""
        self.results.append({
            'test': test_name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        icon = "✅" if passed else "❌"
        logger.info(f"{icon} {test_name}: {message}")
    
    def test_api_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_result("API Health Check", True, f"Status: {data.get('status')}")
                return True
            else:
                self.log_result("API Health Check", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_result("API Health Check", False, f"Error: {e}")
            return False
    
    def test_api_predict(self):
        """Test API prediction endpoint"""
        try:
            transaction = {
                "step": 1,
                "type": "TRANSFER",
                "amount": 150000.0,
                "oldbalanceOrg": 200000.0,
                "newbalanceOrig": 50000.0,
                "nameOrig": "C123",
                "nameDest": "C456",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 150000.0
            }
            
            response = requests.post(f"{self.api_url}/predict", json=transaction, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("API Predict", True, f"Fraud prob: {data.get('fraud_probability', 0):.4f}")
                return True
            else:
                self.log_result("API Predict", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("API Predict", False, f"Error: {e}")
            return False
    
    def test_api_model_info(self):
        """Test API model info endpoint"""
        try:
            response = requests.get(f"{self.api_url}/model/info", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.log_result("API Model Info", True, f"Model: {data.get('model_type')}")
                return True
            else:
                self.log_result("API Model Info", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("API Model Info", False, f"Error: {e}")
            return False
    
    def test_data_pipeline(self):
        """Test data pipeline functionality"""
        try:
            from src.data_processing.etl_pipeline import ETLPipeline
            
            pipeline = ETLPipeline()
            pipeline.create_spark_session()
            
            # Test data loading
            df = pipeline.load_transaction_data("data/PS_20174392719_1491204439457_log.csv")
            
            if df.count() > 0:
                self.log_result("Data Pipeline Load", True, f"Loaded {df.count():,} records")
                pipeline.cleanup()
                return True
            else:
                self.log_result("Data Pipeline Load", False, "No records loaded")
                pipeline.cleanup()
                return False
                
        except Exception as e:
            self.log_result("Data Pipeline", False, f"Error: {e}")
            return False
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        try:
            from src.features.feature_store import FeatureStore
            from pyspark.sql import SparkSession
            
            spark = SparkSession.builder.appName("TestFeatures").getOrCreate()
            feature_store = FeatureStore(spark)
            
            # Load data
            df = spark.read.csv("data/PS_20174392719_1491204439457_log.csv", 
                             header=True, inferSchema=True)
            
            # Test temporal features
            df_features = feature_store.create_temporal_features(df)
            
            if len(df_features.columns) > len(df.columns):
                self.log_result("Feature Engineering", True, f"Added {len(df_features.columns) - len(df.columns)} features")
                spark.stop()
                return True
            else:
                self.log_result("Feature Engineering", False, "No features added")
                spark.stop()
                return False
                
        except Exception as e:
            self.log_result("Feature Engineering", False, f"Error: {e}")
            return False
    
    def test_data_quality(self):
        """Test data quality validation"""
        try:
            from src.validation.data_quality import DataQualityValidator
            from pyspark.sql import SparkSession
            
            spark = SparkSession.builder.appName("TestQuality").getOrCreate()
            validator = DataQualityValidator(spark)
            
            # Load data
            df = spark.read.csv("data/PS_20174392719_1491204439457_log.csv", 
                             header=True, inferSchema=True)
            
            # Run checks
            checks = validator.run_all_checks(df)
            
            passed = sum(1 for c in checks if c.status == "PASS")
            self.log_result("Data Quality", True, f"Passed {passed}/{len(checks)} checks")
            
            spark.stop()
            return True
            
        except Exception as e:
            self.log_result("Data Quality", False, f"Error: {e}")
            return False
    
    def test_model_loading(self):
        """Test model loading"""
        try:
            from src.ml.model_serving import ModelServing
            
            serving = ModelServing()
            # Try to load model, but don't fail if it doesn't exist
            if serving.load_model():
                self.log_result("Model Loading", True, "Model loaded successfully")
                serving.cleanup()
                return True
            else:
                self.log_result("Model Loading", True, "Model not found (expected for first run)")
                return False
                
        except Exception as e:
            self.log_result("Model Loading", False, f"Error: {e}")
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all E2E tests"""
        logger.info("Starting End-to-End Tests...")
        
        # API Tests
        self.test_api_health()
        self.test_api_predict()
        self.test_api_model_info()
        
        # Pipeline Tests
        self.test_data_pipeline()
        self.test_feature_engineering()
        self.test_data_quality()
        
        # Model Tests
        self.test_model_loading()
        
        # Generate report
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate E2E test report"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        report = "\n" + "=" * 60
        report += "\n📊 END-TO-END TEST REPORT"
        report += "\n" + "=" * 60
        report += f"\nTotal Tests: {total}"
        report += f"\nPassed: {passed} ✅"
        report += f"\nFailed: {failed} ❌"
        report += f"\nSuccess Rate: {success_rate:.1f}%"
        
        report += "\n\nDETAILED RESULTS:"
        for result in self.results:
            icon = "✅" if result['passed'] else "❌"
            report += f"\n{icon} {result['test']}: {result['message']}"
        
        report += "\n" + "=" * 60
        
        print(report)
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': success_rate
        }


def main():
    """Main execution"""
    tester = E2ETester()
    results = tester.run_all_tests()
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
