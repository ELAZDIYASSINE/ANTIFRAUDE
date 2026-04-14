#!/usr/bin/env python3
"""
Performance Validation for Projet 1 Requirements
Features: Validate metrics >95% / >90%, latency <100ms, throughput 100K tx/s
"""

import os
import sys
import logging
import time
import requests
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Results of performance validation"""
    precision: float
    recall: float
    avg_latency_ms: float
    max_latency_ms: float
    throughput_tx_s: float
    precision_pass: bool
    recall_pass: bool
    latency_pass: bool
    throughput_pass: bool
    overall_pass: bool


class PerformanceValidator:
    """Performance Validator for Projet 1 requirements"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize performance validator"""
        self.api_url = api_url
        self.results = []
        
    def test_latency(self, num_requests: int = 100) -> Dict:
        """Test API latency requirement (<100ms)"""
        logger.info(f"Testing latency with {num_requests} requests...")
        
        latencies = []
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
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json=transaction,
                    timeout=10
                )
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            except Exception as e:
                logger.error(f"Request {i} failed: {e}")
                latencies.append(9999.0)  # Penalize failed requests
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        latency_pass = avg_latency < 100.0
        
        return {
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'median_latency_ms': median_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'latency_pass': latency_pass,
            'latencies': latencies
        }
    
    def test_throughput(self, duration_seconds: int = 10) -> Dict:
        """Test throughput requirement (100K tx/s)"""
        logger.info(f"Testing throughput for {duration_seconds} seconds...")
        
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
        
        start_time = time.time()
        request_count = 0
        errors = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json=transaction,
                    timeout=5
                )
                if response.status_code == 200:
                    request_count += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
        
        actual_duration = time.time() - start_time
        throughput = request_count / actual_duration
        
        throughput_pass = throughput >= 100000.0
        
        return {
            'throughput_tx_s': throughput,
            'total_requests': request_count,
            'errors': errors,
            'duration_seconds': actual_duration,
            'throughput_pass': throughput_pass
        }
    
    def test_ml_metrics(self, test_data: List[Dict]) -> Dict:
        """Test ML metrics (Precision >95%, Recall >90%)"""
        logger.info("Testing ML metrics...")
        
        predictions = []
        
        for transaction in test_data:
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json=transaction,
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    predictions.append({
                        'actual': transaction.get('isFraud', 0),
                        'predicted': result.get('is_fraud', False),
                        'probability': result.get('fraud_probability', 0.0)
                    })
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
        
        if not predictions:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'precision_pass': False,
                'recall_pass': False,
                'total_predictions': 0
            }
        
        # Calculate metrics
        tp = sum(1 for p in predictions if p['actual'] == 1 and p['predicted'] == True)
        tn = sum(1 for p in predictions if p['actual'] == 0 and p['predicted'] == False)
        fp = sum(1 for p in predictions if p['actual'] == 0 and p['predicted'] == True)
        fn = sum(1 for p in predictions if p['actual'] == 1 and p['predicted'] == False)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_pass = precision >= 0.95
        recall_pass = recall >= 0.90
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_pass': precision_pass,
            'recall_pass': recall_pass,
            'total_predictions': len(predictions),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def generate_test_data(self, num_samples: int = 1000) -> List[Dict]:
        """Generate test data for ML metrics validation"""
        import random
        
        test_data = []
        for i in range(num_samples):
            is_fraud = random.random() < 0.01  # 1% fraud rate
            
            if is_fraud:
                # Fraud transactions tend to be larger
                amount = random.expovariate(200000)
                transaction_type = random.choice(["TRANSFER", "CASH_OUT"])
            else:
                amount = random.expovariate(50000)
                transaction_type = random.choice(["CASH_IN", "CASH_OUT", "PAYMENT", "DEBIT"])
            
            test_data.append({
                "step": random.randint(1, 744),
                "type": transaction_type,
                "amount": round(amount, 2),
                "oldbalanceOrg": round(random.expovariate(500000), 2),
                "newbalanceOrig": round(random.expovariate(500000), 2),
                "nameOrig": f"C{random.randint(1000, 9999)}",
                "nameDest": f"C{random.randint(1000, 9999)}",
                "oldbalanceDest": round(random.expovariate(500000), 2),
                "newbalanceDest": round(random.expovariate(500000), 2),
                "isFraud": 1 if is_fraud else 0
            })
        
        return test_data
    
    def run_all_validations(self) -> ValidationResults:
        """Run all performance validations"""
        logger.info("Starting performance validation...")
        
        # Test latency
        latency_results = self.test_latency(num_requests=100)
        
        # Test throughput
        throughput_results = self.test_throughput(duration_seconds=5)
        
        # Test ML metrics
        test_data = self.generate_test_data(num_samples=500)
        ml_results = self.test_ml_metrics(test_data)
        
        # Create validation results
        results = ValidationResults(
            precision=ml_results['precision'],
            recall=ml_results['recall'],
            avg_latency_ms=latency_results['avg_latency_ms'],
            max_latency_ms=latency_results['max_latency_ms'],
            throughput_tx_s=throughput_results['throughput_tx_s'],
            precision_pass=ml_results['precision_pass'],
            recall_pass=ml_results['recall_pass'],
            latency_pass=latency_results['latency_pass'],
            throughput_pass=throughput_results['throughput_pass'],
            overall_pass=(ml_results['precision_pass'] and 
                         ml_results['recall_pass'] and 
                         latency_results['latency_pass'] and 
                         throughput_results['throughput_pass'])
        )
        
        return results
    
    def generate_report(self, results: ValidationResults) -> str:
        """Generate validation report"""
        report = "\n" + "=" * 70
        report += "\n📊 PROJET 1 - PERFORMANCE VALIDATION REPORT"
        report += "\n" + "=" * 70
        report += f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        report += "\n\n🎯 ML METRICS VALIDATION"
        report += "\n" + "-" * 70
        report += f"\nPrecision: {results.precision:.4f} (target: >0.95)"
        report += f"\n  Status: {'✅ PASS' if results.precision_pass else '❌ FAIL'}"
        report += f"\nRecall: {results.recall:.4f} (target: >0.90)"
        report += f"\n  Status: {'✅ PASS' if results.recall_pass else '❌ FAIL'}"
        
        report += "\n\n⚡ LATENCY VALIDATION"
        report += "\n" + "-" * 70
        report += f"\nAverage Latency: {results.avg_latency_ms:.2f}ms (target: <100ms)"
        report += f"\n  Status: {'✅ PASS' if results.latency_pass else '❌ FAIL'}"
        report += f"\nMax Latency: {results.max_latency_ms:.2f}ms"
        
        report += "\n\n📈 THROUGHPUT VALIDATION"
        report += "\n" + "-" * 70
        report += f"\nThroughput: {results.throughput_tx_s:.0f} tx/s (target: 100K tx/s)"
        report += f"\n  Status: {'✅ PASS' if results.throughput_pass else '❌ FAIL'}"
        
        report += "\n\n" + "=" * 70
        report += f"\n🏆 OVERALL RESULT: {'✅ ALL TESTS PASSED' if results.overall_pass else '❌ SOME TESTS FAILED'}"
        report += "\n" + "=" * 70
        
        return report


def main():
    """Main execution for performance validation"""
    try:
        validator = PerformanceValidator()
        
        # Run all validations
        results = validator.run_all_validations()
        
        # Generate report
        report = validator.generate_report(results)
        print(report)
        
        return 0 if results.overall_pass else 1
        
    except Exception as e:
        logger.error(f"❌ Performance validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
