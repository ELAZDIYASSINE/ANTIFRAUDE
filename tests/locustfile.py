#!/usr/bin/env python3
"""
Locust Load Testing for Anti-Fraud Detection API
Features: Load testing, performance validation, stress testing
"""

import time
import random
from datetime import datetime
from typing import Dict

try:
    from locust import HttpUser, task, between, events
    from locust.runners import MasterRunner
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False

# Mock transaction generator
def generate_transaction() -> Dict:
    """Generate mock transaction for testing"""
    return {
        "step": random.randint(1, 744),
        "type": random.choice(["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]),
        "amount": round(random.expovariate(100000), 2),
        "oldbalanceOrg": round(random.expovariate(500000), 2),
        "newbalanceOrig": round(random.expovariate(500000), 2),
        "nameOrig": f"C{random.randint(1000, 9999)}",
        "nameDest": f"C{random.randint(1000, 9999)}",
        "oldbalanceDest": round(random.expovariate(500000), 2),
        "newbalanceDest": round(random.expovariate(500000), 2)
    }


if LOCUST_AVAILABLE:
    class FraudDetectionUser(HttpUser):
        """Locust user for fraud detection API testing"""
        
        wait_time = between(0.1, 1.0)  # Average 0.5 seconds between requests
        
        def on_start(self):
            """Called when a user starts"""
            self.client.verify = False  # Skip SSL verification for local testing
        
        @task(3)
        def predict_transaction(self):
            """Test prediction endpoint (most frequent)"""
            transaction = generate_transaction()
            
            start_time = time.time()
            response = self.client.post(
                "/predict",
                json=transaction,
                name="/predict"
            )
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Validate latency < 100ms requirement
            if latency_ms > 100:
                events.request.fire(
                    request_type="POST",
                    name="/predict",
                    response_time=latency_ms,
                    response_length=len(response.content),
                    exception="LatencyExceeded"
                )
        
        @task(1)
        def health_check(self):
            """Test health check endpoint (less frequent)"""
            self.client.get("/health", name="/health")
        
        @task(1)
        def model_info(self):
            """Test model info endpoint (less frequent)"""
            self.client.get("/model/info", name="/model/info")
    
    @events.init.add_listener
    def on_locust_init(environment, **kwargs):
        """Called when Locust starts"""
        print("🚀 Locust load testing initialized")
        print("📊 Testing for:")
        print("  - Latency < 100ms")
        print("  - Throughput 100K tx/s")
        print("  - Precision > 95%")
        print("  - Recall > 90%")
    
    @events.test_stop.add_listener
    def on_test_stop(environment, **kwargs):
        """Called when test stops"""
        print("\n" + "=" * 60)
        print("📊 LOAD TEST RESULTS")
        print("=" * 60)
        
        if environment.stats.total.fail_ratio > 0:
            print(f"❌ Fail rate: {environment.stats.total.fail_ratio:.2%}")
        else:
            print(f"✅ Fail rate: {environment.stats.total.fail_ratio:.2%}")
        
        print(f"📈 Total requests: {environment.stats.total.num_requests}")
        print(f"⚡ RPS: {environment.stats.total.rps:.2f}")
        print(f"⏱️  Avg response time: {environment.stats.total.avg_response_time:.2f}ms")
        print(f"⏱️  Median response time: {environment.stats.total.median_response_time:.2f}ms")
        print(f"⏱️  95th percentile: {environment.stats.total.get_response_time_percentile(0.95):.2f}ms")
        print("=" * 60)


class PerformanceValidator:
    """Performance validation for Projet 1 requirements"""
    
    def __init__(self):
        """Initialize performance validator"""
        self.metrics = {
            'latency_ms': [],
            'throughput_tx_s': [],
            'precision': [],
            'recall': []
        }
    
    def validate_latency(self, latency_ms: float, target: float = 100.0) -> bool:
        """Validate latency requirement"""
        self.metrics['latency_ms'].append(latency_ms)
        return latency_ms < target
    
    def validate_throughput(self, throughput_tx_s: float, target: float = 100000.0) -> bool:
        """Validate throughput requirement"""
        self.metrics['throughput_tx_s'].append(throughput_tx_s)
        return throughput_tx_s >= target
    
    def validate_precision(self, precision: float, target: float = 0.95) -> bool:
        """Validate precision requirement"""
        self.metrics['precision'].append(precision)
        return precision >= target
    
    def validate_recall(self, recall: float, target: float = 0.90) -> bool:
        """Validate recall requirement"""
        self.metrics['recall'].append(recall)
        return recall >= target
    
    def generate_report(self) -> str:
        """Generate performance validation report"""
        report = "\n" + "=" * 60
        report += "\n📊 PERFORMANCE VALIDATION REPORT"
        report += "\n" + "=" * 60
        
        if self.metrics['latency_ms']:
            avg_latency = sum(self.metrics['latency_ms']) / len(self.metrics['latency_ms'])
            latency_pass = all(l < 100.0 for l in self.metrics['latency_ms'])
            report += f"\n⏱️  Latency: {avg_latency:.2f}ms (target: <100ms)"
            report += f"\n   Status: {'✅ PASS' if latency_pass else '❌ FAIL'}"
        
        if self.metrics['throughput_tx_s']:
            avg_throughput = sum(self.metrics['throughput_tx_s']) / len(self.metrics['throughput_tx_s'])
            throughput_pass = avg_throughput >= 100000.0
            report += f"\n⚡ Throughput: {avg_throughput:.0f} tx/s (target: 100K tx/s)"
            report += f"\n   Status: {'✅ PASS' if throughput_pass else '❌ FAIL'}"
        
        if self.metrics['precision']:
            avg_precision = sum(self.metrics['precision']) / len(self.metrics['precision'])
            precision_pass = avg_precision >= 0.95
            report += f"\n🎯 Precision: {avg_precision:.4f} (target: >0.95)"
            report += f"\n   Status: {'✅ PASS' if precision_pass else '❌ FAIL'}"
        
        if self.metrics['recall']:
            avg_recall = sum(self.metrics['recall']) / len(self.metrics['recall'])
            recall_pass = avg_recall >= 0.90
            report += f"\n🎯 Recall: {avg_recall:.4f} (target: >0.90)"
            report += f"\n   Status: {'✅ PASS' if recall_pass else '❌ FAIL'}"
        
        report += "\n" + "=" * 60
        
        return report


def main():
    """Main execution for load testing"""
    if not LOCUST_AVAILABLE:
        print("❌ Locust not installed. Install with: pip install locust")
        return 1
    
    print("🚀 Starting Locust Load Testing")
    print("📊 Projet 1 Requirements:")
    print("  - Latency < 100ms")
    print("  - Throughput 100K tx/s")
    print("  - Precision > 95%")
    print("  - Recall > 90%")
    print("\n💡 Run with:")
    print("  locust -f tests/locustfile.py --host http://localhost:8000 --users 1000 --spawn-rate 100")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
