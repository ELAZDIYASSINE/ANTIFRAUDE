#!/usr/bin/env python3
"""
System Monitoring and Observability
Features: System metrics, API monitoring, alerting, performance tracking
"""

import psutil
import time
import logging
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System metrics snapshot"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float


class SystemMonitor:
    """System Monitoring and Observability"""
    
    def __init__(self):
        """Initialize system monitor"""
        self.metrics_history: List[SystemMetrics] = []
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0
        }
        self.alerts: List[Dict] = []
        
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        timestamp = datetime.now().isoformat()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        disk_used_gb = disk.used / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        # Network
        network = psutil.net_io_counters()
        network_sent_mb = network.bytes_sent / (1024**2)
        network_recv_mb = network.bytes_recv / (1024**2)
        
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            disk_used_gb=disk_used_gb,
            disk_free_gb=disk_free_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_alerts(self, metrics: SystemMetrics) -> List[Dict]:
        """Check for alerts based on thresholds"""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'CPU_HIGH',
                'severity': 'HIGH',
                'message': f"CPU usage {metrics.cpu_percent:.1f}% exceeds threshold {self.alert_thresholds['cpu_percent']}%",
                'timestamp': metrics.timestamp
            })
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'MEMORY_HIGH',
                'severity': 'HIGH',
                'message': f"Memory usage {metrics.memory_percent:.1f}% exceeds threshold {self.alert_thresholds['memory_percent']}",
                'timestamp': metrics.timestamp
            })
        
        if metrics.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append({
                'type': 'DISK_HIGH',
                'severity': 'MEDIUM',
                'message': f"Disk usage {metrics.disk_usage_percent:.1f}% exceeds threshold {self.alert_thresholds['disk_usage_percent']}",
                'timestamp': metrics.timestamp
            })
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_average_metrics(self, last_n: int = 10) -> Dict:
        """Get average metrics over last N measurements"""
        if len(self.metrics_history) == 0:
            return {}
        
        recent_metrics = self.metrics_history[-last_n:]
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_disk_usage_percent': avg_disk,
            'sample_size': len(recent_metrics)
        }
    
    def generate_monitoring_report(self) -> str:
        """Generate monitoring report"""
        if not self.metrics_history:
            return "No metrics collected yet"
        
        latest = self.metrics_history[-1]
        averages = self.get_average_metrics()
        
        report = "\n" + "=" * 60
        report += "\n📊 SYSTEM MONITORING REPORT"
        report += "\n" + "=" * 60
        report += f"\n📅 Timestamp: {latest.timestamp}"
        report += "\n\n🔧 CURRENT METRICS:"
        report += f"\n   CPU: {latest.cpu_percent:.1f}%"
        report += f"\n   Memory: {latest.memory_percent:.1f}% ({latest.memory_used_gb:.1f}GB used)"
        report += f"\n   Disk: {latest.disk_usage_percent:.1f}% ({latest.disk_used_gb:.1f}GB used)"
        report += f"\n   Network Sent: {latest.network_sent_mb:.1f}MB"
        report += f"\n   Network Received: {latest.network_recv_mb:.1f}MB"
        
        if averages:
            report += "\n\n📈 AVERAGE METRICS (Last 10):"
            report += f"\n   CPU: {averages['avg_cpu_percent']:.1f}%"
            report += f"\n   Memory: {averages['avg_memory_percent']:.1f}%"
            report += f"\n   Disk: {averages['avg_disk_usage_percent']:.1f}%"
        
        if self.alerts:
            report += f"\n\n⚠️  ALERTS ({len(self.alerts)}):"
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                report += f"\n   {alert['severity']}: {alert['message']}"
        else:
            report += "\n\n✅ NO ALERTS"
        
        report += "\n" + "=" * 60
        
        return report
    
    def save_metrics(self, filepath: str = "logs/system_metrics.json"):
        """Save metrics to file"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        metrics_data = [
            {
                'timestamp': m.timestamp,
                'cpu_percent': m.cpu_percent,
                'memory_percent': m.memory_percent,
                'disk_usage_percent': m.disk_usage_percent
            }
            for m in self.metrics_history
        ]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"✅ Metrics saved to {filepath}")


class APIMonitor:
    """API Performance Monitoring"""
    
    def __init__(self):
        """Initialize API monitor"""
        self.request_history: List[Dict] = []
        self.error_count = 0
        self.total_requests = 0
        
    def log_request(self, endpoint: str, response_time: float, status_code: int):
        """Log API request"""
        self.total_requests += 1
        
        if status_code >= 400:
            self.error_count += 1
        
        request_data = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'response_time_ms': response_time,
            'status_code': status_code
        }
        
        self.request_history.append(request_data)
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_api_metrics(self) -> Dict:
        """Get API performance metrics"""
        if not self.request_history:
            return {}
        
        recent_requests = self.request_history[-100:]
        
        avg_response_time = sum(r['response_time_ms'] for r in recent_requests) / len(recent_requests)
        error_rate = (self.error_count / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'error_rate_percent': error_rate,
            'avg_response_time_ms': avg_response_time,
            'recent_requests': len(recent_requests)
        }


def main():
    """Main execution for monitoring"""
    try:
        monitor = SystemMonitor()
        
        # Collect metrics
        metrics = monitor.collect_metrics()
        
        # Check alerts
        alerts = monitor.check_alerts(metrics)
        
        # Generate report
        report = monitor.generate_monitoring_report()
        print(report)
        
        # Save metrics
        monitor.save_metrics()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Monitoring failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
