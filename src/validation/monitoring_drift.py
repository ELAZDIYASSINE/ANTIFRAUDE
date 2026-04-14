#!/usr/bin/env python3
"""
Feature Drift Monitoring with MLflow
Features: Drift detection, alerting, feature statistics tracking
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg, std, count
import pyspark.sql.functions as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Drift alert"""
    feature_name: str
    alert_type: str  # "MEAN_DRIFT", "STD_DRIFT", "DISTRIBUTION_DRIFT"
    baseline_value: float
    current_value: float
    drift_score: float
    threshold: float
    timestamp: str
    severity: str  # "LOW", "MEDIUM", "HIGH"


@dataclass
class FeatureStatistics:
    """Feature statistics for tracking"""
    feature_name: str
    mean: float
    std: float
    min: float
    max: float
    count: int
    timestamp: str


class DriftMonitor:
    """Feature Drift Monitor with MLflow"""
    
    def __init__(self, spark: SparkSession, mlflow_tracking_uri: str = "mlruns"):
        """Initialize drift monitor"""
        self.spark = spark
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.baseline_stats: Dict[str, FeatureStatistics] = {}
        self.current_stats: Dict[str, FeatureStatistics] = {}
        self.drift_alerts: List[DriftAlert] = []
        self.drift_thresholds = {
            "mean_drift": 0.1,  # 10% change in mean
            "std_drift": 0.15,  # 15% change in std
            "distribution_drift": 0.2  # 20% distribution change
        }
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logger.info(f"✅ MLflow tracking set to: {self.mlflow_tracking_uri}")
            return True
        except ImportError:
            logger.warning("MLflow not installed, drift monitoring without MLflow")
            return False
    
    def calculate_feature_statistics(self, df: DataFrame, feature_cols: List[str]) -> Dict[str, FeatureStatistics]:
        """Calculate statistics for features"""
        logger.info("Calculating feature statistics...")
        
        stats = {}
        timestamp = datetime.now().isoformat()
        
        for feature in feature_cols:
            try:
                result = df.select(
                    avg(col(feature)).alias('mean'),
                    std(col(feature)).alias('std'),
                    F.min(col(feature)).alias('min'),
                    F.max(col(feature)).alias('max'),
                    count(col(feature)).alias('count')
                ).collect()[0]
                
                stat = FeatureStatistics(
                    feature_name=feature,
                    mean=result['mean'] or 0.0,
                    std=result['std'] or 0.0,
                    min=result['min'] or 0.0,
                    max=result['max'] or 0.0,
                    count=result['count'] or 0,
                    timestamp=timestamp
                )
                
                stats[feature] = stat
                logger.debug(f"{feature}: mean={stat.mean:.4f}, std={stat.std:.4f}")
                
            except Exception as e:
                logger.warning(f"Could not calculate statistics for {feature}: {e}")
        
        logger.info(f"✅ Statistics calculated for {len(stats)} features")
        return stats
    
    def set_baseline(self, df: DataFrame, feature_cols: List[str]):
        """Set baseline statistics"""
        logger.info("Setting baseline statistics...")
        
        self.baseline_stats = self.calculate_feature_statistics(df, feature_cols)
        
        # Save baseline to file
        baseline_file = "data/baseline_statistics.json"
        os.makedirs(os.path.dirname(baseline_file), exist_ok=True)
        
        baseline_dict = {
            feature: {
                "mean": stat.mean,
                "std": stat.std,
                "min": stat.min,
                "max": stat.max,
                "count": stat.count,
                "timestamp": stat.timestamp
            }
            for feature, stat in self.baseline_stats.items()
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_dict, f, indent=2)
        
        logger.info(f"✅ Baseline set for {len(self.baseline_stats)} features")
        logger.info(f"Baseline saved to: {baseline_file}")
    
    def load_baseline(self, baseline_file: str = "data/baseline_statistics.json"):
        """Load baseline statistics from file"""
        logger.info(f"Loading baseline from: {baseline_file}")
        
        try:
            with open(baseline_file, 'r') as f:
                baseline_dict = json.load(f)
            
            for feature, stat_dict in baseline_dict.items():
                self.baseline_stats[feature] = FeatureStatistics(
                    feature_name=feature,
                    mean=stat_dict['mean'],
                    std=stat_dict['std'],
                    min=stat_dict['min'],
                    max=stat_dict['max'],
                    count=stat_dict['count'],
                    timestamp=stat_dict['timestamp']
                )
            
            logger.info(f"✅ Baseline loaded: {len(self.baseline_stats)} features")
            return True
            
        except FileNotFoundError:
            logger.warning(f"Baseline file not found: {baseline_file}")
            return False
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")
            return False
    
    def detect_mean_drift(self, feature: str) -> Optional[DriftAlert]:
        """Detect drift in mean value"""
        if feature not in self.baseline_stats or feature not in self.current_stats:
            return None
        
        baseline = self.baseline_stats[feature]
        current = self.current_stats[feature]
        
        if baseline.mean == 0:
            return None
        
        drift_score = abs(current.mean - baseline.mean) / abs(baseline.mean)
        threshold = self.drift_thresholds["mean_drift"]
        
        if drift_score > threshold:
            severity = "HIGH" if drift_score > threshold * 2 else "MEDIUM"
            alert = DriftAlert(
                feature_name=feature,
                alert_type="MEAN_DRIFT",
                baseline_value=baseline.mean,
                current_value=current.mean,
                drift_score=drift_score,
                threshold=threshold,
                timestamp=datetime.now().isoformat(),
                severity=severity
            )
            return alert
        
        return None
    
    def detect_std_drift(self, feature: str) -> Optional[DriftAlert]:
        """Detect drift in standard deviation"""
        if feature not in self.baseline_stats or feature not in self.current_stats:
            return None
        
        baseline = self.baseline_stats[feature]
        current = self.current_stats[feature]
        
        if baseline.std == 0:
            return None
        
        drift_score = abs(current.std - baseline.std) / abs(baseline.std)
        threshold = self.drift_thresholds["std_drift"]
        
        if drift_score > threshold:
            severity = "HIGH" if drift_score > threshold * 2 else "MEDIUM"
            alert = DriftAlert(
                feature_name=feature,
                alert_type="STD_DRIFT",
                baseline_value=baseline.std,
                current_value=current.std,
                drift_score=drift_score,
                threshold=threshold,
                timestamp=datetime.now().isoformat(),
                severity=severity
            )
            return alert
        
        return None
    
    def detect_distribution_drift(self, feature: str) -> Optional[DriftAlert]:
        """Detect distribution drift (simplified using range)"""
        if feature not in self.baseline_stats or feature not in self.current_stats:
            return None
        
        baseline = self.baseline_stats[feature]
        current = self.current_stats[feature]
        
        # Simplified distribution drift using range
        baseline_range = baseline.max - baseline.min
        current_range = current.max - current.min
        
        if baseline_range == 0:
            return None
        
        drift_score = abs(current_range - baseline_range) / abs(baseline_range)
        threshold = self.drift_thresholds["distribution_drift"]
        
        if drift_score > threshold:
            severity = "HIGH" if drift_score > threshold * 2 else "MEDIUM"
            alert = DriftAlert(
                feature_name=feature,
                alert_type="DISTRIBUTION_DRIFT",
                baseline_value=baseline_range,
                current_value=current_range,
                drift_score=drift_score,
                threshold=threshold,
                timestamp=datetime.now().isoformat(),
                severity=severity
            )
            return alert
        
        return None
    
    def monitor_drift(self, df: DataFrame, feature_cols: List[str]) -> List[DriftAlert]:
        """Monitor feature drift"""
        logger.info("Monitoring feature drift...")
        
        # Calculate current statistics
        self.current_stats = self.calculate_feature_statistics(df, feature_cols)
        
        # Check if baseline exists
        if not self.baseline_stats:
            logger.warning("No baseline set, setting current as baseline")
            self.set_baseline(df, feature_cols)
            return []
        
        # Detect drift for each feature
        self.drift_alerts = []
        
        for feature in feature_cols:
            # Check mean drift
            mean_alert = self.detect_mean_drift(feature)
            if mean_alert:
                self.drift_alerts.append(mean_alert)
            
            # Check std drift
            std_alert = self.detect_std_drift(feature)
            if std_alert:
                self.drift_alerts.append(std_alert)
            
            # Check distribution drift
            dist_alert = self.detect_distribution_drift(feature)
            if dist_alert:
                self.drift_alerts.append(dist_alert)
        
        logger.info(f"✅ Drift monitoring complete: {len(self.drift_alerts)} alerts")
        
        return self.drift_alerts
    
    def log_to_mlflow(self, run_name: str = "drift_monitoring"):
        """Log drift metrics to MLflow"""
        try:
            import mlflow
            
            with mlflow.start_run(run_name=run_name):
                # Log drift alerts
                for alert in self.drift_alerts:
                    mlflow.log_metric(
                        f"drift_{alert.feature_name}_{alert.alert_type}",
                        alert.drift_score
                    )
                    mlflow.log_param(
                        f"baseline_{alert.feature_name}_{alert.alert_type}",
                        alert.baseline_value
                    )
                
                # Log summary metrics
                mlflow.log_metric("total_drift_alerts", len(self.drift_alerts))
                mlflow.log_metric("high_severity_alerts", 
                                 sum(1 for a in self.drift_alerts if a.severity == "HIGH"))
                mlflow.log_metric("medium_severity_alerts",
                                 sum(1 for a in self.drift_alerts if a.severity == "MEDIUM"))
                
                logger.info("✅ Drift metrics logged to MLflow")
                
        except ImportError:
            logger.warning("MLflow not installed, skipping MLflow logging")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
    
    def generate_drift_report(self) -> str:
        """Generate drift report"""
        report = "\n" + "=" * 60
        report += "\n📊 FEATURE DRIFT REPORT"
        report += "\n" + "=" * 60
        
        if not self.drift_alerts:
            report += "\n✅ NO DRIFT DETECTED"
            report += "\n" + "=" * 60
            return report
        
        # Summary
        high_alerts = sum(1 for a in self.drift_alerts if a.severity == "HIGH")
        medium_alerts = sum(1 for a in self.drift_alerts if a.severity == "MEDIUM")
        
        report += f"\nTotal Alerts: {len(self.drift_alerts)}"
        report += f"\nHigh Severity: {high_alerts} 🔴"
        report += f"\nMedium Severity: {medium_alerts} 🟡"
        
        # Detailed alerts
        report += "\n\n" + "=" * 60
        report += "\nDETAILED ALERTS"
        report += "\n" + "=" * 60
        
        for alert in self.drift_alerts:
            icon = "🔴" if alert.severity == "HIGH" else "🟡"
            report += f"\n{icon} {alert.feature_name} - {alert.alert_type}"
            report += f"\n   Baseline: {alert.baseline_value:.4f}"
            report += f"\n   Current: {alert.current_value:.4f}"
            report += f"\n   Drift Score: {alert.drift_score:.4f} (threshold: {alert.threshold})"
            report += f"\n   Severity: {alert.severity}"
            report += f"\n   Time: {alert.timestamp}"
            report += "\n"
        
        report += "=" * 60
        
        return report
    
    def save_alerts(self, alert_file: str = "data/drift_alerts.json"):
        """Save drift alerts to file"""
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)
        
        alerts_dict = [
            {
                "feature_name": alert.feature_name,
                "alert_type": alert.alert_type,
                "baseline_value": alert.baseline_value,
                "current_value": alert.current_value,
                "drift_score": alert.drift_score,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp,
                "severity": alert.severity
            }
            for alert in self.drift_alerts
        ]
        
        with open(alert_file, 'w') as f:
            json.dump(alerts_dict, f, indent=2)
        
        logger.info(f"✅ Drift alerts saved to: {alert_file}")


def main():
    """Main execution"""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.appName("DriftMonitoring").getOrCreate()
        
        # Load data
        df = spark.read.csv("data/PS_20174392719_1491204439457_log.csv", 
                           header=True, inferSchema=True)
        
        # Get numeric columns
        numeric_cols = [f.name for f in df.schema.fields if f.dataType.typeName() in ['integer', 'double', 'float', 'long']]
        
        # Initialize monitor
        monitor = DriftMonitor(spark)
        monitor.setup_mlflow()
        
        # Try to load baseline
        if not monitor.load_baseline():
            # Set baseline if not exists
            monitor.set_baseline(df, numeric_cols[:20])  # Use first 20 numeric columns
            print("✅ Baseline set from current data")
        else:
            # Monitor drift
            alerts = monitor.monitor_drift(df, numeric_cols[:20])
            
            # Generate report
            report = monitor.generate_drift_report()
            print(report)
            
            # Log to MLflow
            monitor.log_to_mlflow()
            
            # Save alerts
            monitor.save_alerts()
        
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Drift monitoring failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
