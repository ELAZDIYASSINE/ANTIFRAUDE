#!/usr/bin/env python3
"""
Data Quality Validation with Great Expectations
Features: Automated data profiling, quality checks, anomaly detection
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, sum as spark_sum, avg, std, min as spark_min, max as spark_max, when, lit
from pyspark.sql.types import NumericType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityCheck:
    """Quality check result"""
    check_name: str
    status: str  # "PASS", "FAIL", "WARNING"
    expected_value: float
    actual_value: float
    message: str
    timestamp: str


@dataclass
class DataProfile:
    """Data profile result"""
    column_name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    std_value: Optional[float]


class DataQualityValidator:
    """Data Quality Validator"""
    
    def __init__(self, spark: SparkSession):
        """Initialize data quality validator"""
        self.spark = spark
        self.quality_checks: List[QualityCheck] = []
        self.data_profiles: List[DataProfile] = []
        
    def profile_data(self, df: DataFrame) -> List[DataProfile]:
        """Profile data: distribution, outliers, nulls"""
        logger.info("Starting data profiling...")
        
        profiles = []
        
        for column in df.columns:
            col_type = str(df.schema[column].dataType)
            
            # Null count
            null_count = df.filter(col(column).isNull()).count()
            total_count = df.count()
            null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
            
            # Unique count
            unique_count = df.select(column).distinct().count()
            
            # Numeric statistics
            min_val = None
            max_val = None
            mean_val = None
            std_val = None
            
            if isinstance(df.schema[column].dataType, NumericType):
                try:
                    stats = df.select(
                        spark_min(column).alias('min'),
                        spark_max(column).alias('max'),
                        avg(column).alias('mean'),
                        std(column).alias('std')
                    ).collect()[0]
                    
                    min_val = stats['min']
                    max_val = stats['max']
                    mean_val = stats['mean']
                    std_val = stats['std']
                except Exception as e:
                    logger.warning(f"Could not compute statistics for {column}: {e}")
            
            profile = DataProfile(
                column_name=column,
                data_type=col_type,
                null_count=null_count,
                null_percentage=null_percentage,
                unique_count=unique_count,
                min_value=min_val,
                max_value=max_val,
                mean_value=mean_val,
                std_value=std_val
            )
            
            profiles.append(profile)
            
            logger.debug(f"Profiled {column}: {null_count} nulls ({null_percentage:.2f}%)")
        
        self.data_profiles = profiles
        logger.info(f"✅ Data profiling complete: {len(profiles)} columns profiled")
        
        return profiles
    
    def check_null_values(self, df: DataFrame, max_null_percentage: float = 5.0) -> QualityCheck:
        """Check if null values exceed threshold"""
        logger.info("Checking null values...")
        
        total_columns = len(df.columns)
        columns_with_nulls = 0
        total_nulls = 0
        total_cells = df.count() * total_columns
        
        for profile in self.data_profiles:
            if profile.null_count > 0:
                columns_with_nulls += 1
                total_nulls += profile.null_count
        
        null_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
        
        status = "PASS" if null_percentage <= max_null_percentage else "FAIL"
        message = f"Null values: {null_percentage:.2f}% (threshold: {max_null_percentage}%)"
        
        check = QualityCheck(
            check_name="Null Values Check",
            status=status,
            expected_value=max_null_percentage,
            actual_value=null_percentage,
            message=message,
            timestamp=datetime.now().isoformat()
        )
        
        self.quality_checks.append(check)
        logger.info(f"  {status}: {message}")
        
        return check
    
    def check_duplicates(self, df: DataFrame, max_duplicate_percentage: float = 1.0) -> QualityCheck:
        """Check for duplicate rows"""
        logger.info("Checking for duplicates...")
        
        total_rows = df.count()
        unique_rows = df.dropDuplicates().count()
        duplicate_count = total_rows - unique_rows
        duplicate_percentage = (duplicate_count / total_rows * 100) if total_rows > 0 else 0
        
        status = "PASS" if duplicate_percentage <= max_duplicate_percentage else "FAIL"
        message = f"Duplicates: {duplicate_percentage:.2f}% ({duplicate_count:,} rows)"
        
        check = QualityCheck(
            check_name="Duplicate Check",
            status=status,
            expected_value=max_duplicate_percentage,
            actual_value=duplicate_percentage,
            message=message,
            timestamp=datetime.now().isoformat()
        )
        
        self.quality_checks.append(check)
        logger.info(f"  {status}: {message}")
        
        return check
    
    def check_outliers(self, df: DataFrame, column: str, std_threshold: float = 3.0) -> QualityCheck:
        """Check for outliers using standard deviation"""
        logger.info(f"Checking outliers for {column}...")
        
        try:
            if not isinstance(df.schema[column].dataType, NumericType):
                return QualityCheck(
                    check_name=f"Outlier Check ({column})",
                    status="WARNING",
                    expected_value=0.0,
                    actual_value=0.0,
                    message=f"Column {column} is not numeric",
                    timestamp=datetime.now().isoformat()
                )
            
            stats = df.select(
                avg(column).alias('mean'),
                std(column).alias('std')
            ).collect()[0]
            
            mean_val = stats['mean']
            std_val = stats['std']
            
            if std_val is None or std_val == 0:
                return QualityCheck(
                    check_name=f"Outlier Check ({column})",
                    status="WARNING",
                    expected_value=0.0,
                    actual_value=0.0,
                    message=f"Cannot detect outliers (std=0)",
                    timestamp=datetime.now().isoformat()
                )
            
            # Count outliers (beyond 3 standard deviations)
            lower_bound = mean_val - std_threshold * std_val
            upper_bound = mean_val + std_threshold * std_val
            
            outlier_count = df.filter(
                (col(column) < lower_bound) | (col(column) > upper_bound)
            ).count()
            
            total_count = df.count()
            outlier_percentage = (outlier_count / total_count * 100) if total_count > 0 else 0
            
            status = "PASS" if outlier_percentage <= 5.0 else "WARNING"
            message = f"Outliers in {column}: {outlier_percentage:.2f}% ({outlier_count:,} rows)"
            
            check = QualityCheck(
                check_name=f"Outlier Check ({column})",
                status=status,
                expected_value=5.0,
                actual_value=outlier_percentage,
                message=message,
                timestamp=datetime.now().isoformat()
            )
            
            self.quality_checks.append(check)
            logger.info(f"  {status}: {message}")
            
            return check
            
        except Exception as e:
            logger.error(f"Error checking outliers for {column}: {e}")
            return QualityCheck(
                check_name=f"Outlier Check ({column})",
                status="FAIL",
                expected_value=0.0,
                actual_value=0.0,
                message=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def check_data_range(self, df: DataFrame, column: str, min_val: float, max_val: float) -> QualityCheck:
        """Check if data values are within expected range"""
        logger.info(f"Checking data range for {column}...")
        
        try:
            out_of_range = df.filter(
                (col(column) < min_val) | (col(column) > max_val)
            ).count()
            
            total_count = df.count()
            out_of_range_percentage = (out_of_range / total_count * 100) if total_count > 0 else 0
            
            status = "PASS" if out_of_range_percentage == 0 else "FAIL"
            message = f"Out of range values in {column}: {out_of_range_percentage:.2f}%"
            
            check = QualityCheck(
                check_name=f"Range Check ({column})",
                status=status,
                expected_value=0.0,
                actual_value=out_of_range_percentage,
                message=message,
                timestamp=datetime.now().isoformat()
            )
            
            self.quality_checks.append(check)
            logger.info(f"  {status}: {message}")
            
            return check
            
        except Exception as e:
            logger.error(f"Error checking range for {column}: {e}")
            return QualityCheck(
                check_name=f"Range Check ({column})",
                status="FAIL",
                expected_value=0.0,
                actual_value=0.0,
                message=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def check_balance_consistency(self, df: DataFrame) -> QualityCheck:
        """Check balance consistency for CASH_OUT transactions"""
        logger.info("Checking balance consistency...")
        
        try:
            inconsistent = df.filter(
                (col('type') == 'CASH_OUT') & 
                (col('newbalanceOrig') != col('oldbalanceOrg') - col('amount'))
            ).count()
            
            total_cash_out = df.filter(col('type') == 'CASH_OUT').count()
            inconsistent_percentage = (inconsistent / total_cash_out * 100) if total_cash_out > 0 else 0
            
            status = "PASS" if inconsistent_percentage <= 1.0 else "FAIL"
            message = f"Inconsistent CASH_OUT: {inconsistent_percentage:.2f}% ({inconsistent:,} rows)"
            
            check = QualityCheck(
                check_name="Balance Consistency Check",
                status=status,
                expected_value=1.0,
                actual_value=inconsistent_percentage,
                message=message,
                timestamp=datetime.now().isoformat()
            )
            
            self.quality_checks.append(check)
            logger.info(f"  {status}: {message}")
            
            return check
            
        except Exception as e:
            logger.error(f"Error checking balance consistency: {e}")
            return QualityCheck(
                check_name="Balance Consistency Check",
                status="FAIL",
                expected_value=0.0,
                actual_value=0.0,
                message=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def run_all_checks(self, df: DataFrame) -> List[QualityCheck]:
        """Run all quality checks"""
        logger.info("Running all quality checks...")
        
        # Profile data first
        self.profile_data(df)
        
        # Run checks
        self.check_null_values(df)
        self.check_duplicates(df)
        self.check_balance_consistency(df)
        
        # Check outliers for key numeric columns
        numeric_cols = [c for c in df.columns if isinstance(df.schema[c].dataType, NumericType)]
        for col_name in numeric_cols[:5]:  # Check first 5 numeric columns
            self.check_outliers(df, col_name)
        
        logger.info(f"✅ All quality checks complete: {len(self.quality_checks)} checks")
        
        return self.quality_checks
    
    def generate_quality_report(self) -> str:
        """Generate quality report"""
        report = "\n" + "=" * 60
        report += "\n📊 DATA QUALITY REPORT"
        report += "\n" + "=" * 60
        
        # Summary
        passed = sum(1 for c in self.quality_checks if c.status == "PASS")
        failed = sum(1 for c in self.quality_checks if c.status == "FAIL")
        warnings = sum(1 for c in self.quality_checks if c.status == "WARNING")
        
        report += f"\nTotal Checks: {len(self.quality_checks)}"
        report += f"\nPassed: {passed} ✅"
        report += f"\nFailed: {failed} ❌"
        report += f"\nWarnings: {warnings} ⚠️"
        
        # Detailed checks
        report += "\n\n" + "=" * 60
        report += "\nDETAILED CHECKS"
        report += "\n" + "=" * 60
        
        for check in self.quality_checks:
            icon = "✅" if check.status == "PASS" else "❌" if check.status == "FAIL" else "⚠️"
            report += f"\n{icon} {check.check_name}"
            report += f"\n   Status: {check.status}"
            report += f"\n   Message: {check.message}"
            report += f"\n   Expected: {check.expected_value}, Actual: {check.actual_value:.4f}"
            report += f"\n   Time: {check.timestamp}"
            report += "\n"
        
        # Data profiles
        report += "\n" + "=" * 60
        report += "\nDATA PROFILES"
        report += "\n" + "=" * 60
        
        for profile in self.data_profiles:
            report += f"\n📄 {profile.column_name} ({profile.data_type})"
            report += f"\n   Nulls: {profile.null_count:,} ({profile.null_percentage:.2f}%)"
            report += f"\n   Unique: {profile.unique_count:,}"
            
            if profile.min_value is not None:
                report += f"\n   Range: [{profile.min_value:.2f}, {profile.max_value:.2f}]"
            if profile.mean_value is not None:
                report += f"\n   Mean: {profile.mean_value:.2f}, Std: {profile.std_value:.2f}"
            report += "\n"
        
        report += "=" * 60
        
        return report
    
    def get_anomaly_alerts(self) -> List[str]:
        """Get anomaly alerts"""
        alerts = []
        
        for check in self.quality_checks:
            if check.status in ["FAIL", "WARNING"]:
                alert = f"⚠️ {check.check_name}: {check.message}"
                alerts.append(alert)
        
        return alerts


def main():
    """Main execution"""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.appName("DataQuality").getOrCreate()
        
        # Load data
        df = spark.read.csv("data/PS_20174392719_1491204439457_log.csv", 
                           header=True, inferSchema=True)
        
        # Run quality checks
        validator = DataQualityValidator(spark)
        checks = validator.run_all_checks(df)
        
        # Generate report
        report = validator.generate_quality_report()
        print(report)
        
        # Get alerts
        alerts = validator.get_anomaly_alerts()
        if alerts:
            print("\n⚠️ ANOMALY ALERTS:")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("\n✅ NO ANOMALIES DETECTED")
        
        spark.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Data quality validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
