#!/usr/bin/env python3
"""
Test Suite for Dataset Exploration and Analysis (13h-15h Session)
Validates: Chargement données, analyse statistique, problématiques, qualité, performance
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append('.')

class DatasetExplorationTester:
    def __init__(self):
        self.results = {}
        self.dataset_path = "data/PS_20174392719_1491204439457_log.csv"
        self.spark = None
        self.df = None
        
    def log_result(self, test_name, success, details="", metrics=None):
        """Log test result with metrics"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.results[test_name] = {"success": success, "details": details, "metrics": metrics}
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if metrics:
            for key, value in metrics.items():
                print(f"    • {key}: {value}")
    
    def test_dataset_availability(self):
        """Test if dataset is available and accessible"""
        print("\n📁 Testing Dataset Availability...")
        
        if os.path.exists(self.dataset_path):
            file_size = os.path.getsize(self.dataset_path) / (1024*1024)  # MB
            self.log_result(
                "Dataset File", 
                True, 
                f"Found at {self.dataset_path}",
                {"Size (MB)": f"{file_size:.1f}"}
            )
            
            # Test file permissions
            if os.access(self.dataset_path, os.R_OK):
                self.log_result("Dataset Read Access", True, "File is readable")
            else:
                self.log_result("Dataset Read Access", False, "Cannot read file")
                
        else:
            self.log_result("Dataset File", False, f"Dataset not found at {self.dataset_path}")
            return False
        
        return True
    
    def test_pyspark_loading(self):
        """Test PySpark loading functionality"""
        print("\n⚡ Testing PySpark Loading...")
        
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.functions import col, count, sum as spark_sum, avg
            
            # Create optimized Spark session for testing
            self.spark = SparkSession.builder \
                .appName("DatasetExplorationTest") \
                .config("spark.sql.adaptive.enabled", "false") \
                .config("spark.driver.memory", "4g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .getOrCreate()
            
            self.log_result("Spark Session Creation", True, "Session created successfully")
            
            # Test basic loading
            start_time = time.time()
            
            # Load with error handling
            try:
                self.df = self.spark.read.csv(self.dataset_path, header=True, inferSchema=True)
                load_time = time.time() - start_time
                
                # Get basic info
                row_count = self.df.count()
                col_count = len(self.df.columns)
                
                self.log_result(
                    "Dataset Loading",
                    True,
                    f"Loaded successfully in {load_time:.2f}s",
                    {
                        "Rows": f"{row_count:,}",
                        "Columns": col_count,
                        "Load Time (s)": f"{load_time:.2f}"
                    }
                )
                
                return True
                
            except Exception as e:
                self.log_result("Dataset Loading", False, f"Error loading dataset: {str(e)}")
                return False
                
        except ImportError:
            self.log_result("PySpark Import", False, "PySpark not installed")
            return False
        except Exception as e:
            self.log_result("Spark Setup", False, f"Error setting up Spark: {str(e)}")
            return False
    
    def test_statistical_analysis(self):
        """Test statistical descriptive analysis"""
        print("\n📊 Testing Statistical Analysis...")
        
        if self.df is None:
            self.log_result("Statistical Analysis", False, "No DataFrame available")
            return False
        
        try:
            from pyspark.sql.functions import col, count, sum as spark_sum, avg, max as spark_max, min as spark_min
            
            # Test basic statistics
            start_time = time.time()
            
            # Schema analysis
            schema_info = self.df.schema.fields
            numeric_columns = [field.name for field in schema_info if field.dataType.typeName() in ['double', 'integer', 'long']]
            
            self.log_result(
                "Schema Analysis",
                True,
                f"Found {len(numeric_columns)} numeric columns",
                {"Numeric Columns": len(numeric_columns)}
            )
            
            # Basic descriptive statistics
            stats_df = self.df.describe()
            stats_time = time.time() - start_time
            
            self.log_result(
                "Descriptive Statistics",
                True,
                f"Statistics computed in {stats_time:.2f}s",
                {"Compute Time (s)": f"{stats_time:.2f}"}
            )
            
            # Column-specific analysis
            if 'amount' in self.df.columns:
                start_time = time.time()
                
                amount_stats = self.df.agg(
                    avg("amount").alias("avg_amount"),
                    spark_max("amount").alias("max_amount"),
                    spark_min("amount").alias("min_amount"),
                    spark_sum("amount").alias("total_amount")
                ).collect()[0]
                
                analysis_time = time.time() - start_time
                
                self.log_result(
                    "Amount Analysis",
                    True,
                    "Amount statistics computed",
                    {
                        "Average": f"${amount_stats['avg_amount']:,.2f}",
                        "Maximum": f"${amount_stats['max_amount']:,.2f}",
                        "Minimum": f"${amount_stats['min_amount']:,.2f}",
                        "Total": f"${amount_stats['total_amount']:,.0f}",
                        "Analysis Time (s)": f"{analysis_time:.2f}"
                    }
                )
            
            return True
            
        except Exception as e:
            self.log_result("Statistical Analysis", False, f"Error in analysis: {str(e)}")
            return False
    
    def test_business_problem_identification(self):
        """Test identification of business problems"""
        print("\n🏦 Testing Business Problem Identification...")
        
        if self.df is None:
            self.log_result("Business Problems", False, "No DataFrame available")
            return False
        
        try:
            from pyspark.sql.functions import col, count, sum as spark_sum, avg
            
            # Test 1: Fraud rate analysis
            if 'isFraud' in self.df.columns:
                start_time = time.time()
                
                fraud_stats = self.df.groupBy("isFraud").agg(
                    count("*").alias("transaction_count"),
                    (count("*") / self.df.count() * 100).alias("percentage")
                ).orderBy("isFraud").collect()
                
                analysis_time = time.time() - start_time
                
                normal_count = fraud_stats[0]['transaction_count']
                fraud_count = fraud_stats[1]['transaction_count']
                fraud_rate = fraud_stats[1]['percentage']
                
                self.log_result(
                    "Fraud Rate Analysis",
                    True,
                    f"Fraud rate identified: {fraud_rate:.4f}%",
                    {
                        "Normal Transactions": f"{normal_count:,}",
                        "Fraudulent Transactions": f"{fraud_count:,}",
                        "Fraud Rate (%)": f"{fraud_rate:.4f}",
                        "Analysis Time (s)": f"{analysis_time:.2f}"
                    }
                )
                
                # Business problem identification
                if fraud_rate < 1.0:
                    problem_severity = "SEVERE" if fraud_rate < 0.5 else "MODERATE"
                    self.log_result(
                        "Imbalance Problem",
                        True,
                        f"Dataset severely imbalanced - {problem_severity} class imbalance",
                        {"Severity": problem_severity}
                    )
                
            # Test 2: Transaction type analysis
            if 'type' in self.df.columns and 'isFraud' in self.df.columns:
                start_time = time.time()
                
                type_stats = self.df.groupBy("type").agg(
                    count("*").alias("total_transactions"),
                    spark_sum("isFraud").alias("fraud_count"),
                    avg("amount").alias("avg_amount")
                ).orderBy("total_transactions", ascending=False).collect()
                
                analysis_time = time.time() - start_time
                
                # Identify risky transaction types
                risky_types = [row for row in type_stats if row['fraud_count'] > 0]
                
                self.log_result(
                    "Transaction Type Analysis",
                    True,
                    f"Analyzed {len(type_stats)} transaction types",
                    {
                        "Types Analyzed": len(type_stats),
                        "Risky Types": len(risky_types),
                        "Analysis Time (s)": f"{analysis_time:.2f}"
                    }
                )
                
                if risky_types:
                    total_fraud_in_risky = sum(row['fraud_count'] for row in risky_types)
                    self.log_result(
                        "Risk Concentration",
                        True,
                        f"Fraud concentrated in specific transaction types",
                        {
                            "Risky Types": len(risky_types),
                            "Total Fraud in Risky": f"{total_fraud_in_risky:,}"
                        }
                    )
            
            return True
            
        except Exception as e:
            self.log_result("Business Problems", False, f"Error in business analysis: {str(e)}")
            return False
    
    def test_data_quality(self):
        """Test data quality detection"""
        print("\n🔍 Testing Data Quality Detection...")
        
        if self.df is None:
            self.log_result("Data Quality", False, "No DataFrame available")
            return False
        
        try:
            from pyspark.sql.functions import col, count, when, isnan
            
            quality_score = 100
            issues_found = []
            
            # Test 1: Missing values
            start_time = time.time()
            
            null_counts = self.df.select([
                count(when(col(c).isNull() | isnan(c), c)).alias(c) 
                for c in self.df.columns
            ]).collect()[0]
            
            total_nulls = sum(null_counts)
            missing_time = time.time() - start_time
            
            if total_nulls == 0:
                self.log_result(
                    "Missing Values",
                    True,
                    "No missing values found",
                    {"Missing Count": 0, "Check Time (s)": f"{missing_time:.2f}"}
                )
            else:
                quality_score -= min(total_nulls / 1000, 20)  # Deduct up to 20 points
                issues_found.append(f"{total_nulls} missing values")
                self.log_result(
                    "Missing Values",
                    False,
                    f"Found {total_nulls} missing values",
                    {"Missing Count": total_nulls, "Check Time (s)": f"{missing_time:.2f}"}
                )
            
            # Test 2: Duplicate detection
            start_time = time.time()
            
            total_rows = self.df.count()
            unique_rows = self.df.dropDuplicates().count()
            duplicates = total_rows - unique_rows
            duplicate_time = time.time() - start_time
            
            if duplicates == 0:
                self.log_result(
                    "Duplicate Detection",
                    True,
                    "No duplicates found",
                    {"Duplicates": 0, "Check Time (s)": f"{duplicate_time:.2f}"}
                )
            else:
                quality_score -= min(duplicates / 10000, 15)  # Deduct up to 15 points
                issues_found.append(f"{duplicates} duplicates")
                self.log_result(
                    "Duplicate Detection",
                    False,
                    f"Found {duplicates} duplicate rows",
                    {"Duplicates": duplicates, "Check Time (s)": f"{duplicate_time:.2f}"}
                )
            
            # Test 3: Data consistency checks
            start_time = time.time()
            consistency_issues = 0
            
            # Check balance consistency for CASH_OUT
            if 'type' in self.df.columns and all(col in self.df.columns for col in ['oldbalanceOrg', 'newbalanceOrig', 'amount']):
                cash_out_inconsistent = self.df.filter(
                    (col("type") == "CASH_OUT") & 
                    (col("newbalanceOrig") != col("oldbalanceOrg") - col("amount"))
                ).count()
                
                if cash_out_inconsistent > 0:
                    consistency_issues += cash_out_inconsistent
                    issues_found.append(f"{cash_out_inconsistent} inconsistent CASH_OUT transactions")
            
            consistency_time = time.time() - start_time
            
            if consistency_issues == 0:
                self.log_result(
                    "Data Consistency",
                    True,
                    "No consistency issues found",
                    {"Issues": 0, "Check Time (s)": f"{consistency_time:.2f}"}
                )
            else:
                quality_score -= min(consistency_issues / 10000, 10)  # Deduct up to 10 points
                self.log_result(
                    "Data Consistency",
                    False,
                    f"Found {consistency_issues} consistency issues",
                    {"Issues": consistency_issues, "Check Time (s)": f"{consistency_time:.2f}"}
                )
            
            # Overall quality assessment
            quality_grade = "A" if quality_score >= 90 else "B" if quality_score >= 75 else "C" if quality_score >= 60 else "D"
            
            self.log_result(
                "Overall Data Quality",
                quality_score >= 75,
                f"Quality Score: {quality_score:.1f}/100 (Grade {quality_grade})",
                {
                    "Quality Score": f"{quality_score:.1f}/100",
                    "Grade": quality_grade,
                    "Issues Found": len(issues_found)
                }
            )
            
            return True
            
        except Exception as e:
            self.log_result("Data Quality", False, f"Error in quality check: {str(e)}")
            return False
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks on samples"""
        print("\n⚡ Testing Performance Benchmarks...")
        
        if self.df is None:
            self.log_result("Performance Tests", False, "No DataFrame available")
            return False
        
        try:
            from pyspark.sql.functions import col, count, avg
            
            performance_results = {}
            
            # Test 1: Simple aggregation
            start_time = time.time()
            simple_agg = self.df.groupBy("type").count().collect()
            simple_time = time.time() - start_time
            performance_results['simple_aggregation'] = simple_time
            
            self.log_result(
                "Simple Aggregation",
                simple_time < 10,  # Should be under 10 seconds
                f"Completed in {simple_time:.2f}s",
                {"Time (s)": f"{simple_time:.2f}", "Target": "<10s"}
            )
            
            # Test 2: Complex aggregation
            start_time = time.time()
            complex_agg = self.df.groupBy("type", "isFraud").agg(
                avg("amount"),
                count("*")
            ).collect()
            complex_time = time.time() - start_time
            performance_results['complex_aggregation'] = complex_time
            
            self.log_result(
                "Complex Aggregation",
                complex_time < 20,  # Should be under 20 seconds
                f"Completed in {complex_time:.2f}s",
                {"Time (s)": f"{complex_time:.2f}", "Target": "<20s"}
            )
            
            # Test 3: Filtering operation
            start_time = time.time()
            if 'isFraud' in self.df.columns:
                fraud_filter = self.df.filter(col("isFraud") == 1).count()
            else:
                fraud_filter = self.df.filter(col("amount") > 100000).count()
            filter_time = time.time() - start_time
            performance_results['filtering'] = filter_time
            
            self.log_result(
                "Filtering Operation",
                filter_time < 15,  # Should be under 15 seconds
                f"Completed in {filter_time:.2f}s",
                {"Time (s)": f"{filter_time:.2f}", "Target": "<15s"}
            )
            
            # Test 4: Sampling operation
            start_time = time.time()
            sample_df = self.df.sample(fraction=0.01, seed=42)
            sample_count = sample_df.count()
            sample_time = time.time() - start_time
            performance_results['sampling'] = sample_time
            
            self.log_result(
                "Sampling Operation",
                sample_time < 5,  # Should be under 5 seconds
                f"Completed in {sample_time:.2f}s",
                {
                    "Time (s)": f"{sample_time:.2f}", 
                    "Target": "<5s",
                    "Sample Size": f"{sample_count:,}"
                }
            )
            
            # Overall performance assessment
            total_time = sum(performance_results.values())
            avg_time = total_time / len(performance_results)
            
            performance_grade = "A" if avg_time < 5 else "B" if avg_time < 10 else "C" if avg_time < 20 else "D"
            
            self.log_result(
                "Overall Performance",
                avg_time < 15,
                f"Average operation time: {avg_time:.2f}s (Grade {performance_grade})",
                {
                    "Average Time (s)": f"{avg_time:.2f}",
                    "Grade": performance_grade,
                    "Total Time (s)": f"{total_time:.2f}"
                }
            )
            
            return True
            
        except Exception as e:
            self.log_result("Performance Tests", False, f"Error in performance tests: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up Spark session"""
        if self.spark:
            try:
                self.spark.stop()
                print("✅ Spark session cleaned up")
            except:
                pass
    
    def generate_exploration_report(self):
        """Generate comprehensive exploration report"""
        print("\n" + "="*70)
        print("📊 DATASET EXPLORATION & ANALYSIS TEST REPORT (13h-15h Session)")
        print("="*70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📈 SESSION SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ✅")
        print(f"   Failed: {failed_tests} ❌")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        # Session-specific metrics
        print(f"\n🎯 SESSION 13h-15h DELIVERABLES STATUS:")
        
        deliverables = [
            ("Chargement données PySpark", "Dataset Loading"),
            ("Analyse statistique descriptive", "Statistical Analysis"),
            ("Identification problématiques métier", "Business Problem Identification"),
            ("Détection qualité données", "Data Quality"),
            ("Tests performance", "Performance Tests")
        ]
        
        for deliverable, test_key in deliverables:
            if test_key in self.results:
                status = "✅ COMPLETED" if self.results[test_key]["success"] else "❌ INCOMPLETE"
                print(f"   • {deliverable}: {status}")
            else:
                print(f"   • {deliverable}: ⚠️  NOT TESTED")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        
        # Dataset info
        if "Dataset Loading" in self.results and self.results["Dataset Loading"]["success"]:
            metrics = self.results["Dataset Loading"]["metrics"]
            print(f"   📊 Dataset: {metrics.get('Rows', 'N/A')} rows, {metrics.get('Columns', 'N/A')} columns")
            print(f"   ⚡ Load Time: {metrics.get('Load Time (s)', 'N/A')}s")
        
        # Business insights
        if "Fraud Rate Analysis" in self.results and self.results["Fraud Rate Analysis"]["success"]:
            metrics = self.results["Fraud Rate Analysis"]["metrics"]
            print(f"   🏦 Fraud Rate: {metrics.get('Fraud Rate (%)', 'N/A')}%")
            print(f"   💰 Fraud Cases: {metrics.get('Fraudulent Transactions', 'N/A')}")
        
        # Data quality
        if "Overall Data Quality" in self.results and self.results["Overall Data Quality"]["success"]:
            metrics = self.results["Overall Data Quality"]["metrics"]
            print(f"   🔍 Data Quality: {metrics.get('Quality Score', 'N/A')} (Grade {metrics.get('Grade', 'N/A')})")
        
        # Performance
        if "Overall Performance" in self.results and self.results["Overall Performance"]["success"]:
            metrics = self.results["Overall Performance"]["metrics"]
            print(f"   ⚡ Performance: {metrics.get('Average Time (s)', 'N/A')}s avg (Grade {metrics.get('Grade', 'N/A')})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if passed_tests == total_tests:
            print("   🎉 EXCELLENT! All exploration tasks completed successfully!")
            print("   🚀 Ready to proceed to architecture design (Session 4)")
        elif passed_tests >= total_tests * 0.8:
            print("   🟢 GOOD! Most exploration tasks completed")
            print("   🔧 Address remaining issues before proceeding")
        else:
            print("   🔴 NEEDS ATTENTION! Several exploration tasks incomplete")
            print("   🛠️  Resolve failed tests before continuing")
        
        print(f"\n📋 NEXT STEPS:")
        print("   1. Review failed tests and fix issues")
        print("   2. Document key insights and findings")
        print("   3. Prepare for Session 4: Architecture Design")
        print("   4. Consider feature engineering based on analysis")
        
        print("="*70)
    
    def run_exploration_tests(self):
        """Run all dataset exploration tests"""
        print("🔍 DATASET EXPLORATION & ANALYSIS TEST SUITE (13h-15h Session)")
        print("="*70)
        
        try:
            # Run all tests in sequence
            if not self.test_dataset_availability():
                print("❌ Dataset not available - stopping tests")
                return
            
            if self.test_pyspark_loading():
                self.test_statistical_analysis()
                self.test_business_problem_identification()
                self.test_data_quality()
                self.test_performance_benchmarks()
            
            self.generate_exploration_report()
            
        finally:
            self.cleanup()

def main():
    """Main function"""
    tester = DatasetExplorationTester()
    tester.run_exploration_tests()

if __name__ == "__main__":
    main()
