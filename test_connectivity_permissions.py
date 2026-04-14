#!/usr/bin/env python3
"""
Connectivity and Permissions Test Suite for Anti-Fraud Detection System
"""

import os
import sys
import socket
import subprocess
import requests
import json
import time
from pathlib import Path

class ConnectivityTester:
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def log_result(self, test_name, success, details=""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.results[test_name] = {"success": success, "details": details}
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if not success:
            self.errors.append(f"{test_name}: {details}")
    
    def test_file_permissions(self):
        """Test file and directory permissions"""
        print("\n📁 Testing File Permissions...")
        
        # Test project structure
        required_dirs = [
            "src",
            "src/api", 
            "src/ml",
            "src/utils",
            "src/features",
            "src/streaming",
            "src/data_processing",
            "data",
            "notebooks",
            "tests",
            "dashboards",
            "docker"
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                if os.access(dir_path, os.R_OK | os.W_OK):
                    self.log_result(f"Directory {dir_path}", True, "Read/Write access OK")
                else:
                    self.log_result(f"Directory {dir_path}", False, "Missing read/write permissions")
            else:
                self.log_result(f"Directory {dir_path}", False, "Directory does not exist")
        
        # Test key files
        required_files = [
            "src/api/main.py",
            "load_and_analyze_dataset.py",
            "requirements.txt",
            "pyproject.toml",
            "Dockerfile",
            "docker-compose.yml",
            "README.md"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                if os.access(file_path, os.R_OK):
                    self.log_result(f"File {file_path}", True, "Read access OK")
                else:
                    self.log_result(f"File {file_path}", False, "Missing read permissions")
            else:
                self.log_result(f"File {file_path}", False, "File does not exist")
    
    def test_python_environment(self):
        """Test Python environment and packages"""
        print("\n🐍 Testing Python Environment...")
        
        # Test Python version
        python_version = sys.version_info
        self.log_result(
            "Python Version", 
            python_version.major >= 3 and python_version.minor >= 8,
            f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
        )
        
        # Test required packages
        required_packages = [
            "pyspark",
            "fastapi", 
            "uvicorn",
            "pandas",
            "numpy",
            "mlflow",
            "streamlit",
            "scikit-learn",
            "matplotlib",
            "seaborn"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.log_result(f"Package {package}", True, "Import successful")
            except ImportError:
                self.log_result(f"Package {package}", False, "Package not installed")
    
    def test_spark_connectivity(self):
        """Test PySpark connectivity"""
        print("\n⚡ Testing PySpark Connectivity...")
        
        try:
            from pyspark.sql import SparkSession
            
            # Test Spark session creation
            spark = SparkSession.builder \
                .appName("ConnectivityTest") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
            
            self.log_result("Spark Session Creation", True, "Session created successfully")
            
            # Test basic operations
            try:
                # Create test DataFrame
                test_data = [("test", 1, 100.0)]
                columns = ["name", "id", "amount"]
                df = spark.createDataFrame(test_data, columns)
                
                # Test operations
                count = df.count()
                self.log_result("Spark DataFrame Operations", True, f"Test DataFrame created, count: {count}")
                
                # Test dataset loading (if exists)
                dataset_path = "data/PS_20174392719_1491204439457_log.csv"
                if os.path.exists(dataset_path):
                    start_time = time.time()
                    df_large = spark.read.csv(dataset_path, header=True, inferSchema=True)
                    load_time = time.time() - start_time
                    row_count = df_large.count()
                    self.log_result(
                        "Dataset Loading", 
                        True, 
                        f"Loaded {row_count:,} rows in {load_time:.2f}s"
                    )
                else:
                    self.log_result("Dataset Loading", False, "Dataset file not found")
                    
            except Exception as e:
                self.log_result("Spark Operations", False, f"Error: {str(e)}")
            
            spark.stop()
            self.log_result("Spark Session Cleanup", True, "Session stopped successfully")
            
        except Exception as e:
            self.log_result("Spark Connectivity", False, f"Failed to connect: {str(e)}")
    
    def test_network_connectivity(self):
        """Test network connectivity and ports"""
        print("\n🌐 Testing Network Connectivity...")
        
        # Test common ports
        ports_to_test = [
            (8000, "FastAPI"),
            (8501, "Streamlit"),
            (5000, "MLflow"),
            (6379, "Redis"),
            (5432, "PostgreSQL"),
            (9092, "Kafka")
        ]
        
        for port, service in ports_to_test:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    self.log_result(f"Port {port} ({service})", True, "Port is open")
                else:
                    self.log_result(f"Port {port} ({service})", False, "Port is closed or service not running")
                    
            except Exception as e:
                self.log_result(f"Port {port} ({service})", False, f"Error: {str(e)}")
        
        # Test internet connectivity
        try:
            response = requests.get("https://httpbin.org/ip", timeout=5)
            if response.status_code == 200:
                ip_info = response.json()
                self.log_result("Internet Connectivity", True, f"External IP: {ip_info.get('origin', 'Unknown')}")
            else:
                self.log_result("Internet Connectivity", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_result("Internet Connectivity", False, f"No internet connection: {str(e)}")
    
    def test_api_endpoints(self):
        """Test FastAPI endpoints if server is running"""
        print("\n🚀 Testing API Endpoints...")
        
        endpoints = [
            ("/", "Root"),
            ("/health", "Health Check"),
            ("/model/info", "Model Info"),
            ("/docs", "Documentation")
        ]
        
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
                if response.status_code == 200:
                    self.log_result(f"API {name}", True, f"Status {response.status_code}")
                else:
                    self.log_result(f"API {name}", False, f"HTTP {response.status_code}")
            except requests.exceptions.ConnectionError:
                self.log_result(f"API {name}", False, "API server not running")
            except Exception as e:
                self.log_result(f"API {name}", False, f"Error: {str(e)}")
        
        # Test predict endpoint
        try:
            test_transaction = {
                "type": "TRANSFER",
                "amount": 150000,
                "oldbalanceOrg": 500000,
                "newbalanceOrig": 350000,
                "nameOrig": "C123456789",
                "nameDest": "C987654321",
                "oldbalanceDest": 100000,
                "newbalanceDest": 250000
            }
            
            response = requests.post("http://localhost:8000/predict", json=test_transaction, timeout=5)
            if response.status_code == 200:
                result = response.json()
                self.log_result(
                    "API Predict", 
                    True, 
                    f"Fraud probability: {result.get('fraud_probability', 'N/A')}"
                )
            else:
                self.log_result("API Predict", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("API Predict", False, f"Error: {str(e)}")
    
    def test_docker_connectivity(self):
        """Test Docker connectivity"""
        print("\n🐳 Testing Docker Connectivity...")
        
        try:
            # Test Docker command
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                self.log_result("Docker Installation", True, result.stdout.strip())
                
                # Test Docker daemon
                result = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    self.log_result("Docker Daemon", True, "Docker daemon is running")
                    
                    # Test Docker Compose
                    try:
                        result = subprocess.run(
                            ["docker-compose", "--version"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if result.returncode == 0:
                            self.log_result("Docker Compose", True, result.stdout.strip())
                        else:
                            self.log_result("Docker Compose", False, "Docker Compose not working")
                            
                    except FileNotFoundError:
                        self.log_result("Docker Compose", False, "Docker Compose not installed")
                        
                else:
                    self.log_result("Docker Daemon", False, "Docker daemon not running")
                    
            else:
                self.log_result("Docker Installation", False, "Docker not installed")
                
        except subprocess.TimeoutExpired:
            self.log_result("Docker", False, "Docker command timeout")
        except FileNotFoundError:
            self.log_result("Docker", False, "Docker not found in PATH")
        except Exception as e:
            self.log_result("Docker", False, f"Error: {str(e)}")
    
    def test_system_resources(self):
        """Test system resources and limits"""
        print("\n💻 Testing System Resources...")
        
        try:
            import psutil
            
            # Test memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            self.log_result(
                "Memory",
                available_gb > 4,  # At least 4GB available
                f"Available: {available_gb:.1f}GB / Total: {total_gb:.1f}GB"
            )
            
            # Test disk space
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)
            
            self.log_result(
                "Disk Space",
                free_gb > 10,  # At least 10GB free
                f"Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB"
            )
            
            # Test CPU
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.log_result(
                "CPU",
                cpu_count >= 4,  # At least 4 cores
                f"Cores: {cpu_count}, Usage: {cpu_percent}%"
            )
            
        except ImportError:
            self.log_result("System Resources", False, "psutil package not installed")
        except Exception as e:
            self.log_result("System Resources", False, f"Error: {str(e)}")
    
    def test_git_connectivity(self):
        """Test Git connectivity and repository status"""
        print("\n📦 Testing Git Connectivity...")
        
        try:
            # Test Git installation
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.log_result("Git Installation", True, result.stdout.strip())
                
                # Test repository status
                result = subprocess.run(
                    ["git", "status"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd="."
                )
                
                if result.returncode == 0:
                    if "On branch main" in result.stdout:
                        self.log_result("Git Repository", True, "Repository initialized, on main branch")
                    else:
                        self.log_result("Git Repository", True, "Repository initialized")
                        
                    # Test remote connectivity
                    result = subprocess.run(
                        ["git", "remote", "-v"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        self.log_result("Git Remote", True, "Remote configured")
                    else:
                        self.log_result("Git Remote", False, "No remote configured")
                        
                else:
                    self.log_result("Git Repository", False, "Not a Git repository")
                    
            else:
                self.log_result("Git Installation", False, "Git not installed")
                
        except Exception as e:
            self.log_result("Git", False, f"Error: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 CONNECTIVITY & PERMISSIONS TEST REPORT")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ✅")
        print(f"   Failed: {failed_tests} ❌")
        print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        if self.errors:
            print(f"\n❌ FAILED TESTS:")
            for error in self.errors:
                print(f"   • {error}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        
        # Check critical failures
        critical_failures = []
        for test_name, result in self.results.items():
            if not result["success"]:
                if "Python" in test_name or "Spark" in test_name or "FastAPI" in test_name:
                    critical_failures.append(test_name)
        
        if critical_failures:
            print("   🔴 CRITICAL: Fix these issues first:")
            for failure in critical_failures:
                print(f"      • {failure}")
        else:
            print("   🟢 All critical components are working!")
        
        # Check optional failures
        optional_failures = []
        for test_name, result in self.results.items():
            if not result["success"]:
                if "Docker" in test_name or "Redis" in test_name or "PostgreSQL" in test_name:
                    optional_failures.append(test_name)
        
        if optional_failures:
            print("   🟡 OPTIONAL: Consider fixing for full functionality:")
            for failure in optional_failures:
                print(f"      • {failure}")
        
        print(f"\n✅ NEXT STEPS:")
        if passed_tests == total_tests:
            print("   🎉 Perfect! Your environment is fully ready!")
            print("   🚀 You can start developing the anti-fraud system")
        elif passed_tests >= total_tests * 0.8:
            print("   🟢 Good! Most components are working")
            print("   🔧 Fix the remaining issues for optimal performance")
        else:
            print("   🔴 Several issues need attention")
            print("   🛠️  Please resolve the failed tests above")
        
        print(f"\n💾 Save this report for future reference")
        print("="*60)
    
    def run_all_tests(self):
        """Run all connectivity and permission tests"""
        print("🔍 ANTI-FRAUD SYSTEM - CONNECTIVITY & PERMISSIONS TEST SUITE")
        print("="*60)
        
        self.test_file_permissions()
        self.test_python_environment()
        self.test_spark_connectivity()
        self.test_network_connectivity()
        self.test_api_endpoints()
        self.test_docker_connectivity()
        self.test_system_resources()
        self.test_git_connectivity()
        
        self.generate_report()

def main():
    """Main function"""
    tester = ConnectivityTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
