#!/usr/bin/env python3
"""
Environment verification script for Real-Time Fraud Detection System
"""

def verify_pyspark():
    try:
        import pyspark
        from pyspark.sql import SparkSession
        print("✓ PySpark imported successfully")
        # Create a simple Spark session
        spark = SparkSession.builder.appName("Verification").getOrCreate()
        print("✓ Spark session created")
        spark.stop()
        return True
    except Exception as e:
        print(f"✗ PySpark verification failed: {e}")
        return False

def verify_fastapi():
    try:
        from fastapi import FastAPI
        import uvicorn
        print("✓ FastAPI and Uvicorn imported successfully")

        # Create a simple app
        app = FastAPI()

        @app.get("/")
        def read_root():
            return {"message": "Environment verification successful"}

        @app.get("/health")
        def health():
            return {"status": "healthy"}

        print("✓ FastAPI app created with endpoints")
        print("  Note: To launch the app, run: uvicorn verify_environment:app --reload")
        return True
    except Exception as e:
        print(f"✗ FastAPI verification failed: {e}")
        return False

def verify_mlflow():
    try:
        import mlflow
        print("✓ MLflow imported successfully")
        print("  Note: To launch MLflow UI, run: mlflow ui --port 5000")
        return True
    except Exception as e:
        print(f"✗ MLflow verification failed: {e}")
        return False

def main():
    print("🔍 Verifying environment setup...\n")

    checks = [
        verify_pyspark,
        verify_fastapi,
        verify_mlflow
    ]

    passed = 0
    for check in checks:
        if check():
            passed += 1
        print()

    print(f"📊 Verification complete: {passed}/{len(checks)} checks passed")

    if passed == len(checks):
        print("🎉 Environment is ready for development!")
    else:
        print("⚠️  Some verifications failed. Please check your setup.")

if __name__ == "__main__":
    main()
