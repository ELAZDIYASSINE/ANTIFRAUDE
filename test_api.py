#!/usr/bin/env python3
"""
Test script for FastAPI Anti-Fraud Detection API
"""

import sys
import os
import time
import json

# Add src to path
sys.path.append('src/api')

def test_api_imports():
    """Test if we can import the API components"""
    print("🔍 Testing API imports...")
    
    try:
        from main import app, TransactionRequest, FraudResponse
        print("✅ API imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_prediction_logic():
    """Test the prediction logic without HTTP"""
    print("\n🧠 Testing prediction logic...")
    
    try:
        from main import preprocess_transaction, mock_predict, get_risk_level, TransactionRequest
        
        # Create test transaction
        transaction = TransactionRequest(
            type="TRANSFER",
            amount=150000,
            oldbalanceOrg=500000,
            newbalanceOrig=350000,
            nameOrig="C123456789",
            nameDest="C987654321",
            oldbalanceDest=100000,
            newbalanceDest=250000
        )
        
        # Test preprocessing
        features = preprocess_transaction(transaction)
        print(f"✅ Features generated: {features.shape[1]} columns")
        
        # Test prediction
        probability = mock_predict(features)
        print(f"✅ Mock prediction: {probability:.4f}")
        
        # Test risk level
        risk_level = get_risk_level(probability)
        print(f"✅ Risk level: {risk_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logic test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints if server is running"""
    print("\n🌐 Testing API endpoints...")
    
    try:
        import requests
        
        # Test root endpoint
        try:
            response = requests.get('http://localhost:8000/', timeout=5)
            print(f"✅ Root endpoint: {response.status_code}")
            print(f"   Response: {response.json()}")
        except requests.exceptions.ConnectionError:
            print("⚠️  API server not running on port 8000")
            return False
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            print(f"✅ Health endpoint: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
        
        # Test predict endpoint
        try:
            test_transaction = {
                'type': 'TRANSFER',
                'amount': 150000,
                'oldbalanceOrg': 500000,
                'newbalanceOrig': 350000,
                'nameOrig': 'C123456789',
                'nameDest': 'C987654321',
                'oldbalanceDest': 100000,
                'newbalanceDest': 250000
            }
            
            response = requests.post('http://localhost:8000/predict', json=test_transaction, timeout=5)
            print(f"✅ Predict endpoint: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=4)}")
            
        except Exception as e:
            print(f"❌ Predict endpoint failed: {e}")
        
        return True
        
    except ImportError:
        print("❌ Requests library not available")
        return False

def main():
    """Run all tests"""
    print("🚀 FastAPI Backend Test Suite")
    print("=" * 50)
    
    # Test 1: Imports
    imports_ok = test_api_imports()
    
    # Test 2: Logic
    logic_ok = test_prediction_logic()
    
    # Test 3: Endpoints (if server running)
    endpoints_ok = test_api_endpoints()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   Logic:   {'✅ PASS' if logic_ok else '❌ FAIL'}")
    print(f"   API:     {'✅ PASS' if endpoints_ok else '⚠️  SERVER NOT RUNNING'}")
    
    if imports_ok and logic_ok:
        print("\n🎉 Backend code is working correctly!")
        print("\n💡 To test the full API:")
        print("   1. Run: cd src/api && uvicorn main:app --reload")
        print("   2. Open: http://localhost:8000/docs")
        print("   3. Test endpoints with the Swagger UI")
    else:
        print("\n❌ Some tests failed - check the errors above")

if __name__ == "__main__":
    main()
