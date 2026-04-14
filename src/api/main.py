"""
FastAPI Main Application - Anti-Fraud Detection System
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

app = FastAPI(
    title="Anti-Fraud Detection API",
    description="API for real-time fraud detection using XGBoost",
    version="1.0.0"
)

# Pydantic models for request/response
class TransactionRequest(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    nameOrig: str
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    step: Optional[int] = 1

class FraudResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    processing_time_ms: float
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    predictions_count: int

# Global variables
model = None
model_version = "1.0.0"
start_time = time.time()
predictions_count = 0

# Risk level mapping
def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

# Feature engineering function
def preprocess_transaction(transaction: TransactionRequest) -> pd.DataFrame:
    """Convert transaction to features for ML model"""
    
    # Create feature dictionary
    features = {
        'type': transaction.type,
        'amount': transaction.amount,
        'oldbalanceOrg': transaction.oldbalanceOrg,
        'newbalanceOrig': transaction.newbalanceOrig,
        'oldbalanceDest': transaction.oldbalanceDest,
        'newbalanceDest': transaction.newbalanceDest,
        'step': transaction.step
    }
    
    # Calculate additional features
    features['amount_ratio'] = transaction.amount / max(transaction.oldbalanceOrg, 1)
    features['balance_change_orig'] = transaction.oldbalanceOrg - transaction.newbalanceOrig
    features['balance_change_dest'] = transaction.newbalanceDest - transaction.oldbalanceDest
    features['is_large_amount'] = 1 if transaction.amount > 100000 else 0
    features['is_zero_balance_orig'] = 1 if transaction.newbalanceOrig == 0 else 0
    
    # Type encoding (one-hot)
    transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    for t in transaction_types:
        features[f'type_{t}'] = 1 if transaction.type == t else 0
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Select numeric features for model
    numeric_features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'step',
        'amount_ratio', 'balance_change_orig', 'balance_change_dest',
        'is_large_amount', 'is_zero_balance_orig',
        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]
    
    return df[numeric_features]

# Load model (placeholder - would load actual trained model)
def load_model():
    """Load the trained XGBoost model"""
    global model
    try:
        # In real implementation, load from MLflow or file
        # model = joblib.load('models/xgboost_fraud_model.pkl')
        
        # For now, create a simple mock model
        print("⚠️  Using mock model - replace with trained XGBoost model")
        model = "mock_model"
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Mock prediction function (replace with actual model prediction)
def mock_predict(features: pd.DataFrame) -> float:
    """Mock prediction function - replace with actual XGBoost prediction"""
    
    # Simple rule-based mock for demonstration
    amount = features['amount'].iloc[0]
    transaction_type = features['type_TRANSFER'].iloc[0] if 'type_TRANSFER' in features.columns else 0
    amount_ratio = features['amount_ratio'].iloc[0] if 'amount_ratio' in features.columns else 0
    
    # High risk indicators
    risk_score = 0.0
    
    # Large amounts increase risk
    if amount > 100000:
        risk_score += 0.3
    elif amount > 50000:
        risk_score += 0.2
    
    # TRANSFER transactions are riskier
    if transaction_type:
        risk_score += 0.2
    
    # High amount ratio increases risk
    if amount_ratio > 0.5:
        risk_score += 0.2
    elif amount_ratio > 0.8:
        risk_score += 0.3
    
    # Add some randomness
    risk_score += np.random.uniform(-0.1, 0.1)
    
    return min(max(risk_score, 0.0), 1.0)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the API"""
    success = load_model()
    if success:
        print("✅ Anti-Fraud API started successfully")
    else:
        print("⚠️  API started but model loading failed")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Anti-Fraud Detection API",
        "version": model_version,
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs",
            "dashboard": "/dashboard"
        }
    }

# Dashboard endpoint
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the HTML dashboard"""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Dashboard not found</h1><p>Please ensure dashboard.html exists in src/api/</p>"

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        uptime_seconds=uptime,
        predictions_count=predictions_count
    )

# Prediction endpoint
@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Predict fraud probability for a transaction"""
    
    start_prediction = time.time()
    
    try:
        # Preprocess transaction
        features = preprocess_transaction(transaction)
        
        # Make prediction
        if model == "mock_model":
            fraud_probability = mock_predict(features)
        else:
            # Real XGBoost prediction would go here
            # fraud_probability = model.predict_proba(features)[0][1]
            fraud_probability = 0.5  # placeholder
        
        # Determine if fraud and risk level
        is_fraud = fraud_probability >= 0.5
        risk_level = get_risk_level(fraud_probability)
        
        # Calculate processing time
        processing_time = (time.time() - start_prediction) * 1000
        
        # Update counter
        global predictions_count
        predictions_count += 1
        
        return FraudResponse(
            fraud_probability=round(fraud_probability, 4),
            is_fraud=is_fraud,
            risk_level=risk_level,
            processing_time_ms=round(processing_time, 2),
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(transactions: list[TransactionRequest]):
    """Predict fraud for multiple transactions"""
    
    results = []
    
    for transaction in transactions:
        try:
            prediction = await predict_fraud(transaction)
            results.append({
                "transaction": transaction.dict(),
                "prediction": prediction.dict()
            })
        except Exception as e:
            results.append({
                "transaction": transaction.dict(),
                "error": str(e)
            })
    
    return {
        "batch_size": len(transactions),
        "successful_predictions": len([r for r in results if "prediction" in r]),
        "failed_predictions": len([r for r in results if "error" in r]),
        "results": results
    }

# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    
    return {
        "model_version": model_version,
        "model_type": "XGBoost Classifier" if model != "mock_model" else "Mock Model",
        "features_count": 15,
        "training_dataset": "PaySim 6.3M transactions",
        "accuracy": ">95%" if model != "mock_model" else "Mock Accuracy",
        "last_updated": "2024-01-15",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
