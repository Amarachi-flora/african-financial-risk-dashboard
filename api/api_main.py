"""
FastAPI for Customer Financial Risk Prediction
Endpoints for real-time prediction and segmentation
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json

# Initialize FastAPI app
app = FastAPI(
    title="Customer Financial Risk Prediction API",
    description="API for customer segmentation and risk prediction in African financial markets",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODELS_DIR = "models"

try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    pca_model = joblib.load(os.path.join(MODELS_DIR, "pca_model.pkl"))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    scaler = pca_model = kmeans_model = None

# Pydantic models for request validation
# In your api/api_main.py file, update the CustomerData model:

class CustomerData(BaseModel):
    """Single customer data model"""
    customer_id: str = Field(..., description="Unique customer identifier")
    age: int = Field(..., ge=18, le=70, description="Customer age (18-70)")
    monthly_expenditure: float = Field(..., ge=0, description="Monthly expenditure in local currency")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    transaction_count: int = Field(..., ge=1, description="Number of transactions per month")
    avg_transaction_value: float = Field(..., ge=0, description="Average transaction value")
    uses_pos: int = Field(0, ge=0, le=1, description="Uses POS (0/1)")
    uses_web: int = Field(0, ge=0, le=1, description="Uses Web/Transfer (0/1)")
    uses_ussd: int = Field(0, ge=0, le=1, description="Uses USSD (0/1)")
    uses_mobile_app: int = Field(0, ge=0, le=1, description="Uses Mobile App (0/1)")
    income_level: str = Field("Middle", description="Income level")
    saving_behavior: str = Field("Average", description="Saving behavior")
    location: Optional[str] = Field("Unknown", description="Customer location")
    feedback: Optional[str] = Field("", description="Customer feedback text")
    # Add these optional fields to match your original data structure
    transaction_channel: Optional[str] = Field("Mobile App", description="Transaction channel used")
    spending_category: Optional[str] = Field("Groceries", description="Spending category")
    
    
class BatchCustomerData(BaseModel):
    """Batch customer data model"""
    customers: List[CustomerData]

class PredictionResponse(BaseModel):
    """Prediction response model"""
    customer_id: str
    cluster_id: int
    cluster_name: str
    risk_score: float
    risk_category: str
    digital_adoption_score: float
    recommendations: List[str]
    segment_characteristics: Dict[str, Any]
    processing_time: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    timestamp: str

# Helper functions
def calculate_sentiment(feedback: str) -> float:
    """Calculate sentiment score from feedback"""
    if not feedback or not isinstance(feedback, str):
        return 0.0
    
    feedback = feedback.lower()
    positive_words = ['excellent', 'great', 'good', 'fast', 'smooth', 'easy', 'helpful', 'satisfied']
    negative_words = ['confusing', 'unclear', 'failed', 'crashing', 'slow', 'problem', 'issue', 'bad']
    
    positive_count = sum(1 for word in positive_words if word in feedback)
    negative_count = sum(1 for word in negative_words if word in feedback)
    
    if positive_count + negative_count > 0:
        return (positive_count - negative_count) / (positive_count + negative_count)
    return 0.0

def calculate_risk_score(credit_score: int, saving_behavior: str, loan_status: str = "No Loan") -> float:
    """Calculate risk score"""
    risk = (850 - credit_score) / 550 * 0.4
    
    if saving_behavior == 'Poor':
        risk += 0.3
    elif saving_behavior == 'Average':
        risk += 0.15
    
    if loan_status == 'Default Risk':
        risk += 0.3
    
    return min(max(risk, 0), 1)

def get_cluster_name(cluster_id: int) -> str:
    """Map cluster ID to cluster name"""
    cluster_names = {
        0: "Digital-First High Spenders",
        1: "Traditional Low-Risk Savers",
        2: "High-Risk Low Income",
        3: "Medium Digital Average Spenders",
        4: "Positive Experience Customers"
    }
    return cluster_names.get(cluster_id, f"Segment {cluster_id}")

def get_recommendations(cluster_id: int, risk_score: float) -> List[str]:
    """Get personalized recommendations based on cluster and risk"""
    recommendations = []
    
    if cluster_id == 0:  # Digital-First High Spenders
        recommendations = [
            "Premium mobile banking features",
            "Investment products",
            "Credit card with rewards",
            "Wealth management services"
        ]
    elif cluster_id == 1:  # Traditional Low-Risk Savers
        recommendations = [
            "Fixed deposit accounts",
            "Retirement planning",
            "Insurance products",
            "Secure investment options"
        ]
    elif cluster_id == 2:  # High-Risk Low Income
        recommendations = [
            "Financial literacy programs",
            "Micro-savings accounts",
            "Budgeting assistance",
            "Basic banking education"
        ]
    elif cluster_id == 3:  # Medium Digital Average Spenders
        recommendations = [
            "Digital banking adoption programs",
            "Personalized offers",
            "Credit building products",
            "Mixed channel banking"
        ]
    else:  # Default
        recommendations = [
            "Standard banking products",
            "Customer service improvements",
            "Regular financial reviews"
        ]
    
    # Add risk-based recommendations
    if risk_score > 0.7:
        recommendations.append("Enhanced monitoring and support")
        recommendations.append("Gradual credit increase program")
    
    return recommendations[:5]  # Return top 5 recommendations

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Financial Risk Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "clusters": "/clusters",
            "documentation": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded=all([scaler, pca_model, kmeans_model]),
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_customer(customer: CustomerData):
    """Predict customer segment and risk"""
    try:
        start_time = datetime.now()
        
        # Calculate derived features
        digital_adoption = customer.uses_pos + customer.uses_web + customer.uses_ussd + customer.uses_mobile_app
        sentiment_score = calculate_sentiment(customer.feedback)
        risk_score = calculate_risk_score(customer.credit_score, customer.saving_behavior)
        
        # Prepare feature vector
        features = np.array([[
            customer.age,
            customer.monthly_expenditure,
            customer.credit_score,
            customer.transaction_count,
            customer.avg_transaction_value,
            digital_adoption,
            risk_score,
            sentiment_score
        ]])
        
        # Check if models are loaded
        if scaler is None or pca_model is None or kmeans_model is None:
            raise HTTPException(status_code=503, detail="Models not loaded. Please run training pipeline first.")
        
        # Transform features
        features_scaled = scaler.transform(features)
        features_pca = pca_model.transform(features_scaled)
        
        # Predict cluster
        cluster_id = int(kmeans_model.predict(features_pca)[0])
        cluster_name = get_cluster_name(cluster_id)
        
        # Determine risk category
        if risk_score < 0.3:
            risk_category = "Low Risk"
        elif risk_score < 0.6:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
        
        # Get recommendations
        recommendations = get_recommendations(cluster_id, risk_score)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000  # milliseconds
        
        # Segment characteristics
        segment_characteristics = {
            "digital_adoption": digital_adoption,
            "sentiment_score": sentiment_score,
            "expenditure_level": "High" if customer.monthly_expenditure > 150000 else "Medium" if customer.monthly_expenditure > 50000 else "Low",
            "income_level": customer.income_level,
            "location": customer.location
        }
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            risk_score=round(risk_score, 3),
            risk_category=risk_category,
            digital_adoption_score=float(digital_adoption),
            recommendations=recommendations,
            segment_characteristics=segment_characteristics,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch_customers(batch_data: BatchCustomerData):
    """Predict segments for multiple customers"""
    try:
        predictions = []
        
        for customer in batch_data.customers:
            # Use single prediction endpoint logic
            result = await predict_customer(customer)
            predictions.append(result)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/clusters", response_model=Dict[str, Any])
async def get_cluster_info():
    """Get information about all clusters"""
    try:
        cluster_info = {}
        
        for i in range(5):  # Assuming 5 clusters
            cluster_info[f"cluster_{i}"] = {
                "name": get_cluster_name(i),
                "description": get_cluster_description(i),
                "typical_customers": get_typical_customers(i),
                "recommended_products": get_recommendations(i, 0.5)[:3]
            }
        
        return {
            "total_clusters": 5,
            "clusters": cluster_info,
            "clustering_method": "KMeans with PCA",
            "last_trained": "2024-01-01"  # This should come from model metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster info error: {str(e)}")

def get_cluster_description(cluster_id: int) -> str:
    """Get cluster description"""
    descriptions = {
        0: "Tech-savvy customers with high spending and strong digital adoption",
        1: "Conservative savers with low risk and traditional banking preferences",
        2: "Customers with financial challenges needing support and education",
        3: "Average customers with mixed digital and traditional banking usage",
        4: "Satisfied customers with positive feedback and good relationships"
    }
    return descriptions.get(cluster_id, "General customer segment")

def get_typical_customers(cluster_id: int) -> List[str]:
    """Get typical customer characteristics for cluster"""
    characteristics = {
        0: ["Young professionals", "High income", "Digital natives", "Urban residents"],
        1: ["Middle-aged", "Stable income", "Risk-averse", "Long-term savers"],
        2: ["Low income", "Financial difficulties", "Need support", "Rural areas"],
        3: ["Mixed age groups", "Average income", "Moderate digital use", "Suburban"],
        4: ["Long-term customers", "Positive feedback", "Loyal", "Various demographics"]
    }
    return characteristics.get(cluster_id, ["General customers"])

@app.get("/demo")
async def demo_prediction():
    """Demo endpoint with sample prediction"""
    sample_customer = CustomerData(
        customer_id="DEMO001",
        age=35,
        monthly_expenditure=150000.0,
        credit_score=720,
        transaction_count=25,
        avg_transaction_value=6000.0,
        uses_pos=1,
        uses_web=1,
        uses_ussd=0,
        uses_mobile_app=1,
        income_level="Upper-Middle",
        saving_behavior="Good",
        location="Lagos",
        feedback="Excellent mobile banking experience"
    )
    
    return await predict_customer(sample_customer)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Customer Financial Risk Prediction API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔗 Demo endpoint: http://localhost:8000/demo")
    uvicorn.run(app, host="0.0.0.0", port=8000)