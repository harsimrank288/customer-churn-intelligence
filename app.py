
from pathlib import Path

import joblib
from fastapi import FastAPI

import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal

import json

from database import (
    create_table,
    save_prediction,
    get_prediction_history
)

# Create the FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained machine learning model.",
    version="1.0.0"
)

# Define the model file location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "balanced_logistic_regression.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)
create_table()

class CustomerData(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)



@app.get("/")
def home():
    return {
        "message": "Welcome to the Customer Churn Prediction API!",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert customer details into a DataFrame
    customer_df = pd.DataFrame([customer.model_dump()])

    # Predict churn
    prediction = model.predict(customer_df)[0]

    # Calculate churn probability
    probabilities = model.predict_proba(customer_df)[0]

    classes = list(model.named_steps["classifier"].classes_)
    churn_index = classes.index("Yes")

    churn_probability = float(probabilities[churn_index])

    # Determine risk level
    if churn_probability >= 0.70:
        risk_level = "High"
    elif churn_probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Convert customer information to JSON
    customer_json = json.dumps(customer.model_dump())

    # Save the prediction to SQLite
    prediction_id = save_prediction(
        customer_data=customer_json,
        predicted_churn=str(prediction),
        churn_probability=round(churn_probability * 100, 2),
        risk_level=risk_level
    )

    # Return the prediction and database record ID
    return {
        "prediction_id": prediction_id,
        "predicted_churn": str(prediction),
        "churn_probability": round(churn_probability * 100, 2),
        "risk_level": risk_level
    }

@app.get("/history")
def prediction_history():
    records = get_prediction_history()

    return {
        "total_predictions": len(records),
        "predictions": records
    }



