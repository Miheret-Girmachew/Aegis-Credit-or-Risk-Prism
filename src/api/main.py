import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from .pydantic_models import PredictionRequest, PredictionResponse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Initialize the FastAPI App ---
app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API to predict credit risk probability for customers based on transaction history.",
    version="1.0.0"
)

# --- 2. Load the Model at Startup ---
model = None
MODEL_FEATURES = [
    "TotalTransactionValue",
    "AvgTransactionValue",
    "TransactionCount",
    "StdDevTransactionValue",
    "Recency",
    "Frequency",
    "Monetary"
]

@app.on_event("startup")
def load_model():
    """
    Load the ML model from the MLflow Model Registry when the API starts.
    """
    global model
    
    # Configure MLflow Tracking URI. It's better to use an environment variable.
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logging.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")

    # Define the model to load
    model_name = "CreditRiskModel"  
    model_version = "latest"       
    model_uri = f"models:/{model_name}/{model_version}"
    
    logging.info(f"Loading model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
       
        raise RuntimeError(f"Could not load model '{model_name}'. API cannot start.") from e


# --- 3. Define API Endpoints ---
@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Credit Risk Model API is running. Go to /docs for documentation."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Accepts customer data and returns the credit risk probability.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")

    try:
    
        data_dict = request.model_dump()
        
        data_df = pd.DataFrame([data_dict], columns=MODEL_FEATURES)
        
        logging.info(f"Received prediction request for data: {data_dict}")

        prediction_proba = model.predict(data_df)[0] 
        

        
        logging.info(f"Prediction result: {prediction_proba}")

        return PredictionResponse(risk_probability=float(prediction_proba))

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")