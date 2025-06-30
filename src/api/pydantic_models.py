# src/api/pydantic_models.py

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """
    Defines the shape of the input data for a prediction request.
    The field names MUST EXACTLY MATCH the column names our model was trained on.
    """
    TotalTransactionValue: float
    AvgTransactionValue: float
    TransactionCount: int
    StdDevTransactionValue: float
    Recency: int
    Frequency: int
    Monetary: float
    

class PredictionResponse(BaseModel):
    """
    Defines the shape of the output data our API will send back.
    """
    risk_probability: float