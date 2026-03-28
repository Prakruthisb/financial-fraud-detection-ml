from fastapi import FastAPI
from src.inference.predict import load_pipeline, predict
from src.pydantic_model.models import Transaction
import pandas as pd

app = FastAPI()

# Load pipeline once when API starts
pipeline = load_pipeline('models/fraud_pipeline.pkl')

#Check if API is alive
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    
    # Convert to DataFrame
    df = pd.DataFrame([transaction.dict()])
    
    prob = pipeline.predict_proba(df)[:, 1][0]
    
    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": prob > 0.9
    }