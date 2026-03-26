from fastapi import FastAPI
from src.inference.predict import load_pipeline, predict

app = FastAPI()

# Load pipeline once when API starts
pipeline = load_pipeline('models/fraud_pipeline.pkl')


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict_fraud(transaction: dict):
    result = predict(pipeline, transaction)
    return result