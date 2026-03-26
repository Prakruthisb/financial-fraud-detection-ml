import pandas as pd
from src.training.train import train, save_pipeline
from src.inference.predict import load_pipeline, predict

if __name__ == '__main__':
    df = pd.read_csv('data/Fraud.csv')
 
    # Train and tune
    pipeline = train(df, threshold=0.9)
 
    # Save the full pipeline (transforms + model)
    save_pipeline(pipeline, 'fraud_pipeline.pkl')
 
    # Load and predict on a raw transaction
    pipeline = load_pipeline('fraud_pipeline.pkl')
    # print(pipeline)
 
    sample_transaction = {
        'step': 1,
        'type': 'TRANSFER',
        'amount': 181.0,
        'nameOrig': 'C1305486145',
        'oldbalanceOrg': 181.0,
        'newbalanceOrig': 0.0,
        'nameDest': 'C553264065',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'isFlaggedFraud': 0
    }
 
    result = predict(pipeline, sample_transaction, threshold=0.9)
    print(result)