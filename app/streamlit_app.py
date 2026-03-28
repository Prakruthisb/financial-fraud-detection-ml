import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load pipeline
pipeline = joblib.load("models/fraud_pipeline.pkl")

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Transaction Fraud Detection System")

st.markdown("Enter transaction details to predict fraud probability.")

# -----------------------------
# Input Form
# -----------------------------
with st.form("fraud_form"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        step = st.number_input("Step", min_value=0, value=1)
        type_ = st.selectbox("Transaction Type", 
                             ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    
    with col2:
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
        oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=5000.0)
        nameOrig = st.text_input("Sender ID", "C123")
        nameDest = st.text_input("Receiver ID", "C456")

    submit = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submit:
    
    input_data = pd.DataFrame([{
        "step": step,
        "type": type_,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "oldbalanceDest": oldbalanceDest,
        "nameOrig": nameOrig,
        "nameDest": nameDest
    }])

    prob = pipeline.predict_proba(input_data)[:, 1][0]
    
    st.subheader("Prediction Result")

    if prob > 0.9:
        st.error(f"🚨 FRAUD (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ LEGIT (Confidence: {1 - prob:.2f})")

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    st.subheader("Model Explanation (SHAP)")

    try:
        # Step 1: Copy input
        X_temp = input_data.copy()

        # Step 2: Apply all transformations manually (except model)
        for name, step in pipeline.named_steps.items():
            if name != "model":
                X_temp = step.transform(X_temp)

        # Step 3: Get model
        model = pipeline.named_steps["model"]

        # Step 4: SHAP explainer
        explainer = shap.Explainer(model)

        # Step 5: Compute SHAP values
        shap_values = explainer(X_temp)

        # Step 6: Plot
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation not available.")
        st.text(str(e))