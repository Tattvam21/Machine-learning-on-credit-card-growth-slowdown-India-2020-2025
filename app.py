import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('lasso_model.pkl')

# App title
st.title("Net Growth Impact Predictor")
st.markdown("Predict credit card net growth impact based on key financial indicators.")

# Input fields
monthly_spend = st.number_input("Monthly Spend", value=0.0)
outstanding_balance = st.number_input("Outstanding Balance", value=0.0)
delinquency_status = st.number_input("Delinquency Status", value=0.0)

# Predict button
if st.button("Predict"):
    # Prepare input as numpy array
    features = np.array([[monthly_spend, outstanding_balance, delinquency_status]])
    
    # Prediction
    prediction = model.predict(features)[0]
    
    # Display
    st.success(f"Predicted Net Growth Impact: {prediction:.4f}")
