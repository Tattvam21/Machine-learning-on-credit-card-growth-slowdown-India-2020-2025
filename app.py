# import streamlit as st
# import joblib
# import numpy as np

# # Load the trained model
# model = joblib.load('lasso_model.pkl')

# # App title
# st.title("Net Growth Impact Predictor")
# st.markdown("Predict credit card net growth impact based on key financial indicators.")

# # Input fields
# monthly_spend = st.number_input("Monthly Spend", value=0.0)
# outstanding_balance = st.number_input("Outstanding Balance", value=0.0)
# delinquency_status = st.number_input("Delinquency Status", value=0.0)

# # Predict button
# if st.button("Predict"):
#     # Prepare input as numpy array
#     features = np.array([[monthly_spend, outstanding_balance, delinquency_status]])
    
#     # Prediction
#     prediction = model.predict(features)[0]
    
#     # Display
#     st.success(f"Predicted Net Growth Impact: {prediction:.4f}")



import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained Lasso model and feature names
model = joblib.load('lasso_model.pkl')
feature_names = joblib.load('feature_names.pkl')  # List of expected feature columns

# App Title
st.title("Net Growth Impact Predictor")
st.markdown("Predict credit card net growth impact based on financial and demographic inputs.")

# --- User Inputs ---

# Numeric inputs
monthly_spend = st.number_input("Monthly Spend", value=0.0)
outstanding_balance = st.number_input("Outstanding Balance", value=0.0)
delinquency_status = st.number_input("Delinquency Status", value=0.0)
credit_score = st.number_input("Credit Score", value=0.0)
card_utilization = st.number_input("Card Utilization", value=0.0)

# Categorical inputs
income_level = st.selectbox("Income Level", ['High', 'Medium', 'Low'])
employment_status = st.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Student', 'Unemployed'])

# --- Preprocess Input ---

def preprocess_input():
    data = {
        'Monthly_Spend': monthly_spend,
        'Outstanding_Balance': outstanding_balance,
        'Delinquency_Status': delinquency_status,
        'Credit_Score': credit_score,
        'Card_Utilization': card_utilization,

        # One-hot encoding for Income Level (Low is base class)
        'Income_Level_High': 1 if income_level == 'High' else 0,
        'Income_Level_Medium': 1 if income_level == 'Medium' else 0,

        # One-hot encoding for Employment Status (Unemployed is base class)
        'Employment_Status_Employed': 1 if employment_status == 'Employed' else 0,
        'Employment_Status_Self-Employed': 1 if employment_status == 'Self-Employed' else 0,
        'Employment_Status_Student': 1 if employment_status == 'Student' else 0
    }

    return pd.DataFrame([data])

# --- Prediction Logic ---

if st.button("Predict"):
    input_df = preprocess_input()

    # Ensure all expected columns are present (fill missing with 0)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[feature_names]

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"Predicted Net Growth Impact: {prediction:.4f}")
