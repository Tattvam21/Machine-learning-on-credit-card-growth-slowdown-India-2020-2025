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

# Load trained Lasso model
model = joblib.load('lasso_model.pkl')

st.title("Net Growth Impact Predictor")
st.markdown("Predict credit card net growth impact based on user financial and demographic inputs.")

# --- Input fields ---

# Numeric inputs
monthly_spend = st.number_input("Monthly Spend", value=0.0)
outstanding_balance = st.number_input("Outstanding Balance", value=0.0)
delinquency_status = st.number_input("Delinquency Status", value=0.0)
credit_score = st.number_input("Credit Score", value=0.0)
card_utilization = st.number_input("Card Utilization", value=0.0)

# Categorical inputs
income_level = st.selectbox("Income Level", ['High', 'Medium', 'Low'])
employment_status = st.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Student', 'Unemployed'])

# --- Preprocessing for categorical variables ---

def preprocess_input():
    data = {
        'Monthly_Spend': monthly_spend,
        'Outstanding_Balance': outstanding_balance,
        'Delinquency_Status': delinquency_status,
        'Credit_Score': credit_score,
        'Card_Utilization': card_utilization,

        # One-hot encode Income_Level
        'Income_Level_High': 1 if income_level == 'High' else 0,
        'Income_Level_Medium': 1 if income_level == 'Medium' else 0,

        # One-hot encode Employment_Status
        'Employment_Status_Employed': 1 if employment_status == 'Employed' else 0,
        'Employment_Status_Self-Employed': 1 if employment_status == 'Self-Employed' else 0,
        'Employment_Status_Student': 1 if employment_status == 'Student' else 0,
        # Unemployed is the base case and left out
    }
    return pd.DataFrame([data])

# --- Predict Button ---
if st.button("Predict"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Net Growth Impact: {prediction:.4f}")
