# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

st.set_page_config(page_title="Credit Card Growth Prediction", layout="wide")
st.title("📉 Predict Credit Card Net Growth Impact (India 2020–2025)")
st.write("Enter your own input values below to get a prediction using a trained Lasso + RFE model.")

# Load and preprocess data
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("credit_card_growth_slowdown_india_2020_2025.csv")

    # Drop non-numeric and ID-like columns
    drop_cols = ['Customer_ID', 'Card_Issuance_Date', 'Bank_Name', 'Delinquency_Reason']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

df_num, label_encoders = load_and_prepare_data()

# Set target and features
target_column = "Net_Growth_Impact"
X = df_num.drop(columns=[target_column])
Y = df_num[target_column]

# Feature selection
n_features = min(10, X.shape[1])
model = Lasso(alpha=1.0)
rfe = RFE(model, n_features_to_select=n_features)
rfe.fit(X, Y)

# Train final model
X_rfe = rfe.transform(X)
model.fit(X_rfe, Y)

# Get selected feature names
selected_features = X.columns[rfe.support_]

st.markdown("### 🧾 Enter Input for Prediction")

# Dynamically build form for selected features
input_data = {}
with st.form("prediction_form"):
    for feature in selected_features:
        col_data = df_num[feature]
        if col_data.dtype == 'int64' or col_data.dtype == 'float64':
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            default = float(col_data.mean())
            input_data[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=default)
        else:
            input_data[feature] = st.text_input(f"{feature}")

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    # Ensure order and transformation match training
    input_rfe = input_df[selected_features].values.reshape(1, -1)
    prediction = model.predict(input_rfe)[0]

    st.success(f"✅ Predicted Net Growth Impact: **{prediction:.2f}**")
