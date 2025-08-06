# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Credit Card Growth Prediction", layout="wide")
st.title("📉 Predict Credit Card Net Growth Impact (India 2020–2025)")
st.write("This app uses a pre-trained Lasso model with RFE-selected features to predict Net Growth Impact.")

# Load raw data for preprocessing and encoders
@st.cache_data
def load_data():
    df = pd.read_csv("credit_card_growth_slowdown_india_2020_2025.csv")

    drop_cols = ['Customer_ID', 'Card_Issuance_Date', 'Bank_Name', 'Delinquency_Reason']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    label_encoders = {}
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

# Load model and feature list
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_feature_list():
    with open("features.json", "r") as f:
        return json.load(f)

model = load_model()
features = load_feature_list()
df_num, label_encoders = load_data()

# 🔍 Debug output
st.markdown("### 🛠️ Debug Info (Temporary)")
st.write("📌 Features expected by model (`features.json`):", features)
st.write("📄 Columns available in preprocessed dataset:", df_num.columns.tolist())

st.markdown("### 🧾 Enter Input for Prediction")

# Collect input
input_data = {}
with st.form("prediction_form"):
    for feature in features:
        if feature not in df_num.columns:
            st.error(f"❌ Feature '{feature}' not found in dataset. Please fix your features.json or preprocessing.")
            continue  # Skip missing feature
        
        if feature in label_encoders:
            labels = list(label_encoders[feature].classes_)
            selected_label = st.selectbox(f"{feature}", labels)
            encoded_value = label_encoders[feature].transform([selected_label])[0]
            input_data[feature] = encoded_value
        else:
            col_data = df_num[feature]
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            default = float(col_data.mean())
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=default
            )

    submit = st.form_submit_button("Predict")

# Predict
if submit:
    input_df = pd.DataFrame([input_data])
    
    if len(input_df.columns) != len(features):
        st.error("❌ Input does not match the required number of features. Check warnings above.")
    else:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Net Growth Impact: **{prediction:.2f}**")
