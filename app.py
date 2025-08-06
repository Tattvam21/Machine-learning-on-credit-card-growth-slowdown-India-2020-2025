# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Credit Card Growth Prediction", layout="wide")
st.title("📉 Predict Credit Card Net Growth Impact (India 2020–2025)")
st.write("This app uses a pre-trained Lasso model with RFE-selected features to predict Net Growth Impact based on your inputs.")

# ------------------------------
# Load raw data (used for encoders & ranges)
@st.cache_data
def load_data():
    df = pd.read_csv("credit_card_growth_slowdown_india_2020_2025.csv")

    # Drop irrelevant columns
    drop_cols = ['Customer_ID', 'Card_Issuance_Date', 'Bank_Name', 'Delinquency_Reason']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Label encode categorical columns
    label_encoders = {}
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

# ------------------------------
# Load pre-trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

# ------------------------------
# Load selected features list
@st.cache_data
def load_feature_list():
    with open("features.json", "r") as f:
        return json.load(f)

# ------------------------------
# Load everything
model = load_model()
features = load_feature_list()
df_num, label_encoders = load_data()

# Debug: Show expected and actual features (optional)
# st.write("Expected features from model:", features)
# st.write("Available columns after preprocessing:", df_num.columns.tolist())

st.markdown("### 🧾 Enter Input for Prediction")

# ------------------------------
# Collect user input
input_data = {}
with st.form("prediction_form"):
    for feature in features:
        if feature not in df_num.columns:
            st.error(f"❌ Feature '{feature}' not found in dataset. Please fix your features.json or preprocessing.")
            continue

        if feature in label_encoders:
            # Categorical input via dropdown
            labels = list(label_encoders[feature].classes_)
            selected_label = st.selectbox(f"{feature}", labels)
            encoded_value = label_encoders[feature].transform([selected_label])[0]
            input_data[feature] = encoded_value
        else:
            # Numeric input
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

# ------------------------------
# Validate and predict
if submit:
    input_df = pd.DataFrame([input_data])

    missing_features = [f for f in features if f not in input_df.columns]
    extra_features = [f for f in input_df.columns if f not in features]

    if missing_features:
        st.error(f"❌ The following required features are missing: {missing_features}")
    elif input_df.shape[1] != len(features):
        st.error("❌ Input feature count does not match model requirement.")
    else:
        # Ensure feature order matches model expectation
        input_df = input_df[features]
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Net Growth Impact: **{prediction:.2f}**")
