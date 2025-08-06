# app.py

import streamlit as st
import pandas as pd
import pickle
import json
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Credit Card Net Growth Prediction", layout="wide")
st.title("📊 Predict Net Growth Impact of Credit Cards (India 2020–2025)")

st.write("This app uses a trained Lasso model with RFE to predict the net growth impact based on customer and credit information.")

# ----------------------------
# Load model and feature list
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    with open("features.json", "r") as f:
        return json.load(f)

# ----------------------------
# Load dataset and encoders
@st.cache_data
def load_data_and_encoders():
    df = pd.read_csv("credit_card_growth_slowdown_india_2020_2025.csv")

    # Drop unnecessary columns
    drop_cols = ['Customer_ID', 'Card_Issuance_Date', 'Bank_Name', 'Delinquency_Reason']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Label encode categorical columns
    label_encoders = {}
    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

# ----------------------------
# Load everything
model = load_model()
features = load_features()
df, label_encoders = load_data_and_encoders()

# ----------------------------
# Debug: Show column match (optional)
# st.markdown("### 🛠 Debug Info")
# st.write("Model expects features:", features)
# st.write("Available columns:", df.columns.tolist())

# ----------------------------
# Build input form
st.markdown("### ✍️ Enter Input Data")
input_data = {}

with st.form("input_form"):
    for feature in features:
        if feature not in df.columns:
            st.error(f"❌ Feature '{feature}' not found in dataset. Check preprocessing.")
            continue

        if feature in label_encoders:
            # Show dropdown for categorical variables
            classes = list(label_encoders[feature].classes_)
            selected = st.selectbox(f"{feature}", classes)
            encoded = label_encoders[feature].transform([selected])[0]
            input_data[feature] = encoded
        else:
            # Show numeric input
            col_data = df[feature]
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            mean_val = float(col_data.mean())
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Run prediction
if submitted:
    input_df = pd.DataFrame([input_data])

    missing = [f for f in features if f not in input_df.columns]
    if missing:
        st.error(f"Missing required features: {missing}")
    elif input_df.shape[1] != len(features):
        st.error("Input does not match model's expected feature count.")
    else:
        input_df = input_df[features]  # Ensure correct order
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Net Growth Impact: **{prediction:.2f}**")
