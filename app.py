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
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
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
# Build input form
st.markdown("### ✍️ Enter Input Data")
input_data = {}

with st.form("input_form"):
    for feature in features:
        if feature not in df.columns:
            st.warning(f"⚠️ Feature '{feature}' not found in dataset. Skipping.")
            continue

        if feature in label_encoders:
            # Dropdown for categorical features
            options = list(label_encoders[feature].classes_)
            selected = st.selectbox(f"{feature}", options)
            encoded = label_encoders[feature].transform([selected])[0]
            input_data[feature] = encoded
        else:
            # Numeric input
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    submitted = st.form_submit_button("Predict")

# ----------------------------
# Predict safely
if submitted:
    input_df = pd.DataFrame([input_data])

    # Debug view (optional)
    st.write("🔍 Input shape:", input_df.shape)
    st.write("📋 Input columns:", input_df.columns.tolist())
    st.write("📋 Expected features:", features)

    # Validate input
    missing_features = [f for f in features if f not in input_df.columns]
    extra_features = [f for f in input_df.columns if f not in features]

    if missing_features:
        st.error(f"❌ Missing required features: {missing_features}")
    else:
        # Ensure feature order and only required features
        input_df = input_df[features]
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted Net Growth Impact: **{prediction:.2f}**")
        except Exception as e:
            st.error("❌ Prediction failed. Please check your input.")
            st.exception(e)
