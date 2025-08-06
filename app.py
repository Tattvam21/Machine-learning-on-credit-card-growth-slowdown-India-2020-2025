import streamlit as st
import pandas as pd
import joblib

# Load trained model and RFE
model = joblib.load('model.pkl')
rfe = joblib.load('rfe_selector.pkl')

st.title("📈 Credit Card Growth Slowdown Prediction (India 2020–2025)")

st.markdown("""
This app uses a Lasso Regression model with Recursive Feature Elimination (RFE) to predict credit card growth trends in India.
Upload your dataset or use the default one provided.
""")

# Default sample dataset (from current folder)
default_df = pd.read_csv("credit_card_growth_slowdown_india_2020_2025.csv")

# Upload new data
uploaded_file = st.file_uploader("Upload CSV data (optional)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
else:
    st.info("ℹ️ Using default dataset: credit_card_growth_slowdown_india_2020_2025.csv")
    df = default_df

st.subheader("🔍 Input Data Preview")
st.write(df.head())

# Make predictions
try:
    X_input = rfe.transform(df)
    predictions = model.predict(X_input)

    # Show predictions
    st.subheader("🔮 Model Predictions")
    st.write(predictions)

    # Optionally download results
    output_df = df.copy()
    output_df["Predicted_Value"] = predictions
    csv = output_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Predictions as CSV",
        data=csv,
        file_name="predicted_output.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"🚨 Error: {e}")
