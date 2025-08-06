# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

st.set_page_config(page_title="Credit Card Growth Prediction", layout="wide")
st.title("📉 Credit Card Growth Slowdown (India 2020–2025)")
st.write("Lasso Regression + RFE model to predict `Net_Growth_Impact` using numeric features.")

# Upload dataset
uploaded_file = st.file_uploader("credit_card_growth_slowdown_india_2020_2025.csv", type=["csv"])

if uploaded_file:
    df_num = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Numeric Data")
    st.dataframe(df_num.head())

    # Set default target column
    target_column = "Net_Growth_Impact"

    if target_column not in df_num.columns:
        st.error(f"Target column '{target_column}' not found in uploaded data.")
    else:
        # Define features and label
        X = df_num.drop(columns=[target_column])
        Y = df_num[target_column]

        # Let user select number of features
        st.markdown("### ⚙️ Feature Selection (RFE)")
        n_features = st.slider("Select number of features to keep", 1, min(15, X.shape[1]), value=8)

        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Apply Lasso + RFE
        model = Lasso(alpha=1.0)
        rfe = RFE(model, n_features_to_select=n_features)
        rfe.fit(X_train, Y_train)

        # Reduce features
        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)

        # Train Lasso
        model.fit(X_train_rfe, Y_train)
        Y_pred = model.predict(X_test_rfe)

        # Evaluate
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        st.markdown("### 📈 Model Evaluation")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Show selected features
        st.markdown("### ✅ Selected Features")
        selected_features = X.columns[rfe.support_]
        st.write(selected_features.tolist())

        # Show predictions
        if st.checkbox("📊 Show predictions on test set"):
            pred_df = X_test[selected_features].copy()
            pred_df["Actual"] = Y_test.values
            pred_df["Predicted"] = Y_pred
            st.dataframe(pred_df)

        # Save model
        if st.button("💾 Save Model"):
            joblib.dump(rfe, "model.pkl")
            st.success("Model saved as `model.pkl`")
