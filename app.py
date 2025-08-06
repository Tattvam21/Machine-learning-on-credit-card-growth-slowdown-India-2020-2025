# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

st.set_page_config(page_title="Credit Card Growth Prediction", layout="wide")
st.title("📉 Credit Card Growth Slowdown (India 2020–2025)")
st.write("This app uses **Lasso Regression + RFE** to analyze and predict `Net_Growth_Impact` based on raw credit card data.")

# Load raw data directly from repo
@st.cache_data
def load_data():
    df = pd.read_csv("credit_card_growth_slowdown_india_2020_2025.csv")

    # Drop obviously non-useful columns
    drop_cols = ['Customer_ID', 'Card_Issuance_Date', 'Bank_Name', 'Delinquency_Reason']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

# Load and show data
df_num = load_data()
st.subheader("🔍 Cleaned Data Preview (Numeric Only)")
st.dataframe(df_num.head())

# Check target column
target_column = "Net_Growth_Impact"
if target_column not in df_num.columns:
    st.error(f"Target column '{target_column}' not found.")
else:
    X = df_num.drop(columns=[target_column])
    Y = df_num[target_column]

    # Feature selection config
    st.markdown("### ⚙️ Feature Selection (RFE)")
    n_features = st.slider("Select number of features to keep", 1, min(15, X.shape[1]), value=8)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Apply Lasso + RFE
    model = Lasso(alpha=1.0)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X_train, Y_train)

    # Fit reduced model
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe, Y_train)
    Y_pred = model.predict(X_test_rfe)

    # Evaluation
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    st.markdown("### 📈 Model Performance")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Selected features
    selected_features = X.columns[rfe.support_]
    st.markdown("### ✅ Selected Features")
    st.write(selected_features.tolist())

    # Optional predictions
    if st.checkbox("📊 Show predictions on test set"):
        pred_df = X_test[selected_features].copy()
        pred_df["Actual"] = Y_test.values
        pred_df["Predicted"] = Y_pred
        st.dataframe(pred_df)

    # Save model
    if st.button("💾 Save Model"):
        joblib.dump(rfe, "model.pkl")
        st.success("Model saved as `model.pkl`")
