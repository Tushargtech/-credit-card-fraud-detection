import streamlit as st
import joblib
import numpy as np

# Load trained model and features
model = joblib.load('fraud_xgb_model.pkl')
features = joblib.load('features_list.pkl')

# Page settings
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("💳 Real-time Credit Card Fraud Detection")

st.markdown("Enter transaction details to check if it’s fraudulent.")

# Generate dynamic input fields
input_data = []
for feature in features:
    value = st.number_input(f"{feature}", step=0.01, format="%.5f")
    input_data.append(value)

# Convert to 2D array for prediction
input_array = np.array(input_data).reshape(1, -1)

# Predict
if st.button("Check Transaction"):
    prediction = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Transaction is Legitimate. (Confidence: {(1-prob)*100:.2f}%)")
