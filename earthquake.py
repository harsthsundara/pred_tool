import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Sample function to predict risk level based on magnitude and location
def predict_earthquake_risk(magnitude, depth, latitude, longitude):
    # Basic rule-based prediction (replace with ML model for better accuracy)
    risk = "Low"
    if magnitude > 6.0 or depth < 50:  # Higher magnitude or shallow depth is risky
        risk = "High"
    elif magnitude > 4.0:
        risk = "Medium"
    
    return risk

# Streamlit UI
st.title("Earthquake Risk Prediction App")
st.write("Enter earthquake parameters to estimate the risk level.")

# User Inputs
magnitude = st.slider("Magnitude of Earthquake", 2.0, 10.0, 5.0, step=0.1)
depth = st.slider("Depth (km)", 0, 700, 50, step=10)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)

# Predict Risk
if st.button("Predict Risk"):
    risk_level = predict_earthquake_risk(magnitude, depth, latitude, longitude)
    st.subheader(f"Predicted Earthquake Risk Level: {risk_level}")

    # Show warning for high-risk zones
    if risk_level == "High":
        st.warning("⚠️ This zone has a **high** earthquake risk! Take precautions.")
    elif risk_level == "Medium":
        st.info("⚠️ This zone has a **moderate** earthquake risk.")
    else:
        st.success("✅ This zone has a **low** earthquake risk.")

# Footer
st.write("🔍 This is a simple model. For accurate predictions, use real-time seismic data.")
