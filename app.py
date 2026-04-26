import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load assets
# Using try-except to catch loading errors early
try:
    model = pickle.load(open('house_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or Scaler file not found. Check your GitHub filenames!")

st.title("🏡 House Price Predictor")
st.write("Enter the details below to estimate the house value.")

col1, col2 = st.columns(2)

with col1:
    # Use 0.0 to force float type as per instructor feedback
    sqft = st.number_input("Square Footage", min_value=100.0, value=1500.0)
    beds = st.number_input("Number of Bedrooms", min_value=1.0, value=3.0)
    baths = st.number_input("Number of Bathrooms", min_value=1.0, value=2.0)
    year = st.number_input("Year Built", min_value=1800.0, max_value=2026.0, value=2015.0)

with col2:
    lot = st.number_input("Lot Size (sq ft)", min_value=100.0, value=5000.0)
    garage = st.number_input("Garage Size (Cars)", min_value=0.0, value=2.0)
    quality = st.slider("Neighborhood Quality (1-10)", 1.0, 10.0, 5.0)

if st.button("Predict Price"):
    # 1. Verification of Order
    features = np.array([[sqft, beds, baths, year, lot, garage, quality]])
    
    # 2. Scale
    scaled_features = scaler.transform(features)
    
    # 3. Predict
    prediction = model.predict(scaled_features)
    raw_val = prediction[0]
    
    # --- LOGIC SELECTION ---
    # If raw_val is small (like 5.0), it's a LOG. If it's huge (500,000), it's RAW.
    if raw_val < 30: 
        final_price = np.expm1(raw_val)
    else:
        final_price = raw_val

    # 4. Show Result
    st.success(f"### Estimated Market Value: ${final_price:,.2f}")
    
    # 5. DEBUG (Remove this after it works)
    with st.expander("See Model Debugging Info"):
        st.write(f"Raw Model Output: {raw_val}")
        st.write(f"Scaled Inputs: {scaled_features}")
