import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Ensure these names match your GitHub files exactly
    model = pickle.load(open('house_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# --- 3. UI/SIDEBAR ---
st.sidebar.title("Project Details")
st.sidebar.info("""
**Type:** Regression  
**Model:** Linear Regression  
**R² Score:** 0.94  
**Status:** Internship Submission
""")

st.title("🏡 Real Estate Price Estimator")
st.markdown("Enter property details below to get an AI-driven valuation.")
st.divider()

# --- 4. USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Square Footage", min_value=100.0, value=1500.0, step=50.0, format="%.2f")
    beds = st.number_input("Number of Bedrooms", min_value=1, value=3, step=1)
    baths = st.number_input("Number of Bathrooms", min_value=1.0, value=2.0, step=0.5, format="%.1f")
    year = st.number_input("Year Built", min_value=1800, max_value=2026, value=2010, step=1)

with col2:
    lot = st.number_input("Lot Size (sq ft)", min_value=100.0, value=5000.0, step=100.0, format="%.2f")
    garage = st.number_input("Garage Size (Cars)", min_value=0, value=2, step=1)
    quality = st.select_slider("Neighborhood Quality", options=list(range(1, 11)), value=5)

st.divider()

# --- 5. PREDICTION LOGIC ---
if st.button("Calculate Estimated Price", use_container_width=True):
    # CRITICAL: Order must match your print(list(X.columns)) result:
    # ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']
    features = np.array([[sqft, beds, baths, year, lot, garage, quality]])
    
    # 1. Scale input
    scaled_features = scaler.transform(features)
    
    # 2. Predict (This returns LOG value)
    log_pred = model.predict(scaled_features)
    
    # 3. Inverse Log (Using expm1 because you used log1p in training)
    final_price = np.expm1(log_pred)[0]
    
    # 4. Result Presentation
    st.balloons()
    st.success(f"### Estimated Market Value: ${final_price:,.2f}")
    st.caption("Valuation generated based on historical training data distribution.")
