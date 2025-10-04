# app.py
import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
from src import config

# --- Page Configuration and Styling ---
st.set_page_config(page_title="Exoplanet Explorer", page_icon="🪐", layout="wide")
# (Your CSS remains the same)
st.markdown("""<style>...</style>""", unsafe_allow_html=True) 

# --- Load All Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads the trained model, scaler, and feature selector."""
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    selector = joblib.load(config.FEATURE_SELECTOR_SAVE_PATH)
    return model, scaler, selector

model, scaler, selector = load_artifacts()

# --- App Layout ---
st.title("🪐 Kepler Exoplanet Explorer v2.0 🛰️")
st.write("An enhanced classifier using feature selection and a robust neural network.")

with st.sidebar:
    st.header("🚀 Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    user_df = pd.read_csv(uploaded_file, comment='#')
    st.subheader("Uploaded Data Preview")
    st.write(user_df.head())

    with st.spinner('Running advanced cosmic analysis...'):
        # --- Full Prediction Pipeline ---
        # 1. Engineer features (must match training)
        X_user = user_df.copy()
        X_user = X_user.apply(pd.to_numeric, errors='coerce')
        X_user['signal_to_noise_ratio_composite'] = X_user['koi_max_mult_ev'] / (X_user['koi_max_sngle_ev'] + 1e-6)
        
        # Ensure column order and presence matches the training data
        training_cols = selector.feature_names_in_
        X_user_aligned = X_user.reindex(columns=training_cols).fillna(0)

        # 2. Apply feature selection
        X_user_selected = selector.transform(X_user_aligned)
        
        # 3. Apply scaling
        X_user_scaled = scaler.transform(X_user_selected)
        
        # 4. Predict
        predictions = model.predict(X_user_scaled)
        
        # Format results
        predicted_labels = (predictions > 0.5).astype(int).flatten()
        results_df = user_df.copy()
        results_df['Prediction'] = ['Exoplanet' if lbl == 1 else 'Not an Exoplanet' for lbl in predicted_labels]
        results_df['Confidence'] = predictions.flatten()

    st.success("Analysis complete!")
    st.subheader("Prediction Results")
    st.write(results_df[['kepoi_name', 'Prediction', 'Confidence']])
    
    st.subheader("Prediction Summary")
    st.bar_chart(results_df['Prediction'].value_counts())

else:
    st.info("Awaiting new Kepler data for analysis...")