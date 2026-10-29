import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Store & Item Demand Predictor", page_icon="📦", layout="centered")
st.title("📦 Store & Item Monthly Demand Prediction")
st.write("Predict estimated monthly units sold for a selected store and item.")

MODEL_PATH = Path(__file__).parent / "best_model.joblib"
FEATURES_PATH = Path(__file__).parent / "feature_columns.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    return joblib.load(FEATURES_PATH)

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Model files are missing. Run the Colab notebook through model saving, then copy best_model.joblib and feature_columns.joblib into this folder.")
    st.stop()

model = load_model()
features = load_features()

store_id = st.number_input("Store ID", min_value=1, max_value=10, value=1, step=1)
item_id = st.number_input("Item ID", min_value=1, max_value=50, value=1, step=1)
year = st.number_input("Forecast year", min_value=2013, max_value=2035, value=2018, step=1)
month = st.selectbox("Forecast month", list(range(1, 13)), index=0,
                     format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
quarter = (month - 1) // 3 + 1
input_df = pd.DataFrame([{
    "store": int(store_id),
    "item": int(item_id),
    "year": int(year),
    "month": int(month),
    "quarter": int(quarter),
    "month_sin": np.sin(2 * np.pi * month / 12),
    "month_cos": np.cos(2 * np.pi * month / 12),
}])[features]

if st.button("Predict Monthly Demand", type="primary", use_container_width=True):
    prediction = max(0.0, float(model.predict(input_df)[0]))
    st.metric("Estimated monthly units", f"{prediction:,.0f}")
    st.caption(f"Unrounded model estimate: {prediction:,.2f} units")
    st.info("This is a model estimate, not a guarantee of exact future demand.")
    st.subheader("Prediction inputs")
    st.dataframe(input_df, use_container_width=True)
