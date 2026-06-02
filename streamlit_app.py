from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

from churn_pipeline import load_model_bundle, parse_input_json, predict_churn

MODEL_PATH = Path("artifacts") / "telco_churn_model.joblib"


st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

st.title("Telco Customer Churn Predictor")

st.sidebar.header("Input")

def load_bundle():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run train.py first.")
        st.stop()
    return load_model_bundle(MODEL_PATH)


def ui_input():
    # Minimal set of fields used by the pipeline
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
    senior = st.sidebar.selectbox("SeniorCitizen", [0, 1], index=0)
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"], index=1)
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"], index=1)
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone = st.sidebar.selectbox("PhoneService", ["Yes", "No"], index=0)
    multiple = st.sidebar.text_input("MultipleLines", "No")
    internet = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"], index=1)
    online_security = st.sidebar.text_input("OnlineSecurity", "No")
    online_backup = st.sidebar.text_input("OnlineBackup", "No")
    device_prot = st.sidebar.text_input("DeviceProtection", "No")
    tech_support = st.sidebar.text_input("TechSupport", "No")
    streaming_tv = st.sidebar.text_input("StreamingTV", "No")
    streaming_movies = st.sidebar.text_input("StreamingMovies", "No")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
    paperless = st.sidebar.selectbox("PaperlessBilling", ["Yes", "No"], index=0)
    payment = st.sidebar.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
    monthly = st.sidebar.number_input("MonthlyCharges", min_value=0.0, value=70.0)
    total = st.sidebar.number_input("TotalCharges", min_value=0.0, value=float(monthly * tenure))

    record = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }
    return record


def main():
    st.sidebar.write("Upload JSON or use the controls to set a single customer record.")
    uploaded = st.sidebar.file_uploader("Upload JSON file", type=["json"])

    if uploaded is not None:
        try:
            raw = uploaded.read().decode("utf-8")
            record = parse_input_json(raw)
        except Exception as exc:
            st.sidebar.error(f"Failed to parse uploaded JSON: {exc}")
            st.stop()
    else:
        record = ui_input()

    bundle = load_bundle()

    st.subheader("Customer record")
    st.json(record)

    if st.button("Predict churn"):
        try:
            pred = predict_churn(bundle, record)
            st.success(f"Prediction: {pred['label']} ({pred['probability']:.2%})")
            st.write("Raw probabilities:")
            st.json(pred.get("probabilities", {}))
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
