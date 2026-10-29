
# Liver Patient Prediction Web App
# Purpose: Create a simple interface using Streamlit.

import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Liver Patient Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load the saved model
model = joblib.load("liver_patient_model.pkl")

# App title
st.title("🩺 Liver Patient Prediction")
st.write("Enter the patient details below.")

st.warning(
    "Educational project only. This model is not "
    "validated for medical diagnosis."
)

# Patient details
age = st.number_input("Age", min_value=1, max_value=90, value=45)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

total_bilirubin = st.number_input(
    "Total Bilirubin", min_value=0.0, value=1.0
)

direct_bilirubin = st.number_input(
    "Direct Bilirubin", min_value=0.0, value=0.3
)

alkaline_phosphotase = st.number_input(
    "Alkaline Phosphotase", min_value=0, value=200
)

alamine_aminotransferase = st.number_input(
    "Alamine Aminotransferase", min_value=0, value=30
)

aspartate_aminotransferase = st.number_input(
    "Aspartate Aminotransferase", min_value=0, value=35
)

total_proteins = st.number_input(
    "Total Proteins", min_value=0.0, value=6.5
)

albumin = st.number_input(
    "Albumin", min_value=0.0, value=3.2
)

albumin_globulin_ratio = st.number_input(
    "Albumin/Globulin Ratio", min_value=0.0, value=0.9
)

# Prepare input data
patient_data = pd.DataFrame([{
    "Age": age,
    "Gender": 0 if gender == "Female" else 1,
    "Total_Bilirubin": total_bilirubin,
    "Direct_Bilirubin": direct_bilirubin,
    "Alkaline_Phosphotase": alkaline_phosphotase,
    "Alamine_Aminotransferase": alamine_aminotransferase,
    "Aspartate_Aminotransferase": aspartate_aminotransferase,
    "Total_Proteins": total_proteins,
    "Albumin": albumin,
    "Albumin_Globulin_Ratio": albumin_globulin_ratio
}])

# Predict button
if st.button("Predict Target"):
    prediction = model.predict(patient_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.write("Model output: Class 1")
    else:
        st.write("Model output: Class 2")

    st.caption(
        "This is a machine learning classification, "
        "not a medical diagnosis."
    )
