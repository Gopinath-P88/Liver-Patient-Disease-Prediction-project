
# Liver Patient Prediction Web App
# Purpose: Create a simple interface using Streamlit.

import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Liver Health Prediction | AI/ML Project",
    page_icon="🩺",
    layout="centered"
)


# Professional app header
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #12304A, #0F766E);
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 20px;
        color: white;
    ">
        <h1 style="color: white; margin-bottom: 8px;">
            🩺 Liver Health Prediction
        </h1>
        <p style="color: #E2F5F3; margin-bottom: 0;">
            AI/ML Classification Project
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# Professional healthcare theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F8FB;
        color: #17324D;
    }

    h1, h2, h3 {
        color: #12304A;
    }

    div.stButton > button {
        background-color: #0F766E;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #115E59;
        color: white;
        border: none;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 8px;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Introduction card
st.markdown(
    """
    <div style="
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #0F766E;
        margin-bottom: 20px;
    ">
        <h3 style="color: #12304A; margin-top: 0;">
            Welcome to Liver Health Prediction
        </h3>
        <p style="color: #425466; margin-bottom: 8px;">
            Enter the required input features to get a prediction
            from the trained machine learning model.
        </p>
        <p style="color: #8A4B08; margin-bottom: 0;">
            <strong>Important:</strong> Educational use only.
            This model is not validated for medical diagnosis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Load the saved model
model = joblib.load("liver_patient_model.pkl")

# Patient input form
st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age", min_value=1, max_value=90, value=45
    )
    gender = st.selectbox("Gender", ["Female", "Male"])
    total_bilirubin = st.number_input(
        "Total Bilirubin", min_value=0.0, value=1.0
    )
    direct_bilirubin = st.number_input(
        "Direct Bilirubin", min_value=0.0, value=0.3
    )
    alkaline_phosphotase = st.number_input(
        "Alkaline Phosphotase", min_value=0, value=200
    )

with col2:
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
if st.button(
    "🔍 Predict Result",
    type="primary",
    use_container_width=True
):
    prediction = model.predict(patient_data)[0]

    # Display prediction result
    st.subheader("📊 Prediction Result")
    st.info(f"Model output: Class {prediction}")

    st.caption(
        "This is a machine learning classification, "
        "not a medical diagnosis."
    )

# Footer
st.divider()
st.caption(
    "Built with Python, Scikit-learn, and Streamlit | "
    "AI/ML Portfolio Project"
)
