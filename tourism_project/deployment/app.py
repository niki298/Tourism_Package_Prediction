from pathlib import Path
import json

import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered",
)

# Find the deployment files relative to this script
APP_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(
        APP_DIR / "best_tourism_package_model_v1.joblib"
    )

model = load_model()

with open(APP_DIR / "model_config.json") as file:
    model_config = json.load(file)

classification_threshold = model_config["classification_threshold"]

st.title("Tourism Package Prediction")
st.write(
    "Enter customer details to estimate whether they will "
    "purchase a tourism package."
)
st.info(
    "Use this prediction after customer interaction details "
    "are available, including the sales pitch and follow-ups."
)

st.subheader("Customer and travel details")

Age = st.number_input(
    "Age", min_value=18, max_value=100, value=30, step=1
)

CityTier = st.selectbox("City tier", [1, 2, 3])

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox("Gender", ["Male", "Female"])

MaritalStatus = st.selectbox(
    "Marital status",
    ["Married", "Single", "Divorced", "Unmarried"]
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly income",
    min_value=0.0,
    value=23000.0,
    step=500.0
)

NumberOfPersonVisiting = st.number_input(
    "Total number of people travelling",
    min_value=1,
    value=3,
    step=1
)

NumberOfChildrenVisiting = st.number_input(
    "Number of children below age 5 travelling",
    min_value=0,
    value=1,
    step=1
)

PreferredPropertyStar = st.selectbox(
    "Preferred hotel star rating", [3, 4, 5]
)

NumberOfTrips = st.number_input(
    "Number of trips per year",
    min_value=0,
    value=3,
    step=1
)

Passport = st.selectbox("Has a passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a car?", ["Yes", "No"])

st.subheader("Customer interaction details")

TypeofContact = st.selectbox(
    "Type of contact",
    ["Self Enquiry", "Company Invited"]
)

ProductPitched = st.selectbox(
    "Product pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

DurationOfPitch = st.number_input(
    "Pitch duration in minutes",
    min_value=0.0,
    value=15.0,
    step=1.0
)

NumberOfFollowups = st.number_input(
    "Number of follow-ups",
    min_value=0,
    value=3,
    step=1
)

PitchSatisfactionScore = st.selectbox(
    "Pitch satisfaction score",
    [1, 2, 3, 4, 5],
    index=2
)

if st.button("Predict purchase likelihood"):
    # Check that the travel-party counts are consistent
    if NumberOfChildrenVisiting > NumberOfPersonVisiting:
        st.error(
            "The number of children cannot exceed the total "
            "number of people travelling."
        )
        st.stop()

    # Combine customer inputs into one row
    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": 1 if Passport == "Yes" else 0,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
    }])

    # Match the column order recorded during training
    input_data = input_data[model_config["feature_columns"]]

    # The saved pipeline handles preprocessing automatically
    purchase_score = float(model.predict_proba(input_data)[0, 1])
    prediction = int(purchase_score >= classification_threshold)

    if prediction == 1:
        st.success("Prediction: Likely to purchase")
    else:
        st.info("Prediction: Unlikely to purchase")

    st.caption(
        "This is a model estimate, not a guaranteed outcome."
    )
