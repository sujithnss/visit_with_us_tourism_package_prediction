import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model for Tourism Package Purchase Prediction
model_path = hf_hub_download(repo_id="sujithpv/visit-with-us-tourism-package-prediction", filename="visit_with_us_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Visit WIith Us Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package** 
based on their personal details and interaction data. Please enter the customer details below.
""")

# Collect user input
age = st.number_input("Age", min_value=0, max_value=120, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Student", "Retired", "Others"])
gender = st.selectbox("Gender", ["Male", "Female"])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.text_input("Designation")
productpitched = st.selectbox("Product Pitched", ["Wellness Package", "Adventure Package", "Cultural Package", "Other"])
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=20, value=1)
preferredpropertystar = st.selectbox("Preferred Hotel Star Rating", [1, 2, 3, 4, 5])
numberoftrips = st.number_input("Average Number of Trips per Year", min_value=0, max_value=100, value=1)
passport = st.selectbox("Has Passport?", ["No", "Yes"])
owncar = st.selectbox("Owns Car?", ["No", "Yes"])
numberofchildrenvisiting = st.number_input("Number of Children below 5 years", min_value=0, max_value=10, value=0)
monthlyincome = st.number_input("Monthly Income (in your currency)", min_value=0)
pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=0, max_value=10, value=5)
numberoffollowups = st.number_input("Number of Followups by Salesperson", min_value=0, max_value=50, value=0)
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=180, value=10)

# Prepare input DataFrame for model (match feature order used in training)
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': maritalstatus,
    'Designation': designation,
    'ProductPitched': productpitched,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'NumberOfTrips': numberoftrips,
    'Passport': 1 if passport == "Yes" else 0,
    'OwnCar': 1 if owncar == "Yes" else 0,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'MonthlyIncome': monthlyincome,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'NumberOfFollowups': numberoffollowups,
    'DurationOfPitch': durationofpitch
}])

# On Predict button click
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"The customer is likely to purchase the Wellness Tourism Package. (Confidence: {prediction_proba:.2%})")
    else:
        st.warning(f"The customer is unlikely to purchase the Wellness Tourism Package. (Confidence: {(1-prediction_proba):.2%})")
