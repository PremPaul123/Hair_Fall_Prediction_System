import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load the trained model and preprocessor
model = tf.keras.models.load_model("hairfall_model.h5")
preprocessor = joblib.load("scaler.pkl")  # Contains both scaler & encoders

scaler = preprocessor["scaler"]
encoders = preprocessor.get("encoders", {})  # Load encoders if available

# Define expected features
columns = [
    "What_is_your_age", "What_is_your_gender", 
    "Is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
    "Did_you_face_any_type_of_chronic_illness_in_the_past",
    "Do_you_stay_up_late_at_night", "Do_you_have_any_type_of_sleep_disturbance",
    "Do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems",
    "Do_you_use_chemicals_hair_gel_or_color_in_your_hair", "Do_you_have_anemia",
    "Do_you_have_too_much_stress", "What_is_your_food_habit"
]

# Streamlit UI
st.title("Hair Fall Risk Prediction & Personalized Recommendations")
st.write("Fill in the details below to predict your hair fall risk and receive expert recommendations.")

# User Input Fields
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History of Hair Fall", ["Yes", "No"])
chronic_illness = st.selectbox("Chronic Illness", ["Yes", "No"])
late_night = st.selectbox("Frequent Late Nights", ["Yes", "No"])
sleep_disturbance = st.selectbox("Sleep Disturbance", ["Yes", "No"])
water_quality = st.selectbox("Poor Water Quality", ["Yes", "No"])
chemical_usage = st.selectbox("Frequent Use of Hair Chemicals", ["Yes", "No"])
anemia = st.selectbox("Anemia", ["Yes", "No"])
stress = st.selectbox("High Stress Levels", ["Yes", "No"])
food_habit = st.selectbox("Food Habit", ["Nutritious", "Both", "Dependent on fast food"])

# Convert categorical inputs using encoders
input_data = pd.DataFrame([[age, gender, family_history, chronic_illness, late_night,
                            sleep_disturbance, water_quality, chemical_usage, anemia,
                            stress, food_habit]], columns=columns)

for col in ["What_is_your_gender", "What_is_your_food_habit"]:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

# Convert Yes/No columns to 1/0
binary_map = {"Yes": 1, "No": 0}
binary_columns = [
    "Is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
    "Did_you_face_any_type_of_chronic_illness_in_the_past",
    "Do_you_stay_up_late_at_night", "Do_you_have_any_type_of_sleep_disturbance",
    "Do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems",
    "Do_you_use_chemicals_hair_gel_or_color_in_your_hair", "Do_you_have_anemia",
    "Do_you_have_too_much_stress"
]

for col in binary_columns:
    input_data[col] = input_data[col].map(binary_map)

# Normalize Age Column
input_data["What_is_your_age"] = scaler.transform(input_data[["What_is_your_age"]])

# Function for recommendations
def get_recommendations(features):
    recommendations = []
    
    if features[0] < 25:
        recommendations.append("Ensure adequate protein intake for hair growth.")
    if features[1] == 1 and features[8] == 1:
        recommendations.append("Iron deficiency (Anemia) is a major cause of hair fall. Consider iron-rich foods.")
    if features[9] == 1:
        recommendations.append("Manage stress with meditation and exercise.")
    if features[5] == 1:
        recommendations.append("Improve sleep quality and maintain a consistent sleep schedule.")
    if features[7] == 1:
        recommendations.append("Reduce the use of hair chemicals and switch to natural products.")
    if features[10] == 2:
        recommendations.append("Fast food diets lack essential nutrients. Shift towards a balanced diet.")
    
    return recommendations

# Prediction & Recommendations
if st.button("Predict Hair Fall Risk"):
    try:
        input_transformed = input_data.to_numpy()
        prediction = model.predict(input_transformed)
        risk_score = prediction[0][0] * 100
        
        st.subheader("Prediction Result:")
        if risk_score > 70:
            st.error(f"High Hair Fall Risk: {risk_score:.2f}%")
        elif risk_score > 40:
            st.warning(f"Moderate Hair Fall Risk: {risk_score:.2f}%")
        else:
            st.success(f"Low Hair Fall Risk: {risk_score:.2f}%")
        
        # Show Recommendations
        st.subheader("Personalized Recommendations")
        recommendations = get_recommendations(input_transformed[0])
        if recommendations:
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write("Great! Your hair care habits are on track.")
    except Exception as e:
        st.error(f"Error: {e}")
