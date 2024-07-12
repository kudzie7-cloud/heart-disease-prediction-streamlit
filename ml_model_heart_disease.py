import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier

model = joblib.load('C:/Users/MwedziK/Desktop/random_forest_classifier.joblib')

def predict(features):
    return model.predict([features])

#streamlit App
st.title('Heart Disease Prediction Model')
st.write("Please Enter the following Details to Predict Heart disease")

#Fields for features
age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0)
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholestoral", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2, value=0)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thal (1-3)", min_value=1, max_value=3, value=2)

# Make prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("The model predicts that the patient has heart disease.")
    else:
        st.write("The model predicts that the patient does not have heart disease.")
