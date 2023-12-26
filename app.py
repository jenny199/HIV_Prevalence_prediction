import pandas as pd
import joblib
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

import pandas as pd
import joblib
import streamlit as st

st.title("HIV Prevalence Prediction")
st.caption("This app predicts the HIV prevalence based on input features.It will help policy where to properly allocate healthCare Resources")

# Load the saved model and the scaler
model = joblib.load('Linear_regression.pkl')
scaler = joblib.load('standard_scaler.pkl')

# Get user input for all features except Year
Population = st.number_input('Population', min_value=0)
Health_expenditure_per_GDP = st.number_input('Health expenditure (% of GDP)', min_value=0.0, format="%f")
Health_expenditure_per_capita = st.number_input('Health expenditure per capita (current US$)', min_value=0.0, format="%f")
Physicians = st.number_input('Physicians (per 1,000 people)', min_value=0.0, format="%f")

# Create a DataFrame from the inputs
input_data = {
    'Population': [Population],
    'Health_exp_per_gdp': [Health_expenditure_per_GDP],
    'Health_exp_per_capita': [Health_expenditure_per_capita],
    'physians': [Physicians],
}

# Convert the dictionary to a DataFrame
user_input_df = pd.DataFrame(input_data)

# Scale the features using the same scaler previously fitted
# Ensure you exclude 'Year' as it was not included during scaling
scaled_inputs = scaler.transform(user_input_df)

# Predict HIV prevalence using the trained model
if st.button('Predict'):
    # As Year is not scaled, we drop it for prediction as well
    prediction = model.predict(scaled_inputs)
    predicted_prevalence = prediction[0]  # Assuming the model returns a single prediction
    st.write(f'The Predicted Prevalence of HIV, total (% of population ages 15-49) is: {predicted_prevalence}')

