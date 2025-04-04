# %%
import streamlit as st
import pandas as pd
import joblib
import category_encoders
# %%
# Set up the page title
st.title("Salary Prediction App")

# %%
# Load the pre-trained salary prediction pipeline/model
# Ensure that 'salary_predictor.pkl' is in the same directory as this script.
model = joblib.load('salary_predictor.pkl')

# %%
# Sidebar for user input features
st.sidebar.header("Input Features")

# %%
# Input: Age
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)

# Input: Gender
gender = st.sidebar.selectbox("Gender", options=["Male", "Female", "Other"])

# Input: Education Level
# Adjust the options to match those used during your training/cleaning process.
education = st.sidebar.selectbox("Education Level", options=["High School", "Bachelor's", "Master's", "PhD"])

# Input: Job Title
# For a production app, you might want to use a selectbox if the job titles are fixed.
job_title = st.sidebar.text_input("Job Title", "Data Analyst")

# Input: Years of Experience
experience = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.5)


# %%
# When the user clicks the "Predict Salary" button...
if st.sidebar.button("Predict Salary"):
    # Create a DataFrame for the input data.
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job_title],
        'Years of Experience': [experience]
    })

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)

    # Display the prediction result
    st.subheader("Predicted Salary")
    st.write(f"${prediction[0]:,.2f}")


