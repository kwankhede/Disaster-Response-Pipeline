import streamlit as st
import joblib
from text_processing import (
    tokenize,
)  # Import the tokenize function from text_processing.py

# Load model
model = joblib.load("../models/classifier.pkl")

st.title("Text Classification App")

# Create a text input box for user input
user_input = st.text_area("Enter your text here:")

# Create a button to trigger the prediction
if st.button("Predict"):
    if user_input:
        # Tokenize the input
        tokens = tokenize(user_input)

        # Perform the prediction using your model
        prediction = model.predict(tokens)[0]

        # Display the prediction result
        if prediction == 1:
            st.write("Prediction: Positive")
        else:
            st.write("Prediction: Negative")
    else:
        st.warning("Please enter text for prediction.")
