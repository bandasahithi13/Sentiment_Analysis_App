import streamlit as st
import pickle

# Load model
model = pickle.load(open("model1.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer1.pkl", "rb"))

# Title
st.title("Sentiment Analysis App")

# Input box
review = st.text_area("Enter Review")

# Predict button
if st.button("Predict"):

    # Convert text into vectors
    review_vector = vectorizer.transform([review])

    # Prediction
    prediction = model.predict(review_vector)

    # Store prediction
    result = prediction[0]

    # Display result
    if result == "Positive":
        st.success("Predicted Sentiment: Positive 😊")

    elif result == "Negative":
        st.error("Predicted Sentiment: Negative 😞")

    else:
        st.warning("Predicted Sentiment: Neutral 😐")