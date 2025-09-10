import streamlit as st
import pickle
import re
from preprocessing import clean_text

# Load trained model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Biterate – Restaurant Review Sentiment Analyzer")

review = st.text_area("Enter a restaurant review:")

if st.button("Predict Sentiment"):
    if review:
        # Preprocess the input text
        cleaned_review = clean_text(review)
        X = vectorizer.transform([cleaned_review])
        prediction = model.predict(X)[0]
        st.write("Sentiment:", "Positive" if prediction == 1 else "Negative")
    else:
        st.write("Please enter a review.")