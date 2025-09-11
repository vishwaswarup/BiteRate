import streamlit as st
import pickle
from preprocessing import clean_text

# Load trained model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Biterate – Restaurant Review Sentiment Analyzer", layout="centered")

st.title("🍽️ Biterate – Restaurant Review Sentiment Analyzer")

st.write("Enter a restaurant review below and see the sentiment score, emoji, and intensity bar!")

review = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if review.strip():
        # Preprocess the input text
        cleaned_review = clean_text(review)
        X = vectorizer.transform([cleaned_review])

        # Get prediction probability
        prob = model.predict_proba(X)[0]  # [prob_negative, prob_positive]
        score = prob[1] * 100  # % positive

        # Determine sentiment and emoji
        if score >= 75:
            sentiment_text = "😄 Very Positive!"
        elif score >= 50:
            sentiment_text = "🙂 Positive"
        elif score >= 25:
            sentiment_text = "😐 Slightly Negative"
        else:
            sentiment_text = "😡 Very Negative"

        # Display results
        st.write(f"**Sentiment:** {sentiment_text} ({score:.1f}% positive)")

        # Progress bar for sentiment intensity
        st.progress(int(score))

    else:
        st.warning("⚠️ Please enter a review to analyze.")