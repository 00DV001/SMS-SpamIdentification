import streamlit as st
import re
import joblib

# load the saved model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📱 SMS Spam Detection")
st.write("Enter an SMS message below to check if it's spam or not.")

# input box
user_input = st.text_area("Enter SMS text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean_text = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is **NOT SPAM**.")
