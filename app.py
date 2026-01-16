import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords

nltk.download("stopwords")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Spam SMS/Email Detection")

msg = st.text_area("Enter your message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message!")
    else:
        cleaned = clean_text(msg)
        msg_vec = vectorizer.transform([cleaned])
        pred = model.predict(msg_vec)[0]

        if pred == 1:
            st.error("🚨 SPAM Message")
        else:
            st.success("✅ NOT SPAM Message")
