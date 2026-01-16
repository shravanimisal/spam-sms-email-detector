import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 Spam SMS/Email Detection")
st.write("Enter a message to check whether it is **Spam** or **Not Spam**.")

msg = st.text_area("Enter your message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message!")
    else:
        msg_vec = vectorizer.transform([msg])  # ✅ Directly transform raw input
        pred = model.predict(msg_vec)[0]

        if pred == 1:
            st.error("🚨 SPAM Message")
        else:
            st.success("✅ NOT SPAM Message")

