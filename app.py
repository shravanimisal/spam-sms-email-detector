import streamlit as st
import pickle

# -------------------------------
# Load trained pipeline
# -------------------------------
pipeline = pickle.load(open("spam_pipeline.pkl", "rb"))

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.title("📩 Spam SMS/Email Detection")
st.write("Enter a message below and click Predict to check whether it is Spam or Not Spam.")

msg = st.text_area("✍️ Enter your message here:", height=140)

# Optional: show examples
with st.expander("✅ Try sample messages"):
    st.markdown("""
**Spam Example:**  
URGENT! You have won a 1 week FREE membership. Call now to claim.

**Not Spam Example:**  
Hello, are you coming to college today?
""")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        prediction = pipeline.predict([msg])[0]

        # Probability (confidence)
        prob = pipeline.predict_proba([msg])[0]
        spam_prob = prob[1] * 100
        ham_prob = prob[0] * 100

        if prediction == 1:
            st.error(f"🚨 SPAM Message (Spam probability: {spam_prob:.2f}%)")
        else:
            st.success(f"✅ NOT SPAM Message (Not spam probability: {ham_prob:.2f}%)")
