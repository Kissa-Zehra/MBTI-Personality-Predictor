import streamlit as st
import joblib

# --------------------- CONFIG ---------------------
st.set_page_config(page_title="MBTI Personality Predictor", layout="centered")

# Load pre-trained model and vectorizer (FAST)
@st.cache_resource
def load_model():
    model = joblib.load("mbti_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Personality type descriptions
mbti_meanings = {
    "INTJ": "Introvert, Intuitive, Thinking, Judging",
    "INTP": "Introvert, Intuitive, Thinking, Perceiving",
    "ENTJ": "Extrovert, Intuitive, Thinking, Judging",
    "ENTP": "Extrovert, Intuitive, Thinking, Perceiving",
    "INFJ": "Introvert, Intuitive, Feeling, Judging",
    "INFP": "Introvert, Intuitive, Feeling, Perceiving",
    "ENFJ": "Extrovert, Intuitive, Feeling, Judging",
    "ENFP": "Extrovert, Intuitive, Feeling, Perceiving",
    "ISTJ": "Introvert, Sensing, Thinking, Judging",
    "ISFJ": "Introvert, Sensing, Feeling, Judging",
    "ESTJ": "Extrovert, Sensing, Thinking, Judging",
    "ESFJ": "Extrovert, Sensing, Feeling, Judging",
    "ISTP": "Introvert, Sensing, Thinking, Perceiving",
    "ISFP": "Introvert, Sensing, Feeling, Perceiving",
    "ESTP": "Extrovert, Sensing, Thinking, Perceiving",
    "ESFP": "Extrovert, Sensing, Feeling, Perceiving"
}

# --------------------- UI ---------------------
st.title("🌌 MBTI Personality Predictor")
st.write("Predict your **MBTI personality type** from your text")

# Sidebar info
st.sidebar.header("ℹ️ About the App")
st.sidebar.info("""
**MBTI Personality Predictor** is a machine learning web app  
that analyzes your written text and predicts your MBTI personality.

### 🔹 Features
- Predicts personality types like **INTJ, ENFP, ISFJ, etc.**  
- Uses **TF-IDF** + **Logistic Regression** for predictions  
- Supports **all 16 MBTI categories**  

### 🔹 About MBTI
The **Myers-Briggs Type Indicator (MBTI)** divides personalities into:  
- **E/I** – Extroversion / Introversion  
- **S/N** – Sensing / Intuition  
- **T/F** – Thinking / Feeling  
- **J/P** – Judging / Perceiving  

This app allows you to **explore your personality** based on how you express yourself in writing.
""")

# ----------- TAB 1: Prediction -----------
tab1, = st.tabs(["Predict Personality"])

with tab1:
    st.subheader("💬 Enter your text below")
    user_input = st.text_area("Type a few sentences about yourself:")

    if st.button("🔮 Predict Personality") and user_input.strip():
        input_vect = vectorizer.transform([user_input])
        prediction = model.predict(input_vect)[0]

        st.markdown(f"""
            <div style='background-color:#1c1e24;padding:20px;border-radius:15px;text-align:center;box-shadow:0 0 10px rgba(0,0,0,0.5);font-size:20px;'>
                <span style='font-size:28px;font-weight:bold;color:#00c4ff;'>{prediction}</span><br>
                <span style='font-size:18px;color:#cccccc;'>{mbti_meanings.get(prediction, 'Unknown Type')}</span>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---\nCreated with ❤️ by Kissa Zehra")
