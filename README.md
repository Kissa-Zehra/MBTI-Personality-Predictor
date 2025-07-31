# 🌌 MBTI Personality Predictor  

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)  
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)  
[![Deployed](https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge&logo=vercel)](https://mbti-personality-predictor-ksknslfegj4nzk6yikrqc8.streamlit.app/)

**🔗 Live App:** [MBTI Personality Predictor](https://mbti-personality-predictor-ksknslfegj4nzk6yikrqc8.streamlit.app/)  

---

## 📌 Project Overview
The **MBTI Personality Predictor** is a machine learning web application that predicts a person’s **Myers-Briggs Type Indicator (MBTI)** personality type based on their written text.  

- **Input**: A few sentences about yourself  
- **Output**: Your MBTI personality type (e.g., INTJ, ENFP, ISFJ)  
- **Deployed using**: Streamlit Cloud  

---

## ✨ Features
- 🔹 Predicts **all 16 MBTI personality types**  
- 🔹 Uses **TF-IDF + Logistic Regression** for text classification  
- 🔹 Interactive **dark-themed UI** with live predictions  
- 🔹 **Deployed online** for anyone to try instantly  
- 🔹 Includes **personality type descriptions** for better understanding  

---

## 📊 Tech Stack
- **Programming Language:** Python  
- **Libraries:**  
  - `pandas` & `numpy` → Data processing  
  - `scikit-learn` → Machine Learning (TF-IDF & Logistic Regression)  
  - `matplotlib` & `seaborn` → Visualization  
  - `joblib` → Model serialization  
  - `streamlit` → Web App Deployment  

---

## 🧠 About MBTI
The **Myers-Briggs Type Indicator (MBTI)** classifies people into 16 personality types based on 4 traits:  

- **E/I** – Extroversion / Introversion  
- **S/N** – Sensing / Intuition  
- **T/F** – Thinking / Feeling  
- **J/P** – Judging / Perceiving  

Example personality types:  
`INTJ, ENFP, ISTJ, ESFJ, INTP, ...`  

---

## 🗂️ Project Structure
```
MBTI-Personality-Predictor/
│
├── app.py # Streamlit web app
├── mbti_model.pkl # Pre-trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Project dependencies
├── dataset/
│ └── README.md # Kaggle link for dataset
└── README.md # Project documentation
```


---

## 📦 Installation & Run Locally
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/MBTI-Personality-Predictor.git
cd MBTI-Personality-Predictor
```

2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:
```
streamlit run app.py
```

cat > DATASET_DEPLOYMENT.md << 'EOF'
## 📂 Dataset
The MBTI dataset is available on Kaggle:  
🔗 [Download Here](https://www.kaggle.com/datasets/datasnaek/mbti-type)

> Place `mbti_1.csv` in the project folder before running the app locally.  
> (Not uploaded to GitHub due to large file size)

---

## 🚀 Deployment
The project is deployed on **Streamlit Cloud**.  
🔗 [Try the App Here](https://mbti-personality-predictor-ksknslfegj4nzk6yikrqc8.streamlit.app/)
EOF


---

### 🔹 How to use this:

1. Copy the **entire code block** above and paste it into your **terminal** in the project folder.  
2. Press **Enter**, and it will create a **README.md** file with all content.  
3. Commit and push it to your GitHub repo. ✅  

---


