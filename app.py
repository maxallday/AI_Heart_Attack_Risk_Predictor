import streamlit as st
import numpy as np
import tensorflow as tf
import joblib  # ✅ Used for loading the saved scaler
import requests  # ✅ Used for fetching real-time health data

# ✅ **Load trained AI model**
model = tf.keras.models.load_model("model/heart_disease_model.h5")

# ✅ **Load the saved scaler instead of refitting**
scaler = joblib.load("model/scaler.pkl")

# 🎯 **Streamlit App UI Setup**
st.title("🚑 AI Heart Attack Risk Predictor")
st.write("Enter your health details or connect wearable devices for real-time tracking.")



# 🎥 Add Animated Heartbeat Lottie Player
#optionally, you can use a Lottie animation for a heartbeat effect
import streamlit.components.v1 as components
# ✅ Use a Lottie player component for the heartbeat animation
st.components.v1.html("""
    <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
    <dotlottie-player src="https://lottie.host/5a6857d1-6275-4fec-9b86-fe136e194bc4/NP8ZpjJ7tF.lottie" background="transparent" speed="1" style="width: 100px; height: 100px" loop autoplay></dotlottie-player>
""", height=100)  # ✅ Adjust height for proper display



# 🩺 **User Health Inputs Section**
# ✅ **Manual Health Inputs (for users without wearables)**
age = st.slider("Age", 20, 80)
sex = st.selectbox("Sex", ["Male", "Female"])  # 🟢 Fixed syntax
sex = 1 if sex == "Male" else 0  # ✅ Convert to numeric

cholesterol = st.slider("Cholesterol Level", 100, 300)
blood_pressure = st.slider("Blood Pressure", 80, 200)

cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
cp = {"Typical Angina": 1, "Atypical Angina": 2, "Non-Anginal Pain": 3, "Asymptomatic": 4}[cp]  # ✅ Convert to numeric

fbs = st.slider("Fasting Blood Sugar (>120 mg/dl = 1, else 0)", 0, 1)
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
restecg = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]  # ✅ Convert to numeric

heart_rate = st.slider("Heart Rate (BPM)", 50, 120)
exang = st.slider("Exercise-Induced Angina (1 = Yes, 0 = No)", 0, 1)
oldpeak = st.slider("ST Depression (Induced by Exercise)", 0.0, 6.0, step=0.1)
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
smoking = 1 if smoking == "Yes" else 0  # ✅ Convert to numeric

# 📡 **Real-Time Data Integration (Smartwatch / Medical Devices)**
try:
    response = requests.get("https://api.mockhealthdata.com/heart")  # Example API
    if response.status_code == 200:
        health_data = response.json()
        st.write("✅ Real-time data received.")
        stress_level = health_data["stress_level"]
        oxygen_level = health_data["oxygen_saturation"]
    else:
        st.write("⚠️ Using manual inputs.")
        stress_level = st.slider("Stress Level", 1, 10)
        oxygen_level = st.slider("Oxygen Saturation (%)", 80, 100)
except:
    st.write("⚠️ No wearable device connected.")
    stress_level = st.slider("Stress Level", 1, 10)
    oxygen_level = st.slider("Oxygen Saturation (%)", 80, 100)

# 🏥 **Prepare Input Data (All 13 Features)**
user_data = np.array([[age, sex, cp, cholesterol, fbs, restecg, blood_pressure, heart_rate, exang, oldpeak, stress_level, oxygen_level, smoking]])

# ✅ **Transform user data using the loaded scaler**
user_data_scaled = scaler.transform(user_data)

# 🚨 **Predict and Analyze Risk**
if st.button("Predict Risk"):
    risk_score = model.predict(user_data_scaled)

    st.write(f"🩺 **Predicted Risk Score:** {risk_score[0][0]:.2f}")

    if risk_score[0][0] > 0.80:
        st.error("🚨 High Risk! Consult a doctor immediately.")
    elif risk_score[0][0] > 0.50:
        st.warning("⚠️ Moderate Risk. Maintain a healthy lifestyle.")
    else:
        st.success("✅ Low Risk! Keep up a healthy routine.")


# 🏃‍♂️ **Heart Health Tips Section**
st.header("💖 How to Keep a Healthy Heart")
st.write("""
A healthy heart leads to a longer, better life! Here are expert-backed tips to reduce heart disease risks:
🥦 **Eat a balanced diet**: Focus on fruits, veggies, lean proteins, and whole grains. Avoid excessive sugar and processed foods.  
🏃 **Stay active**: Aim for at least **30 minutes of exercise per day** (walking, jogging, swimming, or cycling).  
🧘 **Manage stress**: Practice relaxation techniques like meditation or deep breathing.  
🚭 **Quit smoking**: Smoking damages blood vessels and significantly raises heart attack risk.  
🍷 **Limit alcohol intake**: Excessive drinking can lead to high blood pressure and heart issues.  
📈 **Monitor blood pressure & cholesterol**: Keep these levels in check through regular screenings.  
😴 **Get enough sleep**: Poor sleep increases the risk of heart disease.  
💧 **Stay hydrated**: Drink plenty of water to maintain good circulation.  
⚖️ **Maintain a healthy weight**: Obesity is a major risk factor for heart disease.  
🩺 **Consult a doctor regularly**: Early detection is key for heart health.

💡 _A small lifestyle change today can lead to a healthier, stronger heart tomorrow!_
""")
st.write("For more tips, visit [Heart Health Tips](https://www.heart.org/en/healthy-living)")   
st.markdown (
    """
    <div style="text-align:center">
    Disclaimer:
    <br>
    <strong>⚠️ Important Notice:</strong> This application is for informational purposes only and does not constitute medical advice.
    This AI model is trained on 2024 heart disease data by Oktay Ördekçi on Kaggle and is intended as an assistive tool for healthcare professionals and casual users.  
        It should <strong>not replace professional medical advice</strong>. Always consult a qualified physician for accurate diagnosis.  
        <br><br>
    💻 Developed by Modiga, 2025

    </div>
   """,
   unsafe_allow_html=True
)
