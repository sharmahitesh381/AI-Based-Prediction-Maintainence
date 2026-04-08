import streamlit as st
import pandas as pd

st.title("AI Predictive Maintenance System")

df = pd.read_csv("machine_dataset.csv")

st.subheader("Dataset Preview")
st.write(df.head())

import matplotlib.pyplot as plt

st.subheader("Temperature vs Cycle")

plt.plot(df["Cycle"], df["Temperature"])
plt.xlabel("Cycle")
plt.ylabel("Temperature")

st.pyplot(plt)


st.sidebar.header("Enter Machine Values")

cycle = st.sidebar.slider("Cycle", 1, 200, 50)
temperature = st.sidebar.slider("Temperature", 30, 100, 50)
vibration = st.sidebar.slider("Vibration", 0.0, 5.0, 1.0)
rpm = st.sidebar.slider("RPM", 1000, 2000, 1500)
pressure = st.sidebar.slider("Pressure", 20, 50, 30)



import pickle

lr_model = pickle.load(open("logistic_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))


model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)



import numpy as np

input_data = np.array([[cycle,temperature, vibration, rpm, pressure]])

if model_choice == "Logistic Regression":
    prediction = lr_model.predict(input_data)
else:
    prediction = rf_model.predict(input_data)


st.subheader("Prediction Result")

if prediction[0] == 1:
    st.error("⚠️ Fault Detected")
else:
    st.success("✅ Machine is Healthy")






import pickle

rul_model = pickle.load(open("rul_model_new.pkl", "rb"))

import pandas as pd

input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Vibration": [vibration],
    "RPM": [rpm],
    "Pressure": [pressure]
})


rul = rul_model.predict(input_data)

st.subheader("Remaining Useful Life")

st.info(f"Estimated RUL: {int(rul[0])} cycles")


st.metric("RUL (cycles)", int(rul[0]))






st.markdown("## About Project")

st.write("""
This project develops an AI-based Predictive Maintenance system for early fault detection and Remaining Useful Life (RUL) estimation.

Using sensor data such as Temperature, Vibration, RPM, and Pressure, machine learning models (Logistic Regression and Random Forest) are used to:
- Detect machine faults (classification)
- Predict remaining life (regression)

The system enables reduced downtime, better maintenance planning, and improved equipment efficiency, supporting smart manufacturing and Industry 4.0.
""")
















