import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("backend/diabetes_model.h5", compile=False)
scaler = joblib.load("backend/scaler.pkl")

def predict(p, g, bp, s, i, bmi, dpf, age):
    data = np.array([[p, g, bp, s, i, bmi, dpf, age]])
    data = scaler.transform(data)

    pred = model.predict(data)[0][0]

    if pred > 0.5:
        return f"🔴 Diabetic ({pred*100:.2f}%)"
    else:
        return f"🟢 Not Diabetic ({(1-pred)*100:.2f}%)"

# UI
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 15, value=2, label="Pregnancies"),
        gr.Slider(50, 200, value=100, label="Glucose"),
        gr.Slider(40, 120, value=70, label="Blood Pressure"),
        gr.Slider(10, 60, value=25, label="Skin Thickness"),
        gr.Slider(0, 300, value=80, label="Insulin"),
        gr.Slider(15, 50, value=25, label="BMI"),
        gr.Slider(0.1, 2.5, value=0.5, label="DPF"),
        gr.Slider(10, 80, value=30, label="Age")
    ],
    outputs="text",
    title="Diabetes Prediction System",
    description="Enter values using sliders to predict diabetes risk"
)

interface.launch()