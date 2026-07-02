# app.py

from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model & scaler
model = load_model("diabetes_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bp']),
            float(request.form['skin']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)

        prediction = model.predict(data)[0][0]

        if prediction > 0.5:
            result = "Diabetic"
        else:
            result = "Not Diabetic"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)