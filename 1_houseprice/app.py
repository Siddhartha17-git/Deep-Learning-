from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Scale input
        features = scaler.transform(features)

        prediction = model.predict(features)
        output = round(prediction[0][0], 2)

        return render_template('index.html', prediction_text=f"Predicted House Price: ${output * 100000}")

    except:
        return render_template('index.html', prediction_text="Error in input")

if __name__ == "__main__":
    app.run(debug=True)