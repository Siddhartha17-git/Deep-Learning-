# DL Assignment Projects

This repository contains several machine learning and deep learning web applications built with Python, Flask, TensorFlow/Keras, and Gradio.

## Included Projects

1. House Price Prediction App
   - Folder: [1_houseprice](1_houseprice)
   - Uses Flask and a trained regression model.

2. MNIST Digit Recognition App
   - Folder: [2_mnist](2_mnist)
   - Uses Flask and a Keras neural network for handwritten digit recognition.

3. Diabetes Prediction App (Flask)
   - Folder: [Diabetic_new/backend](Diabetic_new/backend)
   - Predicts diabetes risk from medical input values.

4. Diabetes Prediction App (Gradio)
   - Folder: [Diabetic_gradio](Diabetic_gradio)
   - Provides a simple Gradio-based UI for prediction.

5. House Price Prediction App (Flask)
   - Folder: [Houseprice_new/backend](Houseprice_new/backend)
   - Predicts house prices from real estate features.

## Requirements

Install Python packages before running any app.

```bash
pip install flask tensorflow numpy joblib pillow scipy flask-cors gradio
```

## How to Run Each Project

### 1. House Price Prediction
```bash
cd 1_houseprice
python app.py
```

### 2. MNIST Digit Recognition
```bash
cd 2_mnist
pip install -r requirements.txt
python app.py
```

### 3. Diabetes Prediction (Flask)
```bash
cd Diabetic_new/backend
python app.py
```

### 4. Diabetes Prediction (Gradio)
```bash
cd Diabetic_gradio
python gradio_app.py
```

### 5. House Price Prediction (New Version)
```bash
cd Houseprice_new/backend
python app.py
```

## GitHub Upload Steps

1. Create a new repository on GitHub.
2. Open a terminal in this project folder.
3. Run the commands below:

```bash
git init -b main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/your-repo-name.git
git push -u origin main
```

## Notes

- Some model files are large binary files, so GitHub may handle them better with Git LFS if needed.
- If you want, you can also add screenshots, screenshots of the UI, and a demo video later.
