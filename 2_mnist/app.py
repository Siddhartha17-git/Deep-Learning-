from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import re

app = Flask(__name__)
CORS(app)

# Load the trained model once at startup
model = tf.keras.models.load_model("mnist_mlp_.keras")
print("✓ Model loaded successfully")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale

        # ── Resize to 28x28 ──────────────────────────────────────────────────
        img = img.resize((28, 28), Image.LANCZOS)

        img_array = np.array(img).astype("float32")

        # ── FIX 1: Canvas is white-on-black (same as MNIST) — no invert needed
        # But if your canvas background is WHITE, invert it:
        # img_array = 255.0 - img_array   ← uncomment if canvas bg is white

        # ── FIX 2: Normalize to [0,1] ────────────────────────────────────────
        img_array = img_array / 255.0

        # ── FIX 3: Center the digit (most important fix) ─────────────────────
        img_array = center_digit(img_array)

        # ── FIX 4: Reshape for model ──────────────────────────────────────────
        img_array = img_array.reshape(1, 784)

        predictions = model.predict(img_array, verbose=0)
        probs = predictions[0].tolist()
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100

        return jsonify({
            "predicted": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": [round(p * 100, 2) for p in probs]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def center_digit(img):
    """
    Replicates MNIST preprocessing:
    - Find bounding box of the digit
    - Center it in a 20x20 box
    - Paste into center of 28x28 image
    """
    from scipy import ndimage

    # Find bounding box of non-zero pixels
    threshold = 0.1
    rows = np.any(img > threshold, axis=1)
    cols = np.any(img > threshold, axis=0)

    if not rows.any() or not cols.any():
        return img  # blank canvas — return as-is

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to digit bounding box
    digit = img[rmin:rmax+1, cmin:cmax+1]

    # Resize to fit in 20x20 while keeping aspect ratio
    h, w = digit.shape
    if h == 0 or w == 0:
        return img

    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    digit_pil = Image.fromarray((digit * 255).astype(np.uint8))
    digit_pil = digit_pil.resize((new_w, new_h), Image.LANCZOS)
    digit = np.array(digit_pil).astype("float32") / 255.0

    # Paste into center of 28x28 blank canvas
    result = np.zeros((28, 28), dtype="float32")
    row_offset = (28 - new_h) // 2
    col_offset = (28 - new_w) // 2
    result[row_offset:row_offset+new_h, col_offset:col_offset+new_w] = digit

    # Use center of mass shift (exactly what MNIST does)
    cy, cx = ndimage.center_of_mass(result)
    shift_y = 14 - cy
    shift_x = 14 - cx
    result = ndimage.shift(result, [shift_y, shift_x], cval=0.0)

    return result

@app.route("/model-info", methods=["GET"])
def model_info():
    layers_info = []
    for layer in model.layers:
        cfg = layer.get_config()
        layers_info.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "units": cfg.get("units"),
            "activation": cfg.get("activation"),
            "rate": cfg.get("rate"),
        })
    total_params = model.count_params()
    return jsonify({
        "layers": layers_info,
        "total_params": total_params
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
