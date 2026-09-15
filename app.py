import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Keep TF quiet-ish and CPU-only friendly
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.keras")
IMG_SIZE = 256

app = Flask(__name__)

# Load once at startup — the model's own Rescaling layer normalizes
# internally, so we feed it raw 0-255 pixel values (do NOT divide by 255).
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Load an uploaded image, resize to IMG_SIZE, return a (1,H,W,3) float32
    array of raw 0-255 pixel values (matching the model's built-in Rescaling
    layer, which expects unnormalized input)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        file_bytes = file.read()
        tensor = preprocess_image(file_bytes)
        prediction = model.predict(tensor, verbose=0)
        value = float(prediction[0][0])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to process image: {exc}"}), 500

    label = "Fracture" if value > 0.5 else "No Fracture"
    return jsonify({"label": label, "score": value})


if __name__ == "__main__":
    # debug=True for local dev only; turn off in production
    app.run(host="0.0.0.0", port=5000, debug=True)
