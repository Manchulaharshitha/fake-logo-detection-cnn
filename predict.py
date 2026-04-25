import tensorflow as tf
import numpy as np
import cv2
import os

# =============================
# 🔧 SETTINGS
# =============================
MODEL_PATH = "logo_model_mobilenet.keras"
IMG_SIZE = 224   

# =============================
# 📦 LOAD MODEL
# =============================
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded!")

# =============================
# 🔍 PREDICTION FUNCTION
# =============================
def predict_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image not found!")
        return

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Failed to read image!")
        return

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img / 255.0

    # Expand dims (model expects batch)
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]

    # Result
    if prediction > 0.7:
        label = "REAL ✅"
    elif prediction < 0.4:
        label = "FAKE ❌"
    else:
        label = "UNCERTAIN ⚠️"

    print(f"\n📷 Image: {image_path}")
    print(f"🔢 Confidence: {prediction:.4f}")
    print(f"🎯 Prediction: {label}")

# =============================
# ▶️ TEST IMAGE HERE
# =============================

# 👉 CHANGE THIS PATH
image_path = "D:/fake-logo-detection-cnn/Testing data/adidas img1.jpeg"

predict_image(image_path)