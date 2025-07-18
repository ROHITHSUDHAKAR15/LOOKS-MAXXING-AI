import cv2
import numpy as np
from typing import Tuple

def detect_acne(img: np.ndarray) -> Tuple[bool, float]:
    """
    Detect acne using image processing. Returns (acne_detected, acne_score [0-1]).
    """
    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Hough Circles to find potential acne spots
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=50, param2=15, minRadius=2, maxRadius=10)
    acne_score = 0.0
    if circles is not None:
        acne_score = min(1.0, len(circles[0]) / 20)  # Normalize to [0,1]
    return (acne_score > 0.2, acne_score)

# Always define load_acne_model and detect_acne_ml for import safety
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    _model = None
    def load_acne_model(path='acne_model.h5'):
        global _model
        if _model is None:
            _model = keras_load_model(path)
        return _model
    def detect_acne_ml(img: np.ndarray) -> Tuple[bool, float]:
        model = load_acne_model()
        img_resized = cv2.resize(img, (224, 224))
        img_norm = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_norm, axis=0)
        pred = model.predict(img_input)
        return (pred[0][0] >= 0.5, float(pred[0][0]))
except ImportError:
    def load_acne_model(path='acne_model.h5'):
        raise RuntimeError("TensorFlow is not installed. ML-based acne detection is unavailable.")
    def detect_acne_ml(img: np.ndarray) -> Tuple[bool, float]:
        raise RuntimeError("TensorFlow is not installed. ML-based acne detection is unavailable.") 