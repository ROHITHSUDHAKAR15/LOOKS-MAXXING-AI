import cv2
import numpy as np

def detect_uv_exposure(img: np.ndarray) -> float:
    """
    Returns UV-exposed percentage (0-100).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_uv = np.array([0, 0, 100])
    upper_uv = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_uv, upper_uv)
    uv_exposed = np.count_nonzero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    uv_percentage = uv_exposed / total_pixels * 100
    return uv_percentage 