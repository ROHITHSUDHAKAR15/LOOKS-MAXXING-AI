import cv2
import numpy as np

def detect_oiliness(img: np.ndarray) -> float:
    """
    Returns oiliness percentage (0-100).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    oily_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    oily_percentage = oily_pixels / total_pixels * 100
    return oily_percentage 