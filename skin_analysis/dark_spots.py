import cv2
import numpy as np

def detect_dark_spots(img: np.ndarray) -> int:
    """
    Returns the number of detected dark spots.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    # Adaptive threshold to find dark regions
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter small contours (noise)
    dark_spots = [cnt for cnt in contours if 10 < cv2.contourArea(cnt) < 200]
    return len(dark_spots) 