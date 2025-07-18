import cv2
import numpy as np

def detect_redness(img: np.ndarray) -> float:
    """
    Returns redness percentage (0-100) based on red hue area.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red hue can wrap around 0, so use two ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    red_pixels = np.count_nonzero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    redness_percentage = red_pixels / total_pixels * 100
    return redness_percentage 