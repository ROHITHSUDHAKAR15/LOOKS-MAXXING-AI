import cv2
import numpy as np

def detect_wrinkles(img: np.ndarray) -> float:
    """
    Returns wrinkle score (edge density, 0-1).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    wrinkle_score = edge_pixels / total_pixels
    return wrinkle_score 