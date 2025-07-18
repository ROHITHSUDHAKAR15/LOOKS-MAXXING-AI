import cv2
import numpy as np
from typing import Dict

def extract_face_regions(img: np.ndarray, face_cascade_path='haarcascade_frontalface_default.xml') -> Dict[str, np.ndarray]:
    """
    Detect face and extract T-zone, left cheek, right cheek, chin regions.
    Returns a dict: region name -> image crop.
    """
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    regions = {}
    if len(faces) == 0:
        return regions
    x, y, w, h = faces[0]
    # Define regions relative to face box
    # T-zone: upper center
    tzone = img[y:y+int(0.4*h), x+int(0.3*w):x+int(0.7*w)]
    # Left cheek
    left_cheek = img[y+int(0.4*h):y+int(0.7*h), x:x+int(0.3*w)]
    # Right cheek
    right_cheek = img[y+int(0.4*h):y+int(0.7*h), x+int(0.7*w):x+w]
    # Chin
    chin = img[y+int(0.7*h):y+h, x+int(0.3*w):x+int(0.7*w)]
    regions['T-zone'] = tzone
    regions['Left Cheek'] = left_cheek
    regions['Right Cheek'] = right_cheek
    regions['Chin'] = chin
    return regions 