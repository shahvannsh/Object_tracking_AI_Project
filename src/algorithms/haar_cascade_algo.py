import cv2
from src.algorithms.base import BaseAlgorithm


class HaarCascadeAlgorithm(BaseAlgorithm):
    """Classical face detector via Haar Cascades (no confidence score — fixed 1.0)."""
    name = "haar_cascade_face"

    def __init__(self, cascade="haarcascade_frontalface_default.xml"):
        path = cv2.data.haarcascades + cascade
        self.classifier = cv2.CascadeClassifier(path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        out = []
        for (x, y, w, h) in faces:
            out.append({
                "box": [float(x), float(y), float(x + w), float(y + h)],
                "label": "face",
                "confidence": 1.0,
            })
        return out
