import cv2
from src.algorithms.base import BaseAlgorithm


class ContourAlgorithm(BaseAlgorithm):
    """Classical motion-based detection via background subtraction + contours.
    No class labels or confidence — detects any moving blob."""
    name = "contour_motion"

    def __init__(self, min_area=500):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.min_area = min_area

    def detect(self, frame):
        fgmask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append({
                "box": [float(x), float(y), float(x + w), float(y + h)],
                "label": "moving_object",
                "confidence": 1.0,
            })
        return out
