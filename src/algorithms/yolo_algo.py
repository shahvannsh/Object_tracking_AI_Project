from ultralytics import YOLO
from src.algorithms.base import BaseAlgorithm


class YoloAlgorithm(BaseAlgorithm):
    def __init__(self, weights="yolov8n.pt", device="cpu", confidence=0.4):
        self.name = weights.replace(".pt", "")
        self.model = YOLO(weights)
        self.device = device
        self.confidence = confidence

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.confidence, device=self.device, verbose=False)[0]
        out = []
        for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(),
                                   results.boxes.conf.cpu().numpy(),
                                   results.boxes.cls.cpu().numpy().astype(int)):
            out.append({
                "box": box.tolist(),
                "label": self.model.names[cls],
                "confidence": float(conf),
            })
        return out
