from ultralytics import YOLO


class Detector:
    def __init__(self, weights_path: str, confidence: float = 0.4,
                 device: str = "cpu", class_filter=None):
        self.model = YOLO(weights_path)
        self.confidence = confidence
        self.device = device
        self.class_filter = set(class_filter) if class_filter else None

    def detect(self, frame):
        results = self.model.predict(
            frame, conf=self.confidence, device=self.device, verbose=False
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = [self.model.names[c] for c in class_ids]

        if self.class_filter:
            keep = [i for i, name in enumerate(class_names) if name in self.class_filter]
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]
            class_names = [class_names[i] for i in keep]

        return boxes, scores, class_ids, class_names
