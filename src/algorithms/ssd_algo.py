import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from src.algorithms.base import BaseAlgorithm

COCO_CLASSES = SSD300_VGG16_Weights.COCO_V1.meta["categories"]


class SSDAlgorithm(BaseAlgorithm):
    name = "ssd300_vgg16"

    def __init__(self, device="cpu", confidence=0.4):
        weights = SSD300_VGG16_Weights.COCO_V1
        self.model = ssd300_vgg16(weights=weights)
        self.model.to(device).eval()
        self.transform = weights.transforms()
        self.device = device
        self.confidence = confidence

    @torch.no_grad()
    def detect(self, frame):
        import cv2
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(torch.from_numpy(img).permute(2, 0, 1)).to(self.device)
        preds = self.model([tensor])[0]

        out = []
        for box, score, label in zip(preds["boxes"].cpu().numpy(),
                                      preds["scores"].cpu().numpy(),
                                      preds["labels"].cpu().numpy()):
            if score < self.confidence:
                continue
            out.append({
                "box": box.tolist(),
                "label": COCO_CLASSES[label],
                "confidence": float(score),
            })
        return out
