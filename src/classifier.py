import torch
import torch.nn.functional as F
import cv2
import numpy as np

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class CifarClassifier:
    """Runs a pretrained CIFAR-10/100 ResNet on cropped detections for a class label."""

    def __init__(self, device: str = "cpu", model_name: str = "cifar100_resnet20"):
        self.device = device
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", model_name, pretrained=True
        )
        self.model.to(device).eval()
        self.classes = CIFAR100_CLASSES if "cifar100" in model_name else CIFAR10_CLASSES

    def _preprocess(self, crop):
        crop = cv2.resize(crop, (32, 32))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - CIFAR_MEAN) / CIFAR_STD
        tensor = torch.tensor(crop).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, frame, box):
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0

        tensor = self._preprocess(crop)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        return self.classes[idx.item()], conf.item()
