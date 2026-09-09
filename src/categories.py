"""Maps raw COCO / CIFAR-100 class names into broad, human-friendly categories."""

COCO_TO_CATEGORY = {
    "person": "human", "Person": "human",
    "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle",
    "airplane": "vehicle", "bus": "vehicle", "train": "vehicle",
    "truck": "vehicle", "boat": "vehicle",
    "Bicycle": "vehicle", "Car": "vehicle", "Motorcycle": "vehicle",
    "Airplane": "vehicle", "Bus": "vehicle", "Train": "vehicle",
    "Truck": "vehicle", "Boat": "vehicle",
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal",
    "sheep": "animal", "cow": "animal", "elephant": "animal", "bear": "animal",
    "zebra": "animal", "giraffe": "animal",
    "Bird": "animal", "Cat": "animal", "Dog": "animal", "Horse": "animal",
    "Sheep": "animal", "Cattle": "animal", "Elephant": "animal", "Bear": "animal",
    "Zebra": "animal", "Giraffe": "animal", "Deer": "animal", "Tiger": "animal",
    "Fox": "animal", "Lion": "animal", "Rabbit": "animal", "Squirrel": "animal",
    "Raccoon": "animal", "Camel": "animal", "Hamster": "animal", "Kangaroo": "animal",
}

CIFAR100_TO_CATEGORY = {
    "baby": "human", "boy": "human", "girl": "human", "man": "human", "woman": "human",
    "bicycle": "vehicle", "bus": "vehicle", "motorcycle": "vehicle", "pickup_truck": "vehicle",
    "streetcar": "vehicle", "tank": "vehicle", "tractor": "vehicle", "train": "vehicle",
    "rocket": "vehicle", "lawn_mower": "vehicle",
    "bear": "animal", "beaver": "animal", "bee": "animal", "beetle": "animal",
    "butterfly": "animal", "camel": "animal", "caterpillar": "animal", "cattle": "animal",
    "chimpanzee": "animal", "cockroach": "animal", "crab": "animal", "crocodile": "animal",
    "dinosaur": "animal", "dolphin": "animal", "elephant": "animal", "fox": "animal",
    "hamster": "animal", "kangaroo": "animal", "leopard": "animal", "lion": "animal",
    "lizard": "animal", "lobster": "animal", "mouse": "animal", "otter": "animal",
    "porcupine": "animal", "possum": "animal", "rabbit": "animal", "raccoon": "animal",
    "ray": "animal", "seal": "animal", "shark": "animal", "shrew": "animal", "skunk": "animal",
    "snail": "animal", "snake": "animal", "spider": "animal", "squirrel": "animal",
    "tiger": "animal", "trout": "animal", "turtle": "animal", "whale": "animal", "wolf": "animal",
    "worm": "animal", "aquarium_fish": "animal", "flatfish": "animal",
}


def to_category(label: str, source: str = "coco") -> str:
    mapping = COCO_TO_CATEGORY if source == "coco" else CIFAR100_TO_CATEGORY
    return mapping.get(label, label)
