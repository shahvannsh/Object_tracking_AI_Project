class BaseAlgorithm:
    """Common interface every detection algorithm implements."""

    name = "base"

    def detect(self, frame):
        """Returns list of dicts: {box:[x1,y1,x2,y2], label:str, confidence:float}"""
        raise NotImplementedError
