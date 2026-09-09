class SpeedEstimator:
    def __init__(self, pixels_per_meter: float, fps: float):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.prev_positions = {}

    def update(self, track_id, box):
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        speed_kmh = None
        if track_id in self.prev_positions:
            px, py = self.prev_positions[track_id]
            dist_px = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5
            dist_m = dist_px / self.pixels_per_meter
            dist_per_sec = dist_m * self.fps
            speed_kmh = dist_per_sec * 3.6

        self.prev_positions[track_id] = center
        return speed_kmh
