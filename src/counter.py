class LineCounter:
    def __init__(self, line_start, line_end):
        self.line_start = line_start
        self.line_end = line_end
        self.counted_ids = set()
        self.track_sides = {}

    def _side(self, point):
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def update(self, track_id, box):
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        side = self._side(center)
        side_sign = 1 if side > 0 else -1

        prev_side = self.track_sides.get(track_id)
        if prev_side is not None and prev_side != side_sign and track_id not in self.counted_ids:
            self.counted_ids.add(track_id)

        self.track_sides[track_id] = side_sign

    @property
    def count(self):
        return len(self.counted_ids)
