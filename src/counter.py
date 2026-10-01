class LineCounter:
    """Counts unique track IDs crossing a line SEGMENT (not an infinite line)."""

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

    def _crosses_segment(self, p1, p2):
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        x3, y3 = p1
        x4, y4 = p2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return False

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

        return 0 <= t <= 1 and 0 <= u <= 1

    def update(self, track_id, box):
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        side = self._side(center)
        side_sign = 1 if side > 0 else -1

        prev_entry = self.track_sides.get(track_id)
        if prev_entry is not None:
            prev_side, prev_center = prev_entry
            if prev_side != side_sign and track_id not in self.counted_ids:
                if self._crosses_segment(prev_center, center):
                    self.counted_ids.add(track_id)

        self.track_sides[track_id] = (side_sign, center)

    @property
    def count(self):
        return len(self.counted_ids)
