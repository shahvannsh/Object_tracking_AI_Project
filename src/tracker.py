import supervision as sv


class Tracker:
    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
        )

    def update(self, boxes, scores, class_ids):
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=class_ids,
        )
        tracked = self.tracker.update_with_detections(detections)
        return tracked  # has .tracker_id per detection
