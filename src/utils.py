import cv2
import os
from src.categories import to_category


def confidence_color(score: float):
    if score >= 0.7:
        return (0, 255, 0)
    elif score >= 0.5:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


def draw_tracks(frame, tracked_detections, class_names, speeds=None, cifar_labels=None):
    for i in range(len(tracked_detections)):
        x1, y1, x2, y2 = tracked_detections.xyxy[i].astype(int)
        track_id = tracked_detections.tracker_id[i]
        conf = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 1.0
        color = confidence_color(conf)

        category = to_category(class_names[i], source="coco")
        label = f"{category} ID:{track_id}"

        if speeds and track_id in speeds:
            label += f" {speeds[track_id]:.1f}km/h"
        if cifar_labels and track_id in cifar_labels:
            cifar_name, cifar_conf = cifar_labels[track_id]
            cifar_category = to_category(cifar_name, source="cifar100")
            label += f" | {cifar_category}({cifar_conf:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return frame


def draw_counting_line(frame, line_start, line_end, count):
    cv2.line(frame, tuple(line_start), tuple(line_end), (255, 0, 0), 2)
    cv2.putText(
        frame, f"Count: {count}", (line_start[0], line_start[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
    )
    return frame


def save_crop(frame, box, save_dir, track_id, frame_idx):
    os.makedirs(save_dir, exist_ok=True)
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    path = os.path.join(save_dir, f"id{track_id}_frame{frame_idx}.jpg")
    cv2.imwrite(path, crop)
