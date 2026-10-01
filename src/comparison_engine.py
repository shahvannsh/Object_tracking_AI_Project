"""
Runs multiple detection algorithms over a video for N passes ("epochs"),
averages their speed/accuracy-proxy metrics, and stores results to disk.

Note: Haar Cascades, Contours, SSD, Faster R-CNN are pretrained/classical —
"epochs" here means repeated inference passes for stable averaged metrics,
not training epochs.
"""
import time
import json
import os
import uuid
from datetime import datetime
import cv2

ALGO_REGISTRY = {}


def register_algorithms(device="cpu", confidence=0.4):
    """Lazily builds the algorithm registry (avoids loading all models unless needed)."""
    from src.algorithms.yolo_algo import YoloAlgorithm
    from src.algorithms.ssd_algo import SSDAlgorithm
    from src.algorithms.faster_rcnn_algo import FasterRCNNAlgorithm
    from src.algorithms.haar_cascade_algo import HaarCascadeAlgorithm
    from src.algorithms.contour_algo import ContourAlgorithm

    return {
        "yolov8n": lambda: YoloAlgorithm("yolov8n.pt", device, confidence),
        "yolov8s": lambda: YoloAlgorithm("yolov8s.pt", device, confidence),
        "yolov8m": lambda: YoloAlgorithm("yolov8m.pt", device, confidence),
        "ssd300_vgg16": lambda: SSDAlgorithm(device, confidence),
        "faster_rcnn_resnet50": lambda: FasterRCNNAlgorithm(device, confidence),
        "haar_cascade_face": lambda: HaarCascadeAlgorithm(),
        "contour_motion": lambda: ContourAlgorithm(),
    }


def _run_single_pass(algo, video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    total_time = 0.0
    total_detections = 0
    total_confidence = 0.0
    frame_count = 0

    while True:
        if max_frames and frame_count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        detections = algo.detect(frame)
        total_time += time.time() - start

        total_detections += len(detections)
        total_confidence += sum(d["confidence"] for d in detections)
        frame_count += 1

    cap.release()

    return {
        "frames": frame_count,
        "time_sec": total_time,
        "detections": total_detections,
        "confidence_sum": total_confidence,
    }


def run_comparison(video_path: str, algorithm_names: list, epochs: int = 3,
                    max_frames: int = 100, device: str = "cpu", confidence: float = 0.4,
                    results_dir: str = "outputs/comparisons"):
    registry = register_algorithms(device, confidence)
    run_id = str(uuid.uuid4())[:8]
    results = {"run_id": run_id, "video": video_path, "epochs": epochs,
               "started_at": datetime.now().isoformat(), "algorithms": {}}

    for name in algorithm_names:
        if name not in registry:
            results["algorithms"][name] = {"error": f"unknown algorithm: {name}"}
            continue

        try:
            algo = registry[name]()
        except Exception as e:
            results["algorithms"][name] = {"error": str(e)}
            continue

        pass_results = []
        for epoch in range(epochs):
            print(f"[{name}] pass {epoch + 1}/{epochs}")
            pass_results.append(_run_single_pass(algo, video_path, max_frames))

        total_frames = sum(p["frames"] for p in pass_results)
        total_time = sum(p["time_sec"] for p in pass_results)
        total_detections = sum(p["detections"] for p in pass_results)
        total_confidence = sum(p["confidence_sum"] for p in pass_results)

        results["algorithms"][name] = {
            "epochs_run": epochs,
            "avg_fps": round(total_frames / total_time, 2) if total_time else 0,
            "avg_inference_ms": round((total_time / total_frames) * 1000, 2) if total_frames else 0,
            "avg_detections_per_frame": round(total_detections / total_frames, 2) if total_frames else 0,
            "avg_confidence": round(total_confidence / total_detections, 3) if total_detections else 0,
            "total_detections": total_detections,
        }

    results["finished_at"] = datetime.now().isoformat()

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
