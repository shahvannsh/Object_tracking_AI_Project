"""
Benchmarks YOLOv8n / s / m on a sample video: FPS, avg inference time,
total detections, avg confidence. Helps pick the accuracy/speed tradeoff.

Usage:
    python src/benchmark.py --video data/raw/traffic_1.mp4 --frames 100
"""
import argparse
import json
import time
import os
import cv2
from ultralytics import YOLO

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def benchmark_model(weights: str, video_path: str, max_frames: int, device: str = "cpu"):
    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)

    total_time = 0.0
    total_detections = 0
    total_confidence = 0.0
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model.predict(frame, device=device, verbose=False)[0]
        elapsed = time.time() - start

        total_time += elapsed
        confs = results.boxes.conf.cpu().numpy()
        total_detections += len(confs)
        total_confidence += confs.sum()
        frame_count += 1

    cap.release()

    avg_inference_ms = (total_time / frame_count) * 1000 if frame_count else 0
    fps = frame_count / total_time if total_time else 0
    avg_conf = total_confidence / total_detections if total_detections else 0

    return {
        "model": weights,
        "frames": frame_count,
        "avg_inference_ms": round(avg_inference_ms, 2),
        "fps": round(fps, 2),
        "total_detections": total_detections,
        "avg_detections_per_frame": round(total_detections / frame_count, 2) if frame_count else 0,
        "avg_confidence": round(float(avg_conf), 3),
    }


def run_benchmark(video_path: str, max_frames: int = 100, device: str = "cpu", models=None,
                   save_path: str = "outputs/benchmarks/results.json"):
    models = models or MODELS
    results = []
    for weights in models:
        print(f"Benchmarking {weights} ...")
        result = benchmark_model(weights, video_path, max_frames, device)
        results.append(result)
        print(result)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    results = run_benchmark(args.video, args.frames, args.device, args.models)

    print("\n| Model | Avg Inference (ms) | FPS | Avg Det/Frame | Avg Confidence |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model']} | {r['avg_inference_ms']} | {r['fps']} | {r['avg_detections_per_frame']} | {r['avg_confidence']} |")
    print("\nSaved to outputs/benchmarks/results.json")
