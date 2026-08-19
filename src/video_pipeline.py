import csv
import cv2

from src.detector import Detector
from src.tracker import Tracker
from src.utils import draw_tracks, draw_counting_line, save_crop
from src.counter import LineCounter
from src.speed_estimator import SpeedEstimator


def run_pipeline(config: dict):
    detector = Detector(
        weights_path=config["model"]["weights"],
        confidence=config["model"]["confidence"],
        device=config["model"]["device"],
        class_filter=config.get("classes", {}).get("filter"),
    )
    tracker = Tracker(
        track_thresh=config["tracker"]["track_thresh"],
    )

    cap = cv2.VideoCapture(config["video"]["input_path"])
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {config['video']['input_path']}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    perf_cfg = config.get("performance", {})
    frame_skip = max(1, perf_cfg.get("frame_skip", 1))
    resize_width = perf_cfg.get("resize_width", 0)

    out_width, out_height = width, height
    if resize_width and resize_width < width:
        scale = resize_width / width
        out_width, out_height = resize_width, int(height * scale)

    out = cv2.VideoWriter(
        config["video"]["output_path"],
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps / frame_skip, (out_width, out_height),
    )

    log_file = open(config["logging"]["log_path"], "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "speed_kmh"])

    count_cfg = config.get("counting", {})
    counter = None
    if count_cfg.get("enabled"):
        counter = LineCounter(count_cfg["line_start"], count_cfg["line_end"])

    speed_cfg = config.get("speed_estimation", {})
    speed_estimator = None
    if speed_cfg.get("enabled"):
        fps_for_speed = speed_cfg.get("fps_override") or (src_fps / frame_skip)
        speed_estimator = SpeedEstimator(speed_cfg["pixels_per_meter"], fps_for_speed)

    crop_cfg = config.get("cropping", {})

    frame_idx = 0
    processed_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            if resize_width and resize_width < width:
                frame = cv2.resize(frame, (out_width, out_height))

            boxes, scores, class_ids, class_names = detector.detect(frame)
            tracked = tracker.update(boxes, scores, class_ids)

            speeds = {}
            for i in range(len(tracked)):
                box = tracked.xyxy[i]
                track_id = tracked.tracker_id[i]
                cls_id = tracked.class_id[i]

                speed_kmh = None
                if speed_estimator:
                    speed_kmh = speed_estimator.update(track_id, box)
                    if speed_kmh is not None:
                        speeds[track_id] = speed_kmh

                if counter:
                    counter.update(track_id, box)

                if crop_cfg.get("enabled"):
                    save_crop(frame, box, crop_cfg["save_dir"], track_id, processed_idx)

                log_writer.writerow([
                    processed_idx, track_id, cls_id,
                    *box, speed_kmh if speed_kmh else ""
                ])

            frame = draw_tracks(frame, tracked, class_names, speeds)
            if counter:
                frame = draw_counting_line(frame, count_cfg["line_start"], count_cfg["line_end"], counter.count)

            out.write(frame)

            if config["video"].get("show_live"):
                cv2.imshow("Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("Stopped by user.")
                    break

            frame_idx += 1
            processed_idx += 1

    except KeyboardInterrupt:
        print("Interrupted (Ctrl+C). Cleaning up...")

    finally:
        cap.release()
        out.release()
        log_file.close()
        cv2.destroyAllWindows()
        if counter:
            print(f"Total line-crossing count: {counter.count}")
        print("Resources released. Video and log saved.")
