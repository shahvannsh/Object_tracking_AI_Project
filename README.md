<<<<<<< HEAD
# Object Tracking Project

Multi-object detection + tracking pipeline using YOLOv8 + ByteTrack. Includes
line-crossing counting, speed estimation, class filtering, and a Streamlit UI.

## Features
- YOLOv8 detection + ByteTrack multi-object tracking
- Class filtering (track only chosen COCO classes)
- Confidence-based bounding box colors
- Frame-skip + resize for CPU performance
- Line-crossing object counter
- Speed estimation (km/h, pixel-based, needs calibration)
- Crop-saving per tracked object
- CSV tracking log (frame, id, class, box, speed)
- MOTA/MOTP/IDF1 evaluation script
- Streamlit web UI for upload-and-run
- FastAPI endpoint for programmatic use

## Setup
```
pip install -r requirements.txt
```

## Usage (CLI)
1. Put a video in `data/raw/`
2. Edit `configs/config.yaml` (input path, classes, counting line, etc.)
3. Run:
```
python main.py
```
Output video → `outputs/videos/`, log → `outputs/logs/tracks.csv`

## Usage (Webcam)
Set in config.yaml:
```yaml
video:
  input_path: 0
  show_live: true
```
Press `q` or `ESC` in the live window, or `Ctrl+C` in terminal, to stop safely.

## Usage (Streamlit UI)
```
streamlit run app.py
```

## Usage (API)
```
uvicorn src.api.endpoints:app --reload
POST /track  { "video_path": "data/raw/traffic_1.mp4" }
```

## Evaluation
Requires ground-truth CSV (`frame,track_id,x1,y1,x2,y2`):
```
python evaluation/metrics.py --gt data/annotations/gt.csv --pred outputs/logs/tracks.csv
```

## Custom Training (optional)
```
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```
Only useful with your own labeled dataset — pretrained COCO classes won't
improve from epochs without custom data.

## Project Structure
```
object-tracking-project/
├── data/               # input videos, annotations
├── models/             # trained/pretrained weights
├── src/
│   ├── detector.py         # YOLOv8 wrapper + class filtering
│   ├── tracker.py          # ByteTrack wrapper
│   ├── counter.py          # line-crossing counter
│   ├── speed_estimator.py  # pixel-based speed estimation
│   ├── utils.py             # drawing, cropping helpers
│   ├── video_pipeline.py    # full pipeline orchestration
│   └── api/                 # FastAPI endpoints, schemas, connections
├── configs/config.yaml
├── outputs/             # videos, logs, crops
├── evaluation/metrics.py
├── app.py               # Streamlit UI
└── main.py              # CLI entry point
```

## Config Reference (`configs/config.yaml`)
| Key | Description |
|---|---|
| `model.weights` | YOLOv8 weights path |
| `model.confidence` | detection confidence threshold |
| `classes.filter` | list of class names to keep (empty = all) |
| `performance.frame_skip` | process every Nth frame |
| `performance.resize_width` | downscale width for speed |
| `counting.line_start/end` | pixel coords of counting line |
| `speed_estimation.pixels_per_meter` | calibrate to your camera |
| `cropping.enabled` | save cropped images per track |
=======
# Object_tracking_AI_Project
>>>>>>> 6afa0a54217c3017f4d1ff2564af48705ef1b605
