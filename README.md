# Object Tracking Project

YOLOv8 + ByteTrack multi-object detection & tracking pipeline. Includes
line-crossing counting, speed estimation, class filtering, TensorBoard
metrics, and a Streamlit UI.

## Setup
```
pip install -r requirements.txt
```

## Usage (CLI)
```
python main.py
```
Output video → `outputs/videos/`, log → `outputs/logs/tracks.csv`

## Usage (Webcam)
Set in config.yaml: `input_path: 0`, `show_live: true`. Press `q`/`ESC` or `Ctrl+C` to stop.

## Usage (Streamlit UI)
```
streamlit run app.py
```
Shows the tracked video plus bar charts of detections/unique objects per class.

## Usage (TensorBoard)
```
tensorboard --logdir outputs/tensorboard_logs
```
Shows per-frame detection counts per class and a text summary of totals.

## Usage (API)
```
uvicorn src.api.endpoints:app --reload
POST /track  { "video_path": "data/raw/traffic_1.mp4" }
```

## Evaluation
```
python evaluation/metrics.py --gt data/annotations/gt.csv --pred outputs/logs/tracks.csv
```

## Config Reference (`configs/config.yaml`)
| Key | Description |
|---|---|
| `model.weights` | YOLOv8 weights path |
| `model.confidence` | detection confidence threshold |
| `classes.filter` | class names to keep (empty = all) |
| `performance.frame_skip` | process every Nth frame |
| `performance.resize_width` | downscale width |
| `counting.line_start/end` | pixel coords of counting line |
| `speed_estimation.pixels_per_meter` | calibrate to camera |
| `cropping.enabled` | save cropped images per track |
| `tensorboard.enabled/log_dir` | TensorBoard logging |
