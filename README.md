# Object Tracking Project

YOLOv8 + ByteTrack multi-object detection & tracking pipeline, with CIFAR-100
secondary classification, broad category tagging (human/vehicle/animal),
line-crossing counting, speed estimation, TensorBoard metrics, a Streamlit UI,
and a model comparison dashboard.

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

## REST API + Algorithm Comparison Website (NEW)
Compares YOLOv8 (n/s/m), SSD300, Faster R-CNN, Haar Cascades, and Contour
(motion) detection — each run for several passes ("epochs") over your video,
averaged for FPS, inference time, detections/frame, and confidence.

```
uvicorn server:app --reload
```
Open: **http://localhost:8000**

Flow: upload a video → pick algorithms + epochs → Run Comparison → results
table + bar charts render live, with run history below.

REST endpoints (for scripting instead of the UI):
- `POST /api/upload` — multipart video upload, returns `video_path`
- `POST /api/compare/start` — form fields: `video_path`, `algorithms` (comma-separated), `epochs`, `max_frames`, `device` → returns `job_id`
- `GET /api/compare/status/{job_id}` — poll until `status: done`
- `GET /api/compare/history` — all past runs
- `GET /api/algorithms` — list of available algorithm names

Note: Haar Cascades / Contours are classical (non-trainable per "epoch") and
SSD/Faster R-CNN/YOLO here use pretrained COCO weights — "epochs" means
repeated inference passes for stable averaged metrics, not model training.

### "Explain with AI" button (every tab)
Each tab has a specialized **agent** (see `src/agents.py`) that explains that
tab's current data via an NVIDIA NIM model (OpenAI-compatible API — no
Anthropic key involved). Requires an NVIDIA API key:
```
# macOS/Linux
export NVIDIA_API_KEY="your-key-here"
export NVIDIA_MODEL="meta/llama-3.1-70b-instruct"   # optional, this is the default
# Windows (PowerShell)
$env:NVIDIA_API_KEY="your-key-here"
```
Get a key at https://build.nvidia.com. Restart `uvicorn` after setting it.

Agents:
| Tab | Agent | Role |
|---|---|---|
| Dashboard | Dashboard Agent | summarizes stats + latest run |
| Version Comparison | Comparison Agent | explains speed/accuracy tradeoffs |
| Run History | History Agent | spots trends across runs |
| Algorithms Info | Algorithms Tutor Agent | teaches the algorithms |
| Version Info | Environment Agent | explains library/model versions |


```
streamlit run benchmark_dashboard.py
```
Benchmarks YOLOv8n / s / m on your video: FPS, avg inference time, avg
confidence, avg detections per frame. Or via CLI:
```
python src/benchmark.py --video data/raw/traffic_1.mp4 --frames 100
```
Results saved to `outputs/benchmarks/results.json`.

## Usage (TensorBoard)
```
tensorboard --logdir outputs/tensorboard_logs
```

## Usage (API)
```
uvicorn src.api.endpoints:app --reload
POST /track  { "video_path": "data/raw/traffic_1.mp4" }
```

## Evaluation
```
python evaluation/metrics.py --gt data/annotations/gt.csv --pred outputs/logs/tracks.csv
```

## Accuracy upgrades applied
- Detector: `yolov8n.pt` → `yolov8s.pt` (better accuracy, still CPU-friendly). Use `yolov8m.pt` for even higher accuracy if GPU available.
- Confidence threshold lowered to 0.35 (catches more true positives; raise if too many false positives)
- Added explicit IoU (NMS) threshold = 0.45
- CIFAR classifier: `cifar100_resnet20` → `cifar100_resnet56` (deeper, more accurate)
- Switched to Open Images V7 weights option (`yolov8n-oiv7.pt`) for 600+ classes (covers animals COCO lacks, e.g. deer)

## Category tagging
Boxes are labeled with broad categories (human / vehicle / animal) mapped
from raw COCO/OIV7 and CIFAR-100 class names — see `src/categories.py` to extend.

## Line counting
`counting.line_start` / `line_end` must be in pixel coordinates of the
**output** frame size (after `performance.resize_width`). Draw the line
**across** the direction of travel (e.g. a vertical line across a road), not
parallel to it. Console prints the actual frame size + line coords on startup.

## Config Reference (`configs/config.yaml`)
| Key | Description |
|---|---|
| `model.weights` | YOLOv8 weights (n/s/m, or `-oiv7` variants) |
| `model.confidence` | detection confidence threshold |
| `model.iou` | NMS IoU threshold |
| `classes.filter` | class names to keep (empty = all) |
| `performance.frame_skip` | process every Nth frame |
| `performance.resize_width` | downscale width |
| `counting.line_start/end` | pixel coords of counting line |
| `speed_estimation.pixels_per_meter` | calibrate to camera |
| `cropping.enabled` | save cropped images per track |
| `cifar_classification.enabled/model_name` | secondary CIFAR-100 classifier |
| `tensorboard.enabled/log_dir` | TensorBoard logging |
