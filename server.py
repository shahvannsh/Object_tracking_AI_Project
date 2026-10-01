"""
REST API + local website for comparing detection algorithms.

Run with:
    uvicorn server:app --reload
Then open:
    http://localhost:8000
"""
import os
import json
import shutil
import threading
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.comparison_engine import run_comparison
from src.explain import explain_with_ai

app = FastAPI(title="Object Detection Algorithm Comparison")

UPLOAD_DIR = "data/raw"
RESULTS_DIR = "outputs/comparisons"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

JOBS = {}  # job_id -> {"status": "running"|"done"|"error", "result": {...}}


def _run_job(job_id, video_path, algorithms, epochs, max_frames, device):
    try:
        JOBS[job_id]["status"] = "running"
        result = run_comparison(
            video_path=video_path, algorithm_names=algorithms, epochs=epochs,
            max_frames=max_frames, device=device, results_dir=RESULTS_DIR,
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"video_path": dest}


@app.post("/api/compare/start")
async def start_comparison(
    video_path: str = Form(...),
    algorithms: str = Form(...),   # comma-separated
    epochs: int = Form(3),
    max_frames: int = Form(100),
    device: str = Form("cpu"),
):
    algo_list = [a.strip() for a in algorithms.split(",") if a.strip()]
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status": "queued"}

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, video_path, algo_list, epochs, max_frames, device),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/compare/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error": "unknown job_id"}, status_code=404)
    return JOBS[job_id]


@app.get("/api/compare/history")
async def get_history():
    runs = []
    for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                runs.append(json.load(f))
    return runs


@app.get("/api/algorithms")
async def list_algorithms():
    return {
        "available": [
            "yolov8n", "yolov8s", "yolov8m",
            "ssd300_vgg16", "faster_rcnn_resnet50",
            "haar_cascade_face", "contour_motion",
        ]
    }


@app.get("/api/versions")
async def get_versions():
    import sys
    versions = {"python": sys.version.split()[0]}
    packages = ["torch", "torchvision", "ultralytics", "opencv-python", "cv2",
                "supervision", "fastapi", "numpy", "streamlit", "tensorboard"]
    for pkg in packages:
        try:
            mod_name = "cv2" if pkg == "opencv-python" else pkg.replace("-", "_")
            mod = __import__(mod_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not installed"

    versions["algorithms"] = {
        "yolov8n": "Ultralytics YOLOv8n (nano) — COCO pretrained",
        "yolov8s": "Ultralytics YOLOv8s (small) — COCO pretrained",
        "yolov8m": "Ultralytics YOLOv8m (medium) — COCO pretrained",
        "ssd300_vgg16": "torchvision SSD300_VGG16_Weights.COCO_V1",
        "faster_rcnn_resnet50": "torchvision FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
        "haar_cascade_face": "OpenCV haarcascade_frontalface_default.xml",
        "contour_motion": "OpenCV MOG2 background subtractor",
        "cifar_classifier": "chenyaofo/pytorch-cifar-models — cifar100_resnet56",
    }
    return versions


@app.post("/api/explain")
async def explain(tab: str = Form(...), context: str = Form("")):
    """Routes to the tab's specialized agent (NVIDIA NIM) for a plain-English explanation."""
    from src.agents import get_agent
    try:
        explanation = explain_with_ai(tab, context)
        return {"explanation": explanation, "agent": get_agent(tab)["name"]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Serve the website
app.mount("/", StaticFiles(directory="static", html=True), name="static")
