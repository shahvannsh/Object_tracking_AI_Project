import yaml
from fastapi import FastAPI
from src.api.schemas import VideoUploadRequest, TrackingResponse
from src.video_pipeline import run_pipeline

app = FastAPI()


@app.post("/track", response_model=TrackingResponse)
def track_video(req: VideoUploadRequest):
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["video"]["input_path"] = req.video_path
    run_pipeline(config)

    return TrackingResponse(
        status="done",
        output_video_path=config["video"]["output_path"],
        log_path=config["logging"]["log_path"],
    )
