from pydantic import BaseModel


class VideoUploadRequest(BaseModel):
    video_path: str


class TrackingResponse(BaseModel):
    status: str
    output_video_path: str
    log_path: str
