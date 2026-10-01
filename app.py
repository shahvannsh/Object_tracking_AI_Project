"""
Streamlit UI: upload a video, run tracking pipeline, view result + class-count chart.

Run with: streamlit run app.py
"""
import streamlit as st
import yaml
import tempfile
import os
import pandas as pd

from src.video_pipeline import run_pipeline

st.set_page_config(page_title="Object Tracking", layout="centered")
st.title("🎯 Object Tracking Demo")

with open("configs/config.yaml", "r") as f:
    default_config = yaml.safe_load(f)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

confidence = st.slider("Detection confidence", 0.1, 0.9, default_config["model"]["confidence"])
class_options = st.multiselect(
    "Classes to track (leave empty for all)",
    ["person", "car", "bicycle", "motorcycle", "bus", "truck"],
    default=default_config.get("classes", {}).get("filter", []),
)

if uploaded_file and st.button("Run Tracking"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    config = default_config.copy()
    config["video"]["input_path"] = input_path
    config["video"]["output_path"] = "outputs/videos/streamlit_output.mp4"
    config["video"]["show_live"] = False
    config["model"]["confidence"] = confidence
    config["classes"]["filter"] = class_options
    config["logging"]["log_path"] = "outputs/logs/streamlit_tracks.csv"
    config["tensorboard"]["log_dir"] = "outputs/tensorboard_logs"

    os.makedirs("outputs/videos", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    with st.spinner("Running detection + tracking..."):
        run_pipeline(config)

    st.success("Done!")
    st.video(config["video"]["output_path"])

    with open(config["logging"]["log_path"], "rb") as f:
        st.download_button("Download tracking log (CSV)", f, file_name="tracks.csv")

    st.subheader("📊 Detections per class")
    df = pd.read_csv(config["logging"]["log_path"])

    from ultralytics import YOLO
    model = YOLO(config["model"]["weights"])
    df["class_name"] = df["class"].map(model.names)

    total_counts = df["class_name"].value_counts()
    st.bar_chart(total_counts)

    unique_counts = df.groupby("class_name")["track_id"].nunique()
    st.write("Unique tracked objects per class:")
    st.bar_chart(unique_counts)

    if "cifar_class" in df.columns and df["cifar_class"].notna().any():
        st.subheader("📊 CIFAR-100 tag distribution")
        cifar_counts = df["cifar_class"].dropna()
        cifar_counts = cifar_counts[cifar_counts != ""].value_counts()
        st.bar_chart(cifar_counts)

    st.info("Want to compare YOLOv8n/s/m speed & accuracy? Run: `streamlit run benchmark_dashboard.py`")
