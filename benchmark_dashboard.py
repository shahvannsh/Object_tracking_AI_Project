"""
Model comparison dashboard: YOLOv8n vs s vs m — accuracy/speed tradeoff.

Run with: streamlit run benchmark_dashboard.py
"""
import streamlit as st
import pandas as pd
import tempfile
import os

from src.benchmark import run_benchmark, MODELS

st.set_page_config(page_title="YOLO Model Comparison", layout="centered")
st.title("⚖️ YOLOv8 Model Comparison — Accuracy vs Speed")

uploaded_file = st.file_uploader("Upload a test video", type=["mp4", "avi", "mov"])
max_frames = st.slider("Frames to benchmark per model", 20, 300, 100)
models_selected = st.multiselect("Models to compare", MODELS, default=MODELS)

if uploaded_file and st.button("Run Benchmark"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    with st.spinner("Running inference across models... this takes a while on CPU"):
        results = run_benchmark(video_path, max_frames=max_frames, models=models_selected)

    df = pd.DataFrame(results)
    st.success("Done!")

    st.subheader("Raw results")
    st.dataframe(df, use_container_width=True)

    st.subheader("Speed (FPS) — higher is faster")
    st.bar_chart(df.set_index("model")["fps"])

    st.subheader("Avg inference time per frame (ms) — lower is faster")
    st.bar_chart(df.set_index("model")["avg_inference_ms"])

    st.subheader("Avg detection confidence — higher suggests better accuracy")
    st.bar_chart(df.set_index("model")["avg_confidence"])

    st.subheader("Avg detections per frame — more can mean better recall (or more false positives)")
    st.bar_chart(df.set_index("model")["avg_detections_per_frame"])

    st.info(
        "Rule of thumb: yolov8n = fastest/least accurate, yolov8s = balanced, "
        "yolov8m = most accurate/slowest. Pick based on your FPS requirement vs "
        "accuracy need. GPU narrows the speed gap significantly."
    )
