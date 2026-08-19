"""
Streamlit UI: upload a video, run tracking pipeline, view result.

Run with: streamlit run app.py
"""
import streamlit as st
import yaml
import tempfile
import os

from src.video_pipeline import run_pipeline

st.set_page_config(page_title="Object Tracking", layout="centered")
st.title("🎯 Object Tracking Demo")

with open("configs/config.yaml", "r") as f:
    default_config = yaml.safe_load(f)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

confidence = st.slider("Detection confidence", 0.1, 0.9, default_config["model"]["confidence"])
class_options = st.multiselect(
    "Classes to track",
    ["person", "car", "bicycle", "motorcycle", "bus", "truck"],
    default=default_config.get("classes", {}).get("filter", ["person", "car"]),
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

    os.makedirs("outputs/videos", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    with st.spinner("Running detection + tracking..."):
        run_pipeline(config)

    st.success("Done!")
    st.video(config["video"]["output_path"])

    with open(config["logging"]["log_path"], "rb") as f:
        st.download_button("Download tracking log (CSV)", f, file_name="tracks.csv")
