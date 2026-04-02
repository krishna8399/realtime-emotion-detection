"""
Streamlit app for real-time emotion detection.

Usage:
    streamlit run src/app/app.py

Features:
    - Upload image or video for emotion detection
    - Real-time webcam feed (if available)
    - Emotion distribution chart
    - Per-face emotion breakdown
"""

import cv2
import numpy as np
import streamlit as st
import tempfile  # for saving uploaded video to a temp file
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.predictor import EmotionPredictor


# Configure the browser tab title, icon, and layout
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide",  # use full browser width
)

st.title("😊 Real-Time Emotion Detection")
st.markdown(
    "Detects faces and classifies emotions using a fine-tuned EfficientNet-B0. "
    "Upload an image or video to get started."
)


@st.cache_resource  # cache the model in memory — only loads once per session
def load_predictor():
    """Load model once and cache it."""
    # Prefer EfficientNet (more accurate) but fall back to baseline CNN if not trained yet
    ckpt_dir = Path("models/checkpoints")
    if (ckpt_dir / "best_efficientnet_b0.pt").exists():
        return EmotionPredictor(str(ckpt_dir / "best_efficientnet_b0.pt"))
    elif (ckpt_dir / "best_baseline_cnn.pt").exists():
        return EmotionPredictor(str(ckpt_dir / "best_baseline_cnn.pt"))
    else:
        st.error("❌ No model checkpoint found! Train a model first.")
        st.code("python src/models/train.py --config configs/baseline_cnn.yaml")
        st.stop()  # halt the app — nothing to do without a model


# Sidebar controls
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Mode", ["📷 Upload Image", "🎬 Upload Video"])
confidence_threshold = st.sidebar.slider(
    "Face Detection Confidence", 0.3, 1.0, 0.5, 0.05,  # min, max, default, step
)

# Load model (cached after first call)
predictor = load_predictor()


def display_results(predictions, col):
    """Display emotion predictions in a column."""
    if not predictions:
        col.warning("No faces detected in the image.")
        return

    for i, pred in enumerate(predictions):
        col.markdown(f"**Face {i + 1}**")
        col.markdown(
            f"Emotion: **{pred.emotion.upper()}** "
            f"({pred.confidence:.1%} confidence)"
        )

        # Bar chart showing probability for each of the 7 emotions
        emotions = list(pred.all_probs.keys())
        probs = list(pred.all_probs.values())
        chart_data = {
            "Emotion": emotions,
            "Probability": probs,
        }
        col.bar_chart(chart_data, x="Emotion", y="Probability", height=200)
        col.divider()  # horizontal separator between faces


if mode == "📷 Upload Image":
    uploaded = st.file_uploader(
        "Upload an image with faces",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        # Decode uploaded bytes into an OpenCV BGR image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run face detection + emotion classification
        predictions = predictor.predict_frame(frame)
        annotated = predictor.annotate_frame(frame, predictions)  # draw boxes + labels

        # Two-column layout: image on left (2/3 width), results on right (1/3 width)
        col1, col2 = st.columns([2, 1])
        col1.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),  # convert BGR → RGB for Streamlit display
            caption=f"Detected {len(predictions)} face(s)",
            use_container_width=True,
        )
        display_results(predictions, col2)


elif mode == "🎬 Upload Video":
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded:
        # Save uploaded video bytes to a temp file so OpenCV can read it frame-by-frame
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)  # open the temp video file
        stframe = st.empty()      # placeholder to update image in-place each frame
        results_area = st.empty() # placeholder for results (unused currently)

        frame_count = 0
        process_every_n = 3  # skip frames to reduce latency: only process every 3rd frame

        while cap.isOpened():
            ret, frame = cap.read()  # read next frame; ret=False at end of video
            if not ret:
                break

            frame_count += 1

            # Skip most frames for speed — emotion doesn't change drastically between frames
            if frame_count % process_every_n == 0:
                predictions = predictor.predict_frame(frame)
                annotated = predictor.annotate_frame(frame, predictions)
                stframe.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),  # BGR → RGB for display
                    use_container_width=True,
                )

        cap.release()  # free video file handle
        st.success("✅ Video processing complete!")


# Footer in sidebar with author info
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by [Krishna Singh](https://github.com/krishna8399)**\n\n"
    "MSc AI @ IU Berlin"
)
