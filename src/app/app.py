"""
Streamlit app for real-time emotion detection.

Usage:
    streamlit run src/app/app.py
"""

import json
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.predictor import EmotionPredictor

# ── Emotion metadata ────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😄",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

# Color (hex) used in the confidence bar for each emotion
EMOTION_COLOR = {
    "angry":    "#e74c3c",
    "disgust":  "#27ae60",
    "fear":     "#8e44ad",
    "happy":    "#f1c40f",
    "neutral":  "#95a5a6",
    "sad":      "#2980b9",
    "surprise": "#e67e22",
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide",
)

st.title("😊 Real-Time Emotion Detection")
st.markdown(
    "Detects faces and classifies emotions using a fine-tuned **EfficientNet-B0** "
    "trained on FER-2013 (68.57% val accuracy). Upload an image or video to get started."
)

# ── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    ckpt_dir = Path("models/checkpoints")
    if (ckpt_dir / "best_efficientnet_b0.pt").exists():
        return EmotionPredictor(str(ckpt_dir / "best_efficientnet_b0.pt"))
    elif (ckpt_dir / "best_baseline_cnn.pt").exists():
        return EmotionPredictor(str(ckpt_dir / "best_baseline_cnn.pt"))
    else:
        st.error("❌ No model checkpoint found! Train a model first.")
        st.code("python src/models/train.py --config configs/baseline_cnn.yaml")
        st.stop()


predictor = load_predictor()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Mode", ["📷 Upload Image", "🎬 Upload Video"])
confidence_threshold = st.sidebar.slider("Face Detection Confidence", 0.3, 1.0, 0.5, 0.05)

# ── Feature 2: Model Info sidebar ────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🧠 Model Info")

# Count parameters from the loaded model
total_params = sum(p.numel() for p in predictor.model.parameters())
trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)

# Measure inference speed: run 10 dummy forward passes and average
import torch
dummy = torch.randn(1, 1, predictor.image_size, predictor.image_size).to(predictor.device)
timings = []
with torch.no_grad():
    for _ in range(10):
        t0 = time.perf_counter()
        predictor.model(dummy)
        timings.append((time.perf_counter() - t0) * 1000)
avg_ms = sum(timings[2:]) / len(timings[2:])  # skip first 2 (warmup)

# Detect which checkpoint is loaded from the model's config
import torch as _torch
ckpt_dir = Path("models/checkpoints")
ckpt_path = (
    ckpt_dir / "best_efficientnet_b0.pt"
    if (ckpt_dir / "best_efficientnet_b0.pt").exists()
    else ckpt_dir / "best_baseline_cnn.pt"
)
ckpt = _torch.load(str(ckpt_path), map_location="cpu")
model_name = ckpt["config"]["model"]["name"]
best_val_acc = ckpt.get("val_acc", "N/A")

st.sidebar.markdown(f"""
| | |
|---|---|
| **Model** | `{model_name}` |
| **Parameters** | {total_params/1e6:.1f}M |
| **Val Accuracy** | {best_val_acc:.2%} |
| **Inference** | {avg_ms:.1f} ms/frame |
| **Device** | `{predictor.device}` |
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by [Krishna Singh](https://github.com/krishna8399)**\n\n"
    "MSc AI @ IU Berlin"
)


# ── Helper: color-coded confidence bar ───────────────────────────────────────
def confidence_bar(emotion: str, confidence: float) -> str:
    """Render an HTML progress bar colored by emotion."""
    color = EMOTION_COLOR.get(emotion, "#888")
    pct = int(confidence * 100)
    return (
        f'<div style="background:#eee;border-radius:6px;height:18px;width:100%">'
        f'<div style="background:{color};width:{pct}%;height:18px;border-radius:6px;'
        f'display:flex;align-items:center;padding-left:6px">'
        f'<span style="color:white;font-size:12px;font-weight:bold">{pct}%</span>'
        f'</div></div>'
    )


# ── Helper: display per-face results ─────────────────────────────────────────
def display_results(predictions, col):
    if not predictions:
        col.warning("No faces detected.")
        return

    for i, pred in enumerate(predictions):
        emoji = EMOTION_EMOJI.get(pred.emotion, "")
        col.markdown(f"#### {emoji} Face {i + 1}")
        col.markdown(f"**{pred.emotion.upper()}** — {pred.confidence:.1%} confidence")

        # Color-coded confidence bar for the top emotion
        col.markdown(
            confidence_bar(pred.emotion, pred.confidence),
            unsafe_allow_html=True,
        )
        col.markdown("")  # spacer

        # Probability breakdown for all 7 emotions
        col.markdown("**All emotions:**")
        for emo, prob in sorted(pred.all_probs.items(), key=lambda x: -x[1]):
            bar = confidence_bar(emo, prob)
            col.markdown(
                f"{EMOTION_EMOJI.get(emo,'')} `{emo:<9}` {bar}",
                unsafe_allow_html=True,
            )
        col.divider()


# ── Feature 3: Download Results helper ───────────────────────────────────────
def download_button(predictions, source_name: str):
    """Render a download button that exports predictions as JSON."""
    if not predictions:
        return
    data = {
        "source": source_name,
        "faces": [
            {
                "face_index": i + 1,
                "emotion": p.emotion,
                "confidence": round(p.confidence, 4),
                "bbox": list(p.bbox),
                "all_probabilities": p.all_probs,
            }
            for i, p in enumerate(predictions)
        ],
    }
    json_str = json.dumps(data, indent=2)
    st.download_button(
        label="⬇️ Download Results (JSON)",
        data=json_str,
        file_name="emotion_predictions.json",
        mime="application/json",
    )


# ════════════════════════════════════════════════════════════════════════════
# IMAGE MODE
# ════════════════════════════════════════════════════════════════════════════
if mode == "📷 Upload Image":
    uploaded = st.file_uploader(
        "Upload an image with faces",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        t0 = time.perf_counter()
        predictions = predictor.predict_frame(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        annotated = predictor.annotate_frame(frame, predictions)

        col1, col2 = st.columns([2, 1])
        col1.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=f"Detected {len(predictions)} face(s) — {elapsed_ms:.0f} ms",
            use_container_width=True,
        )

        with col2:
            display_results(predictions, col2)
            # Feature 3: download button
            download_button(predictions, uploaded.name)


# ════════════════════════════════════════════════════════════════════════════
# VIDEO MODE
# ════════════════════════════════════════════════════════════════════════════
elif mode == "🎬 Upload Video":
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        stframe = st.empty()
        progress_bar = st.progress(0)

        frame_count = 0
        processed_count = 0
        process_every_n = 3

        # Feature 1: Emotion Timeline data
        # Store (timestamp_sec, emotion, confidence) for every processed frame
        timeline_records = []
        # Last seen predictions (to annotate skipped frames too)
        last_predictions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress_bar.progress(min(frame_count / max(total_frames, 1), 1.0))

            if frame_count % process_every_n == 0:
                last_predictions = predictor.predict_frame(frame)
                processed_count += 1

                # Record emotion at this timestamp for the timeline
                timestamp = frame_count / fps
                for pred in last_predictions:
                    timeline_records.append({
                        "time_sec": round(timestamp, 2),
                        "emotion": pred.emotion,
                        "confidence": round(pred.confidence, 3),
                    })

            annotated = predictor.annotate_frame(frame, last_predictions)
            stframe.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        cap.release()
        progress_bar.empty()
        st.success(f"✅ Processed {processed_count} frames from {frame_count} total")

        # ── Feature 1: Emotion Timeline chart ────────────────────────────────
        if timeline_records:
            st.markdown("### 📈 Emotion Timeline")
            st.markdown("How emotions changed throughout the video (confidence over time):")

            df = pd.DataFrame(timeline_records)

            # Pivot: rows = timestamps, columns = emotions, values = confidence
            # Fill missing emotions with 0 so the chart shows all 7 lines
            timeline_pivot = (
                df.pivot_table(
                    index="time_sec",
                    columns="emotion",
                    values="confidence",
                    aggfunc="mean",
                )
                .reindex(columns=list(EMOTION_EMOJI.keys()), fill_value=0)
                .fillna(0)
            )

            # Rename columns to include emoji for the legend
            timeline_pivot.columns = [
                f"{EMOTION_EMOJI.get(c, '')} {c}" for c in timeline_pivot.columns
            ]

            st.line_chart(timeline_pivot, height=300)

            # Summary: most common emotion in the video
            dominant = df.groupby("emotion")["confidence"].mean().idxmax()
            emoji = EMOTION_EMOJI.get(dominant, "")
            st.info(f"**Dominant emotion in video:** {emoji} {dominant.upper()}")

            # ── Feature 3: Download full timeline results ─────────────────
            download_data = {
                "source": uploaded.name,
                "total_frames": frame_count,
                "processed_frames": processed_count,
                "fps": fps,
                "timeline": timeline_records,
            }
            st.download_button(
                label="⬇️ Download Timeline (JSON)",
                data=json.dumps(download_data, indent=2),
                file_name="emotion_timeline.json",
                mime="application/json",
            )
