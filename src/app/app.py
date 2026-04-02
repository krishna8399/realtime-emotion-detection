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
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.predictor import EmotionPredictor
from src.utils.visualize import GradCAM, overlay_heatmap
from src.data.dataset import EMOTION_LABELS

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😄",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

EMOTION_COLOR = {
    "angry":    "#e74c3c",
    "disgust":  "#27ae60",
    "fear":     "#8e44ad",
    "happy":    "#f1c40f",
    "neutral":  "#95a5a6",
    "sad":      "#2980b9",
    "surprise": "#e67e22",
}

# ── Page config ───────────────────────────────────────────────────────────────
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


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor() -> EmotionPredictor:
    """Load model checkpoint once and cache for the lifetime of the server."""
    ckpt_dir = Path("models/checkpoints")
    if (ckpt_dir / "best_efficientnet_b0.pt").exists():
        return EmotionPredictor(str(ckpt_dir / "best_efficientnet_b0.pt"))
    elif (ckpt_dir / "best_baseline_cnn.pt").exists():
        return EmotionPredictor(str(ckpt_dir / "best_baseline_cnn.pt"))
    else:
        st.error("❌ No model checkpoint found! Train a model first.")
        st.code("python src/models/train.py --config configs/baseline_cnn.yaml")
        st.stop()


# Show a spinner only on the very first load — cache_resource means this
# only runs once per server session, so it won't block subsequent reruns.
with st.spinner("Loading model and face detector… (first load only)"):
    predictor = load_predictor()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Input Mode", ["📷 Upload Image", "🎬 Upload Video"])
confidence_threshold = st.sidebar.slider(
    "Face Detection Confidence", 0.3, 1.0, 0.5, 0.05,
    help="Minimum confidence score for a face to be detected",
)

# ── Model Info sidebar — all values come from the cached predictor, no extra
#    computation at startup ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🧠 Model Info")

total_params = sum(p.numel() for p in predictor.model.parameters())
model_name = predictor.config["model"]["name"]

st.sidebar.markdown(f"""
| | |
|---|---|
| **Model** | `{model_name}` |
| **Parameters** | {total_params / 1e6:.1f}M |
| **Val Accuracy** | {predictor.val_acc:.2f}% |
| **Device** | `{predictor.device}` |
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by [Krishna Singh](https://github.com/krishna8399)**\n\n"
    "MSc AI @ IU Berlin"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def confidence_bar(emotion: str, confidence: float) -> str:
    """Return an HTML progress bar string colored by emotion."""
    color = EMOTION_COLOR.get(emotion, "#888")
    pct = int(confidence * 100)
    return (
        f'<div style="background:#eee;border-radius:6px;height:18px;width:100%">'
        f'<div style="background:{color};width:{pct}%;height:18px;border-radius:6px;'
        f'display:flex;align-items:center;padding-left:6px">'
        f'<span style="color:white;font-size:12px;font-weight:bold">{pct}%</span>'
        f'</div></div>'
    )


def display_results(predictions: list, col) -> None:
    """Render per-face emotion results into a Streamlit column."""
    if not predictions:
        col.warning("No faces detected.")
        return

    for i, pred in enumerate(predictions):
        emoji = EMOTION_EMOJI.get(pred.emotion, "")
        col.markdown(f"#### {emoji} Face {i + 1}")
        col.markdown(f"**{pred.emotion.upper()}** — {pred.confidence:.1%} confidence")
        col.markdown(confidence_bar(pred.emotion, pred.confidence), unsafe_allow_html=True)
        col.markdown("")

        col.markdown("**All emotions:**")
        for emo, prob in sorted(pred.all_probs.items(), key=lambda x: -x[1]):
            col.markdown(
                f"{EMOTION_EMOJI.get(emo, '')} `{emo:<9}` {confidence_bar(emo, prob)}",
                unsafe_allow_html=True,
            )
        col.divider()


def download_button(predictions: list, source_name: str) -> None:
    """Render a download button that exports per-face predictions as JSON."""
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
    st.download_button(
        label="⬇️ Download Results (JSON)",
        data=json.dumps(data, indent=2),
        file_name="emotion_predictions.json",
        mime="application/json",
    )


def resize_for_display(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Downscale a frame for display without modifying the original."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


# ── Grad-CAM explainability ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_gradcam(_predictor: EmotionPredictor) -> GradCAM:
    """Build GradCAM instance once — cached so hooks are only registered once."""
    model = _predictor.model
    if _predictor.config["model"]["name"] == "efficientnet_b0":
        target_layer = model.backbone.conv_head
    else:
        target_layer = model.features[-2]
    return GradCAM(model, target_layer)


def render_explainability(face_crop: np.ndarray, all_probs: dict, grad_cam: GradCAM) -> None:
    """
    Show Grad-CAM heatmap overlays for the top 3 predicted emotions in 3 columns.

    Each column shows the heatmap blended onto the face crop for one emotion,
    labelled with the emotion name and its predicted probability.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    resized = cv2.resize(gray, (predictor.image_size, predictor.image_size))
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(predictor.device)
    input_tensor.requires_grad_(True)

    top3 = sorted(all_probs.items(), key=lambda x: -x[1])[:3]
    label_to_idx = {v: k for k, v in EMOTION_LABELS.items()}

    cols = st.columns(3)
    for col, (emotion, prob) in zip(cols, top3):
        class_idx = label_to_idx[emotion]
        heatmap, _ = grad_cam.generate(input_tensor, target_class=class_idx)
        overlay = overlay_heatmap(face_crop, heatmap, alpha=0.45)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        col.image(overlay_rgb, use_container_width=True)
        col.markdown(
            f"**{EMOTION_EMOJI.get(emotion, '')} {emotion.upper()}** — {prob:.1%}  \n"
            f"{confidence_bar(emotion, prob)}",
            unsafe_allow_html=True,
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

        with st.spinner("Detecting faces…"):
            # Apply current slider confidence threshold
            predictor.face_detector.detector = \
                predictor.face_detector.mp_face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=confidence_threshold,
                )
            t0 = time.perf_counter()
            predictions = predictor.predict_frame(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        annotated = predictor.annotate_frame(frame, predictions)
        display_frame = resize_for_display(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        tab_results, tab_explain = st.tabs(["🎯 Results", "🔬 Explainability"])

        with tab_results:
            col1, col2 = st.columns([2, 1])
            col1.image(
                display_frame,
                caption=f"Detected {len(predictions)} face(s) — {elapsed_ms:.0f} ms",
                use_container_width=True,
            )
            with col2:
                display_results(predictions, col2)
                download_button(predictions, uploaded.name)

        with tab_explain:
            if not predictions:
                st.warning("No faces detected — nothing to explain.")
            else:
                grad_cam = get_gradcam(predictor)
                for i, pred in enumerate(predictions):
                    st.markdown(
                        f"#### {EMOTION_EMOJI.get(pred.emotion, '')} Face {i + 1}"
                        f" — Top 3 Grad-CAM Heatmaps"
                    )
                    st.caption("Red/warm regions = areas the model focused on most for each emotion.")
                    render_explainability(pred.face_crop_bgr, pred.all_probs, grad_cam)
                    if i < len(predictions) - 1:
                        st.divider()


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
        progress_bar = st.progress(0, text="Processing video…")
        status_text = st.empty()

        frame_count = 0
        processed_count = 0
        predict_every_n = 3     # run the model every 3rd frame
        display_every_n = 5     # update the Streamlit image every 5th frame (reduce re-renders)

        timeline_records = []
        last_predictions = []

        # Apply current slider confidence threshold
        predictor.face_detector.detector = \
            predictor.face_detector.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=confidence_threshold,
            )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress_bar.progress(
                min(frame_count / max(total_frames, 1), 1.0),
                text=f"Processing frame {frame_count}/{total_frames}…",
            )

            # Run prediction every N frames
            if frame_count % predict_every_n == 0:
                last_predictions = predictor.predict_frame(frame)
                processed_count += 1

                timestamp = frame_count / fps
                for pred in last_predictions:
                    timeline_records.append({
                        "time_sec": round(timestamp, 2),
                        "emotion": pred.emotion,
                        "confidence": round(pred.confidence, 3),
                    })

            # Only push a new image to the browser every N frames — each
            # stframe.image() call triggers a full Streamlit re-render, so
            # calling it on every frame is the main cause of slow video playback.
            if frame_count % display_every_n == 0:
                annotated = predictor.annotate_frame(frame, last_predictions)
                display_frame = resize_for_display(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), max_width=800
                )
                stframe.image(display_frame, use_container_width=True)

        cap.release()
        progress_bar.empty()
        status_text.empty()
        st.success(
            f"✅ Processed {processed_count} frames "
            f"({total_frames} total, {fps:.0f} fps source)"
        )

        # ── Emotion Timeline ──────────────────────────────────────────────────
        if timeline_records:
            st.markdown("### 📈 Emotion Timeline")
            st.caption("Emotion confidence over time (averaged per timestamp).")

            df = pd.DataFrame(timeline_records)
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
            timeline_pivot.columns = [
                f"{EMOTION_EMOJI.get(c, '')} {c}" for c in timeline_pivot.columns
            ]
            st.line_chart(timeline_pivot, height=300)

            dominant = df.groupby("emotion")["confidence"].mean().idxmax()
            st.info(
                f"**Dominant emotion in video:** "
                f"{EMOTION_EMOJI.get(dominant, '')} {dominant.upper()}"
            )

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
