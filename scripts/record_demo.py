"""
Demo recorder: run a video through the emotion pipeline and save annotated output.

Outputs:
    assets/demo_output.mp4  — annotated video at 10 fps, max 800 px wide
    assets/demo.gif         — same content as an animated GIF

Usage:
    # Auto-detect first video in assets/
    python scripts/record_demo.py

    # Specify a video file explicitly
    python scripts/record_demo.py --input path/to/video.mp4

    # Override output paths or fps
    python scripts/record_demo.py --input assets/sample.mp4 --fps 8 --max-width 640

    # Process every Nth frame (speeds up long videos; default every 3rd)
    python scripts/record_demo.py --every-n 5
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image  # used for GIF encoding

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.predictor import EmotionPredictor


# ── Colour palette for per-emotion annotation ────────────────────────────────
EMOTION_COLORS = {
    "angry":    (0,   0,   255),   # red
    "disgust":  (0,   128,   0),   # dark green
    "fear":     (128,   0, 128),   # purple
    "happy":    (0,   255, 255),   # yellow
    "neutral":  (200, 200, 200),   # light grey
    "sad":      (255,   0,   0),   # blue
    "surprise": (0,   255,   0),   # bright green
}


def find_input_video(assets_dir: Path) -> Path:
    """Return the first video file found in assets/, or raise if none exists."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for path in sorted(assets_dir.iterdir()):
        if path.suffix.lower() in video_exts:
            return path
    raise FileNotFoundError(
        f"No video file found in {assets_dir}. "
        "Place a sample video there (e.g. assets/sample.mp4) or use --input."
    )


def find_checkpoint(ckpt_dir: Path) -> Path:
    """Prefer EfficientNet checkpoint; fall back to baseline CNN."""
    for name in ("best_efficientnet_b0.pt", "best_baseline_cnn.pt"):
        path = ckpt_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No model checkpoint found in {ckpt_dir}. "
        "Train a model first: python src/models/train.py --config configs/efficientnet_v2.yaml"
    )


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale frame so its width does not exceed max_width (preserves aspect ratio)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_overlay(frame: np.ndarray, predictions, elapsed_ms: float) -> np.ndarray:
    """
    Draw emotion annotations on a frame.

    In addition to the bounding-box labels drawn by predictor.annotate_frame,
    this adds a small stats bar at the top-left showing inference time and
    the number of faces detected.
    """
    # Use the predictor's built-in annotation (bbox + label + background rect)
    annotated = predictions  # already annotated by caller

    # Stats bar: dark semi-transparent strip at the top
    h, w = annotated.shape[:2]
    strip_h = 28
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, strip_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)  # blend transparency

    # Text on the strip
    text = f"{elapsed_ms:.0f} ms/frame"
    cv2.putText(
        annotated, text,
        (8, strip_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (200, 200, 200), 1, cv2.LINE_AA,
    )
    return annotated


def frames_to_gif(frames_rgb: list, output_path: Path, fps: int) -> None:
    """Save a list of RGB numpy arrays as an animated GIF using Pillow."""
    duration_ms = int(1000 / fps)  # milliseconds per frame
    pil_frames = [Image.fromarray(f) for f in frames_rgb]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,               # loop forever
        duration=duration_ms,
        optimize=False,       # skip palette optimization (much faster to encode)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record annotated demo video and GIF.")
    p.add_argument("--input",     type=str, default=None,
                   help="Path to input video (default: first video in assets/)")
    p.add_argument("--output-mp4", type=str, default="assets/demo_output.mp4",
                   help="Output MP4 path (default: assets/demo_output.mp4)")
    p.add_argument("--output-gif", type=str, default="assets/demo.gif",
                   help="Output GIF path (default: assets/demo.gif)")
    p.add_argument("--fps",       type=int, default=10,
                   help="Output frame rate (default: 10)")
    p.add_argument("--max-width", type=int, default=800,
                   help="Max output width in pixels (default: 800)")
    p.add_argument("--every-n",  type=int, default=3,
                   help="Run prediction every N input frames; interpolate the rest "
                        "(default: 3, i.e. predict on every 3rd frame)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many input frames (useful for quick tests)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).parent.parent  # project root

    # ── Resolve paths ─────────────────────────────────────────────────────────
    input_path = Path(args.input) if args.input else find_input_video(root / "assets")
    output_mp4 = root / args.output_mp4
    output_gif = root / args.output_gif
    ckpt_path  = find_checkpoint(root / "models" / "checkpoints")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input  : {input_path}")
    print(f"Model  : {ckpt_path.name}")
    print(f"Output : {output_mp4}  |  {output_gif}")
    print(f"FPS    : {args.fps}  |  max width: {args.max_width}px")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    predictor = EmotionPredictor(str(ckpt_path))

    # ── Open input video ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Source : {total_frames} frames @ {source_fps:.1f} fps")

    # ── Determine output dimensions from the first frame ─────────────────────
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame from video.")
    first_resized = resize_frame(first_frame, args.max_width)
    out_h, out_w  = first_resized.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start

    # ── Set up MP4 writer ─────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, args.fps, (out_w, out_h))

    # ── Process frames ────────────────────────────────────────────────────────
    gif_frames_rgb = []    # collect resized RGB frames for GIF
    last_predictions = []  # re-use predictions on skipped frames
    frame_idx = 0
    written   = 0

    # Calculate how many input frames map to one output frame
    # (down-sample from source_fps to args.fps)
    input_frames_per_output = max(1, round(source_fps / args.fps))

    print("Processing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames and frame_idx >= args.max_frames:
            break

        # Only keep frames that align to the desired output fps
        if frame_idx % input_frames_per_output != 0:
            frame_idx += 1
            continue

        # Run prediction every N frames; reuse last result on the others
        if frame_idx % args.every_n == 0:
            t0 = time.perf_counter()
            last_predictions = predictor.predict_frame(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        else:
            elapsed_ms = 0.0

        # Annotate and resize
        annotated = predictor.annotate_frame(frame, last_predictions)
        annotated = resize_frame(annotated, args.max_width)
        annotated = draw_overlay(annotated, annotated, elapsed_ms)

        # Write to MP4
        writer.write(annotated)

        # Collect RGB copy for GIF
        gif_frames_rgb.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        written += 1
        if written % 20 == 0 or written == 1:
            pct = (frame_idx / max(total_frames, 1)) * 100
            print(f"  frame {frame_idx:>5} / {total_frames}  ({pct:.0f}%)  "
                  f"— {len(last_predictions)} face(s) detected")

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nProcessed {written} output frames from {frame_idx} input frames.")

    # ── Save MP4 ──────────────────────────────────────────────────────────────
    print(f"Saved MP4 : {output_mp4}  ({output_mp4.stat().st_size / 1e6:.1f} MB)")

    # ── Save GIF ──────────────────────────────────────────────────────────────
    if not gif_frames_rgb:
        print("No frames collected — skipping GIF.")
        return

    print(f"Encoding GIF ({len(gif_frames_rgb)} frames at {args.fps} fps)...")
    frames_to_gif(gif_frames_rgb, output_gif, args.fps)
    print(f"Saved GIF  : {output_gif}  ({output_gif.stat().st_size / 1e6:.1f} MB)")

    print("\nDone!")
    print("  Uncomment the demo line in README.md to show the GIF:")
    print("  ![Demo](assets/demo.gif)")


if __name__ == "__main__":
    main()
