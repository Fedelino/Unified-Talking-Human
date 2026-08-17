#!/usr/bin/env python3
"""Measure face identity drift in generated videos.

This script is intentionally lightweight: it only needs OpenCV and NumPy for
face crops, and optionally onnxruntime plus an ArcFace-compatible ONNX model
for cosine identity scores.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_video_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path)
    path = Path(value)
    return path.stem, path


def largest_face_bgr(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> tuple[tuple[int, int, int, int], bool]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    if len(faces):
        image_h, image_w = frame_bgr.shape[:2]
        plausible = []
        for x, y, w, h in faces:
            center_y = y + h / 2.0
            side = max(w, h)
            if center_y <= image_h * 0.36 and side <= image_w * 0.38:
                plausible.append((x, y, w, h))
        if not plausible:
            return _top_center_fallback(frame_bgr), False
        x, y, w, h = max(plausible, key=lambda box: int(box[2]) * int(box[3]))
        pad = int(max(w, h) * 0.28)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        return (x1, y1, x2, y2), True

    return _top_center_fallback(frame_bgr), False


def _top_center_fallback(frame_bgr: np.ndarray) -> tuple[int, int, int, int]:
    # Full-body videos often make frontal detectors fail. This fallback keeps
    # the script useful for visual sheets, but rows are marked detected=False.
    h, w = frame_bgr.shape[:2]
    side = int(min(w * 0.28, h * 0.22))
    cx = w // 2
    cy = int(h * 0.14)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return x1, y1, x2, y2


def crop_face_rgb(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


class ArcFaceONNX:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_AREA).astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))[None]
        out = self.session.run(None, {self.input_name: face})[0]
        emb = np.asarray(out).reshape(-1).astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def sample_video(path: Path, frame_stride: int, max_frames: int | None) -> Iterable[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride == 0:
            yield frame_idx, frame_idx / fps, frame
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        frame_idx += 1
    cap.release()


def nearest_detected_bbox(items: list[dict], idx: int) -> tuple[int, int, int, int]:
    detected = [j for j, item in enumerate(items) if item["detected"]]
    if not detected:
        return items[idx]["bbox"]
    nearest = min(detected, key=lambda j: abs(j - idx))
    return items[nearest]["bbox"]


def make_sheet(rows: list[dict], out_path: Path, thumb_size: tuple[int, int] = (128, 128)) -> None:
    font = ImageFont.load_default()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    max_cols = max((len(v) for v in grouped.values()), default=1)
    label_w = 170
    cell_w, cell_h = thumb_size[0], thumb_size[1] + 28
    sheet = Image.new("RGB", (label_w + max_cols * cell_w, max(1, len(grouped)) * cell_h), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)

    y = 0
    for label, items in grouped.items():
        draw.text((8, y + 8), label, fill=(255, 255, 255), font=font)
        for col, row in enumerate(items):
            img = Image.open(row["crop_path"]).convert("RGB")
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", thumb_size, (28, 28, 28))
            canvas.paste(img, ((thumb_size[0] - img.width) // 2, (thumb_size[1] - img.height) // 2))
            x = label_w + col * cell_w
            sheet.paste(canvas, (x, y))
            score = row.get("cosine")
            score_txt = "n/a" if score in ("", None) else f"{float(score):.3f}"
            draw.text((x + 4, y + thumb_size[1] + 4), f"{row['time_sec']:.1f}s {score_txt}", fill=(230, 230, 230), font=font)
        y += cell_h

    sheet.save(out_path, quality=92)


def maybe_plot(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = sorted({r["label"] for r in rows if r.get("cosine") not in ("", None)})
    if not labels:
        return
    plt.figure(figsize=(10, 5))
    for label in labels:
        xs, ys = [], []
        for row in rows:
            if row["label"] == label and row.get("cosine") not in ("", None):
                xs.append(float(row["time_sec"]))
                ys.append(float(row["cosine"]))
        plt.plot(xs, ys, marker="o", label=label)
    plt.ylim(0, 1)
    plt.xlabel("time (s)")
    plt.ylabel("ArcFace cosine vs reference")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--video", action="append", required=True, help="Either /path/video.mp4 or label=/path/video.mp4")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--arcface_onnx", type=Path, default=None)
    parser.add_argument("--frame_stride", type=int, default=50)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_dir / "face_crops"
    crops_dir.mkdir(exist_ok=True)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV face cascade: {cascade_path}")

    arcface = None
    ref_emb = None
    if args.arcface_onnx:
        if not args.arcface_onnx.exists():
            raise FileNotFoundError(args.arcface_onnx)
        arcface = ArcFaceONNX(args.arcface_onnx)

    ref_bgr = cv2.imread(str(args.reference))
    if ref_bgr is None:
        raise RuntimeError(f"Could not read reference image: {args.reference}")
    ref_bbox, _ = largest_face_bgr(ref_bgr, cascade)
    ref_face = crop_face_rgb(ref_bgr, ref_bbox)
    Image.fromarray(ref_face).save(args.out_dir / "reference_face.png")
    if arcface is not None:
        ref_emb = arcface.embed(ref_face)

    rows: list[dict] = []
    for label, video_path in map(parse_video_arg, args.video):
        sampled = []
        for frame_idx, time_sec, frame_bgr in sample_video(video_path, args.frame_stride, args.max_frames):
            bbox, detected = largest_face_bgr(frame_bgr, cascade)
            sampled.append({
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "frame_bgr": frame_bgr,
                "bbox": bbox,
                "detected": detected,
            })
        for sample_idx, sample in enumerate(sampled):
            frame_idx = sample["frame_idx"]
            time_sec = sample["time_sec"]
            frame_bgr = sample["frame_bgr"]
            bbox = sample["bbox"] if sample["detected"] else nearest_detected_bbox(sampled, sample_idx)
            face = crop_face_rgb(frame_bgr, bbox)
            crop_path = crops_dir / f"{label}_f{frame_idx:06d}.png"
            Image.fromarray(face).save(crop_path)
            emb = arcface.embed(face) if arcface is not None else None
            score = cosine(ref_emb, emb)
            x1, y1, x2, y2 = bbox
            rows.append({
                "label": label,
                "video": str(video_path),
                "frame": frame_idx,
                "time_sec": round(time_sec, 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "detected": sample["detected"],
                "cosine": "" if score is None or math.isnan(score) else round(score, 6),
                "crop_path": str(crop_path),
            })

    csv_path = args.out_dir / "id_drift_frames.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.out_dir / "id_drift_summary.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = ["label", "frames", "cosine_mean", "cosine_min", "cosine_last"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in sorted({r["label"] for r in rows}):
            vals = [float(r["cosine"]) for r in rows if r["label"] == label and r["cosine"] != ""]
            writer.writerow({
                "label": label,
                "frames": sum(1 for r in rows if r["label"] == label),
                "cosine_mean": "" if not vals else round(float(np.mean(vals)), 6),
                "cosine_min": "" if not vals else round(float(np.min(vals)), 6),
                "cosine_last": "" if not vals else round(vals[-1], 6),
            })

    make_sheet(rows, args.out_dir / "face_crop_sheet.jpg")
    maybe_plot(rows, args.out_dir / "id_drift_plot.png")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.out_dir / 'face_crop_sheet.jpg'}")
    if arcface is None:
        print("ArcFace ONNX not provided: exported face crops only, no identity cosine scores.")


if __name__ == "__main__":
    main()
