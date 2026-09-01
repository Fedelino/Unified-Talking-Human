#!/usr/bin/env python3
"""Apply lightweight masked face enhancement to an already generated video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - remote dependency
    FaceAnalysis = None


def make_detector():
    if FaceAnalysis is None:
        return None
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def fallback_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    return int(0.35 * w), int(0.08 * h), int(0.65 * w), int(0.34 * h)


def detect_bbox(app, frame_bgr: np.ndarray) -> tuple[int, int, int, int]:
    if app is not None:
        faces = app.get(frame_bgr)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = np.round(face.bbox).astype(int).tolist()
            h, w = frame_bgr.shape[:2]
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            bw, bh = (x2 - x1) * 1.45, (y2 - y1) * 1.55
            return (
                max(0, int(round(cx - bw * 0.5))),
                max(0, int(round(cy - bh * 0.5))),
                min(w, int(round(cx + bw * 0.5))),
                min(h, int(round(cy + bh * 0.5))),
            )
    return fallback_bbox(frame_bgr)


def soft_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), np.float32)
    center = (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5))
    axes = (max(2, int((x2 - x1) * 0.55)), max(2, int((y2 - y1) * 0.62)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur = max(11, int(min(w, h) * 0.035) | 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]


def enhance(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    clahe_bgr = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(clahe_bgr, (0, 0), 1.0)
    return cv2.addWeighted(clahe_bgr, 1.45, blur, -0.45, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strength", type=float, required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    app = make_detector()
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bbox = detect_bbox(app, frame)
        mask = soft_mask((height, width), bbox)
        edited = enhance(frame)
        alpha = np.clip(mask * float(args.strength), 0.0, 1.0)
        out = frame.astype(np.float32) * (1.0 - alpha) + edited.astype(np.float32) * alpha
        writer.write(np.clip(out, 0, 255).astype(np.uint8))
        count += 1
    cap.release()
    writer.release()
    print(f"[done] wrote {out_path} frames={count} strength={args.strength}")


if __name__ == "__main__":
    main()
