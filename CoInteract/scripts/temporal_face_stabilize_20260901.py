#!/usr/bin/env python3
"""Training-free face/color stabilization for generated CoInteract videos.

The script edits only a soft face mask. It supports two lightweight modes:
``restore`` sharpens/normalizes each face region, while ``color_anchor`` softly
matches face color statistics to an exponential moving average initialized from
the first detected face.
"""

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


def smooth_bbox(prev: tuple[int, int, int, int] | None, curr: tuple[int, int, int, int], ema: float):
    if prev is None:
        return curr
    return tuple(int(round(ema * p + (1.0 - ema) * c)) for p, c in zip(prev, curr))


def soft_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), np.float32)
    center = (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5))
    axes = (max(2, int((x2 - x1) * 0.56)), max(2, int((y2 - y1) * 0.63)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur = max(11, int(min(w, h) * 0.035) | 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]


def restore_face(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    restored = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(restored, (0, 0), 0.85)
    return cv2.addWeighted(restored, 1.28, blur, -0.28, 0)


def masked_mean_std(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.clip(mask[..., 0], 0.0, 1.0)
    total = float(weights.sum()) + 1e-6
    flat = img.reshape(-1, img.shape[-1]).astype(np.float32)
    wf = weights.reshape(-1, 1)
    mean = (flat * wf).sum(axis=0) / total
    var = (((flat - mean) ** 2) * wf).sum(axis=0) / total
    return mean, np.sqrt(var + 1e-6)


def color_match(frame_bgr: np.ndarray, mask: np.ndarray, anchor_stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_mean, cur_std = masked_mean_std(lab, mask)
    target_mean, target_std = anchor_stats
    adjusted = (lab - cur_mean) / (cur_std + 1e-6) * target_std + target_mean
    adjusted[..., 0] = np.clip(adjusted[..., 0], 0, 255)
    adjusted[..., 1:] = np.clip(adjusted[..., 1:], 0, 255)
    return cv2.cvtColor(adjusted.astype(np.uint8), cv2.COLOR_LAB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["restore", "color_anchor"], required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--bbox_ema", type=float, default=0.70)
    parser.add_argument("--color_ema", type=float, default=0.92)
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
    prev_bbox = None
    anchor_stats = None
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bbox = smooth_bbox(prev_bbox, detect_bbox(app, frame), args.bbox_ema)
        prev_bbox = bbox
        mask = soft_mask((height, width), bbox)
        if args.mode == "restore":
            edited = restore_face(frame)
        else:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            stats = masked_mean_std(lab, mask)
            if anchor_stats is None:
                anchor_stats = stats
            else:
                anchor_stats = (
                    args.color_ema * anchor_stats[0] + (1.0 - args.color_ema) * stats[0],
                    args.color_ema * anchor_stats[1] + (1.0 - args.color_ema) * stats[1],
                )
            edited = color_match(frame, mask, anchor_stats)
        alpha = np.clip(mask * args.strength, 0.0, 1.0)
        out = frame.astype(np.float32) * (1.0 - alpha) + edited.astype(np.float32) * alpha
        writer.write(np.clip(out, 0, 255).astype(np.uint8))
        count += 1

    cap.release()
    writer.release()
    print(f"[done] wrote={out_path} frames={count} mode={args.mode} strength={args.strength}")


if __name__ == "__main__":
    main()
