#!/usr/bin/env python3
"""Keyframe-anchored face stabilization for generated P2V clips.

This is a training-free postprocess inspired by keyframe-anchored identity
methods. It never changes CoInteract weights or the pose. It only blends a
soft face-region detail/color anchor into generated frames, with deterministic
anchor frames such as 0,25,50,75.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - optional remote dependency
    FaceAnalysis = None


def make_detector():
    if FaceAnalysis is None:
        return None
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def read_video(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return frames, float(fps)


def detect_bbox(app, frame_bgr: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame_bgr.shape[:2]
    if app is not None:
        faces = app.get(frame_bgr)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = np.round(face.bbox).astype(int).tolist()
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            bw, bh = max(4, x2 - x1) * 1.35, max(4, y2 - y1) * 1.45
            return (
                max(0, int(round(cx - bw * 0.5))),
                max(0, int(round(cy - bh * 0.5))),
                min(w, int(round(cx + bw * 0.5))),
                min(h, int(round(cy + bh * 0.5))),
            )
    return int(0.35 * w), int(0.08 * h), int(0.65 * w), int(0.34 * h)


def clamp_bbox(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, min(w - 2, x1)), max(0, min(h - 2, y1))
    x2, y2 = max(x1 + 2, min(w, x2)), max(y1 + 2, min(h, y2))
    return x1, y1, x2, y2


def soft_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], feather: float) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), np.float32)
    center = (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5))
    axes = (max(2, int((x2 - x1) * 0.52)), max(2, int((y2 - y1) * 0.60)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur = max(7, int(min(w, h) * feather) | 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]


def crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2].copy()


def lab_stats(img: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    if mask is None:
        flat = lab.reshape(-1, 3)
        return flat.mean(0), flat.std(0) + 1e-6
    weights = np.clip(mask[..., 0], 0.0, 1.0).reshape(-1, 1)
    flat = lab.reshape(-1, 3)
    total = weights.sum() + 1e-6
    mean = (flat * weights).sum(0) / total
    std = np.sqrt((((flat - mean) ** 2) * weights).sum(0) / total + 1e-6)
    return mean, std


def color_match(src_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sm, ss = lab_stats(src_bgr)
    tm, ts = lab_stats(target_bgr)
    out = (src_lab - sm) / ss * ts + tm
    out[..., 0] = np.clip(out[..., 0], 0, 255)
    out[..., 1:] = np.clip(out[..., 1:], 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)


def high_freq_detail(src_bgr: np.ndarray, target_bgr: np.ndarray, amount: float) -> np.ndarray:
    src = color_match(src_bgr, target_bgr).astype(np.float32)
    target = target_bgr.astype(np.float32)
    detail = src - cv2.GaussianBlur(src, (0, 0), 1.2)
    return np.clip(target + amount * detail, 0, 255).astype(np.uint8)


def parse_indices(text: str, total: int) -> list[int]:
    raw = [int(x.strip()) for x in text.split(",") if x.strip()]
    return sorted({max(0, min(total - 1, idx)) for idx in raw}) or [0]


def nearest_anchor(idx: int, anchors: list[int]) -> int:
    return min(anchors, key=lambda a: abs(a - idx))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchor_frames", default="0,25,50,75")
    parser.add_argument("--anchor_source", choices=["video", "reference"], default="video")
    parser.add_argument("--strength", type=float, default=0.20)
    parser.add_argument("--detail_strength", type=float, default=0.55)
    parser.add_argument("--feather", type=float, default=0.035)
    args = parser.parse_args()

    frames, fps = read_video(args.video)
    ref = cv2.imread(args.reference)
    if ref is None:
        raise RuntimeError(f"failed to read reference image: {args.reference}")
    app = make_detector()
    h, w = frames[0].shape[:2]
    anchors = parse_indices(args.anchor_frames, len(frames))

    bboxes = [clamp_bbox(detect_bbox(app, frame), w, h) for frame in frames]
    ref_bbox = clamp_bbox(detect_bbox(app, ref), ref.shape[1], ref.shape[0])
    ref_face = crop(ref, ref_bbox)

    anchor_faces: dict[int, np.ndarray] = {}
    for idx in anchors:
        if args.anchor_source == "reference":
            anchor_faces[idx] = ref_face
        else:
            anchor_faces[idx] = crop(frames[idx], bboxes[idx])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for idx, frame in enumerate(frames):
        bbox = bboxes[idx]
        x1, y1, x2, y2 = bbox
        target = crop(frame, bbox)
        if target.size == 0:
            writer.write(frame)
            continue
        anchor = anchor_faces[nearest_anchor(idx, anchors)]
        anchor = cv2.resize(anchor, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
        edited = high_freq_detail(anchor, target, args.detail_strength)
        mask = soft_mask((h, w), bbox, args.feather)
        patch_mask = mask[y1:y2, x1:x2]
        patch_alpha = np.clip(patch_mask * args.strength, 0.0, 1.0)
        out = frame.copy().astype(np.float32)
        out[y1:y2, x1:x2] = target.astype(np.float32) * (1.0 - patch_alpha) + edited.astype(np.float32) * patch_alpha
        writer.write(np.clip(out, 0, 255).astype(np.uint8))

    writer.release()
    print(f"[done] wrote={out_path} frames={len(frames)} source={args.anchor_source} anchors={anchors}")


if __name__ == "__main__":
    main()
