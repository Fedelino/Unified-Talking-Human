#!/usr/bin/env python3
"""Yaw-gated identity-face retargeting for DWPose control videos.

This is a safer variant of ``id_face_retarget.py``. It keeps the original driving
face landmarks for easy frontal frames and gradually blends in reference-person
3D landmarks only when the driving head yaw becomes large enough to stress ID.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


STABLEANIMATOR_ROOT = "/data1/workspace/linxinliang/StableAnimator"
DWPOSE_ROOT = os.path.join(STABLEANIMATOR_ROOT, "DWPose")
DW_CKPT = os.path.join(STABLEANIMATOR_ROOT, "checkpoints/DWPose")
sys.path.insert(0, DWPOSE_ROOT)
os.environ.setdefault("DWPOSE_DET", os.path.join(DW_CKPT, "yolox_l.onnx"))
os.environ.setdefault("DWPOSE_POSE", os.path.join(DW_CKPT, "dw-ll_ucoco_384.onnx"))
os.chdir(STABLEANIMATOR_ROOT)

from dwpose_utils.dwpose_detector import dwpose_detector_aligned  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402
from skeleton_extraction import draw_pose  # noqa: E402


def euler_R(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = np.deg2rad([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def similarity_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0**2).sum() / max(len(src), 1)
    cov = (d0.T @ s0) / max(len(src), 1)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    scale = np.trace(np.diag(s)) / (var_s + 1e-9)
    return (scale * (src @ r.T)) + (mu_d - scale * (mu_s @ r.T))


def gate_weight(abs_yaw: float, threshold: float, softness: float) -> float:
    if softness <= 1e-6:
        return 1.0 if abs_yaw >= threshold else 0.0
    x = (abs_yaw - threshold) / softness
    return float(np.clip(1.0 / (1.0 + np.exp(-x)), 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driving_video", required=True)
    parser.add_argument("--reference_image", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--blend", type=float, default=0.25)
    parser.add_argument("--yaw_threshold", type=float, default=20.0)
    parser.add_argument("--yaw_softness", type=float, default=8.0)
    args = parser.parse_args()

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    ref_bgr = cv2.imread(args.reference_image)
    if ref_bgr is None:
        raise RuntimeError(f"failed to read reference image: {args.reference_image}")
    ref_faces = app.get(ref_bgr)
    if not ref_faces:
        raise RuntimeError("InsightFace could not detect a reference face")
    rf = max(ref_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    ref_l3d = rf.landmark_3d_68.astype(np.float64)
    ref_centered = ref_l3d - ref_l3d.mean(0)
    canonical = ref_centered @ euler_R(*rf.pose)

    cap = cv2.VideoCapture(args.driving_video)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if args.max_frames > 0 and len(frames) >= args.max_frames:
            break
    cap.release()

    out_frames: list[np.ndarray] = []
    yaw_values: list[float] = []
    weights: list[float] = []
    used = 0
    fallback = 0
    h, w = args.height, args.width

    for bgr in frames:
        pose = dwpose_detector_aligned(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        faces = pose.get("faces")
        det = app.get(bgr)
        if faces is not None and len(faces) > 0 and det:
            tf = max(det, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            yaw = float(tf.pose[0])
            local_weight = args.blend * gate_weight(abs(yaw), args.yaw_threshold, args.yaw_softness)
            yaw_values.append(yaw)
            weights.append(local_weight)
            if local_weight > 1e-4:
                posed = canonical @ euler_R(*tf.pose).T
                projected = posed[:, :2]
                driving_px = faces[0] * np.array([w, h])
                aligned = similarity_align(projected, driving_px) / np.array([w, h])
                faces[0] = local_weight * aligned + (1.0 - local_weight) * faces[0]
                pose["faces"] = faces
                used += 1
            else:
                fallback += 1
        else:
            fallback += 1
        out_frames.append(draw_pose(pose, h, w).transpose(1, 2, 0))

    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for frame in out_frames:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()

    if yaw_values:
        yaws = np.asarray(yaw_values)
        ws = np.asarray(weights)
        print(
            f"[done] wrote={args.out_video} frames={len(out_frames)} "
            f"yaw_min={yaws.min():.2f} yaw_max={yaws.max():.2f} "
            f"mean_weight={ws.mean():.4f} max_weight={ws.max():.4f} "
            f"used={used} fallback={fallback}",
            flush=True,
        )
    else:
        print(f"[done] wrote={args.out_video} frames={len(out_frames)} no_yaw_detected fallback={fallback}", flush=True)


if __name__ == "__main__":
    main()
