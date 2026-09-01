#!/usr/bin/env python3
"""Prepare lightweight reference-face variants for training-free P2V tests.

This script intentionally does not download or require neural restoration weights. If
CodeFormer/GFPGAN/RestoreFormer are unavailable, it creates conservative OpenCV face
enhancement variants inside a soft face mask so CoInteract still receives one normal
full-frame reference image.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - remote dependency
    FaceAnalysis = None


def detect_face_bbox(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    if FaceAnalysis is not None:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        faces = app.get(image_bgr)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = np.round(face.bbox).astype(int).tolist()
            return x1, y1, x2, y2
    h, w = image_bgr.shape[:2]
    return int(0.35 * w), int(0.08 * h), int(0.65 * w), int(0.32 * h)


def expand_bbox(bbox: tuple[int, int, int, int], w: int, h: int, scale: float = 1.55):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1 = max(0, int(round(cx - bw * 0.5)))
    ny1 = max(0, int(round(cy - bh * 0.5)))
    nx2 = min(w, int(round(cx + bw * 0.5)))
    ny2 = min(h, int(round(cy + bh * 0.5)))
    return nx1, ny1, nx2, ny2


def soft_face_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), np.float32)
    center = (int((x1 + x2) * 0.5), int((y1 + y2) * 0.5))
    axes = (max(2, int((x2 - x1) * 0.55)), max(2, int((y2 - y1) * 0.62)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur = max(11, int(min(w, h) * 0.035) | 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]


def clahe_face(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    merged = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def sharpen_face(image_bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image_bgr, (0, 0), 1.2)
    return cv2.addWeighted(image_bgr, 1.55, blur, -0.55, 0)


def blend_masked(base: np.ndarray, edited: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    alpha = np.clip(mask * float(strength), 0.0, 1.0)
    out = base.astype(np.float32) * (1.0 - alpha) + edited.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = Path(args.reference)
    image = cv2.imread(str(ref))
    if image is None:
        raise RuntimeError(f"failed to read reference: {ref}")

    h, w = image.shape[:2]
    bbox = expand_bbox(detect_face_bbox(image), w, h)
    mask = soft_face_mask((h, w), bbox)
    variants = {
        "ref_original.jpg": image,
        "ref_face_clahe035.jpg": blend_masked(image, clahe_face(image), mask, 0.35),
        "ref_face_sharp040.jpg": blend_masked(image, sharpen_face(image), mask, 0.40),
        "ref_face_clahe_sharp045.jpg": blend_masked(image, sharpen_face(clahe_face(image)), mask, 0.45),
    }

    for name, variant in variants.items():
        cv2.imwrite(str(out_dir / name), variant, [cv2.IMWRITE_JPEG_QUALITY, 96])

    status = out_dir / "restoration_tools_status.txt"
    tools_found = []
    for tool in ("CodeFormer", "GFPGAN", "RestoreFormer"):
        for candidate in (
            Path(f"/data1/workspace/linxinliang/{tool}"),
            Path(f"/data1/workspace/linxinliang/CoInteract/{tool}"),
            Path(f"/data1/workspace/{tool}"),
        ):
            if candidate.exists():
                tools_found.append(f"{tool}: {candidate}")
                break
    if not tools_found:
        tools_found.append("No CodeFormer/GFPGAN/RestoreFormer installation found; used OpenCV masked face enhancement variants.")
    status.write_text("\n".join(tools_found) + f"\nface_bbox_expanded={bbox}\n", encoding="utf-8")

    # Keep a copy of the exact source reference next to the variants for auditing.
    shutil.copy2(ref, out_dir / "source_reference.jpg")
    print(f"[done] wrote reference variants to {out_dir}")


if __name__ == "__main__":
    main()
