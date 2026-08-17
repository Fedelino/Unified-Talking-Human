#!/usr/bin/env python3
"""
ArcFace crop/mask debugger for full-body CoInteract outputs.

This evaluates ArcFace only on a chosen face/head crop and exports the exact
crop + image-space mask used for the measurement. It avoids the old Haar-only
failure mode where a full-body frame can be scored on a torso/dress box.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from id_drift_metric import ArcFaceONNX, cos, crop_face, first_frame


def parse_csv_floats(value: str, expected: int):
    vals = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if len(vals) != expected:
        raise ValueError(f"expected {expected} comma-separated values, got {value!r}")
    return vals


def box_from_ratio(frame_bgr: np.ndarray, ratio):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = ratio
    return [
        max(0, min(w - 1, int(round(x1 * w)))),
        max(0, min(h - 1, int(round(y1 * h)))),
        max(1, min(w, int(round(x2 * w)))),
        max(1, min(h, int(round(y2 * h)))),
    ]


def crop_no_margin(frame_bgr: np.ndarray, box):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1)); x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def resize_keep_aspect(img, size):
    target_w, target_h = size
    h, w = img.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    out = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    out[y:y + nh, x:x + nw] = resized
    return out


def make_mask_overlay(frame, box, color=(0, 255, 0), alpha=0.28):
    overlay = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    cv2.rectangle(blended, (x1, y1), (x2, y2), color, thickness=2)
    return blended


def put_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_video_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


class OptionalHaarDetector:
    def __init__(self):
        self.cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            xml = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(xml)
            if not cascade.empty():
                self.cascade = cascade

    def largest_face(self, frame_bgr: np.ndarray):
        if self.cascade is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        return int(x), int(y), int(x + w), int(y + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--arcface_onnx", default="models/arcface/w600k_r50.onnx")
    ap.add_argument("--frames", default="0,25,50,75")
    ap.add_argument(
        "--bbox_ratio",
        default="0.36,0.14,0.57,0.34",
        help="x1,y1,x2,y2 as fractions of generated frame; defaults to current guidance mask",
    )
    ap.add_argument("--ref_bbox", default="", help="optional reference bbox x1,y1,x2,y2; otherwise Haar is used")
    ap.add_argument(
        "--ref_bbox_ratio",
        default="",
        help="optional reference x1,y1,x2,y2 fractions after optional resize; defaults to bbox_ratio",
    )
    ap.add_argument(
        "--resize_reference_to_video",
        action="store_true",
        help="resize reference to the generated video canvas before reference cropping",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    ratio = parse_csv_floats(args.bbox_ratio, 4)
    ref_ratio = parse_csv_floats(args.ref_bbox_ratio, 4) if args.ref_bbox_ratio else ratio

    arc = ArcFaceONNX(args.arcface_onnx)
    det = OptionalHaarDetector()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    ref = first_frame(args.reference)
    if args.resize_reference_to_video and video_w > 0 and video_h > 0:
        ref = cv2.resize(ref, (video_w, video_h), interpolation=cv2.INTER_LINEAR)
    if args.ref_bbox:
        ref_box = [int(x) for x in args.ref_bbox.split(",")]
    else:
        ref_box = box_from_ratio(ref, ref_ratio)
    if ref_box is None:
        raise SystemExit("No reference face box available; pass --ref_bbox or --ref_bbox_ratio")
    ref_crop = crop_face(ref, ref_box, margin=0.0)
    if ref_crop is None:
        raise SystemExit("Reference crop failed")
    ref_emb = arc.embed(ref_crop)
    cv2.imwrite(str(out_dir / "reference_arcface_crop.jpg"), ref_crop)

    tiles = []
    rows = []
    for idx in frames:
        frame = read_video_frame(cap, idx)
        if frame is None:
            continue
        current_box = box_from_ratio(frame, ratio)
        current_crop = crop_no_margin(frame, current_box)
        if current_crop is None:
            continue
        emb = arc.embed(current_crop)
        cos_ref = cos(emb, ref_emb)

        haar_box = det.largest_face(frame)
        overlay = make_mask_overlay(frame, current_box, color=(0, 255, 0))
        if haar_box is not None:
            x1, y1, x2, y2 = haar_box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = current_box
        mask[y1:y2, x1:x2] = 255
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(out_dir / f"frame_{idx:04d}_full_with_mask.jpg"), overlay)
        cv2.imwrite(str(out_dir / f"frame_{idx:04d}_arcface_crop.jpg"), current_crop)
        cv2.imwrite(str(out_dir / f"frame_{idx:04d}_mask.png"), mask)
        rows.append((idx, *current_box, float(cos_ref), haar_box))

        full_tile = resize_keep_aspect(overlay, (240, 416))
        crop_tile = resize_keep_aspect(current_crop, (180, 180))
        mask_tile = resize_keep_aspect(mask_bgr, (240, 416))
        tiles.append([
            put_label(full_tile, f"frame {idx}: green=current mask, red=Haar"),
            put_label(crop_tile, f"ArcFace crop cos={cos_ref:.3f}"),
            put_label(mask_tile, "binary face/head mask"),
        ])

    cap.release()
    if tiles:
        row_imgs = []
        for row in tiles:
            h = max(t.shape[0] for t in row)
            padded = []
            for t in row:
                if t.shape[0] < h:
                    pad = np.full((h - t.shape[0], t.shape[1], 3), 255, dtype=np.uint8)
                    t = np.vstack([t, pad])
                padded.append(t)
            row_imgs.append(np.hstack(padded))
        sheet = np.vstack(row_imgs)
        cv2.imwrite(str(out_dir / "arcface_crop_mask_sheet.jpg"), sheet)

    with open(out_dir / "arcface_crop_mask_metrics.csv", "w", encoding="utf-8") as f:
        f.write("frame,x1,y1,x2,y2,cos_ref,haar_box\n")
        for row in rows:
            idx, x1, y1, x2, y2, c, hb = row
            hb_text = "" if hb is None else ";".join(str(int(v)) for v in hb)
            f.write(f"{idx},{x1},{y1},{x2},{y2},{c:.6f},{hb_text}\n")
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
