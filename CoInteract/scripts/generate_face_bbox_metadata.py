#!/usr/bin/env python3
"""Generate CoInteract face bbox metadata for face-weighted fine-tuning.

The output CSV keeps all original columns/paths unchanged and only replaces the
`face` column with frame-indexed bbox JSON compatible with bbox_utils.py.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2


def resolve_path(dataset_base_path: Path, value: str) -> Path:
    return (dataset_base_path / value).resolve()


def expanded_bbox(x, y, w, h, image_w, image_h, scale):
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * scale
    nh = h * scale
    x1 = max(0, int(round(cx - nw / 2.0)))
    y1 = max(0, int(round(cy - nh / 2.0)))
    x2 = min(image_w - 1, int(round(cx + nw / 2.0)))
    y2 = min(image_h - 1, int(round(cy + nh / 2.0)))
    return [x1, y1, x2, y2]


def detect_largest_face(frame, detector, expand_scale, min_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(min_size, min_size),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    image_h, image_w = frame.shape[:2]
    return expanded_bbox(x, y, w, h, image_w, image_h, expand_scale)


def build_face_metadata(video_path: Path, detector, frame_indices, expand_scale, min_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}, 0, 0

    face_meta = {}
    last_bbox = None
    detected = 0
    reused = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = detect_largest_face(frame, detector, expand_scale, min_size)
        if bbox is not None:
            detected += 1
            last_bbox = bbox
        elif last_bbox is not None:
            bbox = last_bbox
            reused += 1
        if bbox is not None:
            face_meta[f"frame_{frame_idx}"] = bbox
    cap.release()
    return face_meta, detected, reused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--latent_frame_start", type=int, default=2)
    parser.add_argument("--latent_frame_stride", type=int, default=4)
    parser.add_argument("--expand_scale", type=float, default=1.6)
    parser.add_argument("--min_face_size", type=int, default=12)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    dataset_base_path = Path(args.dataset_base_path)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load OpenCV face detector: {cascade_path}")

    frame_indices = list(range(args.latent_frame_start, args.num_frames - 1, args.latent_frame_stride))

    total = 0
    rows_with_faces = 0
    total_detected = 0
    total_reused = 0
    with input_csv.open(newline="", encoding="utf-8") as f_in, output_csv.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if "face" not in fieldnames:
            fieldnames.append("face")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1
            video_value = row.get("input_video", "")
            video_path = resolve_path(dataset_base_path, video_value)
            face_meta, detected, reused = build_face_metadata(
                video_path,
                detector,
                frame_indices,
                args.expand_scale,
                args.min_face_size,
            )
            if face_meta:
                rows_with_faces += 1
            total_detected += detected
            total_reused += reused
            row["face"] = json.dumps(face_meta, separators=(",", ":"))
            writer.writerow(row)

            if total % 50 == 0:
                print(
                    f"processed={total} rows_with_faces={rows_with_faces} "
                    f"detected_boxes={total_detected} reused_boxes={total_reused}",
                    flush=True,
                )

    print(
        f"done rows={total} rows_with_faces={rows_with_faces} "
        f"detected_boxes={total_detected} reused_boxes={total_reused} output={output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
