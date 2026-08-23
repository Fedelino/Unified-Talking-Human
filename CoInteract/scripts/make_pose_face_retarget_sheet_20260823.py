#!/usr/bin/env python3
"""Create a compact visual sheet comparing original and identity-retargeted pose videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, min(frame_index, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {idx} from {path}")
    return frame


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--retargeted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", default="0,25,50,75")
    parser.add_argument("--cell_width", type=int, default=360)
    args = parser.parse_args()

    frame_ids = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    rows = []
    for idx in frame_ids:
        orig = label(resize_to_width(read_frame(args.original, idx), args.cell_width), f"original pose | frame {idx}")
        ret = label(resize_to_width(read_frame(args.retargeted, idx), args.cell_width), f"id-face pose | frame {idx}")
        h = min(orig.shape[0], ret.shape[0])
        rows.append(np.hstack([orig[:h], ret[:h]]))

    sheet = np.vstack(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
