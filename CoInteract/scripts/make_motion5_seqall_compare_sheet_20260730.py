from pathlib import Path

import cv2
import numpy as np


VIDEOS = [
    (
        "baseline",
        Path(
            "output_videos/qalign_fb01_talkinghuman_motions_skip12_baseline_20260721/"
            "qalign_fb01_motion5.mp4"
        ),
    ),
    (
        "aggregate6",
        Path(
            "output_videos/qalign_fb01_motion5_arcface_post_temporalwin6_s0005_full40_manual_20260730/"
            "qalign_fb01_motion5.mp4"
        ),
    ),
    (
        "seq_all",
        Path(
            "output_videos/qalign_fb01_motion5_arcface_post_seqall_s0005_full40_20260730/"
            "qalign_fb01_motion5.mp4"
        ),
    ),
]
FRAMES = [0, 25, 50, 75]
OUT = Path(
    "output_videos/qalign_fb01_motion5_arcface_post_seqall_s0005_full40_20260730/"
    "compare_baseline_aggregate6_seqall_frames.jpg"
)


def grab_frame(path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed reading {path} frame {frame_idx}")
    return frame


rows = []
for label, video_path in VIDEOS:
    row = []
    for frame_idx in FRAMES:
        frame = grab_frame(video_path, frame_idx)
        frame = cv2.resize(frame, (240, 416), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (239, 34), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{label} f{frame_idx}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row.append(frame)
    rows.append(np.concatenate(row, axis=1))

OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), np.concatenate(rows, axis=0))
print(OUT)
