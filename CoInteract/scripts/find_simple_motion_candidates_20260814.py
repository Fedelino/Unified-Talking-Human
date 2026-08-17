from __future__ import annotations

import os
from pathlib import Path

import cv2


ROOTS = [
    Path("/data1/workspace/linxinliang"),
    Path("/data1/workspace/wangqilin"),
    Path("/data1/workspace/leijunwei"),
]

KEYWORDS = (
    "motion",
    "zsy",
    "dianzan",
    "say",
    "hi",
    "wave",
    "walk",
    "turn",
    "pose",
    "full",
    "whole",
    "tiktok",
    "ubcfashion",
    "talkinghuman",
)

SKIP_PARTS = (
    "/output_videos/",
    "/output/",
    "/logs/",
    "/.git/",
    "/__pycache__/",
)


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0:
        return None
    return frames / fps, width, height, fps, int(frames)


def score_candidate(path: Path, duration: float, width: int, height: int) -> tuple[int, str]:
    low = str(path).lower()
    score = 0
    reason = []
    if "wholebody" in low or "fullbody" in low or "full" in low:
        score += 5
        reason.append("fullbody-name")
    if "zsy" in low or "dianzan" in low or "say_hi" in low or "say-hi" in low:
        score += 4
        reason.append("simple-name")
    if "motion" in low:
        score += 2
        reason.append("motion-name")
    if 2.0 <= duration <= 6.0:
        score += 3
        reason.append("short")
    elif duration <= 10.0:
        score += 1
        reason.append("medium")
    if height >= width:
        score += 2
        reason.append("portrait")
    if min(width, height) >= 360:
        score += 1
        reason.append("usable-res")
    return score, ",".join(reason)


def main() -> None:
    rows = []
    for root in ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            normalized = dirpath.replace("\\", "/")
            if any(part in normalized for part in SKIP_PARTS):
                continue
            for filename in filenames:
                if not filename.lower().endswith(".mp4"):
                    continue
                path = Path(dirpath) / filename
                low = str(path).lower()
                if not any(keyword in low for keyword in KEYWORDS):
                    continue
                info = video_info(path)
                if info is None:
                    continue
                duration, width, height, fps, frames = info
                if not (1.0 <= duration <= 30.0):
                    continue
                score, reason = score_candidate(path, duration, width, height)
                rows.append((score, duration, width, height, fps, frames, reason, path))

    rows.sort(key=lambda row: (-row[0], row[1], str(row[-1]).lower()))
    for score, duration, width, height, fps, frames, reason, path in rows[:600]:
        print(
            f"score={score:02d} dur={duration:6.2f}s size={width}x{height} "
            f"fps={fps:6.2f} frames={frames:5d} reason={reason:35s} {path}"
        )


if __name__ == "__main__":
    main()
