from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
OUT = ROOT / "output_videos" / "p2v_refkv_strong_prevconfig_motion1_eval_20260815"
OUT.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    ("00_baseline", ROOT / "output_videos/p2v_altmethods_prevconfig_motion1_baseline_noguidance_20260814/th_fullbody_001_custom_motion1_altmethods.mp4"),
    ("01_s005", ROOT / "output_videos/p2v_altmethods_prevconfig_motion1_refkv_headattn_s005_20260814/th_fullbody_001_custom_motion1_altmethods.mp4"),
    ("02_s010", ROOT / "output_videos/p2v_altmethods_prevconfig_motion1_refkv_headattn_s010_20260814/th_fullbody_001_custom_motion1_altmethods.mp4"),
    ("03_s025", ROOT / "output_videos/p2v_refkv_strong_prevconfig_motion1_refkv_headattn_s025_20260815/th_fullbody_001_custom_motion1_refkv_strong.mp4"),
    ("04_s050", ROOT / "output_videos/p2v_refkv_strong_prevconfig_motion1_refkv_headattn_s050_20260815/th_fullbody_001_custom_motion1_refkv_strong.mp4"),
    ("05_s100", ROOT / "output_videos/p2v_refkv_strong_prevconfig_motion1_refkv_headattn_s100_20260815/th_fullbody_001_custom_motion1_refkv_strong.mp4"),
]
FRAMES = [0, 25, 50, 75]


def read_frame(path: Path, idx: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx} from {path}")
    return frame


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def crop_head(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(0.25 * w), int(0.03 * h), int(0.75 * w), int(0.43 * h)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (300, 390), interpolation=cv2.INTER_CUBIC)


rows = []
for name, path in VIDEOS:
    cells = []
    for idx in FRAMES:
        cells.append(label(crop_head(read_frame(path, idx)), f"{name} f{idx}"))
    rows.append(np.concatenate(cells, axis=1))

sheet = np.concatenate(rows, axis=0)
cv2.imwrite(str(OUT / "headcrop_strong_sheet_frames_0_25_50_75.jpg"), sheet)
print(OUT / "headcrop_strong_sheet_frames_0_25_50_75.jpg")
