import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/data1/workspace/linxinliang/CoInteract")
from p2v_consisid_frequency_identity_postprocess import (  # noqa: E402
    choose_face_box,
    detect_haar_faces,
    expand_box,
    load_haar_face_detector,
)


OUT_DIR = "/data1/workspace/linxinliang/CoInteract/output_videos/p2v_consisid_freq_case1_motion2_20260709"
CASES = [
    (
        "01_cfg4p5_v6",
        "/data1/workspace/linxinliang/CoInteract/output_videos/p2v_othermethods_case1_motion2_20260707/01_single_moe_cfg4p5/th_fullbody_001_motion2_v01_single.mp4",
        f"{OUT_DIR}/01_cfg4p5_consisid_freq_v6_soft_haar.mp4",
    ),
    (
        "02_mixed_lora_v6",
        "/data1/workspace/linxinliang/CoInteract/output_videos/p2v_mixeddata_lora_case1_motion2_20260707/02_mixed_step5000_480x832_nomoe/th_fullbody_001_motion2_v01_single.mp4",
        f"{OUT_DIR}/02_mixed_lora_consisid_freq_v6_soft_haar.mp4",
    ),
    (
        "01_cfg4p5_v7",
        "/data1/workspace/linxinliang/CoInteract/output_videos/p2v_othermethods_case1_motion2_20260707/01_single_moe_cfg4p5/th_fullbody_001_motion2_v01_single.mp4",
        f"{OUT_DIR}/01_cfg4p5_consisid_freq_v7_lighthf_haar.mp4",
    ),
    (
        "02_mixed_lora_v7",
        "/data1/workspace/linxinliang/CoInteract/output_videos/p2v_mixeddata_lora_case1_motion2_20260707/02_mixed_step5000_480x832_nomoe/th_fullbody_001_motion2_v01_single.mp4",
        f"{OUT_DIR}/02_mixed_lora_consisid_freq_v7_lighthf_haar.mp4",
    ),
]


def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def save_sheets(tag, before_path, after_path, detector):
    cap_b = cv2.VideoCapture(before_path)
    cap_a = cv2.VideoCapture(after_path)
    total = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        raise RuntimeError(f"No frames found in {before_path}")

    frame_ids = sorted(set([0, max(0, total // 3), max(0, 2 * total // 3), max(0, total - 1)]))
    full_rows = []
    crop_rows = []
    last_box = None

    for frame_idx in frame_ids:
        before = read_frame(cap_b, frame_idx)
        after = read_frame(cap_a, frame_idx)
        if before is None or after is None:
            continue

        h, w = before.shape[:2]
        full = np.concatenate([before, after], axis=1)
        full_w = 1200
        full_h = max(1, int(full.shape[0] * full_w / full.shape[1]))
        full = cv2.resize(full, (full_w, full_h), interpolation=cv2.INTER_AREA)
        cv2.putText(
            full,
            f"{tag} frame {frame_idx}: before | after",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        full_rows.append(full)

        box = choose_face_box(detect_haar_faces(before, detector), w, h, last_box)
        if box is None:
            box = last_box
        if box is None:
            continue

        last_box = box
        crop_box = expand_box(box, w, h, 1.55, 1.55)
        before_crop = before[crop_box.y1 : crop_box.y2, crop_box.x1 : crop_box.x2]
        after_crop = after[crop_box.y1 : crop_box.y2, crop_box.x1 : crop_box.x2]
        if before_crop.size == 0 or after_crop.size == 0:
            continue

        before_crop = cv2.resize(before_crop, (420, 420), interpolation=cv2.INTER_CUBIC)
        after_crop = cv2.resize(after_crop, (420, 420), interpolation=cv2.INTER_CUBIC)
        crop = np.concatenate([before_crop, after_crop], axis=1)
        cv2.putText(
            crop,
            f"{tag} frame {frame_idx}: before | after",
            (14, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        crop_rows.append(crop)

    cap_b.release()
    cap_a.release()

    if full_rows:
        full_sheet = np.vstack(full_rows)
        full_path = os.path.join(OUT_DIR, f"{tag}_full_sheet.jpg")
        cv2.imwrite(full_path, full_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(full_path)

    if crop_rows:
        crop_sheet = np.vstack(crop_rows)
        crop_path = os.path.join(OUT_DIR, f"{tag}_facecrop_sheet.jpg")
        cv2.imwrite(crop_path, crop_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(crop_path)


def main():
    detector = load_haar_face_detector()
    for tag, before_path, after_path in CASES:
        save_sheets(tag, before_path, after_path, detector)


if __name__ == "__main__":
    main()
