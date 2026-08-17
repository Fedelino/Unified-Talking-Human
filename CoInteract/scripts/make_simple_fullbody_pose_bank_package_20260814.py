from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
OUT = ROOT / "output_videos" / "simple_fullbody_pose_bank_same_image_20260814_package"
CSV_PATH = ROOT / "examples" / "simple_fullbody_pose_bank_same_image_20260814.csv"
REF_IMAGE = Path(
    "/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/"
    "test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
)
PROMPT = (
    "A full-body person follows the provided motion with stable facial identity, stable eyes, "
    "stable nose, stable lips, stable jawline, and stable whole-body proportions."
)

POSES = [
    (
        "01_extracted_motion1_handwave",
        "new-qilin-aligned; simple hand wave; source motion1",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/simple_motion_pose_bank_20260814/dwpose_qilin_aligned/th_motion1_handwave_qilin_aligned_pose.mp4"),
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/simple_motion_pose_bank_20260814/raw_25fps_max4s/th_motion1_handwave.mp4"),
    ),
    (
        "02_extracted_motion2_gentle_shift",
        "new-qilin-aligned; gentle full-body shift/turn; source motion2",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/simple_motion_pose_bank_20260814/dwpose_qilin_aligned/th_motion2_gentle_shift_qilin_aligned_pose.mp4"),
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/simple_motion_pose_bank_20260814/raw_25fps_max4s/th_motion2_gentle_shift.mp4"),
    ),
    (
        "03_extracted_motion5_first4s",
        "new-qilin-aligned; first 4 seconds of motion5; simple full-body control",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/simple_motion_pose_bank_20260814/dwpose_qilin_aligned/th_motion5_first4s_qilin_aligned_pose.mp4"),
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/simple_motion_pose_bank_20260814/raw_25fps_max4s/th_motion5_first4s.mp4"),
    ),
    (
        "04_existing_001_motion1_pose",
        "existing TalkingHuman same-reference pose; manifest case 001 motion1",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion_pose/001_motion1_pose.mp4"),
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/motion1.mp4"),
    ),
    (
        "05_existing_001_motion2_pose",
        "existing TalkingHuman same-reference pose; manifest case 001 motion2",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion_pose/001_motion2_pose.mp4"),
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/motion2.mp4"),
    ),
    (
        "06_existing_001_motion3_pose",
        "existing TalkingHuman same-reference pose; manifest case 001 motion3",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion_pose/001_motion3_pose.mp4"),
        None,
    ),
    (
        "07_existing_001_motion_pose",
        "existing TalkingHuman same-reference pose; manifest case 001 alternate motion",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion_pose/001_motion.mp4"),
        None,
    ),
]


def video_meta(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, frames, width, height, frames / fps


def read_frame(path: Path, idx: int) -> Image.Image:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx} from {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def label_image(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    image.thumbnail((240, 360))
    canvas = Image.new("RGB", (240, 390), (16, 16, 16))
    x = (240 - image.width) // 2
    y = 30 + (340 - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)
    return canvas


def copy_assets_and_manifest() -> list[dict[str, str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id, note, pose_path, source_path in POSES:
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)
        copied_pose = OUT / f"{sample_id}.mp4"
        shutil.copy2(pose_path, copied_pose)
        copied_source = ""
        if source_path is not None and source_path.exists():
            copied_source_path = OUT / f"{sample_id}_source.mp4"
            shutil.copy2(source_path, copied_source_path)
            copied_source = str(copied_source_path)
        fps, frames, width, height, duration = video_meta(pose_path)
        rows.append(
            {
                "sample_id": sample_id,
                "prompt": PROMPT,
                "audio": "",
                "person_image": str(REF_IMAGE),
                "product_image": "",
                "pose_video": str(pose_path),
                "packaged_pose_video": str(copied_pose),
                "source_clip": str(source_path) if source_path else "",
                "motion_note": note,
                "pose_width": str(width),
                "pose_height": str(height),
                "pose_fps": f"{fps:.3f}",
                "pose_frames": str(frames),
                "pose_duration": f"{duration:.3f}",
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sample_id", "prompt", "audio", "person_image", "product_image", "pose_video"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "prompt": row["prompt"],
                    "audio": row["audio"],
                    "person_image": row["person_image"],
                    "product_image": row["product_image"],
                    "pose_video": row["pose_video"],
                }
            )

    with (OUT / "manifest_details.tsv").open("w", encoding="utf-8") as fh:
        keys = list(rows[0].keys())
        fh.write("\t".join(keys) + "\n")
        for row in rows:
            fh.write("\t".join(row[k] for k in keys) + "\n")
    shutil.copy2(CSV_PATH, OUT / CSV_PATH.name)
    shutil.copy2(REF_IMAGE, OUT / "same_reference_image.jpg")
    return rows


def make_sheet(rows: list[dict[str, str]]) -> None:
    frame_fracs = [0.05, 0.50, 0.90]
    sheet_rows = []
    for row in rows:
        pose_path = Path(row["pose_video"])
        fps, frames, *_ = video_meta(pose_path)
        images = []
        for frac in frame_fracs:
            idx = min(frames - 1, max(0, int(round(frac * (frames - 1)))))
            images.append(label_image(read_frame(pose_path, idx), f"{row['sample_id']} pose f{idx}"))
        sheet_rows.append(images)

    cell_w, cell_h = 240, 390
    sheet = Image.new("RGB", (cell_w * len(frame_fracs), cell_h * len(sheet_rows)), (8, 8, 8))
    for y, images in enumerate(sheet_rows):
        for x, image in enumerate(images):
            sheet.paste(image, (x * cell_w, y * cell_h))
    sheet.save(OUT / "simple_fullbody_pose_bank_sheet.jpg", quality=95)


def write_readme(rows: list[dict[str, str]]) -> None:
    lines = [
        "Simple full-body pose bank for CoInteract identity-preservation tests",
        "",
        f"Reference image: {REF_IMAGE}",
        f"CoInteract CSV: {CSV_PATH}",
        "",
        "Notes:",
        "- 01-03 were newly extracted with Qilin-style DWPose alignment to the same reference image.",
        "- 04-07 are existing TalkingHuman case-001 pose videos from the same reference-image manifest entry.",
        "- The failed/cropped zsy clips were intentionally excluded from this stricter full-body bank.",
        "",
        "Rows:",
    ]
    for row in rows:
        lines.append(
            f"- {row['sample_id']}: {row['motion_note']} "
            f"({row['pose_width']}x{row['pose_height']}, {row['pose_frames']} frames, {row['pose_duration']}s)"
        )
    (OUT / "README.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = copy_assets_and_manifest()
    make_sheet(rows)
    write_readme(rows)
    print(OUT)
    print(CSV_PATH)
    print((OUT / "manifest_details.tsv").read_text())


if __name__ == "__main__":
    main()
