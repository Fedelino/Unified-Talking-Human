from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


COINTERACT_ROOT = Path("/data1/workspace/linxinliang/CoInteract")
BANK_ROOT = Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/simple_motion_pose_bank_20260814")
RAW_DIR = BANK_ROOT / "raw_25fps_max4s"
POSE_DIR = BANK_ROOT / "dwpose_qilin_aligned"
PREVIEW_DIR = BANK_ROOT / "preview"
LOG_DIR = COINTERACT_ROOT / "logs" / "simple_motion_pose_bank_20260814"
CSV_PATH = COINTERACT_ROOT / "examples" / "simple_motion_pose_bank_same_image_20260814.csv"

REF_IMAGE = Path(
    "/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/"
    "test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
)
PROMPT = (
    "A full-body person follows the provided motion with stable facial identity, "
    "stable eyes, stable nose, stable lips, stable jawline, and stable whole-body proportions."
)

MOTIONS = [
    (
        "th_motion1_handwave",
        "fullbody-simple",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/motion1.mp4"),
    ),
    (
        "th_motion2_gentle_shift",
        "fullbody-simple",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/motion2.mp4"),
    ),
    (
        "th_motion5_first4s",
        "fullbody-long-trimmed",
        Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/motion5.mp4"),
    ),
    (
        "tiktok_00033_walk",
        "fullbody-natural",
        Path("/data1/workspace/linxinliang/CoInteract/data/tiktok_pose_full/input_video/00033_000.mp4"),
    ),
    (
        "tiktok_00131_gentle_turn",
        "fullbody-natural",
        Path("/data1/workspace/linxinliang/CoInteract/data/tiktok_pose_full/input_video/00131_000.mp4"),
    ),
    (
        "tiktok_00235_outdoor_stand",
        "fullbody-natural",
        Path("/data1/workspace/linxinliang/CoInteract/data/tiktok_pose_full/input_video/00235_000.mp4"),
    ),
    (
        "tiktok_00270_stand_arm",
        "fullbody-natural",
        Path("/data1/workspace/linxinliang/CoInteract/data/tiktok_pose_full/input_video/00270_000.mp4"),
    ),
    (
        "ubc_916iz4hbIJS_turn",
        "fullbody-runway-control",
        Path("/data1/workspace/linxinliang/CoInteract/data/ubcfashion_pose_full/input_video/916iz4hbIJS_000.mp4"),
    ),
    (
        "ubc_91ierZSo5hS_turn",
        "fullbody-runway-control",
        Path("/data1/workspace/linxinliang/CoInteract/data/ubcfashion_pose_full/input_video/91ierZSo5hS_000.mp4"),
    ),
    (
        "ubc_A15g0ekJ1US_turn",
        "fullbody-runway-control",
        Path("/data1/workspace/linxinliang/CoInteract/data/ubcfashion_pose_full/input_video/A15g0ekJ1US_000.mp4"),
    ),
]


def run(cmd: list[str], log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def normalize_raw_clip(name: str, src: Path) -> Path:
    out = RAW_DIR / f"{name}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-t",
        "4",
        "-an",
        "-vf",
        "fps=25",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(out),
    ]
    run(cmd)
    return out


def extract_pose(name: str, raw_clip: Path) -> Path:
    out = POSE_DIR / f"{name}_qilin_aligned_pose.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    log_path = LOG_DIR / f"{name}.log"
    cmd = [
        "python",
        "scripts/extract_qilin_aligned_dwpose_video_cpu_20260814.py",
        "--input-video",
        str(raw_clip),
        "--ref-image",
        str(REF_IMAGE),
        "--output-video",
        str(out),
        "--target-fps",
        "25.0",
    ]
    run(cmd, log_path=log_path)
    return out


def video_meta(path: Path):
    cap = cv2.VideoCapture(str(path))
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
    image.thumbnail((220, 330))
    canvas = Image.new("RGB", (220, 365), (16, 16, 16))
    x = (220 - image.width) // 2
    y = 30 + (315 - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)
    return canvas


def make_preview(rows: list[dict[str, str]]) -> None:
    frame_fracs = [0.05, 0.50, 0.90]
    sheet_rows = []
    for row in rows:
        name = row["sample_id"]
        raw = Path(row["source_clip"])
        pose = Path(row["pose_video"])
        fps, frames, *_ = video_meta(raw)
        pose_fps, pose_frames, *_ = video_meta(pose)
        images = []
        for path, total_frames, tag in [(raw, frames, "raw"), (pose, pose_frames, "pose")]:
            for frac in frame_fracs:
                idx = min(total_frames - 1, max(0, int(round(frac * (total_frames - 1)))))
                images.append(label_image(read_frame(path, idx), f"{name} {tag} f{idx}"))
        sheet_rows.append(images)

    cell_w, cell_h = 220, 365
    sheet = Image.new("RGB", (cell_w * 6, cell_h * len(sheet_rows)), (8, 8, 8))
    for y, images in enumerate(sheet_rows):
        for x, image in enumerate(images):
            sheet.paste(image, (x * cell_w, y * cell_h))
    sheet.save(PREVIEW_DIR / "simple_motion_pose_bank_raw_pose_sheet.jpg", quality=95)


def write_manifest(rows: list[dict[str, str]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "prompt",
                "audio",
                "person_image",
                "product_image",
                "pose_video",
                "source_clip",
                "motion_tag",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, tag, src in MOTIONS:
        print(f"[motion] {name} <- {src}", flush=True)
        if not src.exists():
            raise FileNotFoundError(src)
        raw_clip = normalize_raw_clip(name, src)
        pose_video = extract_pose(name, raw_clip)
        fps, frames, width, height, duration = video_meta(pose_video)
        print(f"[pose] {pose_video} {width}x{height} fps={fps:.3f} frames={frames} dur={duration:.3f}s", flush=True)
        rows.append(
            {
                "sample_id": name,
                "prompt": PROMPT,
                "audio": "",
                "person_image": str(REF_IMAGE),
                "product_image": "",
                "pose_video": str(pose_video),
                "source_clip": str(raw_clip),
                "motion_tag": tag,
            }
        )

    write_manifest(rows)
    make_preview(rows)
    print(f"[done] bank={BANK_ROOT}")
    print(f"[done] csv={CSV_PATH}")
    print(f"[done] preview={PREVIEW_DIR / 'simple_motion_pose_bank_raw_pose_sheet.jpg'}")


if __name__ == "__main__":
    main()
