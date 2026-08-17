from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
OUT_DIR = ROOT / "output_videos" / "simple_motion_pose_bank_20260814"
DATASETS = [
    ("tiktok", ROOT / "data" / "tiktok_pose_full" / "input_video", 36),
    ("ubcfashion", ROOT / "data" / "ubcfashion_pose_full" / "input_video", 36),
]
FRAC_POSITIONS = [0.05, 0.50, 0.90]


def list_videos(folder: Path, limit: int) -> list[Path]:
    videos = sorted(folder.glob("*.mp4"))
    if len(videos) <= limit:
        return videos
    # Use a deterministic spread instead of only the first names.
    step = (len(videos) - 1) / (limit - 1)
    return [videos[round(i * step)] for i in range(limit)]


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
    image.thumbnail((220, 330))
    canvas = Image.new("RGB", (220, 365), (18, 18, 18))
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


def build_sheet(dataset_name: str, folder: Path, limit: int) -> None:
    videos = list_videos(folder, limit)
    rows = []
    meta_lines = []
    for path in videos:
        fps, frames, width, height, duration = video_meta(path)
        name = path.stem
        meta_lines.append(f"{dataset_name}\t{name}\t{duration:.3f}s\t{width}x{height}\tfps={fps:.3f}\tframes={frames}\t{path}")
        row = []
        for frac in FRAC_POSITIONS:
            idx = min(frames - 1, max(0, int(round(frac * (frames - 1)))))
            row.append(label_image(read_frame(path, idx), f"{dataset_name}:{name} f{idx}"))
        rows.append(row)

    sheet = Image.new("RGB", (220 * len(FRAC_POSITIONS), 365 * len(rows)), (10, 10, 10))
    for y, row in enumerate(rows):
        for x, image in enumerate(row):
            sheet.paste(image, (x * 220, y * 365))
    sheet.save(OUT_DIR / f"{dataset_name}_spread_preview_sheet.jpg", quality=95)
    (OUT_DIR / f"{dataset_name}_spread_metadata.tsv").write_text("\n".join(meta_lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name, folder, limit in DATASETS:
        build_sheet(dataset_name, folder, limit)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
