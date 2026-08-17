from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


CANDIDATES = [
    ("motion1", Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/motion1.mp4")),
    ("motion2", Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/motion2.mp4")),
    ("motion5", Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/motion5.mp4")),
    ("zsy_dianzan", Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/zsy_dianzan_0306.mp4")),
    ("zsy_mix", Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/zsy_mix_0306.mp4")),
    ("zsy_say_hi", Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion/zsy_say_hi_1_0306.mp4")),
    ("custom_motion1", Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/custom_motion/raw/motion1.mp4")),
    ("custom_motion2", Path("/data1/workspace/linxinliang/InteractAvatar/InterDemo/custom_motion/raw/motion2.mp4")),
]

OUT_DIR = Path("/data1/workspace/linxinliang/CoInteract/output_videos/simple_motion_pose_bank_20260814")
FRAC_POSITIONS = [0.0, 0.33, 0.66, 0.95]


def read_frame(path: Path, idx: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx} from {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


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


def label_image(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    image.thumbnail((240, 360))
    canvas = Image.new("RGB", (240, 390), (18, 18, 18))
    x = (240 - image.width) // 2
    y = 30 + (340 - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 6), text, fill=(255, 255, 255), font=font)
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    meta_lines = []
    for name, path in CANDIDATES:
        fps, frames, width, height, duration = video_meta(path)
        meta_lines.append(f"{name}\t{duration:.3f}s\t{width}x{height}\tfps={fps:.3f}\tframes={frames}\t{path}")
        row = []
        for frac in FRAC_POSITIONS:
            idx = min(frames - 1, max(0, int(round(frac * (frames - 1)))))
            row.append(label_image(read_frame(path, idx), f"{name} f{idx}"))
        rows.append(row)

    sheet = Image.new("RGB", (240 * len(FRAC_POSITIONS), 390 * len(rows)), (10, 10, 10))
    for y, row in enumerate(rows):
        for x, image in enumerate(row):
            sheet.paste(image, (x * 240, y * 390))
    sheet.save(OUT_DIR / "source_motion_preview_sheet.jpg", quality=95)
    (OUT_DIR / "source_motion_metadata.tsv").write_text("\n".join(meta_lines) + "\n")
    print(OUT_DIR)
    print((OUT_DIR / "source_motion_metadata.tsv").read_text())


if __name__ == "__main__":
    main()
