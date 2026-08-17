import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
OUT = ROOT / "output_videos" / "p2v_idguide_isolated_ablation_motion2_20260814_package"
VIDEO_NAME = "th_fullbody_001_custom_motion2_noaudio_baseline.mp4"
FRAMES = [0, 25, 50, 75]

VARIANTS = [
    (
        "00_baseline_noguidance",
        ROOT / "output_videos" / "p2v_idguide_baseline_noguidance_motion2_20260813" / VIDEO_NAME,
        "No face-reference guidance.",
    ),
    (
        "01_vv_zero_scale1",
        ROOT / "output_videos" / "p2v_idguide_zero_scale1_motion2_20260813" / VIDEO_NAME,
        "Plain identity push: v_full - v_zero, scale=1.",
    ),
    (
        "02_vv_latblur_k5_scale1",
        ROOT / "output_videos" / "p2v_idguide_latblur_k5_scale1_motion2_20260813" / VIDEO_NAME,
        "Latent-blur weak branch: v_full - v_blur, blur kernel=5, scale=1.",
    ),
    (
        "03_vv_latblur_k5_samg_default_scale1",
        ROOT / "output_videos" / "p2v_idguide_latblur_k5_samg_scale1_motion2_20260813" / VIDEO_NAME,
        "Latent blur + SAMG default multipliers 0.5..1.5, scale=1.",
    ),
    (
        "04_vv_latblur_k5_samg_strong_scale1",
        ROOT / "output_videos" / "p2v_idguide_ablate_latblur_k5_samgstrong_scale1_motion2_20260814" / VIDEO_NAME,
        "Latent blur + stronger SAMG multipliers 0.25..2.0, scale=1.",
    ),
    (
        "05_vv_latblur_k5_apg025_scale1",
        ROOT / "output_videos" / "p2v_idguide_ablate_latblur_k5_apg025_scale1_motion2_20260814" / VIDEO_NAME,
        "Latent blur + APG eta=0.25, scale=1.",
    ),
    (
        "06_vv_latblur_k5_apg0_scale1",
        ROOT / "output_videos" / "p2v_idguide_ablate_latblur_k5_apg0_scale1_motion2_20260814" / VIDEO_NAME,
        "Latent blur + APG eta=0.0, scale=1.",
    ),
    (
        "07_vv_latblur_k5_samg_default_apg025_scale1",
        ROOT / "output_videos" / "p2v_idguide_latblur_k5_samg_apg025_scale1_motion2_20260813" / VIDEO_NAME,
        "Latent blur + SAMG default + APG eta=0.25, scale=1.",
    ),
]


def copy_videos() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src, _ in VARIANTS:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, OUT / f"{name}.mp4")


def extract_frame(video_path: Path, frame_idx: int, out_path: Path) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image.save(out_path, quality=95)
    return image


def draw_label(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, bbox[2] + pad * 2, bbox[3] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return image


def make_frame_sheet() -> None:
    thumbs = []
    for name, _, _ in VARIANTS:
        row = []
        video_path = OUT / f"{name}.mp4"
        for frame_idx in FRAMES:
            frame_path = OUT / f"{name}_frame{frame_idx:03d}.jpg"
            image = extract_frame(video_path, frame_idx, frame_path)
            row.append(draw_label(image, f"{name} | frame {frame_idx}"))
        thumbs.append(row)

    cell_w, cell_h = thumbs[0][0].size
    sheet = Image.new("RGB", (cell_w * len(FRAMES), cell_h * len(thumbs)), (20, 20, 20))
    for y, row in enumerate(thumbs):
        for x, image in enumerate(row):
            sheet.paste(image, (x * cell_w, y * cell_h))
    sheet.save(OUT / "isolated_ablation_frames_0_25_50_75.jpg", quality=95)


def write_readme() -> None:
    lines = [
        "CoInteract P2V isolated identity-guidance ablation",
        "",
        "Case: th_fullbody_001_custom_motion2_noaudio_baseline",
        "Reference: TalkingHuman full-body image fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg",
        "Pose: custom motion2_pose.mp4",
        "Canvas/settings: 480x832, 80 frames, 40 steps, cfg=7.0, sigma_shift=7.0, stretch reference compose, no output resize-to-reference.",
        "",
        "Variants:",
    ]
    for name, _, desc in VARIANTS:
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append("Comparison sheet: isolated_ablation_frames_0_25_50_75.jpg")
    (OUT / "README.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    copy_videos()
    make_frame_sheet()
    write_readme()
    print(OUT)


if __name__ == "__main__":
    main()
