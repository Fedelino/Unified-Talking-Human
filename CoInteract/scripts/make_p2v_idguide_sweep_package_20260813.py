from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageDraw


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
DATE = "20260813"
SAMPLE = "th_fullbody_001_custom_motion2_noaudio_baseline.mp4"

VARIANTS = [
    (
        "00_baseline_noguidance",
        "p2v_idguide_baseline_noguidance_motion2_20260813",
        "No identity guidance: normal CoInteract P2V baseline.",
    ),
    (
        "01_zero_scale1",
        "p2v_idguide_zero_scale1_motion2_20260813",
        "Old Stage-A guidance: v_full - v_zero, masked by CoInteract head router.",
    ),
    (
        "02_latblur_k5_scale1",
        "p2v_idguide_latblur_k5_scale1_motion2_20260813",
        "Latent-blur weak branch: v_full - v_blurred_reference, head-mask guided.",
    ),
    (
        "03_latblur_k5_samg_scale1",
        "p2v_idguide_latblur_k5_samg_scale1_motion2_20260813",
        "Latent-blur plus Head-SAMG adaptive local guidance, multiplier 0.5-1.5.",
    ),
    (
        "04_latblur_k5_samg_apg025_scale1",
        "p2v_idguide_latblur_k5_samg_apg025_scale1_motion2_20260813",
        "Latent-blur plus Head-SAMG plus APG eta=0.25.",
    ),
]


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_thumb(frame, title: str, size=(240, 416)) -> Image.Image:
    im = Image.fromarray(frame)
    im.thumbnail((size[0], size[1] - 28), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(im, ((size[0] - im.width) // 2, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill=(0, 0, 0))
    return canvas


def main() -> None:
    package = ROOT / "output_videos" / f"p2v_idguide_sweep_motion2_{DATE}_package"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)

    frames = [0, 25, 50, 75]
    rows = []
    readme = [
        f"CoInteract P2V identity-guidance sweep, {DATE}",
        "",
        "Common settings:",
        "- reference: fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg",
        "- pose: InteractAvatar/InterDemo/custom_motion/dwpose/motion2_pose.mp4",
        "- checkpoint: models/CoInteract/checkpoint_pose.safetensors",
        "- canvas: 480x832, 80 frames, 40 denoising steps",
        "- reference compose mode: stretch",
        "- audio/product: empty, silent fallback",
        "",
        "Variants:",
    ]

    for label, folder, description in VARIANTS:
        src = ROOT / "output_videos" / folder / SAMPLE
        dst_video = package / f"{label}.mp4"
        if not src.exists():
            readme.append(f"- {label}: MISSING {src}")
            continue
        shutil.copy2(src, dst_video)
        readme.append(f"- {label}: {description}")

        thumbs = []
        for frame_idx in frames:
            frame = read_frame(src, frame_idx)
            if frame is None:
                continue
            out_frame = package / f"{label}_frame{frame_idx:03d}.jpg"
            Image.fromarray(frame).save(out_frame, quality=95)
            thumbs.append(make_thumb(frame, f"{label} f{frame_idx}"))
        if thumbs:
            row = Image.new("RGB", (sum(t.width for t in thumbs), max(t.height for t in thumbs)), "white")
            x = 0
            for thumb in thumbs:
                row.paste(thumb, (x, 0))
                x += thumb.width
            rows.append(row)

    if rows:
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(package / "p2v_idguide_sweep_frames_0_25_50_75.jpg", quality=95)

    (package / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(package)


if __name__ == "__main__":
    main()
