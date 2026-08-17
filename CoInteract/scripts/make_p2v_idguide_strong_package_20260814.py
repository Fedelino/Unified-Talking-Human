from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageDraw


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
SAMPLE = "th_fullbody_001_custom_motion2_noaudio_baseline.mp4"
PACKAGE = ROOT / "output_videos" / "p2v_idguide_strong_sweep_motion2_20260814_package"

VARIANTS = [
    ("00_baseline_noguidance", "p2v_idguide_baseline_noguidance_motion2_20260813"),
    ("01_samg_scale1p5", "p2v_idguide_strong_latblur_k5_samg_scale1p5_motion2_20260813"),
    ("02_samg_scale2p0", "p2v_idguide_strong_latblur_k5_samg_scale2p0_motion2_20260813"),
    ("03_samg_scale3p0", "p2v_idguide_strong_latblur_k5_samg_scale3p0_motion2_20260813"),
    ("04_samg_scale5p0", "p2v_idguide_strong_latblur_k5_samg_scale5p0_motion2_20260813"),
    ("05_samg_apg025_scale1p5", "p2v_idguide_strong_latblur_k5_samg_apg025_scale1p5_motion2_20260813"),
    ("06_samg_apg025_scale2p0", "p2v_idguide_strong_latblur_k5_samg_apg025_scale2p0_motion2_20260813"),
    ("07_samg_apg025_scale3p0", "p2v_idguide_strong_latblur_k5_samg_apg025_scale3p0_motion2_20260813"),
    ("08_samg_apg025_scale5p0", "p2v_idguide_strong_latblur_k5_samg_apg025_scale5p0_motion2_20260813"),
]


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_thumb(frame, title: str, size=(220, 384)) -> Image.Image:
    image = Image.fromarray(frame)
    image.thumbnail((size[0], size[1] - 28), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill=(0, 0, 0))
    return canvas


def main() -> None:
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)

    frames = [0, 25, 50, 75]
    rows = []
    readme = [
        "CoInteract P2V stronger face-reference guidance sweep",
        "",
        "Common settings:",
        "- reference: fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg",
        "- pose: InteractAvatar/InterDemo/custom_motion/dwpose/motion2_pose.mp4",
        "- checkpoint: models/CoInteract/checkpoint_pose.safetensors",
        "- canvas: 480x832, 80 frames, 40 denoising steps",
        "- reference compose mode: stretch",
        "- audio/product: empty, silent fallback",
        "- weak-ID branch: latent_blur, blur_kernel=5",
        "",
        "Variants:",
    ]

    for label, folder in VARIANTS:
        src = ROOT / "output_videos" / folder / SAMPLE
        readme.append(f"- {label}: {src}")
        if not src.exists():
            continue

        shutil.copy2(src, PACKAGE / f"{label}.mp4")
        thumbs = []
        for frame_idx in frames:
            frame = read_frame(src, frame_idx)
            if frame is None:
                continue
            Image.fromarray(frame).save(PACKAGE / f"{label}_frame{frame_idx:03d}.jpg", quality=95)
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
        sheet.save(PACKAGE / "strong_sweep_frames_0_25_50_75.jpg", quality=95)

    (PACKAGE / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(PACKAGE)


if __name__ == "__main__":
    main()
