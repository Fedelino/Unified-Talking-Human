#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

OUT="output_videos/trfree_checkpoint_motion5_ranked_20260801"
REF="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
ARC="models/arcface/w600k_r50.onnx"
mkdir -p "$OUT"

python eval/id_drift_metric.py \
  --video output_videos/trfree_seedrank_motion5_seed0_20260801/qalign_fb01_motion5.mp4 \
  --reference "$REF" \
  --arcface_onnx "$ARC" \
  --detector insightface \
  --every 5 \
  --out_csv "$OUT/default_arcface.csv" > "$OUT/default_arcface.log" 2>&1

python eval/id_drift_metric.py \
  --video output_videos/trfree_checkpoint_pose_motion5_20260801/qalign_fb01_motion5.mp4 \
  --reference "$REF" \
  --arcface_onnx "$ARC" \
  --detector insightface \
  --every 5 \
  --out_csv "$OUT/checkpoint_pose_arcface.csv" > "$OUT/checkpoint_pose_arcface.log" 2>&1

python - <<'PY'
import csv
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

out = Path("output_videos/trfree_checkpoint_motion5_ranked_20260801")
items = [
    ("default", out / "default_arcface.csv", "output_videos/trfree_seedrank_motion5_seed0_20260801/qalign_fb01_motion5.mp4"),
    ("checkpoint_pose", out / "checkpoint_pose_arcface.csv", "output_videos/trfree_checkpoint_pose_motion5_20260801/qalign_fb01_motion5.mp4"),
]
rows = []
for name, csv_path, video in items:
    vals = []
    ts = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("detected", "1")) in {"0", "False", "false"}:
                continue
            vals.append(float(row["cos_ref"]))
            ts.append(float(row["time_s"]))
    n = len(vals)
    mean = sum(vals) / n
    mn = min(vals)
    first = vals[0]
    last = vals[-1]
    xmean = sum(ts) / n
    ymean = mean
    denom = sum((x - xmean) ** 2 for x in ts)
    slope = 0.0 if denom == 0 else sum((x - xmean) * (y - ymean) for x, y in zip(ts, vals)) / denom
    rows.append((name, mean, mn, first, last, slope, n, video))

rows.sort(key=lambda x: (x[2], x[1]), reverse=True)
with open(out / "ranking.tsv", "w") as f:
    f.write("rank\tcheckpoint\tmean\tmin\tfirst\tlast\tdrift_slope\tfaces\tvideo\n")
    for i, (name, mean, mn, first, last, slope, n, video) in enumerate(rows, 1):
        f.write(f"{i}\t{name}\t{mean:.6f}\t{mn:.6f}\t{first:.6f}\t{last:.6f}\t{slope:.6f}\t{n}\t{video}\n")

frames = [0, 25, 50, 75]
extract_dir = out / "frames"
extract_dir.mkdir(exist_ok=True)
sheet_rows = []
for name, _, video in items:
    imgs = []
    for frame in frames:
        img = extract_dir / f"{name}_f{frame:03d}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", video,
            "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(img)
        ], check=False)
        im = Image.open(img).convert("RGB")
        im.thumbnail((240, 320))
        canvas = Image.new("RGB", (240, 340), "white")
        canvas.paste(im, ((240 - im.width) // 2, 20))
        ImageDraw.Draw(canvas).text((8, 4), f"{name} frame {frame}", fill=(0, 0, 0))
        imgs.append(canvas)
    row = Image.new("RGB", (240 * len(imgs), 340), "white")
    for i, im in enumerate(imgs):
        row.paste(im, (240 * i, 0))
    sheet_rows.append(row)

sheet = Image.new("RGB", (max(r.width for r in sheet_rows), sum(r.height for r in sheet_rows)), "white")
y = 0
for row in sheet_rows:
    sheet.paste(row, (0, y))
    y += row.height
sheet.save(out / "checkpoint_frame_sheet.jpg", quality=95)

print((out / "ranking.tsv").read_text())
PY
