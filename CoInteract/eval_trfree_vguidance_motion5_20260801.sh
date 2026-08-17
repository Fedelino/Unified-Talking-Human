#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

DATE="20260801"
REF="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
ARC="models/arcface/w600k_r50.onnx"
OUTROOT="output_videos/trfree_vguidance_motion5_ranked_${DATE}"
mkdir -p "$OUTROOT"

SUMMARY="$OUTROOT/summary.tsv"
echo -e "scale\tmean\tmin\tfirst\tlast\tdrift_slope\tfaces\tvideo" > "$SUMMARY"

for label in 0p5 1p0 3p0 5p0; do
  video="output_videos/trfree_vguidance_motion5_scale${label}_${DATE}/qalign_fb01_motion5.mp4"
  csv="$OUTROOT/scale${label}_arcface.csv"
  log="$OUTROOT/scale${label}_arcface.log"
  if [[ ! -s "$video" ]]; then
    echo "missing scale=$label video: $video" | tee "$log"
    continue
  fi

  python eval/id_drift_metric.py \
    --video "$video" \
    --reference "$REF" \
    --arcface_onnx "$ARC" \
    --detector insightface \
    --every 5 \
    --out_csv "$csv" | tee "$log"

  python - "$label" "$csv" "$video" "$SUMMARY" <<'PY'
import csv
import sys

label, csv_path, video_path, summary_path = sys.argv[1:5]
rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        value = row.get("cosine_to_reference") or row.get("cosine")
        if value not in (None, "", "nan"):
            try:
                rows.append((float(row.get("time_sec", len(rows))), float(value)))
            except ValueError:
                pass
if not rows:
    line = [label, "nan", "nan", "nan", "nan", "nan", "0", video_path]
else:
    vals = [v for _, v in rows]
    n = len(rows)
    mean = sum(vals) / n
    mn = min(vals)
    first = vals[0]
    last = vals[-1]
    xs = [x for x, _ in rows]
    xmean = sum(xs) / n
    ymean = mean
    denom = sum((x - xmean) ** 2 for x in xs)
    slope = 0.0 if denom == 0 else sum((x - xmean) * (y - ymean) for x, y in rows) / denom
    line = [label, f"{mean:.6f}", f"{mn:.6f}", f"{first:.6f}", f"{last:.6f}", f"{slope:.6f}", str(n), video_path]
with open(summary_path, "a") as f:
    f.write("\t".join(line) + "\n")
PY
done

python - "$OUTROOT" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

outroot = Path(sys.argv[1])
summary = outroot / "summary.tsv"
rows = []
with open(summary, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        try:
            row["_mean"] = float(row["mean"])
            row["_min"] = float(row["min"])
        except ValueError:
            row["_mean"] = float("-inf")
            row["_min"] = float("-inf")
        rows.append(row)
rows.sort(key=lambda r: (r["_min"], r["_mean"]), reverse=True)
with open(outroot / "ranking.tsv", "w") as f:
    f.write("rank\tscale\tmean\tmin\tfirst\tlast\tdrift_slope\tfaces\tvideo\n")
    for i, row in enumerate(rows, 1):
        f.write("\t".join([
            str(i), row["scale"], row["mean"], row["min"], row["first"], row["last"],
            row["drift_slope"], row["faces"], row["video"],
        ]) + "\n")
print((outroot / "ranking.tsv").read_text())

frames = [0, 25, 50, 75]
extract_dir = outroot / "frames"
extract_dir.mkdir(exist_ok=True)
for row in rows:
    video = row["video"]
    label = row["scale"]
    for frame in frames:
        img = extract_dir / f"scale{label}_f{frame:03d}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", video,
            "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", str(img)
        ], check=False)

try:
    from PIL import Image, ImageDraw
    thumbs = []
    for row in rows:
        label = row["scale"]
        imgs = []
        for frame in frames:
            p = extract_dir / f"scale{label}_f{frame:03d}.jpg"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((240, 320))
            canvas = Image.new("RGB", (240, 340), "white")
            canvas.paste(im, ((240 - im.width) // 2, 20))
            d = ImageDraw.Draw(canvas)
            d.text((8, 4), f"scale {label} frame {frame}", fill=(0, 0, 0))
            imgs.append(canvas)
        if imgs:
            row_canvas = Image.new("RGB", (240 * len(imgs), 340), "white")
            for i, im in enumerate(imgs):
                row_canvas.paste(im, (240 * i, 0))
            thumbs.append(row_canvas)
    if thumbs:
        sheet = Image.new("RGB", (max(t.width for t in thumbs), sum(t.height for t in thumbs)), "white")
        y = 0
        for im in thumbs:
            sheet.paste(im, (0, y))
            y += im.height
        sheet.save(outroot / "vguidance_frame_sheet.jpg", quality=95)
except Exception as exc:
    print(f"frame sheet skipped: {exc}")
PY
