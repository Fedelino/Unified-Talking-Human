#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

REF="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
ARC="models/arcface/w600k_r50.onnx"
OUT="output_videos/p2v_altmethods_prevconfig_motion1_eval_20260815"
mkdir -p "$OUT"

cat > "$OUT/manifest.tsv" <<EOF
01_baseline_noguidance	output_videos/p2v_altmethods_prevconfig_motion1_baseline_noguidance_20260814/th_fullbody_001_custom_motion1_altmethods.mp4
02_refkv_headattn_s005	output_videos/p2v_altmethods_prevconfig_motion1_refkv_headattn_s005_20260814/th_fullbody_001_custom_motion1_altmethods.mp4
03_refkv_headattn_s010	output_videos/p2v_altmethods_prevconfig_motion1_refkv_headattn_s010_20260814/th_fullbody_001_custom_motion1_altmethods.mp4
04_refkv_spatialcopy_s0025	output_videos/p2v_altmethods_prevconfig_motion1_refkv_spatialcopy_s0025_20260814/th_fullbody_001_custom_motion1_altmethods.mp4
05_refkv_spatialcopy_s005	output_videos/p2v_altmethods_prevconfig_motion1_refkv_spatialcopy_s005_20260814/th_fullbody_001_custom_motion1_altmethods.mp4
EOF

printf "method\tmean\tmin\tfirst\tlast\tslope_per_sec\tn_frames\n" > "$OUT/arcface_summary.tsv"

while IFS=$'\t' read -r name video; do
  csv="$OUT/${name}_arcface.csv"
  log="$OUT/${name}_arcface.log"
  echo "[eval] $name"
  python eval/id_drift_metric.py \
    --video "$video" \
    --reference "$REF" \
    --arcface_onnx "$ARC" \
    --detector insightface \
    --every 5 \
    --out_csv "$csv" \
    > "$log" 2>&1
  python - "$name" "$csv" >> "$OUT/arcface_summary.tsv" <<'PY'
import csv
import statistics
import sys

name, csv_path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
values = []
times = []
for row in rows:
    for key in ("cos_ref", "cosine_to_ref", "cosine"):
        if key in row:
            break
    else:
        raise KeyError(f"No cosine column in {csv_path}: {list(rows[0].keys()) if rows else []}")
    values.append(float(row[key]))
    if "time_s" in row:
        times.append(float(row["time_s"]))
    elif "time_sec" in row:
        times.append(float(row["time_sec"]))
    elif "t_sec" in row:
        times.append(float(row["t_sec"]))
    elif "frame" in row:
        times.append(float(row["frame"]) / 25.0)
    else:
        times.append(float(len(times)))

mean = statistics.fmean(values) if values else float("nan")
mn = min(values) if values else float("nan")
first = values[0] if values else float("nan")
last = values[-1] if values else float("nan")
if len(values) >= 2:
    x0 = statistics.fmean(times)
    y0 = statistics.fmean(values)
    denom = sum((x - x0) ** 2 for x in times)
    slope = sum((x - x0) * (y - y0) for x, y in zip(times, values)) / denom if denom else 0.0
else:
    slope = 0.0
print(f"{name}\t{mean:.6f}\t{mn:.6f}\t{first:.6f}\t{last:.6f}\t{slope:.6f}\t{len(values)}")
PY
done < "$OUT/manifest.tsv"

cat "$OUT/arcface_summary.tsv"
