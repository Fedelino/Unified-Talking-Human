#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

REF="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
ARC="models/arcface/w600k_r50.onnx"
OUT="output_videos/p2v_refkv_strong_prevconfig_motion1_eval_20260815"
mkdir -p "$OUT"

cat > "$OUT/manifest.tsv" <<EOF
00_baseline	output_videos/p2v_altmethods_prevconfig_motion1_baseline_noguidance_20260814/th_fullbody_001_custom_motion1_altmethods.mp4
01_headattn_s005	output_videos/p2v_altmethods_prevconfig_motion1_refkv_headattn_s005_20260814/th_fullbody_001_custom_motion1_altmethods.mp4
02_headattn_s010	output_videos/p2v_altmethods_prevconfig_motion1_refkv_headattn_s010_20260814/th_fullbody_001_custom_motion1_altmethods.mp4
03_headattn_s025	output_videos/p2v_refkv_strong_prevconfig_motion1_refkv_headattn_s025_20260815/th_fullbody_001_custom_motion1_refkv_strong.mp4
04_headattn_s050	output_videos/p2v_refkv_strong_prevconfig_motion1_refkv_headattn_s050_20260815/th_fullbody_001_custom_motion1_refkv_strong.mp4
05_headattn_s100	output_videos/p2v_refkv_strong_prevconfig_motion1_refkv_headattn_s100_20260815/th_fullbody_001_custom_motion1_refkv_strong.mp4
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
values = [float(row["cos_ref"]) for row in rows]
times = [float(row["time_s"]) for row in rows]
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
