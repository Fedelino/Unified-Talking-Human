#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/workspace/linxinliang/CoInteract
REF=/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg
ARC="$ROOT/models/arcface/w600k_r50.onnx"
OUT="$ROOT/output_videos/p2v_geometry_face_control_v2_20260828_eval"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd "$ROOT"
mkdir -p "$OUT"

for video in output_videos/p2v_geometry_face_control_v2_20260828/*/*.mp4; do
  [[ -f "$video" ]] || continue
  variant="$(basename "$(dirname "$video")")"
  python eval/id_drift_metric.py \
    --video "$video" \
    --reference "$REF" \
    --arcface_onnx "$ARC" \
    --detector insightface \
    --every 5 \
    --out_csv "$OUT/${variant}.csv" \
    > "$OUT/${variant}.txt" 2>&1
done

grep -H "cos_ref\\|drift slope\\|hsv_ref\\|ssim_ref\\|sharpness" "$OUT"/*.txt || true
