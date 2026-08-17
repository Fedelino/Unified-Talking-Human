#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract

CSV="examples/no_custom_resize_ogpose_motion2_onecase_20260814.csv"
OUT_DIR="output_videos/p2v_noresize_baseline_ogpose_motion2_direct_20260814"
LOG="logs/p2v_noresize_baseline_ogpose_motion2_direct_20260814.log"

cp examples/no_custom_resize_ogpose_onecase_20260814.csv "$CSV"
sed -i \
  -e 's/noresize_ogpose_motion1/noresize_ogpose_motion2/g' \
  -e 's#001_motion1_pose.mp4#001_motion2_pose.mp4#g' \
  "$CSV"

mkdir -p logs "$OUT_DIR"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python batch_infer.py \
  --csv_path "$CSV" \
  --height 832 \
  --width 480 \
  --num_frames 80 \
  --num_clips 1 \
  --num_inference_steps 40 \
  --cfg_scale 7.0 \
  --sigma_shift 7.0 \
  --lora_path ./models/CoInteract/checkpoint_pose.safetensors \
  --reference_compose_mode stretch \
  --no_resize_output_to_reference \
  --output_dir "$OUT_DIR" \
  > "$LOG" 2>&1
