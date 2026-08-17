#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_VISIBLE_DEVICES=7

python scripts/arcface_seed_selector.py \
  --csv_path examples/refkv_smoke_onecase_20260814.csv \
  --row_index 0 \
  --data_base_path . \
  --output_dir output_videos/arcface_seedselect_smoke_20260814 \
  --arcface_onnx models/arcface/w600k_r50.onnx \
  --seeds 0,1 \
  --preview_steps 4 \
  --final_steps 4 \
  --every 10 \
  --detector insightface \
  --score mean_min \
  --extra_args "--height 832 --width 480 --num_frames 80 --num_clips 1 --cfg_scale 7.0 --sigma_shift 7.0 --lora_path ./models/CoInteract/checkpoint_pose.safetensors --reference_compose_mode stretch --no_resize_output_to_reference"
