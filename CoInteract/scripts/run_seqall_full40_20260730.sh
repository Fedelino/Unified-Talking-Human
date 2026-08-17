#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

OUT=output_videos/qalign_fb01_motion5_arcface_post_seqall_s0005_full40_20260730
rm -rf "$OUT"
mkdir -p "$OUT/debug_arcface"

ASCEND_RT_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 python batch_infer.py \
  --csv_path tmp/arcface_midstep_motion5_20260722/qalign_fb01_motion5.csv \
  --output_dir "$OUT" \
  --height 832 --width 480 --num_frames 92 --num_clips 1 \
  --num_inference_steps 40 --cfg_scale 7.0 --sigma_shift 7.0 \
  --reference_compose_mode stretch --face_reference_guidance_scale 0.0 \
  --arcface_guidance_scale 0.005 \
  --arcface_guidance_timing post \
  --arcface_guidance_post_mode sequential \
  --arcface_guidance_detector insightface \
  --arcface_guidance_target_cosine 0.70 \
  --arcface_guidance_max_updates 1 \
  --arcface_guidance_min_updates 0 \
  --arcface_guidance_decode_lat_frames 2 \
  --arcface_guidance_score_frame_count 3 \
  --arcface_guidance_temporal_windows 0 \
  --arcface_guidance_onnx_path models/arcface/w600k_r50.onnx \
  --arcface_guidance_debug_dir "$OUT/debug_arcface"
