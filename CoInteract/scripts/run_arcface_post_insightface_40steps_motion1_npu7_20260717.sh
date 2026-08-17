#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

export ASCEND_RT_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

python batch_infer.py \
  --base_model_path /data1/Wan-AI/wan22_s2v \
  --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
  --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors \
  --csv_path examples/th_fullbody_001_custom_motion1_noaudio_baseline_20260716.csv \
  --data_base_path /data1/workspace/linxinliang/CoInteract \
  --output_dir output_videos/p2v_arcface_post_insightface_40steps_motion1_npu7_20260717 \
  --height 832 \
  --width 480 \
  --num_frames 80 \
  --num_clips 1 \
  --num_inference_steps 40 \
  --cfg_scale 7.0 \
  --sigma_shift 7.0 \
  --reference_compose_mode stretch \
  --no_resize_output_to_reference \
  --arcface_guidance_scale 0.02 \
  --arcface_guidance_timing post \
  --arcface_guidance_detector insightface \
  --arcface_guidance_target_cosine 0.70 \
  --arcface_guidance_max_updates 4 \
  --arcface_guidance_min_updates 1 \
  --arcface_guidance_decode_lat_frames 2 \
  --arcface_guidance_score_frame_count 3 \
  --arcface_guidance_onnx_path models/arcface/w600k_r50.onnx \
  --arcface_guidance_debug_dir output_videos/p2v_arcface_post_insightface_40steps_motion1_npu7_20260717/debug_arcface
