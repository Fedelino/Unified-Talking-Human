#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8

mkdir -p logs output_videos/p2v_stagea_vguidance_scale025_motion2_20260716

python batch_infer.py \
  --base_model_path /data1/Wan-AI/wan22_s2v \
  --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
  --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors \
  --csv_path examples/th_fullbody_001_custom_motion2_noaudio_baseline_20260716.csv \
  --data_base_path /data1/workspace/linxinliang/CoInteract \
  --output_dir output_videos/p2v_stagea_vguidance_scale025_motion2_20260716 \
  --height 832 \
  --width 480 \
  --num_frames 80 \
  --num_clips 1 \
  --num_inference_steps 40 \
  --cfg_scale 7.0 \
  --sigma_shift 7.0 \
  --reference_compose_mode stretch \
  --no_resize_output_to_reference \
  --face_reference_guidance_scale 0.25 \
  --face_reference_guidance_power 1.0 \
  --face_reference_guidance_start_t 0.0 \
  --face_reference_guidance_end_t 0.9
