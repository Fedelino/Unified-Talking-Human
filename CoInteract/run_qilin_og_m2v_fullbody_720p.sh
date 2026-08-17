#!/usr/bin/env bash
set -euo pipefail
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar
export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-7}"
cd /data1/workspace/linxinliang/CoInteract
out_dir="/data1/workspace/linxinliang/CoInteract/output_videos/qilin_og_m2v_fullbody_720p_20260622"
mkdir -p "$out_dir"
echo "===== qilin og m2v fullbody 720p start $(date '+%F %T') ====="
echo "visible_npu=${ASCEND_RT_VISIBLE_DEVICES}"
python batch_infer_qilin_og_m2v.py \
  --base_model_path /data1/Wan-AI/wan22_s2v \
  --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
  --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors \
  --csv_path /data1/workspace/linxinliang/CoInteract/examples/m2v_og/cointeract_qilin_og_m2v_fullbody.csv \
  --data_base_path /data1/workspace/linxinliang/CoInteract \
  --output_dir "$out_dir" \
  --height 1280 \
  --width 720 \
  --num_frames 80 \
  --num_clips 1 \
  --num_inference_steps 40 \
  --cfg_scale 7.0
status=$?
echo "===== qilin og m2v fullbody 720p end $(date '+%F %T') status=$status ====="
