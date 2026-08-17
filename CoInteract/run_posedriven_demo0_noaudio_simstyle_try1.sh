#!/usr/bin/env bash
set -euo pipefail
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar
export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-6}
cd /data1/workspace/linxinliang/CoInteract
out_dir=/data1/workspace/linxinliang/CoInteract/output_videos/posedriven_demo0_noaudio_simstyle_try1
mkdir -p $out_dir
echo ===== CoInteract posedriven demo0 noaudio simstyle start $(date '+%F %T') =====
echo visible_npu=${ASCEND_RT_VISIBLE_DEVICES}
python batch_infer.py \
  --base_model_path /data1/Wan-AI/wan22_s2v \
  --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
  --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors \
  --csv_path /data1/workspace/linxinliang/CoInteract/examples/posedriven_demo0_noaudio_simstyle.csv \
  --data_base_path /data1/workspace/linxinliang/CoInteract \
  --output_dir $out_dir \
  --height 832 \
  --width 480 \
  --num_frames 80 \
  --num_clips 1 \
  --num_inference_steps 40 \
  --cfg_scale 7.0
status=$?
echo ===== CoInteract posedriven demo0 noaudio simstyle end $(date '+%F %T') status=$status =====
