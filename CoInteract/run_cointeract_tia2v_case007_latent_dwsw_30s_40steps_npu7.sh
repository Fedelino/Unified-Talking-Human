#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
CSV_PATH="$ROOT/examples/cointeract_tia2v_case007_stableavatar_20260705.csv"
OUT_DIR="$ROOT/output_videos/tia2v_case007_latent_dwsw_30s_40steps_20260710"
LOG_PATH="$ROOT/logs/tia2v_case007_latent_dwsw_30s_40steps_20260710.log"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="${VISIBLE_NPU:-7}"

mkdir -p "$OUT_DIR" "$(dirname "$LOG_PATH")"

{
  echo "===== CoInteract TIA2V case007 latent DWSW 30s 40-step start $(date '+%F %T') ====="
  echo "visible_npu=$VISIBLE_NPU"
  echo "csv_path=$CSV_PATH"
  echo "output_dir=$OUT_DIR"
} | tee -a "$LOG_PATH"

env ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_NPU" \
  "$PYTHON_BIN" "$ROOT/batch_infer_qilin_og_tia2v.py" \
    --base_model_path /data1/Wan-AI/wan22_s2v \
    --audio_encoder_path "$ROOT/models/chinese-wav2vec2-large" \
    --lora_path "$ROOT/models/CoInteract/checkpoint.safetensors" \
    --csv_path "$CSV_PATH" \
    --output_dir "$OUT_DIR" \
    --height 832 \
    --width 480 \
    --num_frames 76 \
    --num_clips 10 \
    --max_audio_seconds 30 \
    --num_inference_steps 40 \
    --cfg_scale 7.0 \
    --sigma_shift 7.0 \
    --enable_latent_dwsw \
    --latent_window_size 12 \
    --latent_overlap 4 \
    --enable_stableavatar_long_fusion \
    --overlap_frames 40 \
    --overlap_weight_scheme log \
    --clip_seed_mode fixed \
    --base_seed 0 \
    >> "$LOG_PATH" 2>&1

echo "===== CoInteract TIA2V case007 latent DWSW 30s 40-step end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
