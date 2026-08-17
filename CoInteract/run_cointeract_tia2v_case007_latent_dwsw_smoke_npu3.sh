#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
CSV_PATH="$ROOT/examples/cointeract_tia2v_case007_stableavatar_20260705.csv"
OUT_DIR="$ROOT/output_videos/tia2v_case007_latent_dwsw_smoke_20260709"
LOG_PATH="$ROOT/logs/tia2v_case007_latent_dwsw_smoke_20260709.log"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="${VISIBLE_NPU:-3}"

mkdir -p "$OUT_DIR" "$(dirname "$LOG_PATH")"

{
  echo "===== CoInteract TIA2V case007 latent DWSW smoke start $(date '+%F %T') ====="
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
    --num_clips 1 \
    --max_audio_seconds 3.2 \
    --num_inference_steps 20 \
    --cfg_scale 7.0 \
    --sigma_shift 7.0 \
    --enable_latent_dwsw \
    --latent_window_size 12 \
    --latent_overlap 4 \
    --clip_seed_mode fixed \
    --base_seed 0 \
    >> "$LOG_PATH" 2>&1

echo "===== CoInteract TIA2V case007 latent DWSW smoke end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
