#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
CSV_PATH="$ROOT/examples/cointeract_fullbody_case1_motion2_cropboost_20260703.csv"
OUT_DIR="$ROOT/output_videos/fullbody_case1_motion2_cropboost_480p_20260703"
LOG_PATH="$ROOT/logs/fullbody_case1_motion2_cropboost_480p_20260703.log"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="7"

mkdir -p "$OUT_DIR" "$(dirname "$LOG_PATH")"

{
  echo "===== CoInteract fullbody cropboost 480p start $(date '+%F %T') ====="
  echo "visible_npu=$VISIBLE_NPU"
  echo "csv_path=$CSV_PATH"
  echo "output_dir=$OUT_DIR"
} | tee -a "$LOG_PATH"

env ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_NPU" \
  "$PYTHON_BIN" "$ROOT/batch_infer_qilin_og_m2v.py" \
    --base_model_path /data1/Wan-AI/wan22_s2v \
    --lora_path "$ROOT/models/CoInteract/checkpoint_pose.safetensors" \
    --csv_path "$CSV_PATH" \
    --output_dir "$OUT_DIR" \
    --height 832 \
    --width 480 \
    --num_frames 80 \
    --num_clips 3 \
    --num_inference_steps 40 \
    --cfg_scale 7.0 \
    --pose_align_mode none \
    --identity_layout inset_triptych \
    --identity_inset_scale 1.45 \
    --save_identity_debug \
    >> "$LOG_PATH" 2>&1

echo "===== CoInteract fullbody cropboost 480p end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
