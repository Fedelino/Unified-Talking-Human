#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="1"
CSV_PATH="$ROOT/examples/cointeract_p2v_case1_refclean_motion2_20260705.csv"
OUT_ROOT="$ROOT/output_videos/p2v_case1_refclean_20260705"
LOG_ROOT="$ROOT/logs/p2v_case1_refclean_20260705"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_case() {
  local tag="$1"
  shift
  local extra_args=("$@")
  local out_dir="$OUT_ROOT/$tag"
  local log_path="$LOG_ROOT/$tag.log"

  mkdir -p "$out_dir"
  {
    echo "===== $tag start $(date '+%F %T') ====="
    echo "visible_npu=$VISIBLE_NPU"
    echo "csv_path=$CSV_PATH"
    echo "output_dir=$out_dir"
    echo "extra_args=${extra_args[*]:-}"
  } | tee -a "$log_path"

  env ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_NPU" \
    "$PYTHON_BIN" "$ROOT/batch_infer_qilin_og_m2v.py" \
      --base_model_path /data1/Wan-AI/wan22_s2v \
      --audio_encoder_path "$ROOT/models/chinese-wav2vec2-large" \
      --lora_path "$ROOT/models/CoInteract/checkpoint_pose.safetensors" \
      --csv_path "$CSV_PATH" \
      --output_dir "$out_dir" \
      --height 832 \
      --width 480 \
      --num_frames 80 \
      --num_clips 3 \
      --num_inference_steps 40 \
      --cfg_scale 7.0 \
      --pose_align_mode none \
      --identity_layout single \
      --reference_compose_mode stretch \
      --save_identity_debug \
      --save_reference_debug \
      "${extra_args[@]}" \
      >> "$log_path" 2>&1

  echo "===== $tag end $(date '+%F %T') =====" | tee -a "$log_path"
}

run_case \
  "01_single_stretch" \
  --reference_preprocess_mode none

run_case \
  "02_single_stretch_faceboost" \
  --reference_preprocess_mode face_boost

run_case \
  "03_single_stretch_faceupper" \
  --reference_preprocess_mode face_upper_boost

run_case \
  "04_faceonly_stretch_diag" \
  --reference_preprocess_mode face_only
