#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="7"
OUT_ROOT="$ROOT/output_videos/bachvidlite_case1_motioncache_20260705"
LOG_ROOT="$ROOT/logs/bachvidlite_case1_motioncache_20260705"
CACHE_CSV="$ROOT/examples/cointeract_bachvidlite_case1_motion1_cache_20260705.csv"
TARGET_CSV="$ROOT/examples/cointeract_bachvidlite_case1_motion2_target_20260705.csv"
CACHE_OUT="$OUT_ROOT/01_motion1_identity_cache/th_fullbody_001_motion1_identitycache.mp4"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_case() {
  local tag="$1"
  local csv_path="$2"
  shift 2
  local extra_args=("$@")
  local out_dir="$OUT_ROOT/$tag"
  local log_path="$LOG_ROOT/$tag.log"

  mkdir -p "$out_dir"
  {
    echo "===== $tag start $(date '+%F %T') ====="
    echo "visible_npu=$VISIBLE_NPU"
    echo "csv_path=$csv_path"
    echo "output_dir=$out_dir"
    echo "extra_args=${extra_args[*]:-}"
  } | tee -a "$log_path"

  env ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_NPU" \
    "$PYTHON_BIN" "$ROOT/batch_infer_qilin_og_m2v.py" \
      --base_model_path /data1/Wan-AI/wan22_s2v \
      --audio_encoder_path "$ROOT/models/chinese-wav2vec2-large" \
      --lora_path "$ROOT/models/CoInteract/checkpoint_pose.safetensors" \
      --csv_path "$csv_path" \
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
      --reference_preprocess_mode none \
      --save_identity_debug \
      --save_reference_debug \
      "${extra_args[@]}" \
      >> "$log_path" 2>&1

  echo "===== $tag end $(date '+%F %T') =====" | tee -a "$log_path"
}

run_case "01_motion1_identity_cache" "$CACHE_CSV"

if [[ ! -s "$CACHE_OUT" ]]; then
  echo "Missing identity cache video: $CACHE_OUT" | tee -a "$LOG_ROOT/02_motion2_with_cache.log"
  exit 1
fi

run_case \
  "02_motion2_with_motion1_cache" \
  "$TARGET_CSV" \
  --initial_motion_video_path "$CACHE_OUT" \
  --initial_motion_frames 73
