#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="7"
OUT_ROOT="$ROOT/output_videos/p2v_identity_case1_motion2_20260705"
LOG_ROOT="$ROOT/logs/p2v_identity_case1_motion2_20260705"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_case() {
  local tag="$1"
  local csv_path="$2"
  local steps="$3"
  shift 3
  local extra_args=("$@")
  local out_dir="$OUT_ROOT/$tag"
  local log_path="$LOG_ROOT/$tag.log"

  mkdir -p "$out_dir"
  {
    echo "===== $tag start $(date '+%F %T') ====="
    echo "visible_npu=$VISIBLE_NPU"
    echo "csv_path=$csv_path"
    echo "output_dir=$out_dir"
    echo "steps=$steps"
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
      --num_inference_steps "$steps" \
      --cfg_scale 7.0 \
      --pose_align_mode none \
      --save_identity_debug \
      "${extra_args[@]}" \
      >> "$log_path" 2>&1

  echo "===== $tag end $(date '+%F %T') =====" | tee -a "$log_path"
}

run_case \
  "01_single" \
  "$ROOT/examples/cointeract_p2v_identity_case1_motion2_01_single_20260705.csv" \
  "40" \
  --identity_layout single

run_case \
  "02_inset" \
  "$ROOT/examples/cointeract_p2v_identity_case1_motion2_02_inset_20260705.csv" \
  "40" \
  --identity_layout inset_triptych \
  --identity_inset_scale 1.45

run_case \
  "03_inset_enh" \
  "$ROOT/examples/cointeract_p2v_identity_case1_motion2_03_inset_enh_20260705.csv" \
  "40" \
  --identity_layout inset_triptych \
  --identity_inset_scale 1.45 \
  --identity_crop_enhance upscale_sharpen

run_case \
  "04_inset_enh_nomoe" \
  "$ROOT/examples/cointeract_p2v_identity_case1_motion2_04_inset_enh_nomoe_20260705.csv" \
  "40" \
  --identity_layout inset_triptych \
  --identity_inset_scale 1.45 \
  --identity_crop_enhance upscale_sharpen \
  --no_use_moe

run_case \
  "05_inset_enh_nomoe_24step" \
  "$ROOT/examples/cointeract_p2v_identity_case1_motion2_05_inset_enh_nomoe_24step_20260705.csv" \
  "24" \
  --identity_layout inset_triptych \
  --identity_inset_scale 1.45 \
  --identity_crop_enhance upscale_sharpen \
  --no_use_moe
