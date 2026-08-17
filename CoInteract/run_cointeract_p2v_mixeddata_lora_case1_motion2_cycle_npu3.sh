#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="${VISIBLE_NPU:-3}"
CSV_PATH="$ROOT/examples/cointeract_p2v_identity_case1_motion2_01_single_20260705.csv"
MIXED_LORA="/data1/workspace/leijunwei/CoInteract/output/ubcfashion_tiktok_pose_full/version_0/step-5000.safetensors"
OUT_ROOT="$ROOT/output_videos/p2v_mixeddata_lora_case1_motion2_20260707"
LOG_ROOT="$ROOT/logs/p2v_mixeddata_lora_case1_motion2_20260707"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_case() {
  local tag="$1"
  local height="$2"
  local width="$3"
  local frames="$4"
  local steps="$5"
  local cfg="$6"
  local out_dir="$OUT_ROOT/$tag"
  local log_path="$LOG_ROOT/$tag.log"

  mkdir -p "$out_dir"
  {
    echo "===== $tag start $(date '+%F %T') ====="
    echo "visible_npu=$VISIBLE_NPU"
    echo "csv_path=$CSV_PATH"
    echo "mixed_lora=$MIXED_LORA"
    echo "output_dir=$out_dir"
    echo "height=$height width=$width frames=$frames steps=$steps cfg=$cfg"
  } | tee -a "$log_path"

  env ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_NPU" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:512}" \
    "$PYTHON_BIN" "$ROOT/batch_infer_qilin_og_m2v.py" \
      --base_model_path /data1/Wan-AI/wan22_s2v \
      --audio_encoder_path "$ROOT/models/chinese-wav2vec2-large" \
      --lora_path "$MIXED_LORA" \
      --lora_alpha 1.0 \
      --csv_path "$CSV_PATH" \
      --output_dir "$out_dir" \
      --height "$height" \
      --width "$width" \
      --num_frames "$frames" \
      --num_clips 3 \
      --num_inference_steps "$steps" \
      --cfg_scale "$cfg" \
      --pose_align_mode none \
      --identity_layout single \
      --reference_compose_mode stretch \
      --reference_preprocess_mode none \
      --no_use_moe \
      --save_identity_debug \
      --save_reference_debug \
      >> "$log_path" 2>&1

  echo "===== $tag end $(date '+%F %T') =====" | tee -a "$log_path"
}

# First run at the training resolution used by the mixed UBCFashion+TikTok LoRA.
run_case "01_mixed_step5000_trainres_320x480_nomoe" 480 320 80 30 7.0

# Then test our usual taller full-body canvas. This is out-of-training-resolution but useful for direct comparison.
run_case "02_mixed_step5000_480x832_nomoe" 832 480 80 30 7.0
