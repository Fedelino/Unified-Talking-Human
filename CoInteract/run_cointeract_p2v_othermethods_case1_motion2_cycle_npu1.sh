#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="${VISIBLE_NPU:-1}"
CSV_PATH="$ROOT/examples/cointeract_p2v_identity_case1_motion2_01_single_20260705.csv"
OUT_ROOT="$ROOT/output_videos/p2v_othermethods_case1_motion2_20260707"
LOG_ROOT="$ROOT/logs/p2v_othermethods_case1_motion2_20260707"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_case() {
  local tag="$1"
  local cfg="$2"
  local steps="$3"
  local frames="$4"
  local lora_path="$5"
  local lora_alpha="$6"
  shift 6
  local extra_args=("$@")
  local out_dir="$OUT_ROOT/$tag"
  local log_path="$LOG_ROOT/$tag.log"

  mkdir -p "$out_dir"
  {
    echo "===== $tag start $(date '+%F %T') ====="
    echo "visible_npu=$VISIBLE_NPU"
    echo "csv_path=$CSV_PATH"
    echo "output_dir=$out_dir"
    echo "cfg=$cfg steps=$steps frames=$frames"
    echo "lora_path=$lora_path"
    echo "lora_alpha=$lora_alpha"
    echo "extra_args=${extra_args[*]:-}"
  } | tee -a "$log_path"

  env ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_NPU" \
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:512}" \
    "$PYTHON_BIN" "$ROOT/batch_infer_qilin_og_m2v.py" \
      --base_model_path /data1/Wan-AI/wan22_s2v \
      --audio_encoder_path "$ROOT/models/chinese-wav2vec2-large" \
      --lora_path "$lora_path" \
      --lora_alpha "$lora_alpha" \
      --csv_path "$CSV_PATH" \
      --output_dir "$out_dir" \
      --height 832 \
      --width 480 \
      --num_frames "$frames" \
      --num_clips 3 \
      --num_inference_steps "$steps" \
      --cfg_scale "$cfg" \
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

# Method family 1: reduce prompt/CFG oversteering, which can hallucinate face/clothing changes.
run_case "01_single_moe_cfg4p5" 4.5 40 80 "$ROOT/models/CoInteract/checkpoint_pose.safetensors" 1.0

# Method family 2: opposite CFG direction, to see whether stronger conditioning improves pose/appearance or worsens drift.
run_case "02_single_moe_cfg9" 9.0 40 80 "$ROOT/models/CoInteract/checkpoint_pose.safetensors" 1.0

# Method family 3: disable MoE without using any crop/inset board, testing whether region experts are over-editing face details.
run_case "03_single_nomoe_cfg4p5" 4.5 40 80 "$ROOT/models/CoInteract/checkpoint_pose.safetensors" 1.0 --no_use_moe

# Method family 4: reduce LoRA influence while keeping MoE weights, testing whether the pose LoRA itself is over-deforming identity.
run_case "04_single_moe_loraalpha0p6_cfg5p5" 5.5 40 80 "$ROOT/models/CoInteract/checkpoint_pose.safetensors" 0.6

# Method family 5: use the non-pose CoInteract checkpoint as an identity-preservation check against the pose-specialized checkpoint.
run_case "05_single_moe_generalckpt_cfg5p5" 5.5 40 80 "$ROOT/models/CoInteract/checkpoint.safetensors" 1.0

# Method family 6: shorter temporal chunks, same pose duration, to test whether within-window drift is the main issue.
run_case "06_single_moe_shortchunks_cfg5p5" 5.5 40 36 "$ROOT/models/CoInteract/checkpoint_pose.safetensors" 1.0
