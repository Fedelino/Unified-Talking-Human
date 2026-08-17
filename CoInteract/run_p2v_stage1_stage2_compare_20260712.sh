#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
CSV="$ROOT/examples/cointeract_p2v_identity_case1_motion2_01_single_20260705.csv"

STAGE1_LORA="$ROOT/output/ubcfashion_tiktok_pose_full/version_1/step-2050.safetensors"
STAGE2_LORA="$ROOT/output/ubcfashion_tiktok_pose_full_faceweighted/version_3/step-2000.safetensors"

STAMP="20260712"
OUT_ROOT="$ROOT/output_videos/p2v_stage_compare_${STAMP}"
LOG_ROOT="$ROOT/logs/p2v_stage_compare_${STAMP}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_stage() {
  local tag="$1"; local lora="$2"; local npu="$3"
  local out_dir="$OUT_ROOT/$tag"; local log_path="$LOG_ROOT/$tag.log"
  mkdir -p "$out_dir"
  {
    echo "===== $tag start $(date '+%F %T') ====="
    echo "visible_npu=$npu"; echo "lora=$lora"; echo "csv=$CSV"; echo "out=$out_dir"
  } | tee -a "$log_path"

  env ASCEND_RT_VISIBLE_DEVICES="$npu" TOKENIZERS_PARALLELISM=false \
    PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128 \
    "$PYTHON_BIN" "$ROOT/batch_infer_qilin_og_m2v.py" \
      --base_model_path /data1/Wan-AI/wan22_s2v \
      --audio_encoder_path "$ROOT/models/chinese-wav2vec2-large" \
      --lora_path "$lora" \
      --csv_path "$CSV" \
      --data_base_path "$ROOT" \
      --output_dir "$out_dir" \
      --height 832 --width 480 --num_frames 80 --num_clips 1 \
      --num_inference_steps 40 --cfg_scale 7.0 \
      --pose_align_mode none --identity_layout single \
      >> "$log_path" 2>&1

  echo "===== $tag end $(date '+%F %T') =====" | tee -a "$log_path"
}

run_stage "stage1_step2050" "$STAGE1_LORA" "3"
run_stage "stage2_step2000" "$STAGE2_LORA" "3"
echo "ALL DONE $(date '+%F %T')" | tee -a "$LOG_ROOT/_alldone.log"
