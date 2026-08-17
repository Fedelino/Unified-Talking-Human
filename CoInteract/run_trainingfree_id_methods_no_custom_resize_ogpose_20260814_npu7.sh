#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

export ASCEND_RT_VISIBLE_DEVICES=7
export ASCEND_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false

CSV="examples/no_custom_resize_ogpose_onecase_20260814.csv"
COMMON=(
  --csv_path "$CSV"
  --height 832
  --width 480
  --num_frames 80
  --num_clips 1
  --num_inference_steps 40
  --cfg_scale 7.0
  --sigma_shift 7.0
  --lora_path ./models/CoInteract/checkpoint_pose.safetensors
  --reference_compose_mode stretch
  --no_resize_output_to_reference
)

mkdir -p logs

run_case() {
  local name="$1"
  shift
  echo "===== $(date '+%F %T') START ${name} ====="
  python batch_infer.py \
    "${COMMON[@]}" \
    --output_dir "output_videos/${name}" \
    "$@" \
    > "logs/${name}.log" 2>&1
  echo "===== $(date '+%F %T') DONE ${name} ====="
}

run_case "p2v_idmethod_noresize_baseline_ogpose_20260814"

run_case "p2v_idmethod_noresize_refkv_headattn_s010_ogpose_20260814" \
  --reference_kv_guidance_scale 0.10 \
  --reference_kv_guidance_mode head_attn \
  --reference_kv_guidance_blocks 10:22 \
  --reference_kv_guidance_start_t 0.05 \
  --reference_kv_guidance_end_t 0.75

run_case "p2v_idmethod_noresize_refkv_spatial_s005_ogpose_20260814" \
  --reference_kv_guidance_scale 0.05 \
  --reference_kv_guidance_mode spatial_copy \
  --reference_kv_guidance_blocks 10:22 \
  --reference_kv_guidance_start_t 0.05 \
  --reference_kv_guidance_end_t 0.75

python scripts/arcface_seed_selector.py \
  --csv_path "$CSV" \
  --row_index 0 \
  --data_base_path . \
  --output_dir output_videos/p2v_idmethod_noresize_arcface_seedselect_ogpose_20260814 \
  --arcface_onnx models/arcface/w600k_r50.onnx \
  --seeds 0,1,2,3 \
  --preview_steps 12 \
  --final_steps 40 \
  --every 10 \
  --detector insightface \
  --score mean_min \
  --extra_args "--height 832 --width 480 --num_frames 80 --num_clips 1 --cfg_scale 7.0 --sigma_shift 7.0 --lora_path ./models/CoInteract/checkpoint_pose.safetensors --reference_compose_mode stretch --no_resize_output_to_reference" \
  > logs/p2v_idmethod_noresize_arcface_seedselect_ogpose_20260814.log 2>&1

echo "===== $(date '+%F %T') ALL DONE no-custom-resize og-pose ====="
