#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256,garbage_collection_threshold:0.8}"

CSV_PATH="${CSV_PATH:-./examples/demos/ubcfashion_pose_infer.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_videos/ubcfashion_pose_eval}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-320}"
NUM_CLIPS="${NUM_CLIPS:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
CFG_SCALE="${CFG_SCALE:-7.0}"
SIGMA_SHIFT="${SIGMA_SHIFT:-7.0}"

DEFAULT_FULL_CKPT="./output/ubcfashion_pose_full/version_0/step-2500.safetensors"
LATEST_FULL_CKPT="$(find ./output/ubcfashion_pose_full -path '*/step-*.safetensors' -type f 2>/dev/null | sort -V | tail -n 1 || true)"
LATEST_SMALL_CKPT="$(find ./output/ubcfashion_pose_small -path '*/step-*.safetensors' -type f 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -f "${DEFAULT_FULL_CKPT}" ]]; then
  DEFAULT_LORA_PATH="${DEFAULT_FULL_CKPT}"
else
  DEFAULT_LORA_PATH="${LATEST_FULL_CKPT:-${LATEST_SMALL_CKPT:-./models/CoInteract/checkpoint_pose.safetensors}}"
fi
LORA_PATH="${LORA_PATH:-${DEFAULT_LORA_PATH}}"

test -f "${CSV_PATH}" || {
  echo "Missing CSV: ${CSV_PATH}" >&2
  exit 1
}
test -f "${LORA_PATH}" || {
  echo "Missing LoRA checkpoint: ${LORA_PATH}" >&2
  exit 1
}

echo "CSV_PATH=${CSV_PATH}"
echo "LORA_PATH=${LORA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HEIGHT=${HEIGHT} WIDTH=${WIDTH} NUM_CLIPS=${NUM_CLIPS}"

python batch_infer.py \
  --csv_path "${CSV_PATH}" \
  --lora_path "${LORA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_clips "${NUM_CLIPS}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --sigma_shift "${SIGMA_SHIFT}" \
  --no_use_moe
