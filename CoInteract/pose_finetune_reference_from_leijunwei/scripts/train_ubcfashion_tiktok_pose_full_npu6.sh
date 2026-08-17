#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256,garbage_collection_threshold:0.8}"

BASE_MODEL_PATH="./models/Wan2.2-S2V-14B"
AUDIO_ENCODER_PATH="./models/chinese-wav2vec2-large"
TOKENIZER_PATH="${BASE_MODEL_PATH}/google/umt5-xxl"
DATASET_BASE_PATH="./data/ubcfashion_tiktok_pose_full"
DATASET_CSV="${DATASET_BASE_PATH}/data.csv"
OUTPUT_PATH="./output/ubcfashion_tiktok_pose_full"

export AUDIO_ENCODER_DIR="${AUDIO_ENCODER_PATH}"
export TOKENIZER_DIR="${TOKENIZER_PATH}"

MODEL_PATHS="[
  \"${BASE_MODEL_PATH}/models_t5_umt5-xxl-enc-bf16.pth\",
  \"${BASE_MODEL_PATH}/Wan2.1_VAE.pth\",
  \"${AUDIO_ENCODER_PATH}/pytorch_model.bin\",
  [
    \"${BASE_MODEL_PATH}/diffusion_pytorch_model-00001-of-00004.safetensors\",
    \"${BASE_MODEL_PATH}/diffusion_pytorch_model-00002-of-00004.safetensors\",
    \"${BASE_MODEL_PATH}/diffusion_pytorch_model-00003-of-00004.safetensors\",
    \"${BASE_MODEL_PATH}/diffusion_pytorch_model-00004-of-00004.safetensors\"
  ]
]"

test -f "${DATASET_CSV}" || {
  echo "Missing ${DATASET_CSV}. Run: bash scripts/prepare_ubcfashion_tiktok_pose_mix.sh" >&2
  exit 1
}

python examples/wanvideo/model_training/train.py \
  --model_paths "${MODEL_PATHS}" \
  --dataset_metadata_path "${DATASET_CSV}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --height 480 \
  --width 320 \
  --num_frames 81 \
  --extra_inputs "person_image" \
  --trainable_models "dit" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o" \
  --lora_rank 128 \
  --lora_checkpoint "./models/CoInteract/checkpoint_pose.safetensors" \
  --pose_dropout_prob 0.1 \
  --use_gradient_checkpointing_offload \
  --train_shift 5.0 \
  --learning_rate 5e-6 \
  --num_epochs 5 \
  --save_steps 25 \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 0
