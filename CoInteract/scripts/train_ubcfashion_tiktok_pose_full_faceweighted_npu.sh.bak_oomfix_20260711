#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256,garbage_collection_threshold:0.8}"

BASE_MODEL_PATH="./models/Wan2.2-S2V-14B"
AUDIO_ENCODER_PATH="./models/chinese-wav2vec2-large"
TOKENIZER_PATH="${BASE_MODEL_PATH}/google/umt5-xxl"
DATASET_BASE_PATH="./data/ubcfashion_tiktok_pose_full_facebbox"
DATASET_CSV="${DATASET_BASE_PATH}/data.csv"
OUTPUT_PATH="./output/ubcfashion_tiktok_pose_full_faceweighted"
FACE_LOSS_WEIGHT="${FACE_LOSS_WEIGHT:-2.0}"

export AUDIO_ENCODER_DIR="${AUDIO_ENCODER_PATH}"
export TOKENIZER_DIR="${TOKENIZER_PATH}"

LATEST_STAGE1_CKPT="$(find ./output/ubcfashion_tiktok_pose_full -path '*/step-*.safetensors' -type f 2>/dev/null | sort -V | tail -n 1 || true)"
LORA_CHECKPOINT="${STAGE1_LORA_PATH:-${LATEST_STAGE1_CKPT:-./models/CoInteract/checkpoint_pose.safetensors}}"

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
  echo "Missing ${DATASET_CSV}. Generate it with scripts/generate_face_bbox_metadata.py first." >&2
  exit 1
}
test -f "${LORA_CHECKPOINT}" || {
  echo "Missing LoRA checkpoint: ${LORA_CHECKPOINT}" >&2
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
  --lora_checkpoint "${LORA_CHECKPOINT}" \
  --pose_dropout_prob 0.1 \
  --face_loss_weight "${FACE_LOSS_WEIGHT}" \
  --use_gradient_checkpointing_offload \
  --train_shift 5.0 \
  --learning_rate 5e-6 \
  --num_epochs 2 \
  --save_steps 25 \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 0
