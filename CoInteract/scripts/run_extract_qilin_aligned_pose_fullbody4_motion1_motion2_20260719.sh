#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
OUT_ROOT="${ROOT}/output_videos/qilin_keypoint_aligned_pose_fullbody4_motion1_motion2_20260719"
POSE_DIR="${OUT_ROOT}/pose"
LOG_DIR="${ROOT}/logs/qilin_keypoint_aligned_pose_fullbody4_motion1_motion2_20260719"

mkdir -p "${POSE_DIR}" "${LOG_DIR}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

cd "${ROOT}"

declare -A REFS=(
  [fb01]="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
  [fb02]="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImagec13ac940fa4ee5cc9ba7eb4cab1f7535b6007194.jpg"
  [fb03]="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1972da23dfed1470925230ee4fb31c6ede4b624e.jpg"
  [fb04]="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImagefc7e91a057314f4aa00f22b3aa00cafa3e31d756.jpg"
)

declare -A MOTIONS=(
  [motion1]="/data1/workspace/linxinliang/InteractAvatar/InterDemo/custom_motion/raw/motion1.mp4"
  [motion2]="/data1/workspace/linxinliang/InteractAvatar/InterDemo/custom_motion/raw/motion2.mp4"
)

for motion_name in motion1 motion2; do
  for ref_name in fb01 fb02 fb03 fb04; do
    out_pose="${POSE_DIR}/${ref_name}_${motion_name}_qilin_aligned_25fps.mp4"
    log_path="${LOG_DIR}/${ref_name}_${motion_name}.log"
    if [[ -s "${out_pose}" ]]; then
      echo "[skip] ${ref_name}_${motion_name}: ${out_pose}"
      continue
    fi
    echo "[extract] ${ref_name}_${motion_name}"
    python scripts/extract_qilin_aligned_dwpose_video.py \
      --input-video "${MOTIONS[$motion_name]}" \
      --ref-image "${REFS[$ref_name]}" \
      --output-video "${out_pose}" \
      --target-fps 25.0 \
      > "${log_path}" 2>&1
  done
done

echo "[done] ${POSE_DIR}"
