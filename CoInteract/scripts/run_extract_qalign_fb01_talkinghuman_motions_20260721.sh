#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
OUT_ROOT="${ROOT}/output_videos/qilin_keypoint_aligned_pose_fb01_talkinghuman_motions_20260721"
POSE_DIR="${OUT_ROOT}/pose"
TRIM_DIR="${OUT_ROOT}/raw_4s"
LOG_DIR="${ROOT}/logs/qilin_keypoint_aligned_pose_fb01_talkinghuman_motions_20260721"
REF="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
MOTION_ROOT="/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/motion"
MAX_SECONDS="4"

mkdir -p "${POSE_DIR}" "${TRIM_DIR}" "${LOG_DIR}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

cd "${ROOT}"

declare -A MOTIONS=(
  [motion1]="${MOTION_ROOT}/motion1.mp4"
  [motion2]="${MOTION_ROOT}/motion2.mp4"
  [motion5]="${MOTION_ROOT}/motion5.mp4"
  [zsy_dianzan_0306]="${MOTION_ROOT}/zsy_dianzan_0306.mp4"
  [zsy_mix_0306]="${MOTION_ROOT}/zsy_mix_0306.mp4"
  [zsy_say_hi_1_0306]="${MOTION_ROOT}/zsy_say_hi_1_0306.mp4"
)

trim_motion() {
  local motion_name="$1"
  local input_path="$2"
  local output_path="${TRIM_DIR}/${motion_name}_first${MAX_SECONDS}s.mp4"
  if [[ -s "${output_path}" ]]; then
    echo "${output_path}"
    return
  fi
  echo "[trim] ${motion_name}: first ${MAX_SECONDS}s -> ${output_path}" >&2
  ffmpeg -y -hide_banner -loglevel error \
    -i "${input_path}" \
    -t "${MAX_SECONDS}" \
    -an \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "${output_path}"
  echo "${output_path}"
}

for motion_name in motion1 motion2 motion5 zsy_dianzan_0306 zsy_mix_0306 zsy_say_hi_1_0306; do
  out_pose="${POSE_DIR}/fb01_${motion_name}_qilin_aligned_25fps.mp4"
  log_path="${LOG_DIR}/fb01_${motion_name}.log"
  if [[ -s "${out_pose}" ]]; then
    echo "[skip] ${motion_name}: ${out_pose}"
    continue
  fi
  echo "[extract] ${motion_name}"
  trimmed_motion="$(trim_motion "${motion_name}" "${MOTIONS[$motion_name]}")"
  python scripts/extract_qilin_aligned_dwpose_video.py \
    --input-video "${trimmed_motion}" \
    --ref-image "${REF}" \
    --output-video "${out_pose}" \
    --target-fps 25.0 \
    > "${log_path}" 2>&1
done

echo "[done] ${POSE_DIR}"
