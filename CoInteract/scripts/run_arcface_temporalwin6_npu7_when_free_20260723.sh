#!/usr/bin/env bash
set -u

cd /data1/workspace/linxinliang/CoInteract || exit 1
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

mkdir -p logs
WATCH_LOG="logs/arcface_temporalwin6_npu7_watcher_20260723.log"
exec >> "${WATCH_LOG}" 2>&1

echo "[$(date '+%F %T')] watcher started"

get_npu7_hbm_mb() {
  npu-smi info | awk '
    $0 ~ /^\| 7[[:space:]]/ {
      getline
      if (match($0, /[0-9]+[[:space:]]*\/[[:space:]]*65536/)) {
        s = substr($0, RSTART, RLENGTH)
        sub(/[[:space:]]*\/.*/, "", s)
        gsub(/[[:space:]]/, "", s)
        print s
      }
    }
  ' | tail -n 1
}

while true; do
  used="$(get_npu7_hbm_mb)"
  if [ -n "${used}" ] && [ "${used}" -lt 6000 ]; then
    echo "[$(date '+%F %T')] NPU7 free enough: ${used} MB"
    break
  fi
  echo "[$(date '+%F %T')] waiting for NPU7, current HBM=${used:-unknown} MB"
  sleep 120
done

run_case() {
  local out_dir="$1"
  local steps="$2"
  local updates="$3"
  local min_updates="$4"
  local run_log="$5"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}/debug_arcface"
  echo "[$(date '+%F %T')] launching ${out_dir}, steps=${steps}, updates=${updates}"

  env ASCEND_RT_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 python batch_infer.py \
    --csv_path tmp/arcface_midstep_motion5_20260722/qalign_fb01_motion5.csv \
    --output_dir "${out_dir}" \
    --height 832 --width 480 --num_frames 92 --num_clips 1 \
    --num_inference_steps "${steps}" --cfg_scale 7.0 --sigma_shift 7.0 \
    --reference_compose_mode stretch --face_reference_guidance_scale 0.0 \
    --arcface_guidance_scale 0.005 \
    --arcface_guidance_timing post \
    --arcface_guidance_detector insightface \
    --arcface_guidance_target_cosine 0.70 \
    --arcface_guidance_max_updates "${updates}" \
    --arcface_guidance_min_updates "${min_updates}" \
    --arcface_guidance_decode_lat_frames 2 \
    --arcface_guidance_score_frame_count 3 \
    --arcface_guidance_temporal_windows 6 \
    --arcface_guidance_onnx_path models/arcface/w600k_r50.onnx \
    --arcface_guidance_debug_dir "${out_dir}/debug_arcface" \
    > "${run_log}" 2>&1

  echo "[$(date '+%F %T')] finished ${out_dir}"
}

SMOKE_OUT="output_videos/qalign_fb01_motion5_arcface_post_temporalwin6_s0005_smoke2_20260723c"
SMOKE_LOG="logs/qalign_fb01_motion5_arcface_post_temporalwin6_s0005_smoke2_20260723c.log"
run_case "${SMOKE_OUT}" 2 1 0 "${SMOKE_LOG}"

if [ ! -s "${SMOKE_OUT}/qalign_fb01_motion5.mp4" ] || [ ! -s "${SMOKE_OUT}/debug_arcface/guidance_scores.csv" ]; then
  echo "[$(date '+%F %T')] smoke failed; not launching full run"
  tail -n 120 "${SMOKE_LOG}" || true
  exit 2
fi

FULL_OUT="output_videos/qalign_fb01_motion5_arcface_post_temporalwin6_s0005_20260723"
FULL_LOG="logs/qalign_fb01_motion5_arcface_post_temporalwin6_s0005_20260723.log"
run_case "${FULL_OUT}" 40 2 1 "${FULL_LOG}"

if [ -s "${FULL_OUT}/qalign_fb01_motion5.mp4" ]; then
  echo "[$(date '+%F %T')] full run saved ${FULL_OUT}/qalign_fb01_motion5.mp4"
else
  echo "[$(date '+%F %T')] full run did not save MP4"
  tail -n 160 "${FULL_LOG}" || true
  exit 3
fi

echo "[$(date '+%F %T')] watcher complete"
