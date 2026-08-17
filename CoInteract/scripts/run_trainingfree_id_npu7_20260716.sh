#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES=7
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8

BASE_MODEL=/data1/Wan-AI/wan22_s2v
AUDIO_ENCODER=/data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large
LORA=/data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors
REF=/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg
ARCFACE=models/arcface/w600k_r50.onnx
RUN_TAG=20260716

mkdir -p logs

run_metric_and_sheet() {
  local out_dir="$1"
  local sample="$2"
  local method="$3"
  local video="${out_dir}/${sample}.mp4"
  if [[ ! -f "${video}" ]]; then
    echo "[warn] missing video for metric/sheet: ${video}"
    return 0
  fi
  python scripts/make_frame_sheet.py \
    --video "${video}" \
    --out "${out_dir}/${sample}_frames_0_25_50_75.jpg" \
    --frames 0,25,50,75
  python eval/id_drift_metric.py \
    --video "${video}" \
    --reference "${REF}" \
    --arcface_onnx "${ARCFACE}" \
    --every 5 \
    --out_csv "${out_dir}/${sample}_iddrift.csv" \
    2>&1 | tee "logs/${method}_${sample}_iddrift_${RUN_TAG}.log"
}

run_hjb_case() {
  local case_name="$1"
  local csv_path="$2"
  local out_dir=output_videos/p2v_hjb_isolated_${RUN_TAG}
  mkdir -p "${out_dir}"
  echo "[HJB] ${case_name}"
  python run_hjb_eval.py \
    --base_model_path "${BASE_MODEL}" \
    --audio_encoder_path "${AUDIO_ENCODER}" \
    --lora_path "${LORA}" \
    --csv_path "${csv_path}" \
    --data_base_path /data1/workspace/linxinliang/CoInteract \
    --output_dir "${out_dir}" \
    --height 832 \
    --width 480 \
    --num_frames 80 \
    --num_clips 1 \
    --num_inference_steps 40 \
    --cfg_scale 7.0 \
    --sigma_shift 7.0 \
    --reference_compose_mode stretch \
    --hjb_steps 4 \
    --hjb_decode_lat_frames 4 \
    --hjb_lr 0.03 \
    2>&1 | tee "logs/p2v_hjb_isolated_${case_name}_${RUN_TAG}.log"
  run_metric_and_sheet "${out_dir}" "th_fullbody_001_custom_${case_name}_noaudio_baseline" "p2v_hjb_isolated"
}

run_vguidance_case() {
  local scale="$1"
  local label="$2"
  local case_name="$3"
  local csv_path="$4"
  local out_dir="output_videos/p2v_vguidance_scale${label}_${RUN_TAG}"
  mkdir -p "${out_dir}"
  echo "[VGuidance scale=${scale}] ${case_name}"
  python batch_infer.py \
    --base_model_path "${BASE_MODEL}" \
    --audio_encoder_path "${AUDIO_ENCODER}" \
    --lora_path "${LORA}" \
    --csv_path "${csv_path}" \
    --data_base_path /data1/workspace/linxinliang/CoInteract \
    --output_dir "${out_dir}" \
    --height 832 \
    --width 480 \
    --num_frames 80 \
    --num_clips 1 \
    --num_inference_steps 40 \
    --cfg_scale 7.0 \
    --sigma_shift 7.0 \
    --reference_compose_mode stretch \
    --no_resize_output_to_reference \
    --face_reference_guidance_scale "${scale}" \
    --face_reference_guidance_power 1.0 \
    --face_reference_guidance_start_t 0.0 \
    --face_reference_guidance_end_t 0.9 \
    2>&1 | tee "logs/p2v_vguidance_scale${label}_${case_name}_${RUN_TAG}.log"
  run_metric_and_sheet "${out_dir}" "th_fullbody_001_custom_${case_name}_noaudio_baseline" "p2v_vguidance_scale${label}"
}

CASES=(
  "motion1:examples/th_fullbody_001_custom_motion1_noaudio_baseline_20260716.csv"
  "motion2:examples/th_fullbody_001_custom_motion2_noaudio_baseline_20260716.csv"
)

echo "==== Stage 2: HJB isolated on NPU7 ===="
for entry in "${CASES[@]}"; do
  case_name="${entry%%:*}"
  csv_path="${entry#*:}"
  run_hjb_case "${case_name}" "${csv_path}"
  sleep 10
done

echo "==== Stage 3: velocity face-reference guidance on NPU7 ===="
for scale_entry in "0.25:025" "0.5:05" "0.75:075"; do
  scale="${scale_entry%%:*}"
  label="${scale_entry#*:}"
  for entry in "${CASES[@]}"; do
    case_name="${entry%%:*}"
    csv_path="${entry#*:}"
    run_vguidance_case "${scale}" "${label}" "${case_name}" "${csv_path}"
    sleep 10
  done
done

echo "==== Done: training-free ID experiments ${RUN_TAG} ===="
