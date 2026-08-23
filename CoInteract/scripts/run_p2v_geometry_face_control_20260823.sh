#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/workspace/linxinliang/CoInteract
IA_ROOT=/data1/workspace/linxinliang/InteractAvatar
REF="$IA_ROOT/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
OUT_POSE="$IA_ROOT/InterDemo/identity_face_retarget_20260823"
NPU="${NPU:-7}"
ONLY="${1:-}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd "$ROOT"

mkdir -p "$OUT_POSE" logs/p2v_geometry_face_control_20260823

python scripts/id_face_retarget.py \
  --driving_video "$IA_ROOT/InterDemo/custom_motion/raw/motion1.mp4" \
  --reference_image "$REF" \
  --out_video "$OUT_POSE/motion1_idface_pose.mp4" \
  --height 832 --width 480 --fps 25 --max_frames 80 --blend 1.0 \
  > logs/p2v_geometry_face_control_20260823/prepare_motion1_idface_pose.log 2>&1

python scripts/id_face_retarget.py \
  --driving_video "$IA_ROOT/InterDemo/custom_motion/raw/motion2.mp4" \
  --reference_image "$REF" \
  --out_video "$OUT_POSE/motion2_idface_pose.mp4" \
  --height 832 --width 480 --fps 25 --max_frames 80 --blend 1.0 \
  > logs/p2v_geometry_face_control_20260823/prepare_motion2_idface_pose.log 2>&1

python scripts/make_pose_face_retarget_sheet_20260823.py \
  --original "$IA_ROOT/InterDemo/custom_motion/dwpose/motion1_pose.mp4" \
  --retargeted "$OUT_POSE/motion1_idface_pose.mp4" \
  --out "$OUT_POSE/motion1_orig_vs_idface_sheet.jpg"

python scripts/make_pose_face_retarget_sheet_20260823.py \
  --original "$IA_ROOT/InterDemo/custom_motion/dwpose/motion2_pose.mp4" \
  --retargeted "$OUT_POSE/motion2_idface_pose.mp4" \
  --out "$OUT_POSE/motion2_orig_vs_idface_sheet.jpg"

CMD=(python scripts/run_infer_config.py --config configs/p2v_geometry_face_control_20260823.json --npu "$NPU")
if [[ -n "$ONLY" ]]; then
  CMD+=(--only "$ONLY")
fi
"${CMD[@]}"
