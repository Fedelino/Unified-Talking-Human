#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/workspace/linxinliang/CoInteract
IA_ROOT=/data1/workspace/linxinliang/InteractAvatar
REF="$IA_ROOT/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
RAW="$IA_ROOT/InterDemo/custom_motion/raw/motion2.mp4"
ORIG="$IA_ROOT/InterDemo/custom_motion/dwpose/motion2_pose.mp4"
OUT_POSE="$IA_ROOT/InterDemo/identity_face_retarget_v2_20260828"
NPU="${NPU:-7}"
ONLY="${1:-}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd "$ROOT"

mkdir -p "$OUT_POSE" logs/p2v_geometry_face_control_v2_20260828

python scripts/id_face_retarget.py \
  --driving_video "$RAW" --reference_image "$REF" \
  --out_video "$OUT_POSE/motion2_blend05_pose.mp4" \
  --height 832 --width 480 --fps 25 --max_frames 80 --blend 0.5 \
  > logs/p2v_geometry_face_control_v2_20260828/prepare_motion2_blend05.log 2>&1

python scripts/id_face_retarget.py \
  --driving_video "$RAW" --reference_image "$REF" \
  --out_video "$OUT_POSE/motion2_blend075_pose.mp4" \
  --height 832 --width 480 --fps 25 --max_frames 80 --blend 0.75 \
  > logs/p2v_geometry_face_control_v2_20260828/prepare_motion2_blend075.log 2>&1

python scripts/id_face_retarget.py \
  --driving_video "$RAW" --reference_image "$REF" \
  --out_video "$OUT_POSE/motion2_blend05_mesh_pose.mp4" \
  --height 832 --width 480 --fps 25 --max_frames 80 --blend 0.5 \
  --mesh_overlay_alpha 0.45 --mesh_overlay_radius 1 \
  > logs/p2v_geometry_face_control_v2_20260828/prepare_motion2_blend05_mesh.log 2>&1

python scripts/id_face_retarget.py \
  --driving_video "$RAW" --reference_image "$REF" \
  --out_video "$OUT_POSE/motion2_blend075_mesh_pose.mp4" \
  --height 832 --width 480 --fps 25 --max_frames 80 --blend 0.75 \
  --mesh_overlay_alpha 0.45 --mesh_overlay_radius 1 \
  > logs/p2v_geometry_face_control_v2_20260828/prepare_motion2_blend075_mesh.log 2>&1

python scripts/make_pose_face_retarget_sheet_20260823.py \
  --original "$ORIG" \
  --retargeted "$OUT_POSE/motion2_blend05_pose.mp4" \
  --out "$OUT_POSE/motion2_orig_vs_blend05_sheet.jpg"

python scripts/make_pose_face_retarget_sheet_20260823.py \
  --original "$ORIG" \
  --retargeted "$OUT_POSE/motion2_blend05_mesh_pose.mp4" \
  --out "$OUT_POSE/motion2_orig_vs_blend05_mesh_sheet.jpg"

CMD=(python scripts/run_infer_config.py --config configs/p2v_geometry_face_control_v2_20260828.json --npu "$NPU")
if [[ -n "$ONLY" ]]; then
  CMD+=(--only "$ONLY")
fi
"${CMD[@]}"
