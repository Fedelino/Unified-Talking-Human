#!/bin/bash
# OG CoInteract (pose) -> frequency-aligner identity pass, single case (ref1 / th_fullbody_001, motion1)
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar
cd /data1/workspace/linxinliang/CoInteract
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-2}
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
OUT=output_videos/og_freq_ref1_motion1_20260716
mkdir -p "$OUT"

# 1) OG CoInteract pose inference (correct config: checkpoint_pose @ 832x480, 40 steps) -> video + aligned pose
python batch_infer_qilin_og_m2v.py \
  --base_model_path /data1/Wan-AI/wan22_s2v \
  --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
  --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors \
  --csv_path /tmp/ref1_motion1.csv \
  --data_base_path /data1/workspace/linxinliang/CoInteract \
  --output_dir "$OUT" \
  --height 832 --width 480 --num_frames 80 --num_clips 1 --num_inference_steps 40 --cfg_scale 7.0
echo "OG_EXIT=$?"

VID="$OUT/th_fullbody_001_motion1custom.mp4"
AP="$OUT/th_fullbody_001_motion1custom__aligned_pose.mp4"
REFIMG=/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg
REFPOSE=/data1/workspace/linxinliang/InteractAvatar/InterDemo/custom_motion/ref_pose/motion1_firstframe_pose.png

echo "=== aligned pose present? ==="
ls -la "$AP" 2>/dev/null || echo "NO ALIGNED POSE FILE"

# 2) frequency-aligner identity pass (+ side-by-side compare: OG | freq)
python p2v_consisid_frequency_identity_postprocess.py \
  --input_video "$VID" \
  --aligned_pose "$AP" \
  --reference_image "$REFIMG" \
  --reference_pose "$REFPOSE" \
  --output_video "$OUT/th_fullbody_001_freq.mp4" \
  --compare_video "$OUT/th_fullbody_001_compare.mp4"
echo "FREQ_EXIT=$?"
echo "ALL_DONE"
