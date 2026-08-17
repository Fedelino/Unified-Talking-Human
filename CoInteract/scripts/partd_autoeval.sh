#!/usr/bin/env bash
# Overnight auto-eval for Part-D isolated finetunes.
# After a soak, evaluate each method's LATEST checkpoint: run a P2V inference on the
# full-body case, then the ArcFace ID-drift metric. Collect into one results table.
set -u
cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh

RESULTS=logs/partD_autoeval_results_20260713.txt
REF=/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg
CASE=examples/cointeract_fullbody_case1_motion2_aligned_20260630.csv
EVAL_DEV=0            # free card for eval inference
SOAK_SECONDS=21600    # 6h soak so checkpoints mature (~step 800-1000)

echo "==== Part-D auto-eval started $(date) ; soaking ${SOAK_SECONDS}s ====" > "$RESULTS"
sleep "$SOAK_SECONDS"
echo "==== soak done $(date); evaluating latest checkpoints ====" >> "$RESULTS"

# method -> expert_hidden_dim (must match training)
declare -A EXP=( [mD_cointeract_lora]=256 [mD_lora]=256 [mD_cointeract]=512 )

for m in mD_lora mD_cointeract mD_cointeract_lora; do
  ckpt=$(ls -t output/$m/version_*/step-*.safetensors 2>/dev/null | head -1)
  echo "" >> "$RESULTS"; echo "########## $m ##########" >> "$RESULTS"
  if [ -z "$ckpt" ]; then echo "  NO CHECKPOINT FOUND" >> "$RESULTS"; continue; fi
  echo "  ckpt=$ckpt  expert_hidden_dim=${EXP[$m]}  $(date)" >> "$RESULTS"
  outdir=output_videos/partD_eval_${m}_20260713
  mkdir -p "$outdir"
  # --- inference (cointeract env) ---
  conda activate cointeract
  ASCEND_RT_VISIBLE_DEVICES=$EVAL_DEV PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    python batch_infer.py \
      --base_model_path /data1/Wan-AI/wan22_s2v \
      --audio_encoder_path ./models/chinese-wav2vec2-large \
      --lora_path "$ckpt" --expert_hidden_dim ${EXP[$m]} \
      --csv_path "$CASE" --output_dir "$outdir" \
      --height 480 --width 320 --num_frames 80 --num_inference_steps 30 \
      >> logs/partD_eval_${m}_infer.log 2>&1
  vid=$(ls -t "$outdir"/*.mp4 2>/dev/null | head -1)
  if [ -z "$vid" ]; then echo "  INFERENCE PRODUCED NO VIDEO (see logs/partD_eval_${m}_infer.log)" >> "$RESULTS"; continue; fi
  # --- ID metric (facetools env) ---
  conda activate facetools
  python eval/id_drift_insightface.py --tag "$m" --video "$vid" --reference "$REF" --every 5 \
    2>/dev/null | grep -E "cos_ref|drift|frames_with_face" >> "$RESULTS"
done

echo "" >> "$RESULTS"
echo "==== BASELINES for reference ====" >> "$RESULTS"
echo "  Stage-1 pose : cos_ref mean=0.297  drift=-0.052" >> "$RESULTS"
echo "  Stage-2 face : cos_ref mean=0.242  drift=+0.022" >> "$RESULTS"
echo "  (Option B TIA2V, different case: 0.856)" >> "$RESULTS"
echo "==== Part-D auto-eval DONE $(date) ====" >> "$RESULTS"
