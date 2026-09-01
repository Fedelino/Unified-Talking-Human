# P2V Training-Free Overnight Queue

This queue tests training-free identity methods without seed selection:

- Reference-face preprocessing: OpenCV masked face CLAHE/sharpen variants, because CodeFormer/GFPGAN/RestoreFormer were not installed in the inspected workspace.
- Face-only reference guidance: conservative reference K/V head-region guidance at `0.025`, `0.05`, and `0.075`.
- PHiD/FantasyID-lite face proxy: light identity-face DWPose blends at `0.15`, `0.25`, and `0.35`.
- Face-region post-restoration: post-generation masked face enhancement at `0.25`, `0.40`, and `0.55`.

All CoInteract P2V runs use one full-frame reference image, no audio, no product image, `480x832`, `80` frames, `40` steps, `cfg=7`, `sigma_shift=7`, and `reference_compose_mode=stretch`.

Run:

```bash
cd /data1/workspace/linxinliang/CoInteract
NPUS=auto nohup scripts/run_p2v_trainingfree_overnight_20260901.sh \
  > logs/p2v_trainingfree_overnight_20260901/nohup.log 2>&1 &
```

Outputs:

- Videos: `output_videos/p2v_trainingfree_overnight_20260901`
- Metrics and frame sheets: `output_videos/p2v_trainingfree_overnight_20260901_eval`
- Logs: `logs/p2v_trainingfree_overnight_20260901`
