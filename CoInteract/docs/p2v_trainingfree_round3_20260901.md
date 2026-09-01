# CoInteract P2V Training-Free Round3

Round3 appends more identity-preservation trials after the existing 20260901
overnight queue completes. It keeps CoInteract frozen and avoids seed selection.

## Methods

- Yaw-gated identity face retargeting: applies reference-person 3D facial
  landmarks only on high-yaw frames, inspired by ViDS/PHiD/FantasyID.
- Safer delta-v face reference guidance: compares zero weak branch, latent blur,
  latent blur plus SAMG, and latent blur plus SAMG plus APG.
- Temporal postprocess: applies face-only restoration or color anchoring with a
  soft InsightFace mask, inspired by IP-FVR/Ouroboros-style consistency.
- Optional weight probe: records whether DECA, Stand-In, MagicMirror,
  CodeFormer, GFPGAN, or RestoreFormer are locally available. Missing methods
  are skipped rather than downloaded.

## Run

```bash
cd /data1/workspace/linxinliang/CoInteract
nohup bash scripts/run_p2v_trainingfree_round3_after_current_20260901.sh \
  > logs/p2v_trainingfree_overnight_20260901/round3_launcher.log 2>&1 &
```

## Outputs

- Videos: `/data1/workspace/linxinliang/CoInteract/output_videos/p2v_trainingfree_overnight_20260901/round3*`
- Evaluation: `/data1/workspace/linxinliang/CoInteract/output_videos/p2v_trainingfree_overnight_20260901_eval_round3`
- Logs: `/data1/workspace/linxinliang/CoInteract/logs/p2v_trainingfree_overnight_20260901/round3`
