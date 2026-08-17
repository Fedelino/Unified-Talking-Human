#!/usr/bin/env python3
"""Run unadjusted CoInteract full-body cases, one fresh process per sample."""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
CSV_PATH = ROOT / "examples/cointeract_fullbody4_unadjusted_motion1_motion2_20260720.csv"
MODE_TO_FACE_GUIDANCE = {
    "baseline": 0.0,
    "stagea30": 3.0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_TO_FACE_GUIDANCE), required=True)
    parser.add_argument("--npu", type=str, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()

    out_dir = ROOT / "output_videos" / f"fullbody4_unadjusted_{args.mode}_20260720"
    log_dir = ROOT / "logs" / f"fullbody4_unadjusted_{args.mode}_20260720"
    tmp_dir = ROOT / "tmp" / f"fullbody4_unadjusted_{args.mode}_20260720" / f"shard_{args.shard_index}_of_{args.shard_count}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(CSV_PATH.open(newline="")))
    selected = [(i, row) for i, row in enumerate(rows) if i % args.shard_count == args.shard_index]
    print(f"[runner] mode={args.mode} npu={args.npu} shard={args.shard_index}/{args.shard_count} selected={len(selected)}")
    sys.stdout.flush()

    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = args.npu
    env["TOKENIZERS_PARALLELISM"] = "false"
    face_guidance = MODE_TO_FACE_GUIDANCE[args.mode]

    for _, row in selected:
        sample_id = row["sample_id"]
        out_video = out_dir / f"{sample_id}.mp4"
        if out_video.exists() and out_video.stat().st_size > 0:
            print(f"[skip] {sample_id}: output exists")
            sys.stdout.flush()
            continue

        one_csv = tmp_dir / f"{sample_id}.csv"
        with one_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerow(row)

        log_path = log_dir / f"{sample_id}_npu{args.npu}.log"
        cmd = [
            "python", "batch_infer.py",
            "--csv_path", str(one_csv),
            "--output_dir", str(out_dir),
            "--height", "832",
            "--width", "480",
            "--num_frames", "80",
            "--num_clips", "1",
            "--num_inference_steps", "40",
            "--cfg_scale", "7.0",
            "--sigma_shift", "7.0",
            "--reference_compose_mode", "stretch",
            "--face_reference_guidance_scale", str(face_guidance),
        ]
        print(f"[run] {sample_id} frames=80 face_guidance={face_guidance} log={log_path}")
        sys.stdout.flush()
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"[done] {sample_id} status={proc.returncode}")
        sys.stdout.flush()
        time.sleep(8)


if __name__ == "__main__":
    main()
