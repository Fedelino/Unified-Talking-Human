#!/usr/bin/env python3
"""Run Qilin-keypoint-aligned CoInteract cases, one fresh process per case."""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
CSV_PATH = ROOT / "examples/cointeract_fullbody4_qilin_keypoint_aligned_motion1_motion2_20260719.csv"
PROMPT_MODE_TO_SCALE = {
    "baseline": 0.0,
    "stagea30": 3.0,
}


def wait_for_file(path: Path, timeout_seconds: int) -> bool:
    start = time.time()
    while time.time() - start < timeout_seconds:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(30)
    return False


def num_frames_for_sample(sample_id: str) -> int:
    if "motion1" in sample_id:
        return 96
    if "motion2" in sample_id:
        return 76
    return 80


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(PROMPT_MODE_TO_SCALE), required=True)
    parser.add_argument("--npu", type=str, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--pose-wait-timeout", type=int, default=14400)
    args = parser.parse_args()

    out_dir = ROOT / "output_videos" / f"fullbody4_qilin_keypoint_aligned_{args.mode}_20260719"
    log_dir = ROOT / "logs" / f"fullbody4_qilin_keypoint_aligned_{args.mode}_20260719"
    tmp_dir = ROOT / "tmp" / f"fullbody4_qilin_keypoint_aligned_{args.mode}_20260719" / f"shard_{args.shard_index}_of_{args.shard_count}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(CSV_PATH.open(newline="")))
    selected = [(i, row) for i, row in enumerate(rows) if i % args.shard_count == args.shard_index]
    print(f"[runner] mode={args.mode} npu={args.npu} shard={args.shard_index}/{args.shard_count} selected={len(selected)}")
    sys.stdout.flush()

    face_scale = PROMPT_MODE_TO_SCALE[args.mode]
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = args.npu
    env["TOKENIZERS_PARALLELISM"] = "false"

    for _, row in selected:
        sample_id = row["sample_id"]
        pose_path = Path(row["pose_video"])
        out_video = out_dir / f"{sample_id}.mp4"
        if out_video.exists() and out_video.stat().st_size > 0:
            print(f"[skip] {sample_id}: output exists")
            sys.stdout.flush()
            continue

        print(f"[wait-pose] {sample_id}: {pose_path}")
        sys.stdout.flush()
        if not wait_for_file(pose_path, args.pose_wait_timeout):
            print(f"[missing-pose] {sample_id}: {pose_path}", flush=True)
            continue

        one_csv = tmp_dir / f"{sample_id}.csv"
        with one_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerow(row)

        frames = num_frames_for_sample(sample_id)
        log_path = log_dir / f"{sample_id}_npu{args.npu}.log"
        cmd = [
            "python", "batch_infer.py",
            "--csv_path", str(one_csv),
            "--output_dir", str(out_dir),
            "--height", "832",
            "--width", "480",
            "--num_frames", str(frames),
            "--num_clips", "1",
            "--num_inference_steps", "40",
            "--cfg_scale", "7.0",
            "--sigma_shift", "7.0",
            "--reference_compose_mode", "stretch",
            "--face_reference_guidance_scale", str(face_scale),
        ]
        print(f"[run] {sample_id} frames={frames} face_guidance={face_scale} log={log_path}")
        sys.stdout.flush()
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"[done] {sample_id} status={proc.returncode}")
        sys.stdout.flush()
        time.sleep(8)


if __name__ == "__main__":
    main()
