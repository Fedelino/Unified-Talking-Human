#!/usr/bin/env python3
"""Run fb01 Qilin-aligned TalkingHuman motion cases, one fresh process per case."""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
DEFAULT_CSV_PATH = ROOT / "examples/cointeract_qalign_fb01_talkinghuman_motions_20260721.csv"


def wait_for_file(path: Path, timeout_seconds: int) -> bool:
    start = time.time()
    while time.time() - start < timeout_seconds:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(20)
    return False


def pose_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 80
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(count, 1)


def num_frames_for_pose(path: Path) -> int:
    count = pose_frame_count(path)
    # Keep 480x832 within stable memory while still matching short motions well.
    capped = min(count, 96)
    snapped = max(40, ((capped - 1) // 4) * 4)
    return snapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu", type=str, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--pose-wait-timeout", type=int, default=21600)
    parser.add_argument("--face-reference-guidance-scale", type=float, default=0.0)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-name", type=str, default="qalign_fb01_talkinghuman_motions_baseline_20260721")
    args = parser.parse_args()

    out_dir = ROOT / "output_videos" / args.output_name
    log_dir = ROOT / "logs" / args.output_name
    tmp_dir = ROOT / "tmp" / args.output_name / f"shard_{args.shard_index}_of_{args.shard_count}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.csv_path.open(newline="")))
    selected = [(i, row) for i, row in enumerate(rows) if i % args.shard_count == args.shard_index]
    print(f"[runner] npu={args.npu} shard={args.shard_index}/{args.shard_count} selected={len(selected)}")
    sys.stdout.flush()

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

        frames = num_frames_for_pose(pose_path)
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
            "--num_frames", str(frames),
            "--num_clips", "1",
            "--num_inference_steps", "40",
            "--cfg_scale", "7.0",
            "--sigma_shift", "7.0",
            "--reference_compose_mode", "stretch",
            "--face_reference_guidance_scale", str(args.face_reference_guidance_scale),
        ]
        print(f"[run] {sample_id} frames={frames} guidance={args.face_reference_guidance_scale} log={log_path}")
        sys.stdout.flush()
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"[done] {sample_id} status={proc.returncode}")
        sys.stdout.flush()
        time.sleep(8)


if __name__ == "__main__":
    main()
