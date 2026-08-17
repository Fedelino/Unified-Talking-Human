#!/usr/bin/env python3
"""
Config-driven CoInteract experiment runner.

This keeps each inference variant reproducible and auditable:
- one fresh Python process per variant
- exact commands written to the log
- NPU selection lives in the config instead of a hand-written command

Example:
  python scripts/run_infer_config.py --config configs/p2v_id_noresize_ogpose_20260814.json
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument("--only", default="", help="Comma-separated variant names to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--npu", default=None, help="Override config NPU id, e.g. 7.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "variants" not in cfg or not isinstance(cfg["variants"], list):
        raise ValueError("Config must contain a list field: variants")
    return cfg


def add_cli_args(cmd: list[str], args_dict: dict | None, flags: list[str] | None = None):
    for key, value in (args_dict or {}).items():
        if value is None:
            continue
        cli_key = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(cli_key)
            continue
        cmd.extend([cli_key, str(value)])
    for flag in flags or []:
        flag = str(flag).strip()
        if flag:
            cmd.append(flag if flag.startswith("--") else f"--{flag}")
    return cmd


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def build_command(cfg: dict, variant: dict) -> list[str]:
    runner = variant.get("runner", "batch_infer")
    if runner == "batch_infer":
        args_dict = deepcopy(cfg.get("base_batch_args", {}))
        args_dict.update(variant.get("args", {}))
        flags = list(cfg.get("base_batch_flags", [])) + list(variant.get("flags", []))
        cmd = [sys.executable, cfg.get("batch_infer", "batch_infer.py")]
        return add_cli_args(cmd, args_dict, flags)
    if runner == "arcface_seed_selector":
        cmd = [sys.executable, cfg.get("arcface_seed_selector", "scripts/arcface_seed_selector.py")]
        return add_cli_args(cmd, variant.get("args", {}), variant.get("flags", []))
    raise ValueError(f"Unknown variant runner: {runner}")


def run_variant(cfg: dict, variant: dict, env: dict, dry_run: bool):
    name = variant.get("name")
    if not name:
        raise ValueError("Each variant must have a name.")
    cmd = build_command(cfg, variant)
    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"[run] {name}")
    print(f"[cmd] {shell_join(cmd)}")
    print(f"[log] {log_path}")
    if dry_run:
        return
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# variant: {name}\n")
        log.write(f"# command: {shell_join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Variant failed ({proc.returncode}): {name}. See {log_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cwd = cfg.get("cwd")
    if cwd:
        os.chdir(cwd)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.get("env", {}).items()})
    npu = args.npu if args.npu is not None else cfg.get("npu")
    if npu is not None and str(npu).strip() != "":
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu)
        env["ASCEND_VISIBLE_DEVICES"] = str(npu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    print(f"[config] {args.config}")
    print(f"[cwd] {os.getcwd()}")
    print(f"[npu] {env.get('ASCEND_RT_VISIBLE_DEVICES', '<env default>')}")

    for variant in cfg["variants"]:
        if only and variant.get("name") not in only:
            continue
        run_variant(cfg, variant, env, dry_run=args.dry_run)
    print("[done] all requested variants complete")


if __name__ == "__main__":
    main()
