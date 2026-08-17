#!/usr/bin/env python3
"""Run the existing Qilin-aligned DWPose extractor with CPU ONNXRuntime only.

The original extractor is kept untouched. This wrapper monkeypatches
onnxruntime provider discovery before executing it, avoiding intermittent
CANNExecutionProvider crashes during bulk pose extraction.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import onnxruntime as ort


ort.get_available_providers = lambda: ["CPUExecutionProvider"]

SCRIPT = Path("/data1/workspace/linxinliang/CoInteract/scripts/extract_qilin_aligned_dwpose_video.py")
runpy.run_path(str(SCRIPT), run_name="__main__")
