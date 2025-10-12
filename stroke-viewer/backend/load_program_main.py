#!/usr/bin/env python3
"""
main_optimizer.py

Rescale to strokes bbox, then differentiable-fit cuboid sizes to stroke samples,
then export STL and visualize the NEW assembly only.
"""

from __future__ import annotations
from pathlib import Path
import json

from backend.shape_optimizer import  rescale_and_execute, plot_program_only
import numpy as np

def run_once():
    input_dir = Path.cwd().parent / "input"
    ir_path   = input_dir / "sketch_program_ir_editted.json"

    exe = rescale_and_execute(input_dir, ir_path)  # Executor

    # plot_program_only(exe, use_offsets=False, use_scales=False)


run_once()