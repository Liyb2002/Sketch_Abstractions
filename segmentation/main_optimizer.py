#!/usr/bin/env python3
"""
main_optimizer.py

Now delegates all rescaling to rescaling_optimizer.rescale_and_execute().
Then exports STL and visualizes the NEW (post-rescale) executor only.

Fixed I/O:
  Reads/Writes in ../input/
"""

from __future__ import annotations
from pathlib import Path
import json

from rescaling_optimizer import rescale_and_execute
from shape_optimizer import load_perturbed_feature_lines, plot_strokes_and_program


def run_once():
    input_dir = Path.cwd().parent / "input"
    out_stl   = input_dir / "sketch_model.stl"

    # 1) Rescale & get the updated executor
    exe = rescale_and_execute(input_dir)

    # 2) Export STL for the NEW exe
    mesh = exe.to_trimesh()
    mesh.export(out_stl)
    print(f"✅ Wrote {out_stl}  (faces: {len(mesh.faces)}, verts: {len(mesh.vertices)})")

    # 3) Visualize NEW exe with strokes
    sample_points, feature_lines = load_perturbed_feature_lines(input_dir)
    print("📈 Plotting (after rescale) ...")
    plot_strokes_and_program(exe, sample_points, feature_lines)


if __name__ == "__main__":
    run_once()
