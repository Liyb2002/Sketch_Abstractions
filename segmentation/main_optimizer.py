#!/usr/bin/env python3
"""
main_optimizer.py

Rescale to strokes bbox, then differentiable-fit cuboid sizes to stroke samples,
then export STL and visualize the NEW assembly only.
"""

from __future__ import annotations
from pathlib import Path
import json

import shape_optimizer
from shape_optimizer import load_perturbed_feature_lines, plot_strokes_and_program, rescale_and_execute
from differentiable_fitter import run_differentiable_fit   # <— NEW
import stroke_mapping   # <— NEW


def run_once():
    input_dir = Path.cwd().parent / "input"
    out_stl   = input_dir / "sketch_model.stl"

    # 1) Rescale & get the updated executor (bbox alignment)
    exe = rescale_and_execute(input_dir)  # returns an Executor, but we'll refit anyway

    # 2) Load strokes (points + typed feature_lines)
    sample_points, feature_lines = load_perturbed_feature_lines(input_dir)
    # plot_strokes_and_program(exe, sample_points, feature_lines)

    # 3) Differentiable fitting: updates IR on disk and returns a fresh Executor
    exe = run_differentiable_fit(
        input_dir,
        sample_points,
        feature_lines,
        steps=1000,
        lr=1e-2,
    )

    # 4) Export STL and visualize the NEW (post-fit) assembly
    mesh = exe.to_trimesh()
    mesh.export(out_stl)
    print(f"✅ Wrote {out_stl}  (faces: {len(mesh.faces)}, verts: {len(mesh.vertices)})")

    print("📈 Plotting (after differentiable fit) ...")
    plot_strokes_and_program(exe, sample_points, feature_lines, True)
    shape_optimizer.compare_optimized_programs(exe)


    # 5) Stroke → cuboid mapping + visualization
    stroke_labels = stroke_mapping.stroke_to_cuboid_map(exe, sample_points)
    print(f"Stroke mapping: {stroke_labels}")
    # stroke_mapping.vis_stroke_mapping(sample_points, stroke_labels)
    stroke_mapping.vis_strokes_by_cuboid(sample_points, stroke_labels)

if __name__ == "__main__":
    run_once()
