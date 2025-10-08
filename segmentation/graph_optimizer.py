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
from shape_optimizer import load_perturbed_feature_lines, plot_strokes_and_program, rescale_and_execute, plot_program_only
import stroke_mapping   # <— NEW
from program_update import write_updated_program
from program_executor import Executor


import graph_utils

def run_once():
    input_dir = Path.cwd().parent / "input"
    ir_path   = input_dir / "sketch_program_ir_editted.json"

    # 1) Rescale & execute (this is your "old" assembly before the edit)
    exe_old = rescale_and_execute(input_dir, ir_path)  # Executor

    # 2) Load strokes
    perturbed_feature_lines, perturbed_construction_lines, feature_lines = load_perturbed_feature_lines(input_dir)
    sampled_points = perturbed_feature_lines + perturbed_construction_lines

    # 3) Prepare geometric information
    global_thresh = graph_utils.compute_global_threshold(feature_lines)

    intersect_pairs = graph_utils.intersection_pairs(sampled_points, feature_lines, global_thresh)

    perp_pairs = graph_utils.perpendicular_pairs(feature_lines, global_thresh)
    print("perp_pairs", perp_pairs)



    # plot_program_only(exe_old, use_offsets=False, use_scales=False)
    # graph_utils.vis_stroke_node_features(feature_lines)
    # graph_utils.vis_perturbed_strokes(perturbed_feature_lines, perturbed_construction_lines)













    # 3) Pick the first component (prototype) in program order
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    first_name = ir["program"]["cuboids"][0]["var"]

    # Find that prototype instance in the executed assembly
    prims = exe_old.primitives()
    comp = next(p for p in prims if p.name == first_name)
    print(f"[seed] component='{comp.name}' origin={comp.origin} size={comp.size}")

    # 4) Build mask of strokes near this component
    keep_idxs, mask = stroke_mapping.strokes_near_cuboid(
        sample_points=sample_points,
        comp=comp,
        thresh=0.5,
    )
    print(f"[mapping] strokes near '{comp.name}': {len(keep_idxs)} / {len(sample_points)}")

    # 5) Visualize selection (red = selected, black = others; no axes; equal scaling)
    shape_optimizer.plot_strokes_and_program(
        executor=exe_old,
        sample_points=sample_points,
        feature_lines=feature_lines,
        use_optimized=False  # just the current program
    )

    stroke_mapping.plot_stroke_selection(
        sample_points=sample_points,
        mask=mask,
        save_path=input_dir / f"selection_{comp.name}.png",
        show=True
    )

    # 6) Compute new cuboid params from selected strokes
    new_origin, new_size = stroke_mapping.component_params_from_selected_strokes(
        sample_points=sample_points,
        mask=mask,
        margin=0.0,
        min_size_eps=1e-6
    )
    print(f"[fit] new params for '{comp.name}': origin={new_origin.tolist()} size={new_size.tolist()}")

    # 7) Write updated IR with parent-preserving attach
    #    (pass old placements so parent-side fractions are computed correctly)
    old_min_corners = {p.name: p.origin.copy() for p in exe_old.primitives()}
    new_ir_path = write_updated_program(
        ir_path=ir_path,
        cuboid_name=comp.name,
        new_origin=new_origin,
        new_size=new_size,
        old_min_corners=old_min_corners,   # <-- important
    )
    print(f"[ir] wrote updated program -> {new_ir_path}")

    # 8) Execute new IR and overlay old vs new
    ir_new = json.loads(new_ir_path.read_text(encoding="utf-8"))
    exe_new = Executor(ir_new)

    stroke_mapping.plot_programs_overlay(
        exe_old=exe_old,
        exe_new=exe_new,
        save_path=input_dir / f"program_overlay_{comp.name}.png",
        show=True
    )


if __name__ == "__main__":
    run_once()
