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
import stroke_mapping   # <— NEW
from program_update import write_updated_program
from program_executor import Executor


def run_once():
    input_dir = Path.cwd().parent / "input"
    out_stl   = input_dir / "sketch_model.stl"

    # 1) Rescale & get the updated executor (bbox alignment)
    exe = rescale_and_execute(input_dir)  # returns an Executor, but we'll refit anyway

    # 2) Load strokes (points + typed feature_lines)
    sample_points, feature_lines = load_perturbed_feature_lines(input_dir)

    # 3) Pick the first component (prototype) in program order
    ir_path = input_dir / "sketch_program_ir.json"
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    first_name = ir["program"]["cuboids"][0]["var"]  # sequential order, skip 'bbox'

    # Find that exact prototype instance in the executed assembly
    prims = exe.primitives()  # analytic cuboids with world origin & size
    comp = next(p for p in prims if p.name == first_name)

    # 4) Build mask of strokes near this component  —— NEW
    keep_idxs, mask = stroke_mapping.strokes_near_cuboid(
        sample_points=sample_points,
        comp=comp,
        thresh=0.5,     # tune as needed; uses same world units as the executor
    )
    print(f"[mapping] strokes near '{comp.name}': {len(keep_idxs)} / {len(sample_points)}")


    # 5) Visualize selection
    # stroke_mapping.plot_stroke_selection(
    #     sample_points=sample_points,
    #     mask=mask,
    #     save_path=input_dir / f"selection_{comp.name}.png",  # optional: save image
    #     show=True
    # )

    # 6) Compute new cuboid params from selected strokes  —— NEW
    new_origin, new_size = stroke_mapping.component_params_from_selected_strokes(
        sample_points=sample_points,
        mask=mask,
        margin=0.0,       # add padding if needed
        min_size_eps=1e-6
    )


    # 7) Write updated IR with this component edited  —— NEW
    ir_path = input_dir / "sketch_program_ir.json"
    new_ir_path = write_updated_program(
        ir_path=ir_path,
        cuboid_name=comp.name,
        new_origin=new_origin,
        new_size=new_size
    )


    # 8) Load the new program and vis

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
