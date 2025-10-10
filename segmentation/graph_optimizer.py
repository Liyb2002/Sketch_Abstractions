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
import numpy as np


import graph_utils

def run_once():
    input_dir = Path.cwd().parent / "input"
    ir_path   = input_dir / "sketch_program_ir_editted.json"

    # 1) Rescale & execute (this is your "old" assembly before the edit)
    exe_old = rescale_and_execute(input_dir, ir_path)  # Executor

    # 2) Load strokes
    perturbed_feature_lines, perturbed_construction_lines, feature_lines = load_perturbed_feature_lines(input_dir)

    # 3) Prepare geometric information
    global_thresh = graph_utils.compute_global_threshold(feature_lines)

    intersect_pairs = graph_utils.intersection_pairs(perturbed_feature_lines, feature_lines, global_thresh)

    perp_pairs = graph_utils.perpendicular_pairs(feature_lines, global_thresh)

    loops = graph_utils.find_planar_loops(feature_lines, global_thresh, angle_tol_deg=5.0)

    circle_cyl_pairs = graph_utils.find_entity_pairs(feature_lines, global_thresh)

    # 4) Initialize per stroke labels
    components = exe_old.primitives()  # executed cuboids (in order of execution)
    D = graph_utils.stroke_cuboid_distance_matrix(perturbed_feature_lines, components)
    C_init = graph_utils.distances_to_confidence(D, global_thresh)  # shape: (num_strokes, num_cuboids)
    anchor_idx_per_comp, anchor_mask = graph_utils.best_stroke_for_each_component(C_init, D)

    # 5) Propagate confidences (safer)
    C = graph_utils.propagate_confidences_safe(
        C_init=C_init,
        intersect_pairs=intersect_pairs,
        perp_pairs=perp_pairs,
        loops=loops,
        circle_cyl_pairs=circle_cyl_pairs,
        w_self=1.0,
        w_inter=0.1,
        w_perp=0.1,
        w_loop=1,
        w_circle_cyl=1,
        iters=10,
        alpha=0.75,
        use_trust=True,
        anchor_mask=anchor_mask,  # <-- freeze these rows
    )


    # plot_program_only(exe_old, use_offsets=False, use_scales=False)
    # graph_utils.plot_strokes_and_program(perturbed_feature_lines, components)

    # 6) Visualize initial vs propagated
    graph_utils.visualize_strokes_by_confidence(
        perturbed_feature_lines, C_init, components, title="Initial (from distances)"
    )
    graph_utils.visualize_strokes_by_confidence(
        perturbed_feature_lines, C, components, title="After propagation"
    )

    # graph_utils.print_confidence_preview(C, D, components, global_thresh, top_k=3, max_rows=5)
    # graph_utils.vis_stroke_node_features(feature_lines)
    # graph_utils.vis_perturbed_strokes(perturbed_feature_lines, perturbed_construction_lines)




run_once()