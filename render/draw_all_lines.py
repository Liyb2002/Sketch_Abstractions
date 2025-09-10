from pathlib import Path

import brep_read
import numpy as np
import os

import helper
import line_utils
import perturb_strokes

current_folder = Path.cwd().parent
files = (current_folder / "output").glob("final_*.step")

#SetUp:
#We have abstract format, where each line is 10 values
#We have sampled point format, where each line is 10 points, OR, a list of lines (each has 10 points)
#We have feature_lines and intermediate lines in abstract format
#We use line_utils to create more lines in abstract format
#do_perturb() will take abstract format to sampled point format. The number of lines in both format is the same


for filename in files:

    #0)Get file info
    x = filename.stem.split("_")[1]

    #1)Get the feature lines
    edge_features_list, cylinder_features_list = brep_read.sample_strokes_from_step_file(str(filename))  
    feature_lines = edge_features_list + cylinder_features_list


    #2)Get the intermediate lines
    history_dir = current_folder / "output" / "history"

    def hist_key(p: Path):
        something = p.stem.split("_", 1)[1]
        return (0, int(something)) if something.isdigit() else (1, something)

    history_files = sorted(history_dir.glob(f"{x}_*.step"), key=hist_key)

    intermediate_edge_features = []
    intermediate_cylinder_features = []

    for history_file in history_files:
        tmpt_edge_features_list, tmpt_cylinder_features_list = brep_read.sample_strokes_from_step_file(
            str(history_file)
        )

        new_edge_features, new_cylinder_features = helper.find_intermediate_lines(
            edge_features_list,
            cylinder_features_list,
            tmpt_edge_features_list,
            tmpt_cylinder_features_list)

        intermediate_edge_features += new_edge_features
        intermediate_cylinder_features += new_cylinder_features
        edge_features_list += new_edge_features
        cylinder_features_list += new_cylinder_features

    intermediate_lines = intermediate_edge_features + intermediate_cylinder_features


    #3)Get the construction lines
    projection_line = line_utils.projection_lines(feature_lines)
    projection_line += line_utils.derive_construction_lines_for_splines_and_spheres(feature_lines)
    bounding_box_line = line_utils.bounding_box_lines(feature_lines)


    perturbed_feature_lines = perturb_strokes.do_perturb(feature_lines)
    perturbed_construction_lines = perturb_strokes.do_perturb(intermediate_lines + projection_line + bounding_box_line)

    perturb_strokes.vis_perturbed_strokes(perturbed_feature_lines, perturbed_construction_lines)

    # helper.save_strokes(current_folder, feature_lines, construction_lines)
