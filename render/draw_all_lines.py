from pathlib import Path

import brep_read
import numpy as np
import os

import helper
import line_utils
import perturb_strokes

current_folder = Path.cwd().parent
filename = current_folder / "output" / "root.step"
edge_features_list, cylinder_features_list = brep_read.sample_strokes_from_step_file(str(filename))  # STEP reader expects a str
feature_lines = edge_features_list + cylinder_features_list

mating_files = list((current_folder / "output").glob("mated_*.step"))

if mating_files:
    for mating_filename in mating_files:
        mated_edge_features_list, mated_cylinder_features_list = brep_read.sample_strokes_from_step_file(str(mating_filename))
        # brep_read.vis_stroke_node_features(np.array(mated_edge_features_list + mated_cylinder_features_list))




# Now we need to get all the intermediate lines
history_dir = current_folder / "output" / "history"
files = sorted(
    [f for f in os.listdir(history_dir) if f.endswith(".step")],
    key=lambda x: int(x.split(".")[0])
)
intermediate_edge_features = []
intermediate_cylinder_features = []

for file in files:
    tmpt_edge_features_list, tmpt_cylinder_features_list = brep_read.sample_strokes_from_step_file(
        str(os.path.join(history_dir, file))
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



projection_line = line_utils.projection_lines(feature_lines)
bounding_box_line = line_utils.bounding_box_lines(feature_lines)



perturbed_feature_lines = perturb_strokes.do_perturb(feature_lines)
perturbed_construction_lines = perturb_strokes.do_perturb(intermediate_lines + projection_line + bounding_box_line)

perturb_strokes.vis_perturbed_strokes(perturbed_feature_lines, perturbed_construction_lines)


helper.save_strokes(current_folder, feature_lines, construction_lines)

# brep_read.vis_stroke_node_features(np.array(edge_features_list + cylinder_features_list))