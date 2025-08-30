from pathlib import Path

import brep_read
import numpy as np
import os

import helper


current_folder = Path.cwd().parent
filename = current_folder / "output" / "root.step"
edge_features_list, cylinder_features_list = brep_read.sample_strokes_from_step_file(str(filename))  # STEP reader expects a str
feature_lines = edge_features_list + cylinder_features_list



# Now we need to get all the construction lines
history_dir = current_folder / "output" / "history"
files = sorted(
    [f for f in os.listdir(history_dir) if f.endswith(".step")],
    key=lambda x: int(x.split(".")[0])
)
construction_edge_features = []
construction_cylinder_features = []

for file in files:
    tmpt_edge_features_list, tmpt_cylinder_features_list = brep_read.sample_strokes_from_step_file(
        str(os.path.join(history_dir, file))
    )

    new_edge_features, new_cylinder_features = helper.find_construction_lines(
        edge_features_list,
        cylinder_features_list,
        tmpt_edge_features_list,
        tmpt_cylinder_features_list)

    construction_edge_features += new_edge_features
    construction_cylinder_features += new_cylinder_features
    edge_features_list += new_edge_features
    cylinder_features_list += new_cylinder_features


brep_read.vis_stroke_node_features(np.array(edge_features_list + cylinder_features_list))