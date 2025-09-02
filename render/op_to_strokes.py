import pickle
import numpy as np
from pathlib import Path
import os

import brep_read
import helper


# get current folder
current_folder = Path.cwd().parent
all_strokes_file_path = current_folder / "output" / "strokes" / "all_strokes.pkl"

# load the pickle file
with open(all_strokes_file_path, "rb") as f:
    data = pickle.load(f)

# extract strokes
feature_lines = data.get("feature_lines", [])
construction_lines = data.get("construction_lines", [])
all_lines = feature_lines + construction_lines

# Now read the history strokes
history_dir = current_folder / "output" / "history"
files = sorted(
    [f for f in os.listdir(history_dir) if f.endswith(".step")],
    key=lambda x: int(x.split(".")[0])
)

cad_correspondance = np.zeros((len(all_lines), len(files)))

for i, file in enumerate(files):
    tmpt_edge_features_list, tmpt_cylinder_features_list = brep_read.sample_strokes_from_step_file(
        str(os.path.join(history_dir, file))
    )

    op_usage = helper.find_op_mapping(
        all_lines,
        tmpt_edge_features_list,
        tmpt_cylinder_features_list
    )

    # put op_usage into the i-th column
    cad_correspondance[:, i] = np.array(op_usage)

cad_correspondance = helper.clean_cad_correspondance(cad_correspondance)

for col in range(cad_correspondance.shape[1]):
    count = np.sum(cad_correspondance[:, col] == 1)
    print(f"Column {col}: {count} ones")
    brep_read.vis_cad_op(np.array(all_lines), cad_correspondance, col)
