from pathlib import Path

import brep_read
import numpy as np


current_folder = Path.cwd().parent
filename = current_folder / "output" / "root.step"
edge_features_list, cylinder_features = brep_read.sample_strokes_from_step_file(str(filename))  # STEP reader expects a str

brep_read.vis_stroke_node_features(np.array(edge_features_list + cylinder_features))