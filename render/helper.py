from math import pi, atan2, fmod
import numpy as np
from pathlib import Path
import pickle


def _norm_angle(a):
    # normalize to [0, 2π)
    twopi = 2.0 * pi
    a = fmod(a, twopi)
    if a < 0.0:
        a += twopi
    return a

def _angle_diff(a2, a1):
    # signed shortest diff a2 - a1 in (-π, π]
    twopi = 2.0 * pi
    d = _norm_angle(a2) - _norm_angle(a1)
    if d > pi:
        d -= twopi
    if d <= -pi:
        d += twopi
    return d



# ---------------------------------------------------------------------------------------- # 

def point_is_close(p1, p2, tol=0.05):
    """Check if two 3D points are within tolerance."""
    return np.linalg.norm(np.array(p1) - np.array(p2)) < tol


def find_construction_lines(
    edge_features_list,
    cylinder_features_list,
    tmpt_edge_features_list,
    tmpt_cylinder_features_list
):
    new_edge_features = []
    new_cylinder_features = []

    # --- Edge features ---
    for tmpt_edge_feature in tmpt_edge_features_list:
        found = False
        for edge_feature in edge_features_list:
            # forward match
            if (
                point_is_close(tmpt_edge_feature[:3], edge_feature[:3])
                and point_is_close(tmpt_edge_feature[3:6], edge_feature[3:6])
            ):
                found = True
                break
            # reverse match
            if (
                point_is_close(tmpt_edge_feature[:3], edge_feature[3:6])
                and point_is_close(tmpt_edge_feature[3:6], edge_feature[:3])
            ):
                found = True
                break
        if not found:
            new_edge_features.append(tmpt_edge_feature)

    # --- Cylinder features ---
    for tmpt_cyl_feature in tmpt_cylinder_features_list:
        found = False
        for cyl_feature in cylinder_features_list:
            if all(point_is_close(tmpt_cyl_feature[i:i+3], cyl_feature[i:i+3]) for i in range(0, len(tmpt_cyl_feature), 3)):
                found = True
                break
        if not found:
            new_cylinder_features.append(tmpt_cyl_feature)

    return new_edge_features, new_cylinder_features



def find_label(parent_lines, child_edge_features_list, child_cylinder_features_list, label):
    """
    Assigns `label` to parent_lines that match with any child edge or cylinder features.
    
    Args:
        parent_lines (list): List of parent edge features (each is a sequence of 6 coordinates).
        child_edge_features_list (list): List of child edge features.
        child_cylinder_features_list (list): List of child cylinder features.
        label (int): Label to assign when a match is found.
    
    Returns:
        list: Labels for each parent line (default -1 if not matched).
    """

    # initialize all labels as -1
    labels = [-1] * len(parent_lines)

    # loop through all child features
    for child_edge in child_edge_features_list + child_cylinder_features_list:
        for i, edge_feature in enumerate(parent_lines):
            # forward match
            if (
                point_is_close(child_edge[:3], edge_feature[:3]) and
                point_is_close(child_edge[3:6], edge_feature[3:6])
            ):
                labels[i] = label
                continue  # move to next parent line

            # reverse match
            if (
                point_is_close(child_edge[:3], edge_feature[3:6]) and
                point_is_close(child_edge[3:6], edge_feature[:3])
            ):
                labels[i] = label
                continue

    return labels



def find_op_mapping(all_lines, tmpt_edge_features_list, tmpt_cylinder_features_list):

    # initialize all labels as -1
    op_usage = [0] * len(all_lines)

    # loop through all child features
    for child_edge in tmpt_edge_features_list + tmpt_cylinder_features_list:
        for i, edge_feature in enumerate(all_lines):
            # forward match
            if (
                point_is_close(child_edge[:3], edge_feature[:3]) and
                point_is_close(child_edge[3:6], edge_feature[3:6])
            ):
                op_usage[i] = 1
                continue  # move to next parent line

            # reverse match
            if (
                point_is_close(child_edge[:3], edge_feature[3:6]) and
                point_is_close(child_edge[3:6], edge_feature[:3])
            ):
                op_usage[i] = 1
                continue

    return op_usage


# ---------------------------------------------------------------------------------------- # 
def clean_cad_correspondance(cad_correspondance: np.ndarray) -> np.ndarray:
    """
    For each row in cad_correspondance (shape: num_lines, x),
    keep only the first 1 and set all subsequent 1s to 0.
    """
    cleaned = cad_correspondance.copy()
    for i in range(cleaned.shape[0]):
        row = cleaned[i]
        ones = np.where(row == 1)[0]
        if len(ones) > 1:
            # keep only the first 1
            row[ones[1:]] = 0
    return cleaned



# ---------------------------------------------------------------------------------------- # 

def save_strokes(current_folder, feature_lines, construction_lines):
    out_dir = Path(current_folder) / "output" / "strokes"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "feature_lines": feature_lines,
        "construction_lines": construction_lines,
    }

    path = out_dir / "all_strokes.pkl"
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return str(path)
