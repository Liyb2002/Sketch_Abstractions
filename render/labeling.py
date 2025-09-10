from pathlib import Path
import os
import json
from typing import Dict, Any, Union, Tuple, List
import numpy as np


import line_utils
import perturb_strokes
import brep_read

NumberOrArray = Union[float, int, np.ndarray, str]

# ===========================
# Tree construction
# ===========================
def build_label_tree(label_dir: Path) -> Dict[str, Any]:
    label_dir = Path(label_dir)
    stems = [p.stem for p in label_dir.glob("*.step")]

    nested: Dict[str, Dict] = {}
    for stem in stems:
        parts = stem.split('-')
        cur = nested
        for part in parts:
            cur = cur.setdefault(part, {})

    nested.setdefault('0', {})

    def _sort_key(k: str):
        try:
            return int(k)
        except ValueError:
            return k

    def to_node(key: str, children_dict: Dict[str, Dict], prefix: str = "") -> Dict[str, Any]:
        full = key if prefix == "" else f"{prefix}-{key}"
        return {
            "name": f"{full}.step",
            "children": [
                to_node(k, v, prefix=full)
                for k, v in sorted(children_dict.items(), key=lambda kv: _sort_key(kv[0]))
            ],
        }

    return to_node('0', nested.get('0', {}))


# ===========================
# Dummy compute_value
# ===========================
def compute_value(step_path: Path) -> NumberOrArray:
    """
    For now: just return the node's filename as the "value".
    Later, replace this with heavy computation (e.g. volume, features, etc.).
    """
    #1)Get the feature lines
    edge_features_list, cylinder_features_list = brep_read.sample_strokes_from_step_file(str(step_path))  
    feature_lines = edge_features_list + cylinder_features_list

    #2)Get the construction lines
    projection_line = line_utils.projection_lines(feature_lines)
    projection_line += line_utils.derive_construction_lines_for_splines_and_spheres(feature_lines)
    

    perturbed_feature_lines = perturb_strokes.do_perturb(feature_lines)
    perturbed_construction_lines = perturb_strokes.do_perturb(projection_line)

    # perturb_strokes.vis_perturbed_strokes(perturbed_feature_lines, perturbed_construction_lines)

    return feature_lines, perturbed_feature_lines, perturbed_construction_lines


# ===========================
# Post-order aggregation
# ===========================
def compute_tree_values(
    node: Dict[str, Any],
    label_dir: Path,
    value_map: Dict[str, Any] = None,
) -> Dict[str, Any]:
    if value_map is None:
        value_map = {}

    name = node["name"]
    children = node.get("children", [])

    # ---- Leaf ----
    if not children:
        step_path = label_dir / name
        feature_lines, perturbed_feature_lines, perturbed_construction_lines = compute_value(step_path)

        result = {
            "features": feature_lines,
            "perturbed_features": perturbed_feature_lines,
            "perturbed_constructions": perturbed_construction_lines,
            "cuts": {
                "features": [0, len(feature_lines)],
                "perturbed_features": [0, len(perturbed_feature_lines)],
                "perturbed_constructions": [0, len(perturbed_construction_lines)],
            },
        }
        value_map[name] = result
        return result

    # ---- Internal node ----
    all_features, all_pfeatures, all_pconstructions = [], [], []
    cuts_features, cuts_pfeatures, cuts_pconstructions = [0], [0], [0]

    for child in children:
        child_res = compute_tree_values(child, label_dir, value_map=value_map)

        # extend features
        all_features.extend(child_res["features"])
        cuts_features.append(len(all_features))

        # extend perturbed features
        all_pfeatures.extend(child_res["perturbed_features"])
        cuts_pfeatures.append(len(all_pfeatures))

        # extend perturbed constructions
        all_pconstructions.extend(child_res["perturbed_constructions"])
        cuts_pconstructions.append(len(all_pconstructions))

    result = {
        "features": all_features,
        "perturbed_features": all_pfeatures,
        "perturbed_constructions": all_pconstructions,
        "cuts": {
            "features": cuts_features,
            "perturbed_features": cuts_pfeatures,
            "perturbed_constructions": cuts_pconstructions,
        },
    }
    value_map[name] = result
    return result


# ===========================
# Utilities
# ===========================
def visualize_tree_values(tree, value_map):
    """
    Visualize ONLY non-leaf nodes.
    For a node with K children, call vis_labeled_strokes K times (label_id = 0..K-1).
    Leaves (no children) are skipped.
    """
    name = tree["name"]
    children = tree.get("children", [])

    # Recurse first or last doesn't matter for rendering, but we keep parent-first here
    data = value_map.get(name)

    # Only visualize if this node has children AND we have its data
    if children and data is not None:
        feature_lines = data["features"]
        perturbed_feature_lines = data["perturbed_features"]
        perturbed_construction_lines = data["perturbed_constructions"]
        cuts = data["cuts"]  # expects "perturbed_features" and "perturbed_constructions"
        
        # bounding_box_line = line_utils.bounding_box_lines(feature_lines)
        # perturbed_bounding_box_line = perturb_strokes.do_perturb(bounding_box_line)
        # perturbed_construction_lines.extend(perturbed_bounding_box_line)

        # Number of child slices from prefix-sum cuts
        num_labels = len(cuts.get("perturbed_features", [])) - 1
        if num_labels > 0:
            print(f"Visualizing {name} with {num_labels} child groups...")
            for label_id in range(num_labels):
                perturb_strokes.vis_labeled_strokes(
                    perturbed_feature_lines,
                    perturbed_construction_lines,
                    cuts,
                    label_id,
                )

    # Recurse into children
    for child in children:
        visualize_tree_values(child, value_map)


# ===========================
# Example usage
# ===========================
if __name__ == "__main__":
    from pathlib import Path
    from typing import Dict

    # 1) Where your .step files live
    current_folder = Path.cwd().parent
    label_dir = current_folder / "output" / "seperable"

    # 2) Build the tree
    tree = build_label_tree(label_dir)

    # 3) Compute values for all nodes
    #    Each node's value is a tuple:
    #      (perturbed_feature_lines, perturbed_construction_lines)
    value_map: Dict[str, NumberOrArray] = {}
    root_value = compute_tree_values(tree, label_dir, value_map=value_map)

    # 4) Visualize every node
    visualize_tree_values(tree, value_map)
