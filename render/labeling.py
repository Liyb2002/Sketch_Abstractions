from pathlib import Path
import os
import json
from typing import Dict, Any, Union, Tuple, List
import numpy as np
import re


import line_utils
import perturb_strokes
import brep_read
import helper


NumberOrArray = Union[float, int, np.ndarray, str]

# ===========================
# Tree construction
# ===========================
def build_label_trees(label_dir: Path) -> List[Dict[str, Any]]:
    label_dir = Path(label_dir)
    stems = [p.stem for p in label_dir.glob("*.step")]

    nested: Dict[str, Dict] = {}
    for stem in stems:
        parts = stem.split('-')
        cur = nested
        for part in parts:
            cur = cur.setdefault(part, {})

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

    # Build a node for each top-level root (0, 1, 2, ...)
    return [
        to_node(root, nested[root])
        for root in sorted(nested.keys(), key=lambda k: _sort_key(k))
    ]


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
    step_path = label_dir / name

    # Compute this node's own values (for both leaf and internal)
    feature_lines, perturbed_feature_lines, perturbed_construction_lines = op_to_stroke(step_path)

    # ---- Leaf ----
    if not children:
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
    # start with this node's own contribution
    all_features = list(feature_lines)
    all_pfeatures = list(perturbed_feature_lines)
    all_pconstructions = list(perturbed_construction_lines)

    cuts_features = [0, len(all_features)]
    cuts_pfeatures = [0, len(all_pfeatures)]
    cuts_pconstructions = [0, len(all_pconstructions)]

    # then append each child's contribution
    for child in children:
        child_res = compute_tree_values(child, label_dir, value_map=value_map)

        all_features.extend(child_res["features"])
        cuts_features.append(len(all_features))

        all_pfeatures.extend(child_res["perturbed_features"])
        cuts_pfeatures.append(len(all_pfeatures))

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
# Computer Overview
# ===========================


def compute_overview(
    node: Dict[str, Any],
    label_dir: Path,
    value_map: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Overview pass:
      - If `node` has NO children: compute this node's own lines (leaf).
      - If `node` HAS children: TREAT AS ROOT for overview:
          * For each immediate child: compute child's own lines via op_to_stroke(...) and
            store them in value_map.
          * Aggregate those child results into the root's result.
      - Never recurse beyond one level (children never continue).
    """
    if value_map is None:
        value_map = {}

    name = node["name"]
    children = node.get("children", [])

    # ---- Leaf (no children): compute this node only ----
    if not children:
        step_path = label_dir / name
        feature_lines, perturbed_feature_lines, perturbed_construction_lines = op_to_stroke(step_path)
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

    # ---- Root (has children): compute each immediate child ONCE, no recursion ----
    all_features, all_pfeatures, all_pconstructions = [], [], []
    cuts_features, cuts_pfeatures, cuts_pconstructions = [0], [0], [0]

    for child in children:
        child_name = child["name"]
        step_path = label_dir / child_name

        c_feat, c_pfeat, c_pcon = op_to_stroke(step_path)

        # store child's own result
        child_result = {
            "features": c_feat,
            "perturbed_features": c_pfeat,
            "perturbed_constructions": c_pcon,
            "cuts": {
                "features": [0, len(c_feat)],
                "perturbed_features": [0, len(c_pfeat)],
                "perturbed_constructions": [0, len(c_pcon)],
            },
        }
        value_map[child_name] = child_result

        # aggregate into root
        all_features.extend(c_feat)
        cuts_features.append(len(all_features))

        all_pfeatures.extend(c_pfeat)
        cuts_pfeatures.append(len(all_pfeatures))

        all_pconstructions.extend(c_pcon)
        cuts_pconstructions.append(len(all_pconstructions))

    # root result is the concatenation of its children's results
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
# Operation to Stroke
# ===========================
def _load_operations_map(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def op_to_stroke(step_path):

    #1)Path Preparation
    ops_path = current_folder / "output" / "cad_operations.json"
    operations_map = _load_operations_map(ops_path)

    history_dir = current_folder / "output" / "history"

    # Extract stem (e.g., "0-0" from "0-0.step")
    step_path = Path(step_path)
    stem = step_path.stem  

    # Get operations for this step
    ops = operations_map.get(stem, [])

    # Collect related step files in history
    step_files = [
        p.name for p in history_dir.glob(f"{stem}*.step")
        # if "(detail)" not in p.name  # ignore "detail" files
    ]

    # Sort by the last (...) number
    step_files = sorted(
        step_files,
        key=lambda f: int(re.findall(r"\((\d+)\)", f)[-1])  # take last number
    )


    #2)Features Accumulation 
    edge_features_list = []
    cylinder_features_list = []
    feature_lines = []
    cut_off = []

    for step_file in step_files:
        file_path = history_dir / step_file
        tmpt_edge_features_list, tmpt_cylinder_features_list = brep_read.sample_strokes_from_step_file(
            str(file_path)
        )

        # Extend accumulators
        new_edge_features, new_cylinder_features = helper.find_intermediate_lines(
            edge_features_list,
            cylinder_features_list,
            tmpt_edge_features_list,
            tmpt_cylinder_features_list)
        
        edge_features_list += new_edge_features
        cylinder_features_list += new_cylinder_features
        feature_lines += new_edge_features
        feature_lines += new_cylinder_features

        cut_off.append(len(feature_lines))
    


    #3)Get the construction lines
    projection_line = line_utils.projection_lines(feature_lines)
    projection_line += line_utils.derive_construction_lines_for_splines_and_spheres(feature_lines)
    # bounding_box_line = line_utils.bounding_box_lines(feature_lines)


    perturbed_feature_lines = perturb_strokes.do_perturb(feature_lines)
    perturbed_construction_lines = perturb_strokes.do_perturb(projection_line)

    # for idx, _ in enumerate(ops):
    #     perturb_strokes.vis_Op_to_strokes(perturbed_feature_lines, perturbed_construction_lines, cut_off, idx, ops)

    return feature_lines, perturbed_feature_lines, perturbed_construction_lines




# ===========================
# Utilities
# ===========================
def visualize_tree_decomposition(tree, value_map):
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
        visualize_tree_decomposition(child, value_map)


def vis_components(tree, value_map):
    """
    For every node (leaf or non-leaf), if we have its data in value_map,
    call perturb_strokes.vis_perturbed_strokes(perturbed_feature_lines, perturbed_construction_lines).
    Then recurse into children.
    """
    name = tree["name"]
    children = tree.get("children", [])

    data = value_map.get(name)
    if data is not None:
        perturbed_feature_lines = data.get("perturbed_features", [])
        perturbed_construction_lines = data.get("perturbed_constructions", [])
        perturb_strokes.vis_perturbed_strokes(
            perturbed_feature_lines,
            perturbed_construction_lines,
        )

    # Recurse into children regardless
    for child in children:
        vis_components(child, value_map)



def vis_components_overview(tree: Dict[str, Any], value_map: Dict[str, Any]):
    """
    Visualize the overview for root + its immediate children (whatever compute_overview filled).
    No recursion; no computation.
    """
    root_name = tree["name"]

    # root view (aggregation of children)
    root_data = value_map.get(root_name)
    if root_data is not None:
        perturb_strokes.vis_perturbed_strokes(
            root_data.get("perturbed_features", []),
            root_data.get("perturbed_constructions", []),
        )

    # each immediate child view (its own .step only)
    for child in tree.get("children", []):
        c_name = child["name"]
        c_data = value_map.get(c_name)
        if c_data is None:
            continue
        perturb_strokes.vis_perturbed_strokes(
            c_data.get("perturbed_features", []),
            c_data.get("perturbed_constructions", []),
        )


# ===========================
# Example usage
# ===========================
if __name__ == "__main__":
    from pathlib import Path
    from typing import Dict

    # 1) Where your .step files live
    current_folder = Path.cwd().parent
    label_dir = current_folder / "output" / "seperable"

    # 2) Build the trees (one per root .step file)
    trees = build_label_trees(label_dir)

    for tree in trees:
        print("=" * 60)
        print(f"Processing tree rooted at {tree['name']}")

        # 3) Compute values for all nodes
        value_map: Dict[str, NumberOrArray] = {}
        _ = compute_tree_values(tree, label_dir, value_map=value_map)

        # 4) Visualize every non-leaf node
        visualize_tree_decomposition(tree, value_map)
        # vis_components(tree, value_map)

