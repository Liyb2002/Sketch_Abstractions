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
    is_overview: bool = False,
) -> Dict[str, Any]:
    if value_map is None:
        value_map = {}

    name = node["name"]
    children = node.get("children", [])
    step_path = label_dir / name

    # Pass is_overview flag into op_to_stroke
    feature_lines, perturbed_feature_lines, perturbed_construction_lines = op_to_stroke(
        step_path, is_overview=is_overview
    )

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
    all_features = list(feature_lines)
    all_pfeatures = list(perturbed_feature_lines)
    all_pconstructions = list(perturbed_construction_lines)

    if len(all_features) == 0:
        cuts_features = [0]
        cuts_pfeatures = [0]
        cuts_pconstructions = [0]
    else:
        cuts_features = [0, len(all_features)]
        cuts_pfeatures = [0, len(all_pfeatures)]
        cuts_pconstructions = [0, len(all_pconstructions)]

    for child in children:
        child_res = compute_tree_values(child, label_dir, value_map=value_map, is_overview=is_overview)

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
# Operation to Stroke
# ===========================
def _load_operations_map(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def op_to_stroke(step_path, is_overview = False):
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
    if is_overview:
        pattern = re.compile(rf"^{re.escape(stem)}\(overview\)\((\d+)\)\.step$")
        group_idx = 1
    else:
        pattern = re.compile(rf"^{re.escape(stem)}\((overview|detail)\)\((\d+)\)\.step$")
        group_idx = 2


    # Keep Path objects; filter by exact filename pattern
    candidates = []
    for p in history_dir.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                idx = int(m.group(group_idx))
                candidates.append((idx, p))

    # Sort by the numeric index and extract the Paths
    step_files = [p for _, p in sorted(candidates, key=lambda t: t[0])]

    #2)Features Accumulation 
    edge_features_list = []
    cylinder_features_list = []
    feature_lines = []
    cut_off = []

    overview_targets = {"sketch", "extrude", "sweep", "mirror"}

    for i, step_file in enumerate(step_files):
        # step_file is already an absolute Path inside history_dir
        file_path = step_file

        tmpt_edge_features_list, tmpt_cylinder_features_list = brep_read.sample_strokes_from_step_file(
            str(file_path)
        )

        # Extend accumulators
        new_edge_features, new_cylinder_features = helper.find_intermediate_lines(
            edge_features_list,
            cylinder_features_list,
            tmpt_edge_features_list,
            tmpt_cylinder_features_list,
        )

        edge_features_list += new_edge_features
        cylinder_features_list += new_cylinder_features

        # Decide whether to expose these features as "feature_lines"
        if is_overview:
            # ops is expected to have the same length as step_files, but be defensive
            current_op = ops[i] if i < len(ops) else None
            if current_op in overview_targets:
                feature_lines += new_edge_features
                feature_lines += new_cylinder_features
                cut_off.append(len(feature_lines))
        else:
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


def vis_overview_root(root: Dict[str, Any], value_map: Dict[str, Any]) -> None:
    """
    Visualize only the root node's perturbed strokes (features + constructions).
    No recursion, no labels, no cuts.
    """
    name = root["name"]
    data = value_map.get(name)
    if data is None:
        print(f"[vis_overview_root] No data for {name}")
        return

    perturbed_feature_lines = data.get("perturbed_features", [])
    perturbed_construction_lines = data.get("perturbed_constructions", [])

    perturb_strokes.vis_perturbed_strokes(
        perturbed_feature_lines,
        perturbed_construction_lines,
    )



# ===========================
# Example usage
# ===========================
if __name__ == "__main__":
    from pathlib import Path
    from typing import Dict
    import copy

    # 1) Where your .step files live
    current_folder = Path.cwd().parent
    label_dir = current_folder / "output" / "seperable"

    # 2) Build the trees (one per root .step file)
    trees = build_label_trees(label_dir)

    for tree in trees:
        print("=" * 60)
        print(f"Processing tree rooted at {tree['name']}")

        tree_overview = copy.deepcopy(tree)
        # 3) Compute values for all nodes (recursive, normal mode)
        value_map: Dict[str, NumberOrArray] = {}
        _ = compute_tree_values(tree, label_dir, value_map=value_map)

        # 4) (Optional) Visualizations for the full pass
        visualize_tree_decomposition(tree, value_map)
        # vis_components(tree, value_map)

        # 5) Overview-only pass: copy the tree + use a separate value_map
        overview_value_map: Dict[str, NumberOrArray] = {}
        _ = compute_tree_values(tree_overview, label_dir, value_map=overview_value_map, is_overview=True)
        vis_overview_root(tree_overview, overview_value_map)
