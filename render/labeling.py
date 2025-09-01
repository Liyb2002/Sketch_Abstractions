from pathlib import Path
import os
import numpy as np

import brep_read
import helper


def build_label_tree(label_dir):
    label_dir = Path(label_dir)
    stems = [p.stem for p in label_dir.glob("*.step")]

    nested = {}
    for stem in stems:
        parts = stem.split('-')
        cur = nested
        for part in parts:
            cur = cur.setdefault(part, {})

    nested.setdefault('0', {})

    def to_node(key, children_dict, prefix=""):
        # numeric sort on immediate child keys
        def _key(k):
            try:
                return int(k)
            except ValueError:
                return k

        full = key if prefix == "" else f"{prefix}-{key}"
        return {
            "name": f"{full}.step",
            "children": [
                to_node(k, v, prefix=full)
                for k, v in sorted(children_dict.items(), key=lambda kv: _key(kv[0]))
            ],
        }

    return to_node('0', nested.get('0', {}))


def build_segementation(node, label_dir):
    
    child_names = [c["name"] for c in node.get("children", [])]
    if len(child_names) == 0:
        return 

    parent_name = node['name']
    print(f"{node['name']} -> {child_names}")
    parent_path = os.path.join(label_dir, parent_name)


    # read parent strokes 
    parent_edge_features_list, parent_cylinder_features_list = brep_read.sample_strokes_from_step_file(parent_path)  # STEP reader expects a str


    # read child strokes ensure we have all the strokes
    for child_name in child_names:
        child_path = os.path.join(label_dir, child_name)
        child_edge_features_list, child_cylinder_features_list = brep_read.sample_strokes_from_step_file(child_path)  # STEP reader expects a str
        
        new_edge_features, new_cylinder_features = helper.find_construction_lines(
            parent_edge_features_list,
            parent_cylinder_features_list,
            child_edge_features_list,
            child_cylinder_features_list)

        parent_edge_features_list += new_edge_features
        parent_cylinder_features_list += new_cylinder_features


    parent_lines = parent_edge_features_list + parent_cylinder_features_list


    labels = [-1] * len(parent_lines)
    # create labeling
    for child_name in child_names:
        # child_name has format a-b-c-...-Z.step → use the last value (Z) as child_label
        stem, ext = os.path.splitext(child_name)
        last_token = stem.split('-')[-1]
        child_label = int(last_token)

        child_path = os.path.join(label_dir, child_name)

        child_edge_features_list, child_cylinder_features_list = brep_read.sample_strokes_from_step_file(str(child_path))

        # find matches against parent_lines, producing per-parent labels (or -1 when no match)
        child_labels = helper.find_label(parent_lines, child_edge_features_list, child_cylinder_features_list, child_label)

        # merge: if child_labels[i] != -1, overwrite labels[i]
        for i, v in enumerate(child_labels):
            if v != -1:
                labels[i] = v

    brep_read.vis_labels(np.array(parent_lines), labels)



    
    for c in node.get("children", []):
        build_segementation(c, label_dir)



current_folder = Path.cwd().parent
label_dir = current_folder / "output" / "seperable"
tree = build_label_tree(label_dir)
build_segementation(tree, label_dir)
