import json
from pathlib import Path
import component
import random
import os 
import helper

def load_component(json_path, output_path, parent=None, labels = [0]):
    """
    Reads JSON file, picks a variant if any, and returns a Component instance.
    """
    with open(json_path, 'r') as f:
        comp_dict = json.load(f)

    # If component has variants, choose one
    if 'variants' in comp_dict:
        import random
        variant = random.choice(comp_dict['variants'])
        data = {
            'name': variant.get('name', comp_dict['name']),
            'boolean' : variant['boolean'],
            'parameters': variant['parameters'],
            'location': variant.get('location'),
            'cad_operations': variant['cad_operations'],
            'quantity': variant.get("quantity", 1),
            'locations': variant.get("locations"),
            'condition': variant.get('condition', 'None'),
            'path' : variant.get('path', []),
            'mating_reference' : variant.get('mating_reference', None),
            'detail_type' : variant.get('detail_type', 'overview'),
            'extrude_dir' : variant.get('extrude_dir', 'z'),
        }
    else:
        data = {
            'name': comp_dict['name'],
            'boolean' : variant['boolean'],
            'parameters': comp_dict['parameters'],
            'location': comp_dict.get('location'),
            'cad_operations': comp_dict['cad_operations'],
            'quantity': comp_dict.get("quantity", 1),
            'locations': comp_dict.get("locations"),
            'detail_type' : variant.get('detail_type', 'overview')
        }

    c = component.Component(data, parent=parent, labels = labels, output_path = output_path)
    return c


def read_macro(folder_path, output_folder, idx = 0):
    folder_path = Path(folder_path)

    # Recursive loader for children
    def load_with_children(name, parent=None, labels = [idx]):
        json_file = folder_path / f"{name}.json"
        comp = load_component(json_file, output_folder, parent=parent, labels= labels)

        with open(json_file, 'r') as f:
            comp_dict = json.load(f)

        child_names = comp_dict.get("children", [])
        for idx, child_name in enumerate(child_names):
            new_labels = labels.copy()  
            new_labels.append(idx)
            child_comp = load_with_children(child_name, parent=comp, labels=new_labels)
            comp.children.append(child_comp)

        return comp

    # Always start from root.json
    root_component = load_with_children("root")
    macro_name = root_component.name

    return macro_name, root_component


def read_matings(macro_path, output_path):
    # Loop through each subfolder inside macro_path

    seperable_level_count = 1
    for subfolder in os.listdir(macro_path):
        subfolder_path = os.path.join(macro_path, subfolder)

        # Only process directories
        if not os.path.isdir(subfolder_path):
            continue

        # Check if {subfolder}.json exists in output_path
        json_file = os.path.join(output_path, f"{subfolder}.json")
        if os.path.exists(json_file):
            print(f"Executing {subfolder}...")

            # Step 1: Read macro
            macro_name, root_component = read_macro(subfolder_path, output_path, seperable_level_count)

            # Step 2: Execute
            execute(root_component, output_path)
        
        seperable_level_count += 1
    
    for name in os.listdir(output_path):
        if name.startswith("mated") and name.endswith(".json"):
            file_path = os.path.join(output_path, name)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Warning: could not remove {file_path}: {e}")




def execute(component_obj, output_path):
    """
    Execute a Component and all its children recursively.
    Export result to .stl and .step
    Append cad_op_history to cad_operations.json if it exists.
    """
    output_path.mkdir(exist_ok=True)

    canvas, _, cad_op_history = component_obj.build()
    stl_path = output_path / f"final_{component_obj.labels[0]}.stl"
    step_path = output_path / f"final_{component_obj.labels[0]}.step"
    cad_operations_file = output_path / "cad_operations.json"

    helper.func_export_stl(canvas, str(stl_path))
    helper.func_export_step(canvas, str(step_path))

    # --- load existing JSON if available ---
    if cad_operations_file.exists():
        with open(cad_operations_file, "r", encoding="utf-8") as f:
            try:
                existing_ops = json.load(f)
            except json.JSONDecodeError:
                existing_ops = {}
    else:
        existing_ops = {}

    # --- merge histories ---
    # if cad_op_history is a dict, merge dicts
    # if it's a list, append
    if isinstance(existing_ops, dict) and isinstance(cad_op_history, dict):
        existing_ops.update(cad_op_history)
    elif isinstance(existing_ops, list) and isinstance(cad_op_history, list):
        existing_ops.extend(cad_op_history)
    else:
        # fallback: store separately under a new key
        existing_ops[f"run_{len(existing_ops)+1}"] = cad_op_history

    # --- save updated file ---
    with open(cad_operations_file, "w", encoding="utf-8") as f:
        json.dump(existing_ops, f, indent=4, ensure_ascii=False)

    print(f"✅ Exported full assembly: {stl_path} and {step_path}")
    return None


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    macro_folder = base / "macros" / "chair"
    output_folder = base.parent / "output"

    # Clean the two subfolders (auto-create + wipe)
    helper.clean_dir(output_folder)

    # Run your pipeline
    macro_name, root_component = read_macro(macro_folder, output_folder)
    execute(root_component, output_folder)
    read_matings(macro_folder, output_folder)
