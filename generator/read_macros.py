import json
from pathlib import Path
import component
import random
import os 

def load_component(json_path, parent=None, labels = [0]):
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
            'mating_reference' : variant.get('mating_reference', None)
        }
    else:
        data = {
            'name': comp_dict['name'],
            'boolean' : variant['boolean'],
            'parameters': comp_dict['parameters'],
            'location': comp_dict.get('location'),
            'cad_operations': comp_dict['cad_operations'],
            'quantity': comp_dict.get("quantity", 1),
            'locations': comp_dict.get("locations")
        }

    c = component.Component(data, parent=parent, labels = labels)
    return c


def read_macro(folder_path, idx = 0):
    folder_path = Path(folder_path)

    # Recursive loader for children
    def load_with_children(name, parent=None, labels = [idx]):
        json_file = folder_path / f"{name}.json"
        comp = load_component(json_file, parent=parent, labels= labels)

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

    count = 1
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
            macro_name, root_component = read_macro(subfolder_path, count)

            # Step 2: Execute
            execute(root_component, output_path)
        
        count += 1

    


def execute(component_obj, output_path):
    """
    Execute a Component and all its children recursively.
    Export result to .stl and .step
    """
    output_path.mkdir(exist_ok=True)

    canvas, _, _ = component_obj.build()  # build full hierarchy

    stl_path = output_path / f"{component_obj.name}.stl"
    step_path = output_path / f"{component_obj.name}.step"

    canvas.part.export_stl(str(stl_path))
    canvas.part.export_step(str(step_path))

    print(f"✅ Exported full assembly: {stl_path} and {step_path}")
    return None


if __name__ == "__main__":
    macro_folder = Path(__file__).parent / "macros" / "chair"
    output_folder = output_dir = Path(__file__).parent / "output"
    
    macro_name, root_component = read_macro(macro_folder)
    execute(root_component, output_folder)

    read_matings(macro_folder, output_folder)