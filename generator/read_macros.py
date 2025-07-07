import json
from pathlib import Path
import component
import random

def load_component(json_path, parent=None):
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
            'parameters': variant['parameters'],
            'location': variant.get('location'),
            'cad_operations': variant['cad_operations'],
            'quantity': variant.get("quantity", 1),
            'locations': variant.get("locations")
        }
    else:
        data = {
            'name': comp_dict['name'],
            'parameters': comp_dict['parameters'],
            'location': comp_dict.get('location'),
            'cad_operations': comp_dict['cad_operations'],
            'quantity': comp_dict.get("quantity", 1),
            'locations': comp_dict.get("locations")
        }

    c = component.Component(data, parent=parent)
    return c


def read_macro(folder_path):
    folder_path = Path(folder_path)

    # read summary.json
    with open(folder_path / "summary.json", 'r') as f:
        summary = json.load(f)

    macro_name = summary['name']
    component_names = summary['components']

    name_to_component = {}

    # load all components
    for name in component_names:
        json_file = folder_path / f"{name}.json"
        c = load_component(json_file)
        name_to_component[name] = c

    # set parents and children
    for name in component_names:
        json_file = folder_path / f"{name}.json"
        with open(json_file, 'r') as f:
            comp_dict = json.load(f)

        parent_name = comp_dict.get('parent')
        if parent_name:
            parent = name_to_component[parent_name]
            child = name_to_component[name]
            child.parent = parent
            parent.children.append(child)

    root_components = [c for c in name_to_component.values() if c.parent is None]

    return macro_name, root_components


def execute(component_obj):
    """
    Execute a Component and all its children recursively.
    Export result to .stl and .step
    """
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    canvas = component_obj.build()  # build full hierarchy

    stl_path = output_dir / f"{component_obj.name}.stl"
    step_path = output_dir / f"{component_obj.name}.step"

    canvas.part.export_stl(str(stl_path))
    canvas.part.export_step(str(step_path))

    print(f"✅ Exported full assembly: {stl_path} and {step_path}")
    return None


if __name__ == "__main__":
    macro_folder = Path(__file__).parent / "macros" / "chair"
    macro_name, components = read_macro(macro_folder)

    # print(f"Macro: {macro_name}")
    # print("Components:")
    # for comp in components:
    #     comp.describe()

    # Execute the first root component
    root = components[0]
    execute(root)
