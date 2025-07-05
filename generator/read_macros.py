import json
from pathlib import Path
import component


def load_component(comp_dict, parent=None):
    """
    Recursively create a component.Component instance from a dict.
    """
    children = []
    if "children" in comp_dict:
        children = [load_component(child, parent=None) for child in comp_dict["children"]]  # we attach them below

    c = component.Component(
        name=comp_dict['name'],
        parameters=comp_dict['parameters'],
        location=comp_dict.get('location'),
        cad_operations=comp_dict['cad_operations'],
        quantity=comp_dict.get("quantity", 1),
        locations=comp_dict.get("locations"),
        children=[],  # fill children below
        parent=parent
    )

    # attach children with proper parent
    c.children = [load_component(child_dict, parent=c) for child_dict in comp_dict.get("children", [])]

    return c


def read_macro(file_path):
    with open(file_path, 'r') as f:
        macro = json.load(f)

    components_data = macro['components']
    name_to_component = {}

    # First pass: create all components
    for comp_dict in components_data:
        c = component.Component(
            name=comp_dict['name'],
            parameters=comp_dict['parameters'],
            location=comp_dict.get('location'),
            cad_operations=comp_dict['cad_operations'],
            quantity=comp_dict.get('quantity', 1),
            locations=comp_dict.get('locations'),
            parent=None  # will set in second pass
        )
        name_to_component[c.name] = c

    # Second pass: set parents and children
    for comp_dict in components_data:
        name = comp_dict['name']
        parent_name = comp_dict.get('parent')
        if parent_name:
            parent = name_to_component[parent_name]
            child = name_to_component[name]
            child.parent = parent
            parent.children.append(child)

    # Find root components (no parent)
    root_components = [c for c in name_to_component.values() if c.parent is None]

    return macro['name'], root_components


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
    macro_path = Path(__file__).parent / "macros" / "chair.json"
    macro_name, components = read_macro(macro_path)

    # print(f"Macro: {macro_name}")
    # print("Components:")
    # for comp in components:
    #     comp.describe()

    # Execute the root component (seat)
    seat = components[0]
    execute(seat)
