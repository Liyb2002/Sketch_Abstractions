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
    """
    Read the macro JSON file and parse its components.
    """
    with open(file_path, 'r') as f:
        macro = json.load(f)

    root_components = [load_component(comp) for comp in macro['components']]
    return macro['name'], root_components


if __name__ == "__main__":
    macro_path = Path(__file__).parent / "macros" / "chair.json"
    macro_name, components = read_macro(macro_path)

    print(f"Macro: {macro_name}")
    print("Components:")
    for comp in components:
        comp.describe()
