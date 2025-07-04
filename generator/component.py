import random

import build123.protocol


class Component:
    def __init__(self, name, parameters, location, cad_operations, quantity=1, locations=None, children=None, parent=None):
        """
        Represents a component of a macro.

        :param name: Name of the component
        :param parameters: Dict of parameter ranges
        :param location: Default relative location (dict)
        :param cad_operations: List of CAD operations
        :param quantity: Number of this component
        :param locations: Optional list of relative locations (one per instance)
        :param children: List of child Components
        :param parent: Parent Component instance
        """
        self.name = name
        self.parent = parent
        self.cad_operations = cad_operations
        self.quantity = quantity
        self.children = children if children else []

        # Randomize parameters
        self.chosen_parameters = {
            k: random.uniform(v[0], v[1]) for k, v in parameters.items()
        }

        # Compute absolute locations
        if quantity > 1 and locations:
            self.absolute_locations = [self.compute_absolute_location(loc) for loc in locations]
        else:
            self.absolute_locations = [self.compute_absolute_location(location)]

    def compute_absolute_location(self, rel_loc):
        if not rel_loc:
            return {"x": 0, "y": 0, "z": 0}

        abs_loc = rel_loc.copy()
        if self.parent:
            parent_loc = self.parent.absolute_locations[0]  # always relative to parent's first instance
            abs_loc["x"] += parent_loc["x"]
            abs_loc["y"] += parent_loc["y"]
            abs_loc["z"] += parent_loc["z"]
        return abs_loc

    def describe(self, indent=0):
        pad = "  " * indent
        print(f"{pad}- {self.name} (x{self.quantity}): {' → '.join(self.cad_operations)}")
        print(f"{pad}  Parameters: {self.chosen_parameters}")
        for idx, loc in enumerate(self.absolute_locations):
            print(f"{pad}  Location {idx+1}: {loc}")
        for child in self.children:
            child.describe(indent + 1)

    def choose_parameters(self, parameters: dict) -> dict:
        """
        Randomly choose a value within each parameter's range.

        :param parameters: dict of parameter: [min, max]
        :return: dict of parameter: chosen_value
        """
        chosen = {}
        for k, v in parameters.items():
            val = random.uniform(v[0], v[1])
            chosen[k] = round(val, 4)  # optionally round for readability
        return chosen

    def build(self):
        """
        Build the shape of this component using build123d.
        Only supports sketch_rectangle + extrude for now.
        Returns: BuildPart canvas
        """
        print(f"Building component: {self.name}")

        width = self.chosen_parameters.get("width")
        depth = self.chosen_parameters.get("depth")
        thickness = self.chosen_parameters.get("thickness")

        # Get absolute location
        loc = self.absolute_locations[0]
        x, y, z = loc["x"], loc["y"], loc["z"]

        # Prepare the point list for the rectangle in 3D
        half_w, half_d = width / 2, depth / 2
        new_point_list = [
            [ x - half_w, y - half_d, z ], [ x + half_w, y - half_d, z ],
            [ x + half_w, y - half_d, z ], [ x + half_w, y + half_d, z ],
            [ x + half_w, y + half_d, z ], [ x - half_w, y + half_d, z ],
            [ x - half_w, y + half_d, z ], [ x - half_w, y - half_d, z ]
        ]

        prev_sketch = build123.protocol.build_sketch(0, None, new_point_list, False, None)

        canvas = build123.protocol.build_extrude(0, None, prev_sketch, 1, False, None)
        
        return canvas
