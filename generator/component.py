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

        self.parameters = parameters
        self.quantity = quantity
        self.location = location
        self.locations = locations


    def param_init(self):
        # Randomize parameters
        self.chosen_parameters = {
            k: random.uniform(v[0], v[1]) for k, v in self.parameters.items()
        }

        # Compute absolute locations
        if self.quantity > 1 and self.locations:
            self.absolute_locations = [self.compute_absolute_location(loc) for loc in self.locations]
        else:
            self.absolute_locations = [self.compute_absolute_location(self.location)]

   
    def compute_absolute_location(self, rel_loc):
        if not rel_loc:
            return {"x": 0, "y": 0, "z": 0}

        abs_loc = {}

        for axis in ["x", "y", "z"]:
            val = rel_loc.get(axis, 0)

            if isinstance(val, str):
                expr = val.strip()

                if expr.startswith("parent "):
                    # expression refers to parent
                    if not self.parent:
                        raise ValueError(f"Location expression '{expr}' refers to parent but {self.name} has no parent.")
                    expr = expr[len("parent "):]  # remove 'parent ' prefix

                    # replace parent's parameters
                    for param_name, param_value in self.parent.chosen_parameters.items():
                        expr = expr.replace(param_name, str(param_value))

                else:
                    # replace own parameters
                    for param_name, param_value in self.chosen_parameters.items():
                        expr = expr.replace(param_name, str(param_value))

                try:
                    val = eval(expr)
                except Exception as e:
                    raise ValueError(
                        f"Failed to evaluate location expression '{rel_loc[axis]}' for {self.name}: {e}"
                    )

            abs_loc[axis] = val

        if self.parent:
            parent_loc = self.parent.absolute_locations[0]
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


    def build_one_instance(self, idx, location, canvas=None):
        """
        Build one instance of this component at the given location.
        """
        print(f"Building {self.name} instance {idx+1}")

        x_len = self.chosen_parameters.get("x_length")
        y_len = self.chosen_parameters.get("y_length")
        z_len = self.chosen_parameters.get("z_length")

        x, y, z = location["x"], location["y"], location["z"]

        half_x, half_y = x_len / 2, y_len / 2

        # Define rectangle perimeter points in XY plane at z
        new_point_list = [
            [x - half_x, y - half_y, z], [x + half_x, y - half_y, z],
            [x + half_x, y - half_y, z], [x + half_x, y + half_y, z],
            [x + half_x, y + half_y, z], [x - half_x, y + half_y, z],
            [x - half_x, y + half_y, z], [x - half_x, y - half_y, z]
        ]

        # Build sketch and extrude
        sketch = build123.protocol.build_sketch(
            idx, canvas, new_point_list, False, None
        )

        canvas = build123.protocol.build_extrude(
            idx, canvas, sketch, z_len, False, None
        )

        return canvas


    def build(self, canvas=None):
        """
        Build this component and its children recursively.
        """

        self.param_init()
        
        for idx, loc in enumerate(self.absolute_locations):
            canvas = self.build_one_instance(idx, loc, canvas)

        for child in self.children:
            canvas = child.build(canvas)

        return canvas
