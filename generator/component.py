import random

import build123.protocol
import re

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

                # Replace all 'parent param' with parent's parameter value
                if self.parent:
                    def replace_parent(match):
                        param = match.group(1)
                        if param not in self.parent.chosen_parameters:
                            raise ValueError(f"Parent parameter '{param}' not found for {self.name}")
                        return str(self.parent.chosen_parameters[param])

                    expr = re.sub(r'parent\s+([a-zA-Z_]\w*)', replace_parent, expr)

                # Replace own parameters
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


    def build_one_instance(self, idx, location, tempt_canvas):
        """
        Build one instance of this component at the given location, following cad_operations.
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

        sketch = None  # for reference during operations

        for op in self.cad_operations:
            op_name = op["name"]

            if op_name == "sketch_rectangle":
                sketch = build123.protocol.build_sketch(
                    idx, tempt_canvas, new_point_list, False, None
                )

            elif op_name == "extrude":
                if sketch is None:
                    raise RuntimeError(f"Cannot extrude: no sketch created for {self.name}")
                tempt_canvas = build123.protocol.build_extrude(
                    idx, tempt_canvas, sketch, z_len, False, None
                )

            elif op_name == "fillet":
                # Respect probability if specified
                prob = op.get("probability", 1.0)
                if random.random() > prob:
                    print(f"Skipping fillet on {self.name} instance {idx+1} (probability={prob})")
                    continue

                params = op.get("parameters", {})
                radius = params.get("radius", 0.01)
                edge_indices = params.get("edges", [])

                if tempt_canvas is None:
                    raise RuntimeError(f"Cannot fillet: no geometry generated for {self.name}")

                # Get the actual edge objects from tempt_canvas
                edges_in_canvas = tempt_canvas.edges()
                for edge_idx in edge_indices:
                    target_edge = edges_in_canvas[edge_idx]
                    tempt_canvas = build123.protocol.build_fillet(
                        tempt_canvas, target_edge, radius
                    )

            else:
                raise NotImplementedError(f"Unsupported CAD operation: {op_name}")

        return tempt_canvas


    def build(self, canvas=None):
        """
        Build this component and its children recursively.
        """

        self.param_init()
        tempt_canvas = None

        for idx, loc in enumerate(self.absolute_locations):
            tempt_canvas = self.build_one_instance(idx, loc, tempt_canvas)

        canvas = build123.protocol.merge_canvas(tempt_canvas, canvas)

        for child in self.children:
            canvas = child.build(canvas)

        return canvas
