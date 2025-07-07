import random

import build123.protocol
import re
import helper

class Component:
    def __init__(self, data: dict, parent=None):
        """
        Initializes the component from its chosen variant dict.
        :param data: dict of chosen variant
        :param parent: Parent Component
        """
        self.parent = parent
        self.children = []

        self.name = data['name']
        self.parameters = data['parameters']
        self.location = data.get('location')
        self.cad_operations = data['cad_operations']
        self.quantity = data.get('quantity', 1)
        self.locations = data.get('locations')


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
        x_len = self.chosen_parameters.get("x_length")
        y_len = self.chosen_parameters.get("y_length")
        z_len = self.chosen_parameters.get("z_length")

        x, y, z = location["x"], location["y"], location["z"]

        half_x, half_y = x_len / 2, y_len / 2

        # Define rectangle perimeter points in XY plane at z
        new_point_list = [
            [x - half_x, y - half_y, z],
            [x + half_x, y - half_y, z],
            [x + half_x, y + half_y, z],
            [x - half_x, y + half_y, z]
        ]

        sketch = None  # for reference during operations

        # context for eval()
        context = {
            "x_length": x_len,
            "y_length": y_len,
            "z_length": z_len
        }

        for op in self.cad_operations:
            op_name = op["name"]

            if op_name == "sketch_rectangle":
                sketch = build123.protocol.build_sketch(
                    tempt_canvas, new_point_list
                )

            elif op_name == "extrude":
                if sketch is None:
                    raise RuntimeError(f"Cannot extrude: no sketch created for {self.name}")
                tempt_canvas = build123.protocol.build_extrude(
                    tempt_canvas, sketch, z_len
                )

            elif op_name == "fillet_or_chamfer":
                prob = op.get("probability", 1.0)
                if random.random() > prob:
                    print(f"Skipping fillet/chamfer on {self.name} instance {idx+1} (probability={prob})")
                    continue

                sub_params = op.get("sub_parameters", {})
                radius_expr = sub_params.get("radius", 0.01)
                edge_indices = sub_params.get("edges", [])
                radius = helper.eval_with_range(str(radius_expr), {})


                if tempt_canvas is None:
                    raise RuntimeError(f"Cannot fillet/chamfer: no geometry for {self.name}")

                edges_in_canvas = tempt_canvas.edges()

                if random.random() > 0.5:
                    for edge_idx in edge_indices:
                        target_edge = edges_in_canvas[edge_idx]
                        tempt_canvas = build123.protocol.build_fillet(
                            tempt_canvas, target_edge, radius
                        )
                else:
                    for edge_idx in edge_indices:
                        target_edge = edges_in_canvas[edge_idx]
                        tempt_canvas = build123.protocol.build_chamfer(
                            tempt_canvas, target_edge, radius
                        )

            elif op_name == "subtraction":
                sub_params = op.get("sub_parameters", {})
                parameters = sub_params.get("parameters", sub_params)  # fallback
                shape_details = parameters["shape_details"]
                chosen_shape = parameters["shape"][0]  # assuming always 1 shape

                detail = shape_details[chosen_shape]

                # Base center of the solid
                center = [x, y, z]

                # Safe context for eval
                context = {
                    "x_length": x_len,
                    "y_length": y_len,
                    "z_length": z_len
                }

                # Adjust center based on shape-specific location
                location = detail.get("location", {"x": 0, "y": 0, "z": 0})
                offset_x = eval(str(location.get("x", 0)), {}, context)
                offset_y = eval(str(location.get("y", 0)), {}, context)
                offset_z = eval(str(location.get("z", 0)), {}, context)

                adjusted_center = [
                    center[0] + offset_x,
                    center[1] + offset_y,
                    center[2] + offset_z
                ]

                size = detail["size"]
                plane_normal = [0, 0, 0]
                for idx2, val in enumerate(size):
                    if val == 0:
                        plane_normal[idx2] = 1  # normal to axis with 0 extent

                subtract_height = z_len / 2  # or configurable

                if chosen_shape == "triangle":
                    x_size = helper.eval_with_range(str(size[0]), context)
                    z_size = helper.eval_with_range(str(size[2]), context)

                    half_x, half_z = x_size / 2, z_size / 2

                    Points_list = [
                        [adjusted_center[0] - half_x, adjusted_center[1], adjusted_center[2] - half_z],
                        [adjusted_center[0] + half_x, adjusted_center[1], adjusted_center[2] - half_z],
                        [adjusted_center[0],          adjusted_center[1], adjusted_center[2] + half_z]
                    ]

                    subtract_sketch = build123.protocol.build_sketch(
                        tempt_canvas, Points_list
                    )

                elif chosen_shape == "square":
                    x_size = helper.eval_with_range(str(size[0]), context)
                    z_size = helper.eval_with_range(str(size[2]), context)

                    half_x, half_z = x_size / 2, z_size / 2

                    Points_list = [
                        [adjusted_center[0] - half_x, adjusted_center[1], adjusted_center[2] - half_z],
                        [adjusted_center[0] + half_x, adjusted_center[1], adjusted_center[2] - half_z],
                        [adjusted_center[0] + half_x, adjusted_center[1], adjusted_center[2] + half_z],
                        [adjusted_center[0] - half_x, adjusted_center[1], adjusted_center[2] + half_z]
                    ]

                    subtract_sketch = build123.protocol.build_sketch(
                        tempt_canvas, Points_list
                    )

                elif chosen_shape == "cylinder":
                    dims = [x_len, y_len, z_len]
                    radius_expr = detail.get("radius", None)
                    if radius_expr:
                        radius = helper.eval_with_range(str(radius_expr), context)
                    else:
                        radius = random.choice(dims) * 0.5

                    subtract_sketch = build123.protocol.build_circle(
                        radius=radius,
                        center=adjusted_center,
                        normal=plane_normal
                    )

                else:
                    raise ValueError(f"Unsupported subtraction shape: {chosen_shape}")


                tempt_canvas = build123.protocol.build_subtract(
                    tempt_canvas, subtract_sketch, subtract_height
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
