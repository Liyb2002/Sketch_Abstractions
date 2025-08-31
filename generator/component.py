import random

import build123.protocol
import re
import helper
from pathlib import Path
import ast
import os
import json


class Component:
    def __init__(self, data: dict, parent=None, labels = [0], output_path = Path(__file__).parent.parent / "output"):
        """
        Initializes the component from its chosen variant dict.
        :param data: dict of chosen variant
        :param parent: Parent Component
        """
        self.output_folder = output_path

        self.parent = parent
        self.children = []

        self.name = data['name']
        self.parameters = data['parameters']
        self.location = data.get('location')
        self.cad_operations = data['cad_operations']
        self.quantity = data.get('quantity', 1)
        self.locations = data.get('locations')
        self.boolean = data.get('boolean')
        self.condition = data.get('condition')
        self.path = data.get('path')
        
        self.mating_reference = data.get('mating_reference')
        self.set_mating_params(data.get('mating_reference'))

        self.labels = labels

    def param_init(self):
        """
        Initialize chosen parameters and absolute locations.
        - Numeric ranges [min, max] → random uniform value
        - Parent references (e.g., 'parent x_length') → inherit from parent
        - Parent refs with multiplier (e.g., 'parent x_length * [0.2, 0.8]')
        → inherit from parent, multiply by random in range
        - Mating references (e.g., 'mating x_length') → inherit from mating_params
        - Categorical (list of strings) → first element
        - Fixed values → used directly
        """
        def parse_range(s: str):
            """Safely parse a '[min, max]' string into a Python list of floats."""
            try:
                val = ast.literal_eval(s)
                if (isinstance(val, list) and len(val) == 2 
                        and all(isinstance(n, (int, float)) for n in val)):
                    return val
            except Exception:
                pass
            return None

        self.chosen_parameters = {}

        for k, v in self.parameters.items():
            # Case 1: numeric range [min, max]
            if isinstance(v, list) and len(v) == 2 and all(isinstance(n, (int, float)) for n in v):
                self.chosen_parameters[k] = random.uniform(v[0], v[1])

            # Case 2: parent reference (with optional multiplier)
            elif isinstance(v, str) and v.startswith("parent "):
                if self.parent is None:
                    raise ValueError(f"Parameter {k} requires a parent, but no parent found.")

                # Split at '*' if multiplier exists
                if "*" in v:
                    parent_part, multiplier_part = v.split("*", 1)
                    parent_param = parent_part.replace("parent", "").strip()
                    multiplier_range = parse_range(multiplier_part.strip())
                    if multiplier_range is None:
                        raise ValueError(f"Invalid multiplier format for {k}: {multiplier_part}")
                else:
                    parent_param = v.replace("parent", "").strip()
                    multiplier_range = None

                # Ensure parent params are initialized
                if parent_param not in self.parent.chosen_parameters:
                    self.parent.param_init()

                base_value = self.parent.chosen_parameters[parent_param]

                if multiplier_range:
                    factor = random.uniform(multiplier_range[0], multiplier_range[1])
                    self.chosen_parameters[k] = base_value * factor
                else:
                    self.chosen_parameters[k] = base_value

            # Case 3: mating reference
            elif isinstance(v, str) and v.startswith("mating "):
                mating_param = v.replace("mating", "").strip()
                if not hasattr(self, "mating_params") or self.mating_params is None:
                    raise ValueError(f"Parameter {k} requires mating_params, but none found.")
                if mating_param not in self.mating_params:
                    raise KeyError(f"Mating parameter '{mating_param}' not found in mating_params.")
                self.chosen_parameters[k] = self.mating_params[mating_param]

            # Case 4: categorical or fixed value
            else:
                self.chosen_parameters[k] = v[0] if isinstance(v, list) else v

        # Compute absolute locations
        if self.quantity > 1 and self.locations:
            self.absolute_locations = [self.compute_absolute_location(loc) for loc in self.locations]
        else:
            self.absolute_locations = [self.compute_absolute_location(self.location)]


    def set_mating_params(self, mating_reference):
        if mating_reference is None:
            return
        
        # Build the file path
        file_path = self.output_folder / f"mating_{mating_reference}.json"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                mating_json = json.load(f)
            self.mating_params = mating_json["parameters"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {file_path}")
        except KeyError:
            raise KeyError(f"'parameters' key not found in {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {file_path}")


    def compute_absolute_location(self, rel_loc):
        if not rel_loc:
            return {"x_location": 0, "y_location": 0, "z_location": 0}

        abs_loc = {}

        for axis in ["x_location", "y_location", "z_location"]:
            val = rel_loc.get(axis, 0)

            if isinstance(val, str):
                expr = val.strip()

                # Replace all 'parent param' with parent's parameter value
                if self.parent:
                    def replace_parent(match):
                        param = match.group(1)

                        if param not in self.parent.chosen_parameters and not param.endswith("_location"):
                            raise ValueError(f"Parent parameter '{param}' not found for {self.name}")

                        if param.endswith("_length"):
                            return str(self.parent.chosen_parameters[param])
                        elif param.endswith("_location"):
                            return str(self.parent.absolute_locations[0][param])
                        else:
                            return str(self.parent.chosen_parameters[param])

                    expr = re.sub(r'parent\s+([a-zA-Z_]\w*)', replace_parent, expr)

                # Replace mating params (like "mating z_length")
                def replace_mating(match):
                    param = match.group(1)
                    if param not in self.mating_params:
                        raise ValueError(f"Mating parameter '{param}' not found for {self.name}")
                    return str(self.mating_params[param])

                expr = re.sub(r'mating\s+([a-zA-Z_]\w*)', replace_mating, expr)

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
            abs_loc["x_location"] += parent_loc["x_location"]
            abs_loc["y_location"] += parent_loc["y_location"]
            abs_loc["z_location"] += parent_loc["z_location"]

        return abs_loc


    def describe(self, indent=0):
        pad = "  " * indent
        print(f"{pad}- {self.name} (x{self.quantity}): {' → '.join(self.cad_operations)}")
        print(f"{pad}  Parameters: {self.chosen_parameters}")
        for idx, loc in enumerate(self.absolute_locations):
            print(f"{pad}  Location {idx+1}: {loc}")
        for child in self.children:
            child.describe(indent + 1)


    def build_one_instance(self, idx, location, tempt_canvas):

        if self.name == "root":
            return 
        """
        Build one instance of this component at the given location, following cad_operations.
        """
        x_len = self.chosen_parameters.get("x_length")
        y_len = self.chosen_parameters.get("y_length")
        z_len = self.chosen_parameters.get("z_length")

        x, y, z = location["x_location"], location["y_location"], location["z_location"]

        # context for eval()
        context = {
            "x_length": x_len,
            "y_length": y_len,
            "z_length": z_len
        }

        for op in self.cad_operations:
            op_name = op["name"]

            if op_name == "sketch_rectangle":
                half_x, half_y = x_len / 2, y_len / 2
                new_point_list = [
                    [x - half_x, y - half_y, z],
                    [x + half_x, y - half_y, z],
                    [x + half_x, y + half_y, z],
                    [x - half_x, y + half_y, z]
                ]
                sketch = build123.protocol.build_sketch(
                    tempt_canvas, new_point_list
                )

            elif op_name == "sketch_circle":
                radius = x_len / 2
                center = [x,y,z]


                normal_axis = self.chosen_parameters.get("normal", ["z"])[0]  # default to z
                if normal_axis.startswith("-"):
                    axis = normal_axis[1:]
                    sign = -1
                else:
                    axis = normal_axis
                    sign = 1

                normal_lookup = {
                    "x": [1, 0, 0],
                    "y": [0, 1, 0],
                    "z": [0, 0, 1]
                }

                base_normal = normal_lookup.get(axis, [0, 0, 1])  # fallback to z if unknown
                normal = [sign * n for n in base_normal]


                sketch = build123.protocol.build_circle(
                    radius, center, normal
                )

            elif op_name == "sketch_triangle":
                # Isosceles triangle centered at (x, y).
                # Base length = x_len, height = y_len.
                half_x, half_y = x_len / 2, y_len / 2
                new_point_list = [
                    [x - half_x, y - half_y, z],  # left base
                    [x + half_x, y - half_y, z],  # right base
                    [x,          y + half_y, z],  # apex
                ]
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

            elif op_name == "sweep":
                self.path = self.parse_eval_list(self.path)
                if sketch is None:
                    raise RuntimeError(f"Cannot extrude: no sketch created for {self.name}")
                tempt_canvas = build123.protocol.build_sweep(
                    tempt_canvas, sketch, self.path
                )
                           
            elif op_name == "mirror":
                tempt_canvas = build123.protocol.build_mirror(
                    tempt_canvas
                )

            elif op_name == "mating_subtraction":
                # ensure output folder exists
                os.makedirs(self.output_folder, exist_ok=True)
                
                # build the file path
                file_path = os.path.join(self.output_folder, f"mating_{self.name}.json")
                
                # prepare the data
                data = {
                    "name": self.name,
                    "parameters": self.chosen_parameters
                }
                
                # write to the file
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

            elif op_name == "build_sphere":
                radius = x_len / 2
                location = [x,y,z]
                tempt_canvas = build123.protocol.build_sphere(
                    tempt_canvas, radius, location
                    )


            else:
                raise NotImplementedError(f"Unsupported CAD operation: {op_name}")

            self.save_process(tempt_canvas, self.main_canvas)
        return tempt_canvas


    def save_process(self, tempt_canvas, full_canvas):
        to_save_canvas = build123.protocol.merge_canvas(tempt_canvas, full_canvas, self.boolean)
        output_dir = self.output_folder / "history"
        output_dir.mkdir(exist_ok=True)

        tmp_stl = output_dir / f"{self.process_count}.stl"
        tmp_step = output_dir / f"{self.process_count}.step"
        if to_save_canvas is not None and to_save_canvas.part is not None:
            self.process_count += 1

            helper.func_export_stl(to_save_canvas, str(tmp_stl))
            helper.func_export_step(to_save_canvas, str(tmp_step))


    def parse_eval_list(self, path_list):
        coords = []
        for point in path_list:
            parsed_point = []
            for val in point:
                if isinstance(val, (int, float)):
                    parsed_point.append(float(val))

                elif isinstance(val, list) and len(val) == 2:
                    # literal [min, max] range → sample
                    lo, hi = float(val[0]), float(val[1])
                    if lo > hi: lo, hi = hi, lo
                    parsed_point.append(random.uniform(lo, hi))

                elif isinstance(val, str):
                    expr = val
                    # Replace parent references
                    for pk, pv in self.parent.chosen_parameters.items():
                        expr = expr.replace(f"parent {pk}", str(pv))

                    # Replace inline [a,b] ranges with sampled numbers
                    def repl(m):
                        lo, hi = float(m.group(1)), float(m.group(2))
                        if lo > hi: lo, hi = hi, lo
                        return str(random.uniform(lo, hi))

                    expr = re.sub(r"\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]", repl, expr)

                    try:
                        parsed_val = eval(expr, {"__builtins__": {}}, {})
                    except Exception as e:
                        raise ValueError(f"Failed to evaluate expression '{val}' -> '{expr}': {e}")
                    parsed_point.append(float(parsed_val))

                else:
                    raise ValueError(f"Unsupported value type in path: {val}")
            coords.append(parsed_point)

        return coords


    def build(self, canvas=None, process_count = 0):
        """
        Build this component and its children recursively.
        """

        if self.parent and self.condition !="None" and self.condition != self.parent.name:
            return canvas, process_count, None

        self.process_count = process_count
        self.main_canvas = canvas

        self.param_init()
        tempt_canvas = None

        for idx, loc in enumerate(self.absolute_locations):
            tempt_canvas = self.build_one_instance(idx, loc, tempt_canvas)

        self.main_canvas = build123.protocol.merge_canvas(tempt_canvas, self.main_canvas, self.boolean)

        for child in self.children:
            self.main_canvas, self.process_count, child_tempt_canvas = child.build(self.main_canvas, self.process_count)
            tempt_canvas = build123.protocol.merge_canvas(child_tempt_canvas, tempt_canvas, child.boolean)

        # Also save tempt canvas
        if tempt_canvas is not None:

            file_name = "-".join(str(n) for n in self.labels)

            output_dir = self.output_folder / "seperable"
            output_dir.mkdir(parents=True, exist_ok=True)

            tmp_stl = output_dir / f"{file_name}.stl"
            tmp_step = output_dir / f"{file_name}.step"

            helper.func_export_stl(tempt_canvas, str(tmp_stl))
            helper.func_export_step(tempt_canvas, str(tmp_step))


        return self.main_canvas, self.process_count, tempt_canvas
