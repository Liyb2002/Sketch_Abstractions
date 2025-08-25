from pathlib import Path
from build123d import *
import os
import numpy as np
from build123d import BuildPart, add


def build_sketch(canvas, Points_list):

    # if tempt_idx == 0:
    #     brep_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")
    #     stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    # else:
    #     brep_dir = os.path.join(data_dir, "canvas", f"tempt_{tempt_idx}.step")
    #     stl_dir = os.path.join(data_dir, "canvas", f"tempt_{tempt_idx}.stl")

    if canvas is None:
        with BuildSketch():
            with BuildLine():
                lines = []
                for i in range(len(Points_list)):
                    start_point_sublist = Points_list[i]
                    end_point_sublist = Points_list[(i+1) % len(Points_list)]  # wrap around

                    start_point = (
                        start_point_sublist[0],
                        start_point_sublist[1],
                        start_point_sublist[2]
                    )
                    end_point = (
                        end_point_sublist[0],
                        end_point_sublist[1],
                        end_point_sublist[2]
                    )

                    line = Line(start_point, end_point)
                    lines.append(line)
            perimeter = make_face()

        return perimeter
    
    else:
        with canvas: 
            with BuildSketch():
                with BuildLine():
                    lines = []
                    for i in range(len(Points_list)):
                        start_point_sublist = Points_list[i]
                        end_point_sublist = Points_list[(i+1) % len(Points_list)]  # wrap around

                        start_point = (
                            start_point_sublist[0],
                            start_point_sublist[1],
                            start_point_sublist[2]
                        )
                        end_point = (
                            end_point_sublist[0],
                            end_point_sublist[1],
                            end_point_sublist[2]
                        )

                        line = Line(start_point, end_point)
                        lines.append(line)

                perimeter = make_face()

    return perimeter





def build_circle(radius, center, normal):

    with BuildSketch(Plane(origin=(center[0], center[1], center[2]), z_dir=(normal[0], normal[1], normal[2])) )as perimeter:
        Circle(radius = radius)


    return perimeter.sketch.face()




def build_extrude(canvas, target_face, extrude_amount):
    # stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    # step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    if canvas != None:
        with canvas: 
            extrude( target_face, amount=extrude_amount)

    else:
        with BuildPart() as canvas:
            extrude( target_face, amount=extrude_amount)


    return canvas


def build_subtract(canvas, target_face, extrude_amount):
    
    num_lines = 0

    with canvas:
        extrude( target_face, amount= extrude_amount, mode=Mode.SUBTRACT)

    return canvas


def build_fillet(canvas, target_edge, radius):
    with canvas:
        fillet(target_edge, radius)

    return canvas


def build_chamfer(canvas, target_edge, radius):

    with canvas:
        chamfer(target_edge, radius)
    
    return canvas


def build_sweep(canvas, target_face, control_points, *, is_frenet=True, mode=Mode.ADD):
    """
    Build a sweep from `target_face` along a spline through `control_points`.
    - If `canvas` is None, returns a new BuildPart containing ONLY the sweep.
    - Otherwise, booleans the sweep into `canvas.part` using `mode`.
    """
    # 1) Build the sweep as a standalone solid
    pts = [Vector(*p) for p in control_points]
    with BuildPart() as tmp:
        with BuildLine() as path_ln:
            Spline(pts)
        path_wire = path_ln.wire()
        section = target_face.face() if isinstance(target_face, Sketch) else target_face
        sweep(sections=section, path=path_wire, is_frenet=is_frenet, mode=Mode.ADD)
    swept_solid = tmp.part  # solid built in temp context

    # If no canvas provided, return the sweep as the "canvas"
    if canvas is None:
        return tmp  # contains .part == swept_solid

    # 2) Boolean it onto the existing canvas.part
    if mode == Mode.ADD:
        canvas.part = canvas.part + swept_solid
    elif mode == Mode.SUBTRACT:
        canvas.part = canvas.part - swept_solid
    elif mode == Mode.INTERSECT:
        canvas.part = canvas.part & swept_solid
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return canvas


def build_mirror(canvas):
    mirrored = mirror(objects=canvas.part, about=Plane.YZ)
    canvas.part = (canvas.part + mirrored).clean()
    return canvas

def build_sphere(canvas, radius, location):
    x, y, z = (float(location[0]), float(location[1]), float(location[2]))

    # Build the sphere as its own Part at the given location
    with BuildPart() as _tmp:
        with Locations(Location((x, y, z))):
            Sphere(radius)
    sphere_part = _tmp.part

    # If no canvas provided, return the BuildPart containing just the sphere
    if canvas is None:
        return _tmp

    # Otherwise, expect something with a `.part` attribute and merge
    if not hasattr(canvas, "part"):
        raise TypeError("canvas must have a 'part' attribute (e.g., a BuildPart)")

    if canvas.part is None:
        canvas.part = sphere_part
    else:
        # Combine like your mirror example: (existing + new).clean()
        canvas.part = (canvas.part + sphere_part).clean()

    return canvas



def simulate_extrude(sketch, amount):
    with BuildPart() as temp:
        extrude(sketch, amount=amount)
    return temp.part


def has_volume_overlap(canvas, new_part):
    intersection = canvas & new_part
    return intersection.volume > new_part.volume * 0.5


def get_part(obj):
    """Extract the Part shape from a BuildPart or pass-through if already a Part."""
    if hasattr(obj, "part"):
        return obj.part
    else:
        return obj


def merge_canvas(canvas1, canvas2, boolean):
    add_parts = []
    subtract_parts = []

    if canvas1 is not None:
        if boolean == "subtraction":
            subtract_parts.append(get_part(canvas1))
        else:
            add_parts.append(get_part(canvas1))
    if canvas2 is not None:
        add_parts.append(get_part(canvas2))

    if len(add_parts) == 0 or add_parts[0] == None:
        return None
    
    with BuildPart() as merged:
        for part in add_parts:
            add(part)
        
        for part in subtract_parts:
            add(part, mode=Mode.SUBTRACT)  

    return merged
