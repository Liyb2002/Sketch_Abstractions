from pathlib import Path
from build123d import *
import os
import numpy as np
from build123d import BuildPart, add
from build123d import BuildSketch, Plane, Circle, Vector


from build123d import BuildSketch, BuildLine, Line, Plane, Vector, Location, make_face

def _plane_from_canvas(canvas):
    if canvas is None:
        return Plane.XY
    if isinstance(canvas, Plane):
        return canvas
    try:
        return Plane(canvas)  # works for planar faces
    except Exception:
        pass
    for attr in ("plane", "workplane"):
        if hasattr(canvas, attr) and isinstance(getattr(canvas, attr), Plane):
            return getattr(canvas, attr)
    return Plane.XY

def build_sketch(canvas, Points_list):
    # 1) Get the plane we want to use
    plane = _plane_from_canvas(canvas)

    # 2) Build a brand-new, empty sketch on that plane (standalone)
    with BuildSketch(plane) as standalone_ctx:
        with BuildLine():
            for i in range(len(Points_list)):
                p0w = Vector(*Points_list[i])
                p1w = Vector(*Points_list[(i + 1) % len(Points_list)])

                # Convert world -> plane-local 2D
                p0 = plane.to_local_coords(p0w)
                p1 = plane.to_local_coords(p1w)

                Line((p0.X, p0.Y), (p1.X, p1.Y))
        _ = make_face()

    # 3) IMPORTANT: bake the plane transform so the returned sketch is in world coords
    #    (some downstream code/tools look only at geometry, not the sketch's plane)
    try:
        standalone_sketch_world = standalone_ctx.sketch.moved(plane.location)
    except AttributeError:
        # some versions prefer the @ operator
        standalone_sketch_world = standalone_ctx.sketch @ plane.location

    # ---- Your original perimeter behavior (unchanged) ----
    if canvas is None:
        with BuildSketch(plane) as ctx:
            with BuildLine():
                for i in range(len(Points_list)):
                    a = Points_list[i]; b = Points_list[(i + 1) % len(Points_list)]
                    pa = plane.to_local_coords(Vector(*a))
                    pb = plane.to_local_coords(Vector(*b))
                    Line((pa.X, pa.Y), (pb.X, pb.Y))
            perimeter = make_face()
    else:
        with canvas:
            with BuildSketch(plane) as ctx:
                with BuildLine():
                    for i in range(len(Points_list)):
                        a = Points_list[i]; b = Points_list[(i + 1) % len(Points_list)]
                        pa = plane.to_local_coords(Vector(*a))
                        pb = plane.to_local_coords(Vector(*b))
                        Line((pa.X, pa.Y), (pb.X, pb.Y))
                perimeter = make_face()

    return perimeter, standalone_sketch_world


def build_circle(radius, center, normal, canvas=None):
    # --- Standalone sketch (new, empty, only plane info copied) ---
    circle_plane = Plane(origin=Vector(*center), z_dir=Vector(*normal))

    with BuildSketch(circle_plane) as standalone_ctx:
        Circle(radius=radius)

    standalone_sketch = standalone_ctx.sketch
    standalone_face = standalone_sketch.face()  # optional, but parallel to sketch

    # --- Perimeter (original behavior) ---
    if canvas is None:
        with BuildSketch(circle_plane) as ctx:
            Circle(radius=radius)
        perimeter = ctx.sketch.face()
    else:
        with canvas:
            with BuildSketch(circle_plane) as ctx:
                Circle(radius=radius)
            perimeter = ctx.sketch.face()

    return perimeter, standalone_sketch



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
