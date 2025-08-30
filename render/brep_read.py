from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from OCC.Core.Geom import Geom_CylindricalSurface
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_SurfaceType
from OCC.Core.gp import gp_Vec
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Circle
from OCC.Core.Geom import Geom_Circle, Geom_Line

from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_BSplineCurve
from OCC.Core.Geom import Geom_BSplineCurve


from typing import List, Dict, Tuple, Any

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev, CubicSpline

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status == 1:  # Check if the read was successful
        step_reader.TransferRoot()  # Transfers the whole STEP file
        shape = step_reader.Shape()  # Retrieves the translated shape
        return shape
    else:
        raise Exception("Error reading STEP file.")



def sample_strokes_from_step_file(step_path):
    shape = read_step_file(step_path)
    edge_features_list = []
    cylinder_features = []

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        cylinders = create_face_node_gnn(face)
        cylinder_features += cylinders

        # Explore edges of the face
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)

        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            edge_features = create_edge_node(edge)

            edge_duplicate_id = check_duplicate(edge_features, edge_features_list)
            if edge_duplicate_id != -1:
                edge_explorer.Next()
                continue
            
            edge_features_list.append(edge_features)
            
            edge_explorer.Next()
        
        face_explorer.Next()

    return edge_features_list, cylinder_features




# ---------------------------------------------------------------- #

# What this code does:
# 1)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 2)Cylinder face: Center (3 value), normal (3 value), height, radius, 0, 3
def create_face_node_gnn(face):
    
    adaptor_surface = BRepAdaptor_Surface(face)
    circle_features = []


    # cylinder surface
    if adaptor_surface.GetType() == GeomAbs_Cylinder:
        
        # we also need to compute the angle to see if this is cylinder or an arc
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        total_angle = 0.0
        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge_curve_handle, first, last = BRep_Tool.Curve(edge)
            
            curve_adaptor = GeomAdaptor_Curve(edge_curve_handle)
            curve_type = curve_adaptor.GetType()

            if curve_type == GeomAbs_Circle:
                angle_radians = abs(last - first)
                total_angle += angle_radians
            
            edge_explorer.Next()
        if total_angle < 6.27:
            return []

        cylinder = adaptor_surface.Cylinder()
        radius = cylinder.Radius()

        axis = cylinder.Axis()
        axis_direction = axis.Direction()
        axis_location = axis.Location()
        axis_direction = [axis_direction.X(), axis_direction.Y(), axis_direction.Z()]
        axis_location = [axis_location.X(), axis_location.Y(), axis_location.Z()]

        u_min = adaptor_surface.FirstUParameter()
        u_max = adaptor_surface.LastUParameter()
        v_min = adaptor_surface.FirstVParameter()
        v_max = adaptor_surface.LastVParameter()

        surface = BRep_Tool.Surface(face)
        point_start = surface.Value(u_min, v_min)
        point_end = surface.Value(u_min, v_max)

        height_vector = gp_Vec(point_start, point_end)
        height = height_vector.Magnitude()
        cylinder_data = axis_location + axis_direction + [height, radius] + [0, 3]
        circle_features.append(cylinder_data)


    if adaptor_surface.GetType() == GeomAbs_Plane:

        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            curve_handle, first, last = BRep_Tool.Curve(edge)
            adaptor_curve = GeomAdaptor_Curve(curve_handle, first, last)
            curve_type = adaptor_curve.GetType()

            # Check if the curve is a circle
            if curve_type == GeomAbs_Circle:
                geom_circle = adaptor_curve.Circle()
                angle_radians = abs(last - first)
                
                if abs(angle_radians - 6.283) < 0.1:  # Full circle (approximately 2π)
                    # Extract circle parameters
                    circle_axis = geom_circle.Axis()
                    circle_center = geom_circle.Location()
                    circle_radius = geom_circle.Radius()
                    circle_normal = circle_axis.Direction()
                

                    center_coords = [circle_center.X(), circle_center.Y(), circle_center.Z()]
                    normal_coords = [circle_normal.X(), circle_normal.Y(), circle_normal.Z()]
                    radius = circle_radius

                    cylinder_data = center_coords + normal_coords + [0, circle_radius] + [0, 2]
                    circle_features.append(cylinder_data)
            
            edge_explorer.Next()

    return circle_features





# What this code does:
# 1)Straight Line: Point_1 (3 value), Point_2 (3 value), 0, 0, 0, 1
# 2)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 3)Cylinder face: Center (3 value), normal (3 value), height, radius, 0, 3
# 4)Arc: Point_1 (3 value), Point_2 (3 value), Center (3 value), 4
# 4)Spline: Control_point_1 (3 value), Control_point_2 (3 value), Control_point_3 (3 value), 5

def create_edge_node(edge):
    # Underlying curve + edge's trimmed parameter range
    h_curve, first, last = BRep_Tool.Curve(edge)
    adaptor = GeomAdaptor_Curve(h_curve)
    curve_type = adaptor.GetType()

    def p2t(p):
        return (float(p.X()), float(p.Y()), float(p.Z()))

    if curve_type == GeomAbs_BSplineCurve:
        # Downcast directly
        bs = Geom_BSplineCurve.DownCast(h_curve)
        if bs is None:
            raise RuntimeError("Curve claimed to be BSpline, but DownCast failed")

        # Always 3 control points in your setup
        cp1 = p2t(bs.Pole(1))
        cp2 = p2t(bs.Pole(2))
        cp3 = p2t(bs.Pole(3))
        return [*cp1, *cp2, *cp3, 5]  # 9 coords + type

    # Straight line (everything else)
    p_start = adaptor.Value(first)
    p_end   = adaptor.Value(last)
    start = p2t(p_start)
    end   = p2t(p_end)
    return [*start, *end, 0.0, 0.0, 0.0, 1]  # 10 values total



def check_duplicate(new_feature, feature_list):
    for existing_feature in feature_list:
        if existing_feature == new_feature:
            return 0
    return -1


# ------------------------------------------------------# 


def vis_stroke_node_features(stroke_node_features):
    # Initialize the 3D plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')  # Turn off axis background and borders

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    perturb_factor = 0.000002  # Adjusted perturbation factor for hand-drawn effect

    # Plot all strokes in blue with perturbations
    for idx, stroke in enumerate(stroke_node_features):
        start, end = stroke[:3], stroke[3:6]
        

        # Update min and max limits based on strokes (ignoring circles)
        if stroke[-1] == 1:
            continue
            # Straight line: [start(x,y,z), end(x,y,z), 0,0,0, 1]
            start = np.array(stroke[0:3], dtype=float)
            end   = np.array(stroke[3:6], dtype=float)

            x_values = [start[0], end[0]]
            y_values = [start[1], end[1]]
            z_values = [start[2], end[2]]

            # Update bounds
            x_min, x_max = min(x_min, *x_values), max(x_max, *x_values)
            y_min, y_max = min(y_min, *y_values), max(y_max, *y_values)
            z_min, z_max = min(z_min, *z_values), max(z_max, *z_values)

            ax.plot(x_values, y_values, z_values, color='black', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 2:
            # Circle face
            x_values, y_values, z_values = plot_circle(stroke)
            ax.plot(x_values, y_values, z_values, color='red', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 4:
            # Arc
            x_values, y_values, z_values = plot_arc(stroke)
            ax.plot(x_values, y_values, z_values, color='blue', alpha=1, linewidth=0.5)
            continue
        

        if stroke[-1] == 5:
            # Spline encoded as 3 control points + type 5
            p0 = np.array(stroke[0:3], dtype=float)
            p1 = np.array(stroke[3:6], dtype=float)
            p2 = np.array(stroke[6:9], dtype=float)

            # Quadratic Bézier sampling: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
            t = np.linspace(0.0, 1.0, 100)
            one_minus_t = 1.0 - t
            bez_x = (one_minus_t**2) * p0[0] + 2 * one_minus_t * t * p1[0] + (t**2) * p2[0]
            bez_y = (one_minus_t**2) * p0[1] + 2 * one_minus_t * t * p1[1] + (t**2) * p2[1]
            bez_z = (one_minus_t**2) * p0[2] + 2 * one_minus_t * t * p1[2] + (t**2) * p2[2]

            # Optional slight perturbation for hand-draw effect (kept tiny)
            perturbations = np.random.normal(0, perturb_factor, (100, 3))
            bez_x = bez_x + perturbations[:, 0]
            bez_y = bez_y + perturbations[:, 1]
            bez_z = bez_z + perturbations[:, 2]

            # Update bounds
            x_min, x_max = min(x_min, bez_x.min()), max(x_max, bez_x.max())
            y_min, y_max = min(y_min, bez_y.min()), max(y_max, bez_y.max())
            z_min, z_max = min(z_min, bez_z.min()), max(z_max, bez_z.max())

            # Plot the spline
            ax.plot(bez_x, bez_y, bez_z, color='black', alpha=1, linewidth=0.5)
            continue



    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])



    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Show plot
    plt.show()



def plot_arc(stroke):
    import numpy as np

    # Extract start and end points from the stroke
    start_point = np.array(stroke[:3])
    end_point = np.array(stroke[3:6])

    # Generate a straight line with 100 points between start_point and end_point
    t = np.linspace(0, 1, 100)  # Parameter for interpolation
    line_points = (1 - t)[:, None] * start_point + t[:, None] * end_point

    # Return x, y, z coordinates of the line points
    return line_points[:, 0], line_points[:, 1], line_points[:, 2]
