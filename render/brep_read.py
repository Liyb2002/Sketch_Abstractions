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
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve

from typing import List, Dict, Tuple, Any

import os
from pathlib import Path

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

    print("edge_features_list", edge_features_list)
    return edge_features_list, cylinder_features




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
def create_edge_node(edge):

    # Get the underlying geometry of the edge
    edge_curve_handle, first, last = BRep_Tool.Curve(edge)
    adaptor = GeomAdaptor_Curve(edge_curve_handle)
    curve_type = adaptor.GetType()


    if curve_type == GeomAbs_Circle and abs(last - first) < 6.27:
        start_point = adaptor.Value(first)
        end_point = adaptor.Value(last)
        radius = adaptor.Circle().Radius()
        center = adaptor.Circle().Location()

        return [start_point.X(), start_point.Y(), start_point.Z(), end_point.X(), end_point.Y(), end_point.Z(), center.X(),center.Y(), center.Z() , 4]
 



    properties = GProp_GProps()
    brepgprop.LinearProperties(edge, properties)
    length = properties.Mass()

    vertices = []
    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while vertex_explorer.More():

        vertex = topods.Vertex(vertex_explorer.Current())
        vertex_coords = BRep_Tool.Pnt(vertex)
        vertices.append([vertex_coords.X(), vertex_coords.Y(), vertex_coords.Z()])
        vertex_explorer.Next()
        
    return [vertices[0][0], vertices[0][1], vertices[0][2], vertices[1][0], vertices[1][1], vertices[1][2], 0, 0, 0, 1]



def check_duplicate(new_feature, feature_list):
    for existing_feature in feature_list:
        if existing_feature == new_feature:
            return 0
    return -1


# ------------------------------------------------------# 

