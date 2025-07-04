from pathlib import Path
from build123d import *
import os
import numpy as np


def build_sketch(count, canvas, Points_list, output, data_dir, tempt_idx = 0):

    # if tempt_idx == 0:
    #     brep_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")
    #     stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    # else:
    #     brep_dir = os.path.join(data_dir, "canvas", f"tempt_{tempt_idx}.step")
    #     stl_dir = os.path.join(data_dir, "canvas", f"tempt_{tempt_idx}.stl")

    if count == 0:
        with BuildSketch():
            with BuildLine():
                lines = []
                for i in range(0, len(Points_list), 2):
                    start_point_sublist = Points_list[i]
                    end_point_sublist = Points_list[i+1]
                    start_point = (start_point_sublist[0],
                                start_point_sublist[1], 
                                start_point_sublist[2])
                    
                    
                    end_point = (end_point_sublist[0],
                                end_point_sublist[1], 
                                end_point_sublist[2])


                    line = Line(start_point, end_point)
                    lines.append(line)

            perimeter = make_face()

        if output:
            _ = perimeter.export_stl(stl_dir)
            _ = perimeter.export_step(brep_dir)

        return perimeter
    
    else:
        with canvas: 
            with BuildSketch():
                with BuildLine():
                    lines = []
                    for i in range(0, len(Points_list), 2):
                        start_point_sublist = Points_list[i]
                        end_point_sublist = Points_list[i+1]
                        start_point = (start_point_sublist[0],
                                    start_point_sublist[1], 
                                    start_point_sublist[2])
                        
                        
                        end_point = (end_point_sublist[0],
                                    end_point_sublist[1], 
                                    end_point_sublist[2])


                        line = Line(start_point, end_point)
                        lines.append(line)

                perimeter = make_face()

            if output:
                _ = canvas.part.export_stl(stl_dir)
                _ = canvas.part.export_step(brep_dir)


    return perimeter





def build_circle(count, radius, point, normal, output, data_dir):
    brep_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")
    stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")

    
    with BuildSketch(Plane(origin=(point[0], point[1], point[2]), z_dir=(normal[0], normal[1], normal[2])) )as perimeter:
        Circle(radius = radius)
    
    if output:
        _ = perimeter.sketch.export_stl(stl_dir)
        _ = perimeter.sketch.export_step(brep_dir)

    return perimeter.sketch




def build_extrude(count, canvas, target_face, extrude_amount, output, data_dir):
    # stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    # step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    if canvas != None:
        with canvas: 
            extrude( target_face, amount=extrude_amount)

    else:
        with BuildPart() as canvas:
            extrude( target_face, amount=extrude_amount)

    if output:
        _ = canvas.part.export_stl(stl_dir)
        _ = canvas.part.export_step(step_dir)


    return canvas


def build_subtract(count, canvas, target_face, extrude_amount, output, data_dir):
    stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    with canvas:
        extrude( target_face, amount= extrude_amount, mode=Mode.SUBTRACT)

    if output:
        _ = canvas.part.export_stl(stl_dir)
        _ = canvas.part.export_step(step_dir)


    return canvas


def build_fillet(count, canvas, target_edge, radius, output, data_dir):
    stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    with canvas:
        fillet(target_edge, radius)
    
    if output:
        _ = canvas.part.export_stl(stl_dir)
        _ = canvas.part.export_step(step_dir)


    return canvas


def build_chamfer(count, canvas, target_edge, radius, output, data_dir):
    stl_dir = os.path.join(data_dir, "canvas", f"vis_{count}.stl")
    step_dir = os.path.join(data_dir, "canvas", f"brep_{count}.step")

    with canvas:
        chamfer(target_edge, radius)
    
    if output:
        _ = canvas.part.export_stl(stl_dir)
        _ = canvas.part.export_step(step_dir)


    return canvas



def simulate_extrude(sketch, amount):
    with BuildPart() as temp:
        extrude(sketch, amount=amount)
    return temp.part

def has_volume_overlap(canvas, new_part):
    intersection = canvas & new_part
    return intersection.volume > new_part.volume * 0.5

