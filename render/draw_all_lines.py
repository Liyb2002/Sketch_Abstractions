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

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

import os
from pathlib import Path

def read_step_file(filename: str):

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(filename))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP ReadFile failed (status={code}) for {filename}")

    nb_roots = reader.NbRootsForTransfer()
    if nb_roots == 0:
        raise RuntimeError("STEP file has no transferable roots.")
    ok = reader.TransferRoots()

    shape = reader.OneShape()  # combined TopoDS_Shape
    return shape




current_folder = Path.cwd().parent
filename = current_folder / "output" / "root.step"
shape = read_step_file(str(filename))  # STEP reader expects a str
