from math import pi, atan2, fmod

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX

from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps

from OCC.Core.Geom import (
    Geom_CylindricalSurface,
    Geom_Circle,
    Geom_Line,
    Geom_BSplineCurve,
)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_SurfaceType,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Circle,
    GeomAbs_BSplineCurve,
)


from math import pi
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Circle, GeomAbs_Sphere
from OCC.Core.gp import gp_Vec

from OCC.Core.gp import gp_Vec, gp_Ax2
from OCC.Core.gp import gp_Pnt, gp_Vec


from typing import List, Dict, Tuple, Any

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev, CubicSpline


import helper

def read_step_file(filename):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status == 1:  # Check if the read was successful
        step_reader.TransferRoot()  # Transfers the whole STEP file
        shape = step_reader.Shape()  # Retrieves the translated shape
        return shape
    else:
        print("filename", filename)
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
            if edge_features is None:
                edge_explorer.Next()
                continue

            edge_duplicate_id = check_duplicate(edge_features, edge_features_list)
            if edge_duplicate_id != -1:
                edge_explorer.Next()
                continue
            
            edge_features_list.append(edge_features)
            
            edge_explorer.Next()
        
        face_explorer.Next()

    return edge_features_list, cylinder_features




# ---------------------------------------------------------------- #


def _safe_curve_and_range(edge):
    # Degenerated edges have no 3D curve
    try:
        if BRep_Tool.Degenerated(edge):
            return None, None, None
    except Exception:
        pass

    res = BRep_Tool.Curve(edge)
    if isinstance(res, tuple) and len(res) == 3:
        return res
    h_curve = res
    if h_curve is None:
        return None, None, None
    adap = BRepAdaptor_Curve(edge)
    return h_curve, adap.FirstParameter(), adap.LastParameter()



# What this code does:
# 1)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 2)Cylinder face: lower Center (3 value), upper Center (3 value), 0, radius, 0, 3
# 3) Sphere: center_x, center_y, center_z, axis_nx,  axis_ny,  axis_nz, 0,        radius,   0,     6
def create_face_node_gnn(face):
    """
    Extract per-face features for a GNN.

    Outputs a list of feature vectors (feats). Encodings:
      - Sphere (full): [center(3), axis_dir(3), 0.0, radius, 0.0, 6]
      - Cylinder (full): [lower_center(3), upper_center(3), 0, radius, 0.0, 3]
      - Plane with full circle: [center(3), normal(3), 0.0, radius, 0.0, 2]
      - Otherwise: []
    """
    feats = []
    surf = BRepAdaptor_Surface(face)
    stype = surf.GetType()

    # --- Sphere (full) -> [center(3), axis_dir(3), 0, radius, 0, 6]
    if stype == GeomAbs_Sphere:
        sph = surf.Sphere()               # gp_Sphere
        center = sph.Location()
        radius = sph.Radius()
        ax3 = sph.Position()              # gp_Ax3
        axis_dir = ax3.Direction()        # gp_Dir (north-pole)

        # NOTE: u/v spans are computed but not used to exclude spherical patches (yet)
        u_min, u_max = surf.FirstUParameter(), surf.LastUParameter()
        v_min, v_max = surf.FirstVParameter(), surf.LastVParameter()
        _u_span = abs(u_max - u_min)
        _v_span = abs(v_max - v_min)

        feats.append([
            center.X(), center.Y(), center.Z(),
            axis_dir.X(), axis_dir.Y(), axis_dir.Z(),
            0.0, float(radius), 0.0, 6
        ])
        return feats

    # --- Cylinder (reworked) ---
    if stype == GeomAbs_Cylinder:
        # Decide if it's a "full" cylinder wall by summing circular edge spans
        total_angle = 0.0
        it = TopExp_Explorer(face, TopAbs_EDGE)
        while it.More():
            e = topods.Edge(it.Current())
            h_curve, first, last = _safe_curve_and_range(e)
            if h_curve is not None:
                cadap = BRepAdaptor_Curve(e)
                if cadap.GetType() == GeomAbs_Circle and first is not None and last is not None:
                    total_angle += abs(last - first)
            it.Next()

        # Looks like an open/partial cylinder → skip
        if total_angle < 2*pi - 1e-3:
            return feats

        cyl = surf.Cylinder()             # gp_Cylinder
        radius = float(cyl.Radius())
        axis = cyl.Axis()                 # gp_Ax1
        axis_loc = axis.Location()        # gp_Pnt
        axis_dir = axis.Direction()       # gp_Dir (unit)

        # Use surface params; on cylinders, v is linear along axis
        u_min, u_max = surf.FirstUParameter(), surf.LastUParameter()
        v_min, v_max = surf.FirstVParameter(), surf.LastVParameter()

        # Evaluate two surface points at extremes of v (any valid u works)
        S = BRep_Tool.Surface(face)       # Handle(Geom_Surface)
        u_ref = u_min
        p_low = S.Value(u_ref, v_min)     # gp_Pnt
        p_up  = S.Value(u_ref, v_max)     # gp_Pnt

        # Helper: orthogonal projection of a point P to axis (axis_loc + t * axis_dir)
        def _project_to_axis(P: gp_Pnt):
            # Vector from axis origin to P
            dx = P.X() - axis_loc.X()
            dy = P.Y() - axis_loc.Y()
            dz = P.Z() - axis_loc.Z()

            # axis_dir is gp_Dir (unit); t = dot(v, dir)
            t = dx * axis_dir.X() + dy * axis_dir.Y() + dz * axis_dir.Z()

            # Projected center C = axis_loc + t * axis_dir
            C = gp_Pnt(
                axis_loc.X() + t * axis_dir.X(),
                axis_loc.Y() + t * axis_dir.Y(),
                axis_loc.Z() + t * axis_dir.Z()
            )
            return C, t

        c_low, t_low = _project_to_axis(p_low)
        c_up,  t_up  = _project_to_axis(p_up)

        # Ensure ordering: lower center first (smaller t along axis direction)
        if t_up < t_low:
            c_low, c_up = c_up, c_low
            t_low, t_up = t_up, t_low

        # 2) Cylinder face features (new):
        #    [lower_center(3), upper_center(3), 0, radius, 0.0, 3]
        feats.append([
            c_low.X(), c_low.Y(), c_low.Z(),
            c_up.X(),  c_up.Y(),  c_up.Z(),
            0, 
            radius,
            0.0, 3
        ])
        return feats

    # --- Plane with full circle (unchanged logic) ---
    if stype == GeomAbs_Plane:
        it = TopExp_Explorer(face, TopAbs_EDGE)
        while it.More():
            e = topods.Edge(it.Current())
            h_curve, first, last = _safe_curve_and_range(e)
            if h_curve is None:
                it.Next(); continue

            cadap = BRepAdaptor_Curve(e)
            if cadap.GetType() == GeomAbs_Circle and first is not None and last is not None:
                if abs(abs(last - first) - 2*pi) < 0.1:
                    gcirc = cadap.Circle()
                    center = gcirc.Location()
                    normal = gcirc.Axis().Direction()
                    r = gcirc.Radius()
                    feats.append([
                        center.X(), center.Y(), center.Z(),
                        normal.X(), normal.Y(), normal.Z(),
                        0.0, float(r), 0.0, 2
                    ])
            it.Next()
        return feats

    return feats





# What this code does:
# 1)Straight Line: Point_1 (3 value), Point_2 (3 value), 0, 0, 0, 1
# 2)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 3)Cylinder face: Center (3 value), normal (3 value), height, radius, 0, 3
# 4)Arc: center (3 value), normal (3 value), radius (1 value), angle_start (1 value), sweep (1 value), 4
# 5)Spline: Control_point_1 (3 value), Control_point_2 (3 value), Control_point_3 (3 value), 5
# 6) Sphere: center_x, center_y, center_z, axis_nx,  axis_ny,  axis_nz, 0,        radius,   0,     6

def create_edge_node(edge):
    h_curve, u_first, u_last = _safe_curve_and_range(edge)
    if h_curve is None:
        return None

    adap = BRepAdaptor_Curve(edge)
    ctype = adap.GetType()

    def p2t(p):
        return (float(p.X()), float(p.Y()), float(p.Z()))

    # --- ARC (quarter circle) detection ---
    if ctype == GeomAbs_Circle:
        circ   = adap.Circle()
        center = circ.Location()
        radius = circ.Radius()

        p_start = adap.Value(u_first)
        p_end   = adap.Value(u_last)

        # Measure swept angle to confirm quarter arc
        # Build OCC local frame so angles are consistent:
        ax2  = gp_Ax2(center, circ.Axis().Direction())
        xdir = ax2.XDirection(); ydir = ax2.YDirection()

        def angle_of(p):
            vx = (p.X()-center.X())*xdir.X() + (p.Y()-center.Y())*xdir.Y() + (p.Z()-center.Z())*xdir.Z()
            vy = (p.X()-center.X())*ydir.X() + (p.Y()-center.Y())*ydir.Y() + (p.Z()-center.Z())*ydir.Z()
            return atan2(vy, vx)

        a0 = angle_of(p_start); a1 = angle_of(p_end)
        # shortest signed difference into (-pi, pi]
        d = a1 - a0
        if d >  pi: d -= 2*pi
        if d <= -pi: d += 2*pi

        if abs(abs(d) - pi/2) < 1e-4:
            return [
                float(p_start.X()), float(p_start.Y()), float(p_start.Z()),
                float(p_end.X()),   float(p_end.Y()),   float(p_end.Z()),
                float(center.X()),  float(center.Y()),  float(center.Z()),
                4
            ]

    # --- BSPLINE (3 control points) ---
    if ctype == GeomAbs_BSplineCurve:
        bs = Geom_BSplineCurve.DownCast(h_curve)
        if bs is not None and bs.NbPoles() == 3:
            cp1 = p2t(bs.Pole(1))
            cp2 = p2t(bs.Pole(2))
            cp3 = p2t(bs.Pole(3))
            return [*cp1, *cp2, *cp3, 5]

        # Fallback: sample start/mid/end to preserve your 3-pt encoding
        p_start = adap.Value(u_first)
        p_mid   = adap.Value(0.5 * (u_first + u_last))
        p_end   = adap.Value(u_last)
        return [*p2t(p_start), *p2t(p_mid), *p2t(p_end), 5]

    # --- STRAIGHT LINE (default) ---
    p_start = adap.Value(u_first)
    p_end   = adap.Value(u_last)
    return [*p2t(p_start), *p2t(p_end), 0.0, 0.0, 0.0, 1]





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


    # Plot all strokes in blue with perturbations
    for idx, stroke in enumerate(stroke_node_features):
        start, end = stroke[:3], stroke[3:6]
        

        # Update min and max limits based on strokes (ignoring circles)
        if stroke[-1] == 1:
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
            # Circle: [center(3), normal(3), 0, radius, 0, 2]
            cx, cy, cz, nx, ny, nz, _, r, _ = (float(v) for v in stroke[:9])
            center = np.array([cx, cy, cz], dtype=float)
            normal = np.array([nx, ny, nz], dtype=float)

            # Normalize normal; if degenerate, skip
            nlen = np.linalg.norm(normal)
            if nlen < 1e-12:
                continue
            normal /= nlen

            # Build in-plane orthonormal basis (xdir, ydir) ⟂ normal
            up = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(normal, up)
            if np.linalg.norm(xdir) < 1e-12:
                up = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(normal, up)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(normal, xdir)

            # Sample full circle
            theta = np.linspace(0.0, 2.0 * np.pi, 200)
            pts = center[None, :] + r * (np.cos(theta)[:, None] * xdir + np.sin(theta)[:, None] * ydir)
            x_values, y_values, z_values = pts[:, 0], pts[:, 1], pts[:, 2]

            # (By your convention, we do NOT update bounds for circles)
            ax.plot(x_values, y_values, z_values, color='black', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 3:
            # Cylinder face: [lower_center(3), upper_center(3), 0.0, radius, 0.0, 3]
            # total length = 10; last is type code (3)
            lx, ly, lz, ux, uy, uz, _zero0, r, _zero1 = (float(v) for v in stroke[:9])

            L = np.array([lx, ly, lz], dtype=float)  # lower circle center
            U = np.array([ux, uy, uz], dtype=float)  # upper circle center
            axis_vec = U - L
            h = np.linalg.norm(axis_vec)             # cylinder height

            if h < 1e-12 or r <= 0.0:
                continue

            # unit axis direction (lower -> upper)
            n = axis_vec / h

            # build orthonormal basis in the rim plane (perpendicular to n)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(n, ref)
            if np.linalg.norm(xdir) < 1e-12:
                ref = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(n, ref)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(n, xdir)

            # four boundary directions at 0°, 90°, 180°, 270°
            for ang in (0.0, 0.5*np.pi, np.pi, 1.5*np.pi):
                radial = np.cos(ang)*xdir + np.sin(ang)*ydir

                # corresponding boundary points on lower/upper rims
                p_low = L + r * radial
                p_up  = U + r * radial

                ax.plot([p_low[0], p_up[0]],
                        [p_low[1], p_up[1]],
                        [p_low[2], p_up[2]],
                        color='black', alpha=1, linewidth=0.5)

                # bounds
                x_min = min(x_min, p_low[0], p_up[0]); x_max = max(x_max, p_low[0], p_up[0])
                y_min = min(y_min, p_low[1], p_up[1]); y_max = max(y_max, p_low[1], p_up[1])
                z_min = min(z_min, p_low[2], p_up[2]); z_max = max(z_max, p_low[2], p_up[2])
            continue
            
        if stroke[-1] == 4:
            # Arc encoded as: [sx,sy,sz, ex,ey,ez, cx,cy,cz, 4]
            sx, sy, sz, ex, ey, ez, cx, cy, cz = (float(v) for v in stroke[:9])
            S = np.array([sx, sy, sz])
            E = np.array([ex, ey, ez])
            C = np.array([cx, cy, cz])

            vS = S - C; vE = E - C
            rS = np.linalg.norm(vS); rE = np.linalg.norm(vE)
            r  = 0.5 * (rS + rE)
            if r < 1e-12:
                continue

            vS /= rS; vE /= rE  # unit
            # normal from start to end (right-hand)
            n = np.cross(vS, vE)
            nlen = np.linalg.norm(n)
            if nlen < 1e-12:
                # degenerate (collinear) — just draw a line
                ax.plot([sx, ex], [sy, ey], [sz, ez], color='black', alpha=1, linewidth=0.5)
                continue
            n /= nlen

            # In-plane basis: xdir along start vector, ydir = n × xdir
            xdir = vS
            ydir = np.cross(n, xdir)

            # signed sweep from S to E
            cosang = np.clip(np.dot(vS, vE), -1.0, 1.0)
            sweep  = np.arccos(cosang)
            # orientation sign
            if np.dot(n, np.cross(vS, vE)) < 0:
                sweep = -sweep

            # sample (should be ±pi/2 for quarter arcs)
            theta = np.linspace(0.0, sweep, 100)
            pts = C + r*(np.cos(theta)[:,None]*xdir + np.sin(theta)[:,None]*ydir)
            x_values, y_values, z_values = pts[:,0], pts[:,1], pts[:,2]

            # update bounds
            x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
            y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
            z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

            ax.plot(x_values, y_values, z_values, color='black', alpha=1, linewidth=0.5)
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

            # Update bounds
            x_min, x_max = min(x_min, bez_x.min()), max(x_max, bez_x.max())
            y_min, y_max = min(y_min, bez_y.min()), max(y_max, bez_y.max())
            z_min, z_max = min(z_min, bez_z.min()), max(z_max, bez_z.max())

            # Plot the spline
            ax.plot(bez_x, bez_y, bez_z, color='black', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 6:
            # Sphere: [cx,cy,cz, nx,ny,nz, 0, r, 0, 6]
            cx, cy, cz, nx, ny, nz, _, r, _ = (float(v) for v in stroke[:9])
            C = np.array([cx, cy, cz], dtype=float)
            n = np.array([nx, ny, nz], dtype=float)

            # Normalize axis
            nlen = np.linalg.norm(n)
            if nlen < 1e-12:
                continue
            n /= nlen

            # Build an orthonormal basis in the equatorial plane (xdir, ydir) ⟂ n
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(n, ref)
            if np.linalg.norm(xdir) < 1e-12:
                ref = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(n, ref)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(n, xdir)

            # Four plane normals for four great circles
            normals = [
                n,                                   # equator
                xdir,                                # meridian 1
                ydir,                                # meridian 2
                (xdir + ydir) / np.linalg.norm(xdir + ydir)  # tilted meridian (45°)
            ]

            theta = np.linspace(0.0, 2.0*np.pi, 200)
            c, s = np.cos(theta), np.sin(theta)

            for pn in normals:
                # Build in-plane basis (u,v) for this circle plane (normal = pn)
                # Choose a ref not parallel to pn
                ref2 = np.array([0.0, 0.0, 1.0]) if abs(pn[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
                u = np.cross(pn, ref2)
                if np.linalg.norm(u) < 1e-12:
                    ref2 = np.array([0.0, 1.0, 0.0])
                    u = np.cross(pn, ref2)
                u /= np.linalg.norm(u)
                v = np.cross(pn, u)

                pts = C[None, :] + r * (c[:, None]*u[None, :] + s[:, None]*v[None, :])
                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

                # update bounds
                x_min = min(x_min, x.min()); x_max = max(x_max, x.max())
                y_min = min(y_min, y.min()); y_max = max(y_max, y.max())
                z_min = min(z_min, z.min()); z_max = max(z_max, z.max())

                ax.plot(x, y, z, color='black', alpha=1, linewidth=0.5)
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



def vis_labels(stroke_node_features, labels):
    """
    stroke_node_features: array-like of shape (n, 11)
    labels: list/array of length n with values in {-1,0,1,2,3,4}
    """
    # simple palette: -1 gray, others distinct
    palette = {
        -1: (0, 0, 0),            # black
         0: plt.cm.tab10(0),
         1: plt.cm.tab10(1),
         2: plt.cm.tab10(2),
         3: plt.cm.tab10(3),
         4: plt.cm.tab10(4),
    }

    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')  # Turn off axis background and borders

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    # Plot strokes with label-based colors
    for idx, stroke in enumerate(stroke_node_features):
        start, end = stroke[:3], stroke[3:6]
        col = palette.get(labels[idx], (0.2, 0.2, 0.2))  # fallback dark gray


        # Update min and max limits based on strokes (ignoring circles)
        if stroke[-1] == 1:
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

            ax.plot(x_values, y_values, z_values, color=col, alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 2:
            # Circle: [center(3), normal(3), 0, radius, 0, 2]
            cx, cy, cz, nx, ny, nz, _, r, _ = (float(v) for v in stroke[:9])
            center = np.array([cx, cy, cz], dtype=float)
            normal = np.array([nx, ny, nz], dtype=float)

            # Normalize normal; if degenerate, skip
            nlen = np.linalg.norm(normal)
            if nlen < 1e-12:
                continue
            normal /= nlen

            # Build in-plane orthonormal basis (xdir, ydir) ⟂ normal
            up = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(normal, up)
            if np.linalg.norm(xdir) < 1e-12:
                up = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(normal, up)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(normal, xdir)

            # Sample full circle
            theta = np.linspace(0.0, 2.0 * np.pi, 200)
            pts = center[None, :] + r * (np.cos(theta)[:, None] * xdir + np.sin(theta)[:, None] * ydir)
            x_values, y_values, z_values = pts[:, 0], pts[:, 1], pts[:, 2]

            # (By your convention, we do NOT update bounds for circles)
            ax.plot(x_values, y_values, z_values, color=col, alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 3:
            # Cylinder face: [lower_center(3), upper_center(3), 0.0, radius, 0.0, 3]
            # total length = 10; last is type code (3)
            lx, ly, lz, ux, uy, uz, _zero0, r, _zero1 = (float(v) for v in stroke[:9])

            L = np.array([lx, ly, lz], dtype=float)  # lower circle center
            U = np.array([ux, uy, uz], dtype=float)  # upper circle center
            axis_vec = U - L
            h = np.linalg.norm(axis_vec)             # cylinder height

            if h < 1e-12 or r <= 0.0:
                continue

            # unit axis direction (lower -> upper)
            n = axis_vec / h

            # build orthonormal basis in the rim plane (perpendicular to n)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(n, ref)
            if np.linalg.norm(xdir) < 1e-12:
                ref = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(n, ref)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(n, xdir)

            # four boundary directions at 0°, 90°, 180°, 270°
            for ang in (0.0, 0.5*np.pi, np.pi, 1.5*np.pi):
                radial = np.cos(ang)*xdir + np.sin(ang)*ydir

                # corresponding boundary points on lower/upper rims
                p_low = L + r * radial
                p_up  = U + r * radial

                ax.plot([p_low[0], p_up[0]],
                        [p_low[1], p_up[1]],
                        [p_low[2], p_up[2]],
                        color=col, alpha=1, linewidth=0.5)

                # bounds
                x_min = min(x_min, p_low[0], p_up[0]); x_max = max(x_max, p_low[0], p_up[0])
                y_min = min(y_min, p_low[1], p_up[1]); y_max = max(y_max, p_low[1], p_up[1])
                z_min = min(z_min, p_low[2], p_up[2]); z_max = max(z_max, p_low[2], p_up[2])
            continue
            
        if stroke[-1] == 4:
            # Arc encoded as: [sx,sy,sz, ex,ey,ez, cx,cy,cz, 4]
            sx, sy, sz, ex, ey, ez, cx, cy, cz = (float(v) for v in stroke[:9])
            S = np.array([sx, sy, sz])
            E = np.array([ex, ey, ez])
            C = np.array([cx, cy, cz])

            vS = S - C; vE = E - C
            rS = np.linalg.norm(vS); rE = np.linalg.norm(vE)
            r  = 0.5 * (rS + rE)
            if r < 1e-12:
                continue

            vS /= rS; vE /= rE  # unit
            # normal from start to end (right-hand)
            n = np.cross(vS, vE)
            nlen = np.linalg.norm(n)
            if nlen < 1e-12:
                # degenerate (collinear) — just draw a line
                ax.plot([sx, ex], [sy, ey], [sz, ez], color=col, alpha=1, linewidth=0.5)
                continue
            n /= nlen

            # In-plane basis: xdir along start vector, ydir = n × xdir
            xdir = vS
            ydir = np.cross(n, xdir)

            # signed sweep from S to E
            cosang = np.clip(np.dot(vS, vE), -1.0, 1.0)
            sweep  = np.arccos(cosang)
            # orientation sign
            if np.dot(n, np.cross(vS, vE)) < 0:
                sweep = -sweep

            # sample (should be ±pi/2 for quarter arcs)
            theta = np.linspace(0.0, sweep, 100)
            pts = C + r*(np.cos(theta)[:,None]*xdir + np.sin(theta)[:,None]*ydir)
            x_values, y_values, z_values = pts[:,0], pts[:,1], pts[:,2]

            # update bounds
            x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
            y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
            z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

            ax.plot(x_values, y_values, z_values, color=col, alpha=1, linewidth=0.5)
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

            # Update bounds
            x_min, x_max = min(x_min, bez_x.min()), max(x_max, bez_x.max())
            y_min, y_max = min(y_min, bez_y.min()), max(y_max, bez_y.max())
            z_min, z_max = min(z_min, bez_z.min()), max(z_max, bez_z.max())

            # Plot the spline
            ax.plot(bez_x, bez_y, bez_z, color=col, alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 6:
            # Sphere: [cx,cy,cz, nx,ny,nz, 0, r, 0, 6]
            cx, cy, cz, nx, ny, nz, _, r, _ = (float(v) for v in stroke[:9])
            C = np.array([cx, cy, cz], dtype=float)
            n = np.array([nx, ny, nz], dtype=float)

            # Normalize axis
            nlen = np.linalg.norm(n)
            if nlen < 1e-12:
                continue
            n /= nlen

            # Build an orthonormal basis in the equatorial plane (xdir, ydir) ⟂ n
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(n, ref)
            if np.linalg.norm(xdir) < 1e-12:
                ref = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(n, ref)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(n, xdir)

            # Four plane normals for four great circles
            normals = [
                n,                                   # equator
                xdir,                                # meridian 1
                ydir,                                # meridian 2
                (xdir + ydir) / np.linalg.norm(xdir + ydir)  # tilted meridian (45°)
            ]

            theta = np.linspace(0.0, 2.0*np.pi, 200)
            c, s = np.cos(theta), np.sin(theta)

            for pn in normals:
                # Build in-plane basis (u,v) for this circle plane (normal = pn)
                # Choose a ref not parallel to pn
                ref2 = np.array([0.0, 0.0, 1.0]) if abs(pn[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
                u = np.cross(pn, ref2)
                if np.linalg.norm(u) < 1e-12:
                    ref2 = np.array([0.0, 1.0, 0.0])
                    u = np.cross(pn, ref2)
                u /= np.linalg.norm(u)
                v = np.cross(pn, u)

                pts = C[None, :] + r * (c[:, None]*u[None, :] + s[:, None]*v[None, :])
                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

                # update bounds
                x_min = min(x_min, x.min()); x_max = max(x_max, x.max())
                y_min = min(y_min, y.min()); y_max = max(y_max, y.max())
                z_min = min(z_min, z.min()); z_max = max(z_max, z.max())

                ax.plot(x, y, z, color=col, alpha=1, linewidth=0.5)
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





def vis_cad_op(stroke_node_features, cad_correspondance, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')

    # bounds
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    black_segments = []  # list of (x_values, y_values, z_values)
    red_segments   = []

    def add_segment(xs, ys, zs, update_bounds=True, color_is_red=False):
        nonlocal x_min, x_max, y_min, y_max, z_min, z_max
        if update_bounds:
            x_min = min(x_min, np.min(xs)); x_max = max(x_max, np.max(xs))
            y_min = min(y_min, np.min(ys)); y_max = max(y_max, np.max(ys))
            z_min = min(z_min, np.min(zs)); z_max = max(z_max, np.max(zs))
        (red_segments if color_is_red else black_segments).append((xs, ys, zs))

    # --------- pass 1: prepare all polylines and update bounds ----------
    for idx, stroke in enumerate(stroke_node_features):
        is_red = bool(cad_correspondance[idx, i] == 1)

        if stroke[-1] == 1:
            # straight line
            start = np.array(stroke[0:3], dtype=float)
            end   = np.array(stroke[3:6], dtype=float)
            xs = np.array([start[0], end[0]]); ys = np.array([start[1], end[1]]); zs = np.array([start[2], end[2]])
            add_segment(xs, ys, zs, update_bounds=True, color_is_red=is_red)
            continue

        if stroke[-1] == 2:
            # circle (do NOT update bounds per your convention)
            cx, cy, cz, nx, ny, nz, _, r, _ = (float(v) for v in stroke[:9])
            center = np.array([cx, cy, cz]); normal = np.array([nx, ny, nz])
            nlen = np.linalg.norm(normal)
            if nlen < 1e-12: 
                continue
            normal /= nlen
            up = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(normal, up)
            if np.linalg.norm(xdir) < 1e-12:
                up = np.array([0.0, 1.0, 0.0]); xdir = np.cross(normal, up)
            xdir /= np.linalg.norm(xdir); ydir = np.cross(normal, xdir)
            theta = np.linspace(0.0, 2.0*np.pi, 200)
            pts = center[None,:] + r*(np.cos(theta)[:,None]*xdir + np.sin(theta)[:,None]*ydir)
            add_segment(pts[:,0], pts[:,1], pts[:,2], update_bounds=False, color_is_red=is_red)
            continue

        if stroke[-1] == 3:
            # Cylinder face: [lower_center(3), upper_center(3), 0.0, radius, 0.0, 3]
            # total length = 10; last is type code (3)
            lx, ly, lz, ux, uy, uz, _zero0, r, _zero1 = (float(v) for v in stroke[:9])

            L = np.array([lx, ly, lz], dtype=float)  # lower circle center
            U = np.array([ux, uy, uz], dtype=float)  # upper circle center
            axis_vec = U - L
            h = np.linalg.norm(axis_vec)             # cylinder height

            if h < 1e-12 or r <= 0.0:
                continue

            # unit axis direction (lower -> upper)
            n = axis_vec / h

            # build orthonormal basis in the rim plane (perpendicular to n)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(n, ref)
            if np.linalg.norm(xdir) < 1e-12:
                ref = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(n, ref)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(n, xdir)

            # four boundary directions at 0°, 90°, 180°, 270°
            for ang in (0.0, 0.5*np.pi, np.pi, 1.5*np.pi):
                radial = np.cos(ang)*xdir + np.sin(ang)*ydir

                # corresponding boundary points on lower/upper rims
                p_low = L + r * radial
                p_up  = U + r * radial

                ax.plot([p_low[0], p_up[0]],
                        [p_low[1], p_up[1]],
                        [p_low[2], p_up[2]],
                        color=col, alpha=1, linewidth=0.5)

                # bounds
                x_min = min(x_min, p_low[0], p_up[0]); x_max = max(x_max, p_low[0], p_up[0])
                y_min = min(y_min, p_low[1], p_up[1]); y_max = max(y_max, p_low[1], p_up[1])
                z_min = min(z_min, p_low[2], p_up[2]); z_max = max(z_max, p_low[2], p_up[2])
            continue

        if stroke[-1] == 4:
            # arc: [S(3), E(3), C(3), 4]
            sx, sy, sz, ex, ey, ez, cx, cy, cz = (float(v) for v in stroke[:9])
            S = np.array([sx, sy, sz]); E = np.array([ex, ey, ez]); C = np.array([cx, cy, cz])
            vS = S - C; vE = E - C
            rS = np.linalg.norm(vS); rE = np.linalg.norm(vE); r = 0.5*(rS + rE)
            if r < 1e-12:
                continue
            if rS < 1e-12 or rE < 1e-12:
                xs = np.array([S[0], E[0]]); ys = np.array([S[1], E[1]]); zs = np.array([S[2], E[2]])
                add_segment(xs, ys, zs, update_bounds=True, color_is_red=is_red)
                continue
            vS /= rS; vE /= rE
            n = np.cross(vS, vE); nlen = np.linalg.norm(n)
            if nlen < 1e-12:
                xs = np.array([S[0], E[0]]); ys = np.array([S[1], E[1]]); zs = np.array([S[2], E[2]])
                add_segment(xs, ys, zs, update_bounds=True, color_is_red=is_red)
                continue
            n /= nlen
            xdir = vS; ydir = np.cross(n, xdir)
            cosang = np.clip(np.dot(vS, vE), -1.0, 1.0)
            sweep  = np.arccos(cosang)
            if np.dot(n, np.cross(vS, vE)) < 0: sweep = -sweep
            theta = np.linspace(0.0, sweep, 100)
            pts = C + r*(np.cos(theta)[:,None]*xdir + np.sin(theta)[:,None]*ydir)
            add_segment(pts[:,0], pts[:,1], pts[:,2], update_bounds=True, color_is_red=is_red)
            continue

        if stroke[-1] == 5:
            # quadratic Bézier (3 control points)
            p0 = np.array(stroke[0:3], dtype=float)
            p1 = np.array(stroke[3:6], dtype=float)
            p2 = np.array(stroke[6:9], dtype=float)
            t = np.linspace(0.0, 1.0, 100); omt = 1.0 - t
            xs = (omt**2)*p0[0] + 2*omt*t*p1[0] + (t**2)*p2[0]
            ys = (omt**2)*p0[1] + 2*omt*t*p1[1] + (t**2)*p2[1]
            zs = (omt**2)*p0[2] + 2*omt*t*p1[2] + (t**2)*p2[2]
            add_segment(xs, ys, zs, update_bounds=True, color_is_red=is_red)
            continue

        if stroke[-1] == 6:
            # sphere: draw 4 great circles
            cx, cy, cz, nx, ny, nz, _, r, _ = (float(v) for v in stroke[:9])
            C = np.array([cx, cy, cz]); n = np.array([nx, ny, nz]); nlen = np.linalg.norm(n)
            if nlen < 1e-12: 
                continue
            n /= nlen
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(n, ref)
            if np.linalg.norm(xdir) < 1e-12:
                ref = np.array([0.0, 1.0, 0.0]); xdir = np.cross(n, ref)
            xdir /= np.linalg.norm(xdir); ydir = np.cross(n, xdir)
            normals = [
                n,
                xdir,
                ydir,
                (xdir + ydir) / np.linalg.norm(xdir + ydir)
            ]
            theta = np.linspace(0.0, 2.0*np.pi, 200); c, s = np.cos(theta), np.sin(theta)
            for pn in normals:
                ref2 = np.array([0.0, 0.0, 1.0]) if abs(pn[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
                u = np.cross(pn, ref2)
                if np.linalg.norm(u) < 1e-12:
                    ref2 = np.array([0.0, 1.0, 0.0]); u = np.cross(pn, ref2)
                u /= np.linalg.norm(u); v = np.cross(pn, u)
                pts = C[None,:] + r*(c[:,None]*u[None,:] + s[:,None]*v[None,:])
                add_segment(pts[:,0], pts[:,1], pts[:,2], update_bounds=True, color_is_red=is_red)
            continue

    # rescale (keep your "ignore circle bounds" behavior already encoded above)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim([x_center - max_diff/2, x_center + max_diff/2])
    ax.set_ylim([y_center - max_diff/2, y_center + max_diff/2])
    ax.set_zlim([z_center - max_diff/2, z_center + max_diff/2])

    # --------- pass 2: draw blacks first, then reds ----------
    for xs, ys, zs in black_segments:
        ax.plot(xs, ys, zs, color='black', alpha=1, linewidth=0.5, zorder=1)
    for xs, ys, zs in red_segments:
        ax.plot(xs, ys, zs, color='red',   alpha=1, linewidth=0.5, zorder=2)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.show()
