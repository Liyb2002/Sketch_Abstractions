#!/usr/bin/env python3
"""
shape_optimizer.py

Step 2: Load the pre-sampled 3D strokes from
../input/perturbed_feature_lines.json.

Usage:
  Imported into main_optimizer.py
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, Tuple, Optional



# 1)Straight Line: Point_1 (3 value), Point_2 (3 value), 0, 0, 0, 1
# 2)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 3)Cylinder face: Center (3 value), normal (3 value), height, radius, 0, 3
# 4)Arc: Start S (3 values), End E (3 values), Center C (3 values), 4
# 5)Spline: Control_point_1 (3 value), Control_point_2 (3 value), Control_point_3 (3 value), 5
# 6)Sphere: center_x, center_y, center_z, axis_nx,  axis_ny,  axis_nz, 0,        radius,   0,     6

def load_perturbed_feature_lines(input_dir: Path):
    path_sampled_points = input_dir / "perturbed_feature_lines.json"
    path_feature_lines = input_dir / "feature_lines.json"

    with open(path_sampled_points, "r", encoding="utf-8") as f:
        sample_points = json.load(f)

    with open(path_feature_lines, "r", encoding="utf-8") as f:
        feature_lines = json.load(f)

    print(f"📄 Loaded {len(sample_points)} strokes as sampled points, and {len(feature_lines)} feature lines")
    return sample_points, feature_lines






# ====== Step 3: normalization + union-SDF metrics ======
import numpy as np
from typing import Dict, Tuple, List

def compute_sdf_metrics(executor, sample_points, feature_lines, margin: float = 0.01) -> Dict:
    """
    Compute dry-run SDF metrics for current program vs. sampled strokes.

    Args:
      executor: an instance of program_executor.Executor (already executed)
      sample_points: List[List[[x,y,z], ...]]  # per-stroke sampled points (your loader output)
      feature_lines: List[List[float]]         # per-stroke params, last value == type tag (1..6)
      margin: containment margin in *normalized* units (default 0.01)

    Returns:
      metrics dict with keys:
        - mean_abs_sdf_surface
        - inlier_rate_surface (|sdf| < 0.01)
        - mean_hinge_inside (ReLU(sdf + margin))
        - inside_rate (sdf < -margin)
        - counts: {'surface_pts': int, 'inside_pts': int, 'total_pts': int}
      Also prints a short summary.
    """
    # ---- 1) Collect boxes (min corner o, size s) from executor, exclude bbox
    boxes_world: List[Tuple[np.ndarray, np.ndarray]] = []
    for name, inst in executor.instances.items():
        if name == "bbox":
            continue
        o = inst.T[:3, 3].astype(float).copy()  # min corner
        s = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float)
        boxes_world.append((o, s))

    if len(boxes_world) == 0:
        raise ValueError("No cuboids (excluding bbox) found in executor.")

    # BBox diagonal for normalization (bbox origin is (0,0,0) in this executor)
    L, W, H = executor.bbox.l, executor.bbox.w, executor.bbox.h
    diag = float(np.linalg.norm([L, W, H]))
    if diag <= 0:
        raise ValueError("Invalid bbox diagonal (zero).")

    # ---- 2) Normalize boxes
    boxes_norm = []
    for o, s in boxes_world:
        boxes_norm.append((o / diag, s / diag))

    # ---- 3) Flatten and normalize stroke points; build per-point type array
    pts_list = []
    types_list = []
    for stroke_pts, fl in zip(sample_points, feature_lines):
        t = int(round(float(fl[-1])))  # 1..6 per your spec
        arr = np.asarray(stroke_pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Each stroke in sample_points must be an (M,3) list/array.")
        pts_list.append(arr)
        types_list.append(np.full((arr.shape[0],), t, dtype=np.int32))
    if not pts_list:
        raise ValueError("No stroke points provided.")
    P_world = np.vstack(pts_list)                  # (N,3)
    T = np.concatenate(types_list, axis=0)         # (N,)
    P = P_world / diag                              # normalize

    # ---- 4) Union SDF in normalized space
    sdf = _sdf_box_union(P, boxes_norm)            # (N,)

    # ---- 5) Split by type: 1 -> surface, 2..6 -> inside
    surface_mask = (T == 1)
    inside_mask  = (T != 1)

    # tolerances in normalized units
    tau_surface = 0.01

    # Surface metrics
    if np.any(surface_mask):
        sdf_s = sdf[surface_mask]
        mean_abs_sdf_surface = float(np.mean(np.abs(sdf_s)))
        inlier_rate_surface  = float(np.mean(np.abs(sdf_s) < tau_surface))
    else:
        mean_abs_sdf_surface = float('nan')
        inlier_rate_surface  = float('nan')

    # Inside metrics (with margin)
    if np.any(inside_mask):
        sdf_i = sdf[inside_mask]
        hinge_i = np.maximum(sdf_i + margin, 0.0)
        mean_hinge_inside = float(np.mean(hinge_i))
        inside_rate       = float(np.mean(sdf_i < -margin))
    else:
        mean_hinge_inside = float('nan')
        inside_rate       = float('nan')

    # ---- 6) Print summary
    counts = {
        "surface_pts": int(surface_mask.sum()),
        "inside_pts":  int(inside_mask.sum()),
        "total_pts":   int(P.shape[0]),
    }
    print(
        "SDF metrics (normalized):\n"
        f"  • surface: mean|sdf|={mean_abs_sdf_surface:.5f}, inliers(|sdf|<{tau_surface})={inlier_rate_surface:.3f}\n"
        f"  • inside : mean ReLU(sdf+{margin})={mean_hinge_inside:.5f}, inside_rate(sdf<-{margin})={inside_rate:.3f}\n"
        f"  • counts : {counts}"
    )

    return {
        "mean_abs_sdf_surface": mean_abs_sdf_surface,
        "inlier_rate_surface": inlier_rate_surface,
        "mean_hinge_inside": mean_hinge_inside,
        "inside_rate": inside_rate,
        "counts": counts,
        "diag_scale": diag,
    }


# ---- helpers (internal) ----
def _sdf_box_union(P: np.ndarray, boxes: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    P: (N,3) normalized points
    boxes: list of (o, s) in normalized units
    Returns: (N,) union SDF (negative inside)
    """
    N = P.shape[0]
    sdf = np.full((N,), np.inf, dtype=P.dtype)
    for (o, s) in boxes:
        h = 0.5 * s
        c = o + h
        p = P - c                 # center frame
        q = np.abs(p) - h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside  = np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0)
        sdf_box = outside + inside
        sdf = np.minimum(sdf, sdf_box)
    return sdf



# ====== Vis Function ======
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def _cuboid_edges_from_origin_size(origin: np.ndarray, size: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return 12 boundary edges for an axis-aligned cuboid (min corner `origin`, size `size`)."""
    o = np.asarray(origin, dtype=float)
    s = np.asarray(size, dtype=float)
    x0, y0, z0 = o
    l, w, h = s
    x1, y1, z1 = x0 + l, y0 + w, z0 + h

    v = [
        np.array([x0,y0,z0]), np.array([x1,y0,z0]),
        np.array([x0,y1,z0]), np.array([x1,y1,z0]),
        np.array([x0,y0,z1]), np.array([x1,y0,z1]),
        np.array([x0,y1,z1]), np.array([x1,y1,z1]),
    ]
    E = []
    # bottom face
    E += [(v[0],v[1]), (v[1],v[3]), (v[3],v[2]), (v[2],v[0])]
    # top face
    E += [(v[4],v[5]), (v[5],v[7]), (v[7],v[6]), (v[6],v[4])]
    # verticals
    E += [(v[0],v[4]), (v[1],v[5]), (v[2],v[6]), (v[3],v[7])]
    return E


def plot_strokes_and_program(executor, sample_points, feature_lines=None):
    """
    Plot sampled strokes and program cuboid edges together in a clean view.
    - Auto scales to show the full bbox.
    - Removes grid, ticks, labels, panes, and axis lines.
    - Avoids tight_layout() (problematic for 3D with no ticks/labels).
    Returns (fig, ax).
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=22, azim=-62)

    # optional: ensure white canvas
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- program cuboid edges
    for name, inst in executor.instances.items():
        if name == "bbox":
            continue
        o = inst.T[:3, 3]
        s = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float)
        for (p0, p1) in _cuboid_edges_from_origin_size(o, s):
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                "-", color="black", linewidth=1.0, alpha=0.9
            )

    # ---- strokes
    type_colors = {
        1: "blue", 2: "green", 3: "orange",
        4: "red", 5: "purple", 6: "brown",
    }
    for i, pts in enumerate(sample_points):
        pts_arr = np.asarray(pts, dtype=float)
        if pts_arr.shape[0] < 2:
            continue
        color = "gray"
        if feature_lines is not None and i < len(feature_lines):
            try:
                t = int(round(float(feature_lines[i][-1])))
                color = type_colors.get(t, "gray")
            except Exception:
                pass
        ax.plot(
            pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2],
            "-", color=color, linewidth=1.5, alpha=0.9
        )

    # ---- axis scaling: full bbox, equal aspect cube
    L, W, H = executor.bbox.l, executor.bbox.w, executor.bbox.h
    maxlen = max(L, W, H)
    xmid, ymid, zmid = L/2, W/2, H/2
    ax.set_xlim(xmid - maxlen/2, xmid + maxlen/2)
    ax.set_ylim(ymid - maxlen/2, ymid + maxlen/2)
    ax.set_zlim(zmid - maxlen/2, zmid + maxlen/2)

    # (If you’re on Matplotlib >= 3.3) keep a true cube:
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass  # older Matplotlib

    # ---- style cleanup
    ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # Remove background panes (gray planes)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        # also hide the axis line itself
        axis.line.set_visible(False)

    # Remove ticks & labels
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    # Fill the canvas (no margins) without tight_layout (avoids bbox error)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.show()
    return fig, ax
