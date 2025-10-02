#!/usr/bin/env python3
"""
stroke_mapping.py

Utilities to score & select strokes that are near a given cuboid component.
Distance definition:
  For a stroke (N≈10 sampled 3D points), its distance to a cuboid is
  the average over points of the minimum Euclidean distance to any of the
  cuboid's 12 edges.

Public API:
  - stroke_to_cuboid_distance(stroke_points, cuboid_edges) -> float
  - strokes_near_cuboid(sample_points, comp, thresh=0.5) -> (indices, mask)
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable
from program_executor import Executor, CuboidGeom


# ------------ geometry helpers ------------

def _points_to_segments_min_dists(p: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Vectorized point->segment distance for a *single* point against many segments.

    p: (3,) point
    A: (E,3) segment starts
    B: (E,3) segment ends
    returns: (E,) distances from p to each segment A_i B_i
    """
    AB = B - A                    # (E,3)
    AP = p[None, :] - A          # (E,3)
    AB2 = np.einsum("ij,ij->i", AB, AB)  # (E,)
    # Guard zero-length segments
    denom = np.maximum(AB2, 1e-18)
    t = np.einsum("ij,ij->i", AP, AB) / denom
    t = np.clip(t, 0.0, 1.0)     # (E,)
    proj = A + t[:, None] * AB   # (E,3)
    dp = p[None, :] - proj       # (E,3)
    return np.linalg.norm(dp, axis=1)  # (E,)


def stroke_to_cuboid_distance(stroke_points: np.ndarray,
                              cuboid_edges: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Average-of-min-edge distance for a stroke.

    stroke_points: (N,3) array of sampled 3D points along the stroke
    cuboid_edges : list of 12 (a,b) endpoints from CuboidGeom.edges()
    returns: scalar mean distance
    """
    if stroke_points.size == 0:
        return float("inf")
    # Pack edges once
    A = np.stack([e[0] for e in cuboid_edges], axis=0)  # (E,3)
    B = np.stack([e[1] for e in cuboid_edges], axis=0)  # (E,3)

    mins = []
    for p in stroke_points:
        d_all = _points_to_segments_min_dists(np.asarray(p, float), A, B)
        mins.append(d_all.min())
    return float(np.mean(mins))


def strokes_near_cuboid(sample_points: List[List[List[float]]],
                        comp: CuboidGeom,
                        thresh: float = 0.5) -> Tuple[List[int], np.ndarray]:
    """
    Compute which strokes are near a given cuboid (by the average-of-min-edge metric).

    sample_points: List over strokes -> list over points -> [x,y,z]
    comp         : CuboidGeom for the target component (from Executor.primitives())
    thresh       : distance threshold in model units (after any rescaling)
    returns:
      - indices: list of stroke indices with distance < thresh
      - mask   : boolean numpy array of shape (num_strokes,) with True where selected
    """
    edges = comp.edges()  # list of 12 (a,b) 3D endpoints
    num_strokes = len(sample_points)
    keep_idxs: List[int] = []
    mask = np.zeros(num_strokes, dtype=bool)

    for si, stroke in enumerate(sample_points):
        P = np.asarray(stroke, dtype=float)
        if P.size == 0:
            continue
        dist = stroke_to_cuboid_distance(P, edges)
        if dist < thresh:
            keep_idxs.append(si)
            mask[si] = True

    return keep_idxs, mask



def plot_stroke_selection(sample_points,
                          mask,
                          save_path: str | None = None,
                          show: bool = True):
    """
    Plot all strokes as polylines: selected ones in RED, others in BLACK.
    No grid/axis/ticks, rescaled equally using [center - half, center + half].
    """
    mask = np.asarray(mask, dtype=bool)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    all_pts = []
    for i, stroke in enumerate(sample_points):
        P = np.asarray(stroke, dtype=float)
        if P.shape[0] < 2:
            continue
        clr = "red" if mask[i] else "black"
        lw = 1.5 if mask[i] else 0.8
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=clr, linewidth=lw)
        all_pts.append(P)

    if all_pts:
        all_pts = np.vstack(all_pts)
        x_min, y_min, z_min = all_pts.min(axis=0)
        x_max, y_max, z_max = all_pts.max(axis=0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
        half = max_diff / 2

        ax.set_xlim([x_center - half, x_center + half])
        ax.set_ylim([y_center - half, y_center + half])
        ax.set_zlim([z_center - half, z_center + half])

    # Remove grid, ticks, and axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_axis_off()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)





import numpy as np
from typing import Tuple, List

def component_params_from_selected_strokes(
    sample_points: List[List[List[float]]],
    mask: np.ndarray,
    margin: float = 0.0,
    min_size_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From selected strokes (mask=True), compute an axis-aligned bounding box:
      origin = (xmin, ymin, zmin)
      size   = (l, w, h) = (xmax - xmin, ymax - ymin, zmax - zmin)

    Args:
      sample_points : list of strokes -> list of [x,y,z] points
      mask          : boolean array (num_strokes,), True = selected
      margin        : optional padding added on *both* sides along each axis
      min_size_eps  : floor to avoid zero sizes

    Returns:
      origin (3,), size (3,) as float64 numpy arrays
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != len(sample_points):
        raise ValueError(f"mask length {mask.shape[0]} != number of strokes {len(sample_points)}")

    # Gather all points from selected strokes
    pts_list = []
    for i, use in enumerate(mask):
        if not use:
            continue
        P = np.asarray(sample_points[i], dtype=float)
        if P.size > 0:
            pts_list.append(P)

    if not pts_list:
        raise ValueError("No selected strokes (mask has no True entries).")

    pts = np.vstack(pts_list)  # (N,3)

    # Axis-aligned bounding box
    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)

    if margin > 0.0:
        xyz_min = xyz_min - margin
        xyz_max = xyz_max + margin

    size = xyz_max - xyz_min
    # Prevent degenerate sizes
    size = np.maximum(size, min_size_eps)

    origin = xyz_min.copy()  # min corner = translation
    return origin, size




def _collect_edges_points(prims: Iterable[CuboidGeom]) -> np.ndarray:
    """Collect all edge endpoints from a list of cuboids into an (M,3) array."""
    pts = []
    for p in prims:
        for a, b in p.edges():
            pts.append(a)
            pts.append(b)
    return np.asarray(pts, dtype=float) if pts else np.zeros((0,3), dtype=float)

def plot_programs_overlay(
    exe_old: Executor,
    exe_new: Executor,
    save_path: str | None = None,
    show: bool = True,
    color_old: str = "black",
    color_new: str = "red",
    lw_old: float = 1.0,
    lw_new: float = 1.5,
):
    """
    Overlay wireframes of two executed programs (old vs new) in one plot.
    - Old in `color_old`, new in `color_new`
    - No grid/axis/ticks; equal scaling via [center - half, center + half]
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    prims_old = exe_old.primitives()
    prims_new = exe_new.primitives()

    # draw old
    for p in prims_old:
        for a, b in p.edges():
            A = np.asarray(a, float); B = np.asarray(b, float)
            ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color=color_old, linewidth=lw_old)

    # draw new
    for p in prims_new:
        for a, b in p.edges():
            A = np.asarray(a, float); B = np.asarray(b, float)
            ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color=color_new, linewidth=lw_new)

    # equal rescale using combined points
    pts_old = _collect_edges_points(prims_old)
    pts_new = _collect_edges_points(prims_new)
    all_pts = np.vstack([pts_old, pts_new]) if pts_old.size and pts_new.size else (pts_old if pts_old.size else pts_new)
    if all_pts.size:
        x_min, y_min, z_min = all_pts.min(axis=0)
        x_max, y_max, z_max = all_pts.max(axis=0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
        half = max_diff / 2

        ax.set_xlim([x_center - half, x_center + half])
        ax.set_ylim([y_center - half, y_center + half])
        ax.set_zlim([z_center - half, z_center + half])

    # strip everything
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_axis_off()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
