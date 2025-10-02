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
from typing import List, Tuple
import numpy as np
from program_executor import CuboidGeom


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


import matplotlib.pyplot as plt
import numpy as np

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
