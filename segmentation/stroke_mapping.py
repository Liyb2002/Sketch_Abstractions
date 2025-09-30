#!/usr/bin/env python3
"""
stroke_mapping.py

Map each stroke to a cuboid by checking the distance between stroke sample points
and the edges of cuboids. One stroke maps to at most one cuboid. If no cuboid is
within the threshold distance, the stroke is labeled -1.

Also provides a visualization helper to plot strokes colored by cuboid assignment.
"""

from __future__ import annotations
from typing import List
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from program_executor import Executor


# ---------------- Mapping ----------------

def stroke_to_cuboid_map(
    exe: Executor,
    sample_points: List[List[List[float]]],
    dist_thresh: float = 0.5,
    device: str = "cpu"
) -> List[int]:
    """
    stroke→cuboid distance = average over the stroke's points of
    (point→closest edge of that cuboid).
    """
    prims = exe.primitives()
    if not prims:
        return [-1] * len(sample_points)

    # Build cuboid edge segments (N,12,2,3)
    boxes = []
    for g in prims:
        o = np.array(g.origin, float)
        s = np.array(g.size, float)
        corners = np.array([
            [0,0,0],[1,0,0],[0,1,0],[1,1,0],
            [0,0,1],[1,0,1],[0,1,1],[1,1,1],
        ], float) * s + o
        idx_pairs = np.array([
            [0,1],[1,3],[3,2],[2,0],   # bottom
            [4,5],[5,7],[7,6],[6,4],   # top
            [0,4],[1,5],[2,6],[3,7],   # verticals
        ], dtype=int)
        a = corners[idx_pairs[:,0]]
        b = corners[idx_pairs[:,1]]
        segs = np.stack([a,b], axis=1)  # (12,2,3)
        boxes.append(segs)
    boxes = torch.tensor(np.stack(boxes, axis=0), dtype=torch.float32, device=device) # (N,12,2,3)

    def point_to_segs(points: torch.Tensor, segs: torch.Tensor) -> torch.Tensor:
        # points: (M,3), segs: (N,12,2,3)
        M = points.size(0)
        a = segs[:, :, 0, :].unsqueeze(0)  # (1,N,12,3)
        b = segs[:, :, 1, :].unsqueeze(0)  # (1,N,12,3)
        p = points.view(M,1,1,3)           # (M,1,1,3)
        ab = b - a
        ap = p - a
        denom = (ab*ab).sum(-1).clamp_min(1e-12)
        t = (ap*ab).sum(-1) / denom
        t = t.clamp(0,1).unsqueeze(-1)
        proj = a + t*ab
        d = ((p - proj)**2).sum(-1).sqrt() # (M,N,12)
        return d.min(-1).values            # (M,N): per-point distance to each cuboid (closest edge)

    stroke_labels = []
    for stroke in sample_points:
        pts = torch.tensor(stroke, dtype=torch.float32, device=device)
        if pts.numel() == 0:
            stroke_labels.append(-1)
            continue

        d_point2box = point_to_segs(pts, boxes)  # (M,N), already min over edges
        score = d_point2box.mean(dim=0)          # (N,) average over ALL points in the stroke
        best_idx = int(torch.argmin(score).item())
        best_val = float(score[best_idx].item())

        if best_val < dist_thresh:
            stroke_labels.append(best_idx)
        else:
            stroke_labels.append(-1)

    return stroke_labels


# ---------------- Visualization ----------------

def vis_stroke_mapping(sample_points, stroke_labels, show=True):
    """
    Visualize strokes with colors based on cuboid mapping.
    -1 (unmapped) strokes are plotted in gray.
    Axis/grid/ticks removed, rescaled equally to shape bounds.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Assign random colors per cuboid label
    unique_labels = sorted(set(lbl for lbl in stroke_labels if lbl != -1))
    cmap = {lbl: (random.random(), random.random(), random.random()) for lbl in unique_labels}

    # Track min/max for rescaling
    xs_all, ys_all, zs_all = [], [], []

    for stroke, lbl in zip(sample_points, stroke_labels):
        if not stroke:
            continue
        xs, ys, zs = zip(*stroke)
        xs_all.extend(xs)
        ys_all.extend(ys)
        zs_all.extend(zs)
        if lbl == -1:
            color = "gray"
        else:
            color = cmap[lbl]
        ax.plot(xs, ys, zs, color=color, linewidth=1.0)

    # Compute center and half range for equal scaling
    if xs_all and ys_all and zs_all:
        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = min(ys_all), max(ys_all)
        z_min, z_max = min(zs_all), max(zs_all)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        half = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

        # Equalize axes
        ax.set_xlim([x_center - half, x_center + half])
        ax.set_ylim([y_center - half, y_center + half])
        ax.set_zlim([z_center - half, z_center + half])
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    # Remove axes, grid, ticks
    ax.set_axis_off()
    ax.grid(False)

    # Set viewing angle
    ax.view_init(elev=100, azim=45)

    if show:
        plt.show()

    return fig, ax
