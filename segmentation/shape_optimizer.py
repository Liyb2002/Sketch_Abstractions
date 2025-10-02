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




def plot_strokes_and_program(executor, sample_points, feature_lines=None, use_optimized=False):
    """
    Plot sampled strokes and program cuboid edges together in a clean view.
    - If use_optimized=True, apply per-part translations from fit_translations.json.
    - Auto scales to show the full bbox.
    - Removes grid, ticks, labels, panes, and axis lines.
    - Avoids tight_layout() (problematic for 3D with no ticks/labels).
    Returns (fig, ax).
    """
    import json
    from pathlib import Path

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=22, azim=-62)

    # optional: ensure white canvas
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- optional optimized translations
    offsets = {}
    if use_optimized:
        ir_path = getattr(executor, "ir_path", None)
        if ir_path is None:
            # fallback: assume ../input relative to CWD
            input_dir = Path.cwd().parent / "input"
        else:
            input_dir = Path(ir_path).parent
        trans_file = input_dir / "fit_translations.json"
        if trans_file.exists():
            try:
                data = json.loads(trans_file.read_text())
                offsets = data.get("offsets_xyz", {})
            except Exception as e:
                print(f"[warn] Failed to load translations: {e}")

    # ---- program cuboid edges
    for name, inst in executor.instances.items():
        if name == "bbox":
            continue
        o = inst.T[:3, 3].copy()
        if name in offsets:
            o = o + np.array(offsets[name], dtype=float)  # apply learned offset
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

    try:
        ax.set_box_aspect((1, 1, 1))  # keep a true cube (matplotlib >= 3.3)
    except Exception:
        pass

    # ---- style cleanup
    ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.show()
    return fig, ax



def compare_optimized_programs(executor, use_offsets=True, use_scales=True):
    """
    Plot program cuboid edges before (black) and after applying optimized
    translations + scales (red).
    - executor: a program_executor.Executor instance
    - use_offsets: if True, loads fit_translations.json and applies offsets
    - use_scales: if True, loads fit_scales.json and applies scales
    Returns (fig, ax).
    """
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=22, azim=-62)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- load optimized offsets & scales
    offsets, scales = {}, {}
    if use_offsets or use_scales:
        ir_path = getattr(executor, "ir_path", None)
        if ir_path is None:
            input_dir = Path.cwd().parent / "input"
        else:
            input_dir = Path(ir_path).parent

        if use_offsets:
            trans_file = input_dir / "fit_translations.json"
            if trans_file.exists():
                try:
                    data = json.loads(trans_file.read_text())
                    offsets = data.get("offsets_xyz", {})
                except Exception as e:
                    print(f"[warn] Failed to load translations: {e}")

        if use_scales:
            scale_file = input_dir / "fit_scales.json"
            if scale_file.exists():
                try:
                    data = json.loads(scale_file.read_text())
                    scales = data.get("scales_lwh", {})
                except Exception as e:
                    print(f"[warn] Failed to load scales: {e}")

    # ---- original program (black)
    for name, inst in executor.instances.items():
        if name == "bbox":
            continue
        o = inst.T[:3, 3]
        s = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float)
        for (p0, p1) in _cuboid_edges_from_origin_size(o, s):
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                "-", color="black", linewidth=1.0, alpha=0.8
            )

    # ---- optimized program (red)
    if offsets or scales:
        for name, inst in executor.instances.items():
            if name == "bbox":
                continue
            # start with original
            o = inst.T[:3, 3].copy()
            s = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float)
            # apply offset if available
            if name in offsets:
                o = o + np.array(offsets[name], dtype=float)
            # apply scale if available
            if name in scales:
                s = s * np.array(scales[name], dtype=float)
            for (p0, p1) in _cuboid_edges_from_origin_size(o, s):
                ax.plot(
                    [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    "-", color="red", linewidth=1.2, alpha=0.9
                )
    else:
        print("[info] No optimized translations/scales found; only original program is shown.")

    # ---- axis scaling: full bbox, equal aspect cube
    L, W, H = executor.bbox.l, executor.bbox.w, executor.bbox.h
    maxlen = max(L, W, H)
    xmid, ymid, zmid = L / 2, W / 2, H / 2
    ax.set_xlim(xmid - maxlen / 2, xmid + maxlen / 2)
    ax.set_ylim(ymid - maxlen / 2, ymid + maxlen / 2)
    ax.set_zlim(zmid - maxlen / 2, zmid + maxlen / 2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # ---- style cleanup
    ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.show()
    return fig, ax


# ====== Rescaling ======

#!/usr/bin/env python3
"""
rescaling_optimizer.py

Single-responsibility: align the executed program's Z placement to the strokes'
bbox from info.json by editing program.bblock.min/max.z (pure translation),
then return a fresh Executor built from the updated IR.

- Reads:
    ../input/sketch_program_ir.json
    ../input/info.json   (expects bbox{x_min..z_max})
- Overwrites:
    ../input/sketch_program_ir.json   (with Z-shifted min/max)
- Returns:
    Executor over the updated IR

No visualization here; caller can handle plotting/export.
"""

from pathlib import Path
import json
import numpy as np
from typing import Tuple

from program_executor import Executor


# ---------- Helpers ----------
def _geom_aabb_from_executor(exe: Executor) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """Geometry AABB (excluding bbox) from placed cuboids."""
    prims = exe.primitives()
    if not prims:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    mins = np.full(3, float("inf"))
    maxs = np.full(3, float("-inf"))
    for p in prims:
        o = np.array(p.origin, float)
        s = np.array(p.size, float)
        lo = o
        hi = o + s
        mins = np.minimum(mins, lo)
        maxs = np.maximum(maxs, hi)
    return tuple(map(float, mins)), tuple(map(float, maxs))


def _strokes_aabb_from_info(info: dict) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """Extract stroke bbox from info.json (expects bbox with x_min..z_max)."""
    b = info.get("bbox")
    if not b:
        raise SystemExit("info.json missing 'bbox' with x_min..z_max")
    mins = (float(b["x_min"]), float(b["y_min"]), float(b["z_min"]))
    maxs = (float(b["x_max"]), float(b["y_max"]), float(b["z_max"]))
    return mins, maxs


def _ensure_bbox_minmax_inplace(ir: dict) -> None:
    """Ensure program.bblock has min/max; if absent, assume min=(0,0,0), max=(l,w,h)."""
    bb = ir["program"]["bblock"]
    if "min" not in bb or "max" not in bb:
        L, W, H = float(bb["l"]), float(bb["w"]), float(bb["h"])
        bb["min"] = [0.0, 0.0, 0.0]
        bb["max"] = [L, W, H]


def _shift_ir_bbox_z_inplace(ir: dict, dz: float) -> None:
    """Translate the entire program in Z by shifting bbox min/max.z in place."""
    _ensure_bbox_minmax_inplace(ir)
    bb = ir["program"]["bblock"]
    bb["min"][2] = float(bb["min"][2]) + dz
    bb["max"][2] = float(bb["max"][2]) + dz
    # l/w/h unchanged (size preserved)


def _print_diagnostics(tag: str,
                       strokes_min, strokes_max,
                       geom_min, geom_max) -> None:
    def fmt(v): return f"[{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]"
    Ls = (strokes_max[0]-strokes_min[0], strokes_max[1]-strokes_min[1], strokes_max[2]-strokes_min[2])
    Lg = (geom_max[0]-geom_min[0],       geom_max[1]-geom_min[1],       geom_max[2]-geom_min[2])
    # print(f"\n===== BBOX DIAGNOSTICS ({tag}) =====")
    # print(f"Strokes  min: {fmt(strokes_min)}   max: {fmt(strokes_max)}   size: {fmt(Ls)}")
    # print(f"Geometry min: {fmt(geom_min)}     max: {fmt(geom_max)}     size: {fmt(Lg)}")
    dmin = (geom_min[0]-strokes_min[0], geom_min[1]-strokes_min[1], geom_min[2]-strokes_min[2])
    dmax = (geom_max[0]-strokes_max[0], geom_max[1]-strokes_max[1], geom_max[2]-strokes_max[2])
    # print(f"Δmin (geom - strokes): {fmt(dmin)}")
    # print(f"Δmax (geom - strokes): {fmt(dmax)}")
    # print("====================================\n")


# ---------- Public API ----------
def rescale_and_execute(input_dir: Path) -> Executor:
    ir_path   = input_dir / "sketch_program_ir.json"
    info_path = input_dir / "info.json"
    if not ir_path.exists():
        raise SystemExit(f"IR not found: {ir_path}")
    if not info_path.exists():
        raise SystemExit(f"info.json not found: {info_path}")

    # Load IR & info
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    info = json.loads(info_path.read_text(encoding="utf-8"))

    # Execute current IR & compute AABBs
    exe_before = Executor(ir)
    strokes_min, strokes_max = _strokes_aabb_from_info(info)
    geom_min, geom_max = _geom_aabb_from_executor(exe_before)
    _print_diagnostics("before shift", strokes_min, strokes_max, geom_min, geom_max)

    # Decide Z shift
    dz_min = strokes_min[2] - geom_min[2]
    dz_max = strokes_max[2] - geom_max[2]
    dz = dz_min if abs(dz_min) <= abs(dz_max) else dz_max
    print(f"Proposed Z shift dz = {dz:.6f}")

    if abs(dz) >= 1e-9:
        _shift_ir_bbox_z_inplace(ir, dz)
        # overwrite original so downstream tools see the shift
        ir_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
        print(f"🧩 Overwrote IR with Z-shift at: {ir_path}")
    else:
        print("dz negligible; IR not modified.")

    # ---- NEW: normalize attaches to a parent tree, preserving current world placement
    ir_norm = json.loads(ir_path.read_text(encoding="utf-8"))
    normalize_attaches_to_parent_tree_inplace(ir_norm)
    # Write a sibling file for debugging/inspection (optional)
    out_norm_path = ir_path.with_name(ir_path.stem + "_parenttree.json")
    out_norm_path.write_text(json.dumps(ir_norm, indent=2), encoding="utf-8")
    print(f"🔧 Wrote normalized parent-tree IR: {out_norm_path}")

    # Re-execute normalized IR
    exe_after = Executor(ir_norm)
    geom_min2, geom_max2 = _geom_aabb_from_executor(exe_after)
    _print_diagnostics("after shift+normalize", strokes_min, strokes_max, geom_min2, geom_max2)

    return exe_after





# ====== Attach normalization: make a parent tree rooted at bbox ======
from typing import Set, List
import numpy as np
from program_executor import Executor  # already used elsewhere in this file

def _ir_bbox_min_and_size(P: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    bb = P["bblock"]
    if "min" in bb and "max" in bb:
        bb_min = np.array(bb["min"], dtype=float).reshape(3)
        bb_max = np.array(bb["max"], dtype=float).reshape(3)
        return bb_min, (bb_max - bb_min).astype(float)
    return np.zeros(3, dtype=float), np.array([bb["l"], bb["w"], bb["h"]], dtype=float)

def _ir_spec_size(P: Dict[str, Any], name: str) -> np.ndarray:
    if name == "bbox":
        _, sz = _ir_bbox_min_and_size(P)
        return sz
    for c in P.get("cuboids", []):
        if str(c["var"]) == name:
            return np.array([float(c["l"]), float(c["w"]), float(c["h"])], dtype=float)
    raise KeyError(f"Unknown cuboid spec: {name}")

def _ir_nodes(P: Dict[str, Any]) -> List[str]:
    return ["bbox"] + [str(c["var"]) for c in P.get("cuboids", [])]

def _ir_adj(P: Dict[str, Any]) -> Dict[str, Set[str]]:
    nbrs: Dict[str, Set[str]] = {n: set() for n in _ir_nodes(P)}
    for a in P.get("attach", []):
        A, B = str(a["a"]), str(a["b"])
        if A in nbrs and B in nbrs:
            nbrs[A].add(B); nbrs[B].add(A)
    return nbrs

def _world_min_corners_from_executor(exe: Executor) -> Dict[str, np.ndarray]:
    """Prototype min corners only (exclude bbox)."""
    names = {k for k in exe.specs.keys() if k != "bbox"}
    out: Dict[str, np.ndarray] = {}
    for nm, inst in exe.instances.items():
        if nm in names:
            out[nm] = inst.T[:3, 3].astype(float).copy()
    out["bbox"] = exe.instances["bbox"].T[:3, 3].astype(float).copy()
    return out

def normalize_attaches_to_parent_tree_inplace(ir: Dict[str, Any]) -> None:
    """
    Rewrite program.attach so every prototype has exactly one parent (a tree rooted at bbox),
    while preserving current world placement. Reflect/translate lists remain untouched.

    This executes the current IR to read world poses, then rebuilds attaches as:
        child (vA=[0,0,0]) attaches to parent with parent-side fractions vB
        s.t. child's min corner equals its current world origin.
    """
    P = ir["program"]

    # Execute as-is to get world placements
    exe = Executor(ir)
    world_min = _world_min_corners_from_executor(exe)

    # BFS from bbox to choose one parent per node; islands will be anchored to bbox
    from collections import deque
    nbrs = _ir_adj(P)
    parent: Dict[str, str] = {}
    seen = {"bbox"}
    dq = deque(["bbox"])
    while dq:
        u = dq.popleft()
        for v in nbrs.get(u, ()):
            if v not in seen:
                seen.add(v); parent[v] = u; dq.append(v)

    for n in _ir_nodes(P):
        if n == "bbox": 
            continue
        if n not in seen:
            parent[n] = "bbox"  # island → direct parent bbox

    # Build new attaches (child -> parent) that freeze current world placement
    bb_min, bb_size = _ir_bbox_min_and_size(P)
    new_attaches: List[Dict[str, Any]] = []

    for child, par in parent.items():
        if child == "bbox":
            continue
        oA = world_min[child]                  # current child min-corner (world)
        sA = _ir_spec_size(P, child)

        if par == "bbox":
            oB, sB = bb_min, bb_size
        else:
            oB = world_min[par]
            sB = _ir_spec_size(P, par)

        vA = np.array([0.0, 0.0, 0.0], dtype=float)   # child min corner
        vB = (oA + sA * vA - oB) / np.maximum(sB, 1e-12)

        new_attaches.append({
            "a": child, "b": par,
            "x1": float(vA[0]), "y1": float(vA[1]), "z1": float(vA[2]),
            "x2": float(vB[0]), "y2": float(vB[1]), "z2": float(vB[2]),
        })

    # Replace attaches with the normalized tree
    P["attach"] = new_attaches

def normalize_attaches_file(ir_path: Path) -> Path:
    """
    Load IR, normalize attaches to parent tree in place, and write a sibling
    file with suffix '_parenttree.json'. Return new path.
    """
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    normalize_attaches_to_parent_tree_inplace(ir)
    out_path = ir_path.with_name(ir_path.stem + "_parenttree.json")
    out_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
    return out_path
