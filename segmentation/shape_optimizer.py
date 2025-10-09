#!/usr/bin/env python3
"""
shape_optimizer.py

Step 2: Load the pre-sampled 3D strokes from
../input/perturbed_lines.json.

Usage:
  Imported into main_optimizer.py
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Any, Dict, Tuple, Optional
from typing import Iterable, List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import math

EPS_Z = 1e-9  # dz tolerance


# 1)Straight Line: Point_1 (3 value), Point_2 (3 value), 0, 0, 0, 1
# 2)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 3)Cylinder face: Center (3 value), normal (3 value), height, radius, 0, 3
# 4)Arc: Start S (3 values), End E (3 values), Center C (3 values), 4
# 5)Spline: Control_point_1 (3 value), Control_point_2 (3 value), Control_point_3 (3 value), 5
# 6)Sphere: center_x, center_y, center_z, axis_nx,  axis_ny,  axis_nz, 0,        radius,   0,     6

def load_perturbed_feature_lines(input_dir: Path):
    path_stroke_lines = input_dir / "stroke_lines.json"

    # Load all line data from one file
    with open(path_stroke_lines, "r", encoding="utf-8") as f:
        stroke_lines = json.load(f)

    # Extract each line type
    perturbed_feature_lines = stroke_lines.get("perturbed_feature_lines", [])
    perturbed_construction_lines = stroke_lines.get("perturbed_construction_lines", [])
    feature_lines = stroke_lines.get("feature_lines", [])

    print(
        f"📄 Loaded {len(perturbed_feature_lines)} perturbed feature lines, "
        f"{len(perturbed_construction_lines)} perturbed construction lines, "
        f"and {len(feature_lines)} feature lines"
    )

    return perturbed_feature_lines, perturbed_construction_lines, feature_lines





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
def _height_from_aabb(min3, max3):
    return max3[2] - min3[2]

def _cuboids_height_from_executor(exe):
    """
    Compute height (Z span) of cuboid/box primitives inside an already-executed program.
    Falls back to whole-geometry height if no cuboids are found.
    No extra execution is performed.
    """
    def _looks_like_cuboid(obj):
        tag = (type(obj).__name__ + " " + str(getattr(obj, "type", "")) + " " + str(getattr(obj, "kind", ""))).lower()
        return any(k in tag for k in ("cuboid", "box", "rectangular_prism", "rect-prism", "block"))

    def _try_get_aabb(obj):
        for meth in ("aabb", "bbox", "bounds", "get_aabb", "get_bbox", "get_bounds"):
            if hasattr(obj, meth):
                b = getattr(obj, meth)
                b = b() if callable(b) else b
                if isinstance(b, (list, tuple)) and len(b) == 2 and all(len(x) == 3 for x in b):
                    return tuple(b[0]), tuple(b[1])
                if hasattr(b, "min") and hasattr(b, "max") and len(b.min) == 3 and len(b.max) == 3:
                    return tuple(b.min), tuple(b.max)
                if isinstance(b, (list, tuple)) and len(b) == 6:
                    return (b[0], b[1], b[2]), (b[3], b[4], b[5])
        for attr in ("vertices", "points", "verts"):
            if hasattr(obj, attr):
                pts = getattr(obj, attr)
                try:
                    zs = [p[2] for p in pts]
                    zmin, zmax = min(zs), max(zs)
                    # we only need height; return dummy xy
                    return (0.0, 0.0, zmin), (0.0, 0.0, zmax)
                except Exception:
                    pass
        return None

    def _iter_shapes(exe):
        for attr in ("solids", "objects", "nodes", "instances", "shapes", "prims"):
            if hasattr(exe, attr):
                bag = getattr(exe, attr)
                if isinstance(bag, dict):
                    yield from bag.values()
                else:
                    try:
                        for v in bag:
                            yield v
                    except TypeError:
                        pass

    zmin, zmax = math.inf, -math.inf
    count = 0
    for obj in _iter_shapes(exe):
        if not _looks_like_cuboid(obj):
            continue
        ab = _try_get_aabb(obj)
        if not ab:
            continue
        a, b = ab
        zmin = min(zmin, a[2]); zmax = max(zmax, b[2])
        count += 1

    if count > 0 and zmin < math.inf and zmax > -math.inf:
        return zmax - zmin, count

    # Fallback: use overall geometry AABB height
    gmin, gmax = _geom_aabb_from_executor(exe)
    return _height_from_aabb(gmin, gmax), 0

def rescale_and_execute(input_dir: Path, ir_path) -> Executor:
    info_path = input_dir / "info.json"
    if not ir_path.exists():
        raise SystemExit(f"IR not found: {ir_path}")
    if not info_path.exists():
        raise SystemExit(f"info.json not found: {info_path}")

    # Load IR & info
    ir_original_text = ir_path.read_text(encoding="utf-8")
    ir = json.loads(ir_original_text)
    info = json.loads(info_path.read_text(encoding="utf-8"))

    # Execute current IR & compute AABBs
    exe_before = Executor(ir)
    strokes_min, strokes_max = _strokes_aabb_from_info(info)
    geom_min, geom_max = _geom_aabb_from_executor(exe_before)

    # ---- Heights (no extra execution) ----
    strokes_h = _height_from_aabb(strokes_min, strokes_max)
    cuboids_h_before, cub_count_before = _cuboids_height_from_executor(exe_before)
    print(f"[BEFORE] heights — strokes: {strokes_h:.6f}, cuboids: {cuboids_h_before:.6f} "
          f"(Δ={cuboids_h_before - strokes_h:.6f}, cuboids_found={cub_count_before})")

    # Decide Z shift based on overall geometry (unchanged from your logic)
    dz_min = strokes_min[2] - geom_min[2]
    dz_max = strokes_max[2] - geom_max[2]
    dz = dz_min if abs(dz_min) <= abs(dz_max) else dz_max
    print(f"Proposed Z shift dz = {dz:.6f}")

    # Mutate on a copy; compare to avoid unnecessary writes
    ir_mut = copy.deepcopy(ir)

    if abs(dz) >= EPS_Z:
        _shift_ir_bbox_z_inplace(ir_mut, dz)
        print("🧩 Applied Z shift.")
    else:
        print("dz negligible; no Z shift needed.")

    before_norm = json.dumps(ir_mut, sort_keys=True)
    normalize_attaches_to_parent_tree_inplace(ir_mut)
    after_norm = json.dumps(ir_mut, sort_keys=True)
    if before_norm != after_norm:
        print("🔧 Normalized parent-tree attaches.")

    # Write only if changed
    ir_mut_text = json.dumps(ir_mut, indent=2)
    if ir_mut_text != ir_original_text:
        ir_path.write_text(ir_mut_text, encoding="utf-8")
        print(f"💾 Saved updated IR to: {ir_path}")
    else:
        print("No changes required; IR not modified.")

    # Re-execute what’s on disk (not “extra”; this is your existing after step)
    ir_after = json.loads(ir_path.read_text(encoding="utf-8"))
    exe_after = Executor(ir_after)
    geom_min2, geom_max2 = _geom_aabb_from_executor(exe_after)

    # ---- Heights AFTER (using the executor we already created) ----
    cuboids_h_after, cub_count_after = _cuboids_height_from_executor(exe_after)
    strokes_h_after = strokes_h  # strokes come from info.json; unchanged by IR edits
    print(f"[AFTER ] heights — strokes: {strokes_h_after:.6f}, cuboids: {cuboids_h_after:.6f} "
          f"(Δ={cuboids_h_after - strokes_h_after:.6f}, cuboids_found={cub_count_after})")

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





# ====== Multi-view export for SAM pre-segmentation ======

def plot_strokes_and_program_multiview(
    executor,
    sample_points: List[List[List[float]]],
    *,
    iso_dirs: Tuple[Tuple[float, float, float], ...] = (
        ( 1,  1,  1),
        (-1,  1,  1),
        ( 1, -1,  1),
        (-1, -1,  1),
    ),
    image_size: Tuple[int, int] = (1024, 1024),
    stroke_linewidth: float = 1.2,
    pad_frac: float = 0.05,
    out_dir: Path | None = None,
    # how the spec dimensions are referenced in the local cuboid frame:
    #  - "min": local origin at a *corner* (use [0,l]x[0,w]x[0,h])
    #  - "center": local origin at the *center* (use [-l/2,l/2] etc.)
    box_frame_origin: str = "min",
) -> None:
    """
    Render strokes-only PNGs from multiple orthographic isometric directions and
    save one JSON per view with program cuboid bounding boxes in *pixel coords*.

    Files written (index i corresponds to iso_dirs[i]):
      <out_dir>/<i>.png   # strokes only
      <out_dir>/<i>.json  # {"size":[H,W],"boxes":[{"name":..., "bbox":[xmin,ymin,xmax,ymax]}, ...]}

    Notes
    -----
    - Each 3D instance uses its full 4x4 transform (rotation + translation).
    - Per view, we orthographically project the 8 world-space corners, then
      take min/max in pixel space to build the 2D AABB SAM expects.
    """
    W, H = int(image_size[0]), int(image_size[1])
    if out_dir is None:
        out_dir = Path.cwd().parent / "SAM" / "bbx_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Build local-corner sets once (in each instance's local frame) --------
    def local_corners(l, w, h, origin="min"):
        if origin == "center":
            hx, hy, hz = 0.5*l, 0.5*w, 0.5*h
            xs = [-hx, +hx]
            ys = [-hy, +hy]
            zs = [-hz, +hz]
        else:  # "min" (default)
            xs = [0.0, l]
            ys = [0.0, w]
            zs = [0.0, h]
        cc = np.array([[x, y, z, 1.0]
                       for x in xs for y in ys for z in zs], dtype=float)  # (8,4)
        return cc

    # (name, world-corners 8x3)
    cuboids = []
    for name, inst in executor.instances.items():
        if name == "bbox":
            continue
        T = np.asarray(inst.T, dtype=float)      # (4,4)
        l = float(inst.spec.l); w = float(inst.spec.w); h = float(inst.spec.h)
        lc = local_corners(l, w, h, origin=box_frame_origin)   # (8,4) in local
        wc_h = (T @ lc.T).T                        # (8,4) world (homogeneous)
        wc = wc_h[:, :3] / wc_h[:, 3:4]           # (8,3)
        cuboids.append((name, wc))

    # -------- Flatten stroke points once (world) --------
    def flatten_strokes_3d(strokes) -> np.ndarray:
        out = []
        for s in strokes:
            a = np.asarray(s, float)
            if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] > 0:
                out.append(a)
        return np.vstack(out) if out else np.zeros((0,3), float)

    P3_all = flatten_strokes_3d(sample_points)

    # -------- Orthographic projection utilities --------
    def ortho_project(P3: np.ndarray, cam_dir: np.ndarray) -> np.ndarray:
        """
        cam_dir: viewing direction (camera looks along +w toward the origin at infinity).
        Build ONB {u,v,w}. Project onto (u,v) plane: [dot(p,u), dot(p,v)].
        """
        w = cam_dir / (np.linalg.norm(cam_dir) + 1e-12)
        world_up = np.array([0.0, 0.0, 1.0], float)
        if abs(np.dot(w, world_up)) > 0.95:
            world_up = np.array([1.0, 0.0, 0.0], float)
        u = np.cross(world_up, w); u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(w, u)
        x = P3 @ u
        y = P3 @ v
        return np.stack([x, y], axis=1)  # (N,2)

    def compute_mapping(P2_all: np.ndarray):
        if P2_all.size == 0:
            xmin=ymin=0.0; xmax=ymax=1.0
        else:
            xmin, ymin = P2_all.min(axis=0)
            xmax, ymax = P2_all.max(axis=0)
        span = max(xmax - xmin, ymax - ymin, 1e-9)
        pad = pad_frac * span
        xmin -= pad; xmax += pad
        ymin -= pad; ymax += pad
        span_pad = max(xmax - xmin, ymax - ymin)
        s = min(W / span_pad, H / span_pad)
        cw = s * (xmax - xmin); ch = s * (ymax - ymin)
        mx = 0.5 * (W - cw); my = 0.5 * (H - ch)
        return (xmin, xmax, ymin, ymax), s, s, mx, my

    def world2pix(P2: np.ndarray, bounds, sx, sy, mx, my) -> np.ndarray:
        xmin, xmax, ymin, ymax = bounds
        x = sx * (P2[:, 0] - xmin) + mx
        y = sy * (P2[:, 1] - ymin) + my
        y_pix = H - y  # flip to top-left origin
        return np.stack([x, y_pix], axis=1)

    # -------- Render & export per view --------
    for i, d in enumerate(iso_dirs):
        cam_dir = np.asarray(d, float)

        # strokes in view space
        P2_strokes = []
        for s in sample_points:
            a = np.asarray(s, float)
            if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 2:
                continue
            P2_strokes.append(ortho_project(a, cam_dir))

        # mapping based on all stroke points (could include cuboids too)
        P2_all_for_bounds = ortho_project(P3_all, cam_dir) if P3_all.size else np.zeros((0,2), float)
        bounds2d, sx, sy, mx, my = compute_mapping(P2_all_for_bounds)

        # ----- strokes-only PNG -----
        fig = plt.figure(figsize=(W/100.0, H/100.0), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis()
        ax.axis("off")
        for P2 in P2_strokes:
            PP = world2pix(P2, bounds2d, sx, sy, mx, my)
            ax.plot(PP[:, 0], PP[:, 1], "-", linewidth=stroke_linewidth,
                    color="black", solid_capstyle="round")
        png_path = out_dir / f"{i}.png"
        fig.savefig(png_path, dpi=100, facecolor="white")
        plt.close(fig)

        # ----- per-view bbox JSON (all cuboids) -----
        boxes = []
        for name, world_corners in cuboids:
            P2 = ortho_project(world_corners, cam_dir)       # (8,2)
            PP = world2pix(P2, bounds2d, sx, sy, mx, my)     # (8,2) pixel coords
            xmin = float(np.min(PP[:, 0])); xmax = float(np.max(PP[:, 0]))
            ymin = float(np.min(PP[:, 1])); ymax = float(np.max(PP[:, 1]))
            # clamp to image bounds
            xmin = max(0.0, min(xmin, W)); xmax = max(0.0, min(xmax, W))
            ymin = max(0.0, min(ymin, H)); ymax = max(0.0, min(ymax, H))
            boxes.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})

        # IMPORTANT: store size as [H, W] to match the SAM scripts that expect (H, W)
        json_path = out_dir / f"{i}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"size": [H, W], "boxes": boxes}, f, indent=2)





def plot_strokes_and_program_mask(
    executor,
    sample_points: List[List[List[float]]],
    *,
    iso_dirs: Tuple[Tuple[float,float,float], ...] = (
        ( 1,  1,  1),
        (-1,  1,  1),
        ( 1, -1,  1),
        (-1, -1,  1),
    ),
    image_size: Tuple[int,int] = (1024,1024),
    stroke_linewidth: float = 1.6,
    pad_frac: float = 0.05,
    out_dir: Path|None = None,
    box_frame_origin: str = "min",   # or "center"
) -> None:
    """
    Render strokes-only PNGs and component-specific non-axis-aligned prior masks
    (projected 3D cuboids, orthographic) for each view.

    For each view index i:
      bbx_mask_input/{i}.png                 # strokes-only
      bbx_mask_input/{i}_{component}.png     # binary mask (0/255) per component
    """
    W, H = int(image_size[0]), int(image_size[1])
    if out_dir is None:
        out_dir = Path.cwd() / "bbx_mask_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- utilities ---
    def local_corners(l, w, h, origin="min"):
        if origin == "center":
            xs = [-0.5*l, 0.5*l]; ys = [-0.5*w, 0.5*w]; zs = [-0.5*h, 0.5*h]
        else:
            xs = [0, l]; ys = [0, w]; zs = [0, h]
        # order: z0 then z1; 8 corners (homogeneous)
        return np.array([[x,y,z,1.0] for z in zs for y in ys for x in xs], dtype=float)

    # faces (indices into the 8-corner array above)
    FACES = [
        [0,1,3,2],  # bottom
        [4,5,7,6],  # top
        [0,1,5,4],  # +x
        [2,3,7,6],  # -x
        [0,2,6,4],  # +y
        [1,3,7,5],  # -y
    ]

    def ortho_basis(cam_dir: np.ndarray):
        wv = cam_dir / (np.linalg.norm(cam_dir)+1e-12)
        world_up = np.array([0,0,1], float)
        if abs(np.dot(wv, world_up)) > 0.95:
            world_up = np.array([1,0,0], float)
        u = np.cross(world_up, wv); u /= (np.linalg.norm(u)+1e-12)
        v = np.cross(wv, u)
        return u, v

    def project_points(P3: np.ndarray, u: np.ndarray, v: np.ndarray):
        x = P3 @ u; y = P3 @ v
        return np.stack([x,y], axis=1)

    def compute_mapping(P2_all: np.ndarray):
        if P2_all.size == 0:
            xmin=ymin=0.0; xmax=ymax=1.0
        else:
            xmin, ymin = P2_all.min(axis=0); xmax, ymax = P2_all.max(axis=0)
        span = max(xmax-xmin, ymax-ymin, 1e-9)
        pad = pad_frac*span
        xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
        span_pad = max(xmax-xmin, ymax-ymin)
        s = min(W/span_pad, H/span_pad)
        mx = 0.5*(W - s*(xmax-xmin))
        my = 0.5*(H - s*(ymax-ymin))
        return (xmin,ymin), s, mx, my

    def world2pix(P2: np.ndarray, origin, s, mx, my):
        xmin,ymin = origin
        x = s*(P2[:,0]-xmin) + mx
        y = s*(P2[:,1]-ymin) + my
        # IMPORTANT: pixel coordinates: top-left origin -> flip y
        return np.stack([x, H - y], axis=1)

    # --- flatten strokes (world) ---
    P3_all = np.vstack([np.asarray(s, float) for s in sample_points if len(s)>0]) \
             if sample_points else np.zeros((0,3))

    # --- process views ---
    for i, d in enumerate(iso_dirs):
        cam_dir = np.asarray(d, float)
        u, v = ortho_basis(cam_dir)

        # 1) determine mapping from all stroke points (if none, use a dummy box)
        P2_all = project_points(P3_all, u, v) if P3_all.size else np.array([[0.0,0.0],[1.0,1.0]])
        origin, s, mx, my = compute_mapping(P2_all)

        # 2) strokes-only PNG (Matplotlib display must invert y to match pixels)
        fig = plt.figure(figsize=(W/100, H/100), dpi=100)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.axis("off")
        ax.invert_yaxis()  # <-- FIX: make display use top-left origin

        for s_pts in sample_points:
            a = np.asarray(s_pts, float)
            if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 2: 
                continue
            P2 = project_points(a, u, v)
            PP = world2pix(P2, origin, s, mx, my)  # [x, y_pix]
            ax.plot(PP[:,0], PP[:,1], "-", color="black", linewidth=stroke_linewidth, solid_capstyle="round")

        fig.savefig(out_dir / f"{i}.png", dpi=100, facecolor="white")
        plt.close(fig)

        # 3) per-component masks (OpenCV uses top-left origin, matches world2pix)
        for name, inst in executor.instances.items():
            if name == "bbox":
                continue
            l, w, h = float(inst.spec.l), float(inst.spec.w), float(inst.spec.h)
            T = np.asarray(inst.T, float)           # (4,4)

            lc = local_corners(l, w, h, box_frame_origin)   # (8,4) local
            wc = (T @ lc.T).T[:, :3]                         # (8,3) world
            P2 = project_points(wc, u, v)
            PP = world2pix(P2, origin, s, mx, my)            # (8,2) pixels

            mask = np.zeros((H, W), np.uint8)
            for face in FACES:
                pts = PP[face].astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
            cv2.imwrite(str(out_dir / f"{i}_{name}.png"), mask)




def plot_program_only(
    executor,
    *,
    use_offsets: bool = False,
    use_scales: bool = False,
    linewidth: float = 1.2,
    color: str = "black",
    elev: float = 22,
    azim: float = -62,
    figsize=(8, 7),
):
    """
    Plot only the cuboid edges of the program (excluding 'bbox').

    Args:
        executor: a program_executor.Executor instance
        use_offsets: if True, applies translations from fit_translations.json
        use_scales: if True, applies scales from fit_scales.json
        linewidth: line width for drawing cuboids
        color: color of cuboid edges (default: black)
        elev, azim: view angles for 3D visualization
        figsize: size of the matplotlib figure

    Returns:
        (fig, ax): the matplotlib figure and 3D axis
    """
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- Load optional optimization files ----
    offsets, scales = {}, {}
    if use_offsets or use_scales:
        ir_path = getattr(executor, "ir_path", None)
        input_dir = Path(ir_path).parent if ir_path is not None else Path.cwd().parent / "input"

        if use_offsets:
            trans_file = input_dir / "fit_translations.json"
            if trans_file.exists():
                try:
                    data = json.loads(trans_file.read_text())
                    offsets = data.get("offsets_xyz", {})
                    print(f"Loaded {len(offsets)} translation offsets.")
                except Exception as e:
                    print(f"[warn] Could not read fit_translations.json: {e}")

        if use_scales:
            scale_file = input_dir / "fit_scales.json"
            if scale_file.exists():
                try:
                    data = json.loads(scale_file.read_text())
                    scales = data.get("scales_lwh", {})
                    print(f"Loaded {len(scales)} scale factors.")
                except Exception as e:
                    print(f"[warn] Could not read fit_scales.json: {e}")

    # ---- Create figure ----
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- Draw all cuboid edges ----
    for name, inst in executor.instances.items():
        if name == "bbox":
            continue

        o = inst.T[:3, 3].astype(float).copy()
        s = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float)

        if name in offsets:
            o = o + np.array(offsets[name], dtype=float)
        if name in scales:
            s = s * np.array(scales[name], dtype=float)

        for p0, p1 in _cuboid_edges_from_origin_size(o, s):
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                "-", color=color, linewidth=linewidth, alpha=0.9
            )

    # ---- Match aspect ratio to bbox ----
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

    # ---- Clean styling ----
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
