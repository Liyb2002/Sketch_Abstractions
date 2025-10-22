# stroke-viewer/backend/load_program_main.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json

import numpy as np

__all__ = [
    "INPUT_DIR",
    "UI_DIST",
    "load_strokes_payload",
    "execute_default_to_cuboids",
]

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]   # .../stroke-viewer
INPUT_DIR = ROOT.parent / "input"            # sibling input/
UI_DIST = ROOT / "dist"                      # vite build output
DEFAULT_IR = INPUT_DIR / "sketch_program_ir.json"

# -------- Project imports (ensure backend/__init__.py exists) --------
from backend.shape_optimizer import rescale_and_execute
from backend import graph_utils  # must provide distance/anchor helpers


# ============================================================================
# Strokes I/O + normalize
# ============================================================================
def _is_point(p: Any) -> bool:
    return (
        isinstance(p, (list, tuple))
        and len(p) == 3
        and all(isinstance(x, (int, float)) for x in p)
    )

def _flatten_polylines(raw: Any) -> List[List[List[float]]]:
    """
    Accepts nested lists of points and returns a flat list of polylines.
    Keeps only polylines with >=2 points.
    """
    out: List[List[List[float]]] = []
    if not isinstance(raw, list):
        return out
    if all(_is_point(p) for p in raw):
        if len(raw) >= 2:
            out.append([[float(a), float(b), float(c)] for a, b, c in raw])
        return out
    for it in raw:
        if isinstance(it, list):
            if all(_is_point(p) for p in it):
                if len(it) >= 2:
                    out.append([[float(a), float(b), float(c)] for a, b, c in it])
            else:
                out.extend(_flatten_polylines(it))
    return out

def _norm_bundle(bundle: Any) -> List[Dict[str, Any]]:
    return [{"points": pl} for pl in _flatten_polylines(bundle)]

def _load_perturbed_feature_polylines() -> List[List[List[float]]]:
    """
    Return perturbed_feature_lines as pure polylines (no wrapping dicts).
    """
    path = INPUT_DIR / "stroke_lines.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("perturbed_feature_lines", [])
    return _flatten_polylines(raw)

def load_strokes_payload() -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns the UI payload:
    {
      "perturbed_feature_lines": [{"points":[...]}],
      "perturbed_construction_lines": [{"points":[...]}],
      "feature_lines": [{"points":[...]}]
    }
    """
    path = INPUT_DIR / "stroke_lines.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pf = _norm_bundle(data.get("perturbed_feature_lines", []))
    pc = _norm_bundle(data.get("perturbed_construction_lines", []))
    fl = _norm_bundle(data.get("feature_lines", []))
    return {
        "perturbed_feature_lines": pf,
        "perturbed_construction_lines": pc,
        "feature_lines": fl,
    }


# ============================================================================
# Offsets / scales (match your matplotlib plotter semantics)
# ============================================================================
def _load_offsets_scales(
    use_offsets: bool,
    use_scales: bool,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    offsets: Dict[str, List[float]] = {}
    scales: Dict[str, List[float]] = {}

    if use_offsets:
        tf = INPUT_DIR / "fit_translations.json"
        if tf.exists():
            try:
                data = json.loads(tf.read_text())
                offsets = data.get("offsets_xyz", {}) or {}
            except Exception as e:
                print(f"[warn] could not read {tf}: {e}")

    if use_scales:
        sf = INPUT_DIR / "fit_scales.json"
        if sf.exists():
            try:
                data = json.loads(sf.read_text())
                scales = data.get("scales_lwh", {}) or {}
            except Exception as e:
                print(f"[warn] could not read {sf}: {e}")

    return offsets, scales


# ============================================================================
# Executor -> cuboids (skip 'bbox'; convert origin->center)
# ============================================================================
def _executor_to_cuboids(
    executor: Any,
    *,
    use_offsets: bool,
    use_scales: bool,
) -> List[Dict[str, Any]]:
    """
    - iterate executor.instances (skip 'bbox')
    - origin o = inst.T[:3,3]
    - size   s = [inst.spec.l, inst.spec.w, inst.spec.h]
    - apply optional per-instance offsets/scales
    - convert to center = o + s/2
    """
    offsets, scales = _load_offsets_scales(use_offsets, use_scales)
    out: List[Dict[str, Any]] = []

    instances: Dict[str, Any] = getattr(executor, "instances", {})
    for name, inst in instances.items():
        if name == "bbox":
            continue

        o = inst.T[:3, 3].astype(float)  # origin (min corner)
        s = [float(inst.spec.l), float(inst.spec.w), float(inst.spec.h)]

        if use_offsets:
            off = offsets.get(name)
            if isinstance(off, (list, tuple)) and len(off) == 3:
                o = o + [float(off[0]), float(off[1]), float(off[2])]

        if use_scales:
            sc = scales.get(name)
            if isinstance(sc, (list, tuple)) and len(sc) == 3:
                s = [s[0] * float(sc[0]), s[1] * float(sc[1]), s[2] * float(sc[2])]

        center = [float(o[0] + s[0] / 2), float(o[1] + s[1] / 2), float(o[2] + s[2] / 2)]

        out.append({
            "id": name,
            "name": name,
            "center": center,
            "size": s,
            "rotationEuler": None,  # axis-aligned
        })

    return out


# ============================================================================
# Anchors: one stroke per cuboid
# ============================================================================
def _components_from_instances(exe: Any) -> List[Dict[str, Any]]:
    """
    Fallback: build minimal component list from executor.instances, skipping 'bbox'.
    """
    comps: List[Dict[str, Any]] = []
    instances: Dict[str, Any] = getattr(exe, "instances", {}) or {}
    for name, inst in instances.items():
        if name == "bbox":
            continue
        o = inst.T[:3, 3].astype(float)  # origin
        s = [float(inst.spec.l), float(inst.spec.w), float(inst.spec.h)]
        comps.append({
            "name": name,
            "origin": [float(o[0]), float(o[1]), float(o[2])],
            "size": s,
        })
    return comps

def _compute_anchors_for_executor(
    exe: Any,
    *,
    use_offsets: bool,
    use_scales: bool,
    global_thresh: float = 0.0,
    anchor_idx_per_comp: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns:
      [
        {
          "cuboidId": str,
          "cuboidName": str,
          "strokeIndex": int,
          # "strokePoints": [[x,y,z], ...]  # optional, UI fetches by index
        }, ...
      ]
    """
    # 1) Strokes
    strokes = _load_perturbed_feature_polylines() or []
    S = len(strokes)
    if S == 0:
        print("[anchor] no perturbed_feature_lines; returning empty anchors")
        return []

    # 2) Components: prefer exe.primitives(), else fallback to instances
    try:
        components = exe.primitives() if hasattr(exe, "primitives") else None
    except Exception as e:
        print(f"[anchor] exe.primitives() raised: {e}")
        components = None

    if components is None:
        components = _components_from_instances(exe)

    if components is None:
        components = []
    elif not isinstance(components, (list, tuple)):
        components = list(components)

    C = len(components)
    if C == 0:
        print("[anchor] no components; returning empty anchors")
        return []

    # Build instance name order (for output IDs/names)
    instance_names = [n for n in getattr(exe, "instances", {}).keys() if n != "bbox"]
    if len(instance_names) != C:
        cand_names = []
        for k in range(C):
            nm = getattr(components[k], "name", None) if hasattr(components[k], "name") else None
            if nm is None and isinstance(components[k], dict):
                nm = components[k].get("name")
            cand_names.append(nm or f"comp_{k}")
        instance_names = cand_names

    # 3) Preferred path: graph_utils
    try:
        D = graph_utils.stroke_cuboid_distance_matrix(strokes, components)  # (S x C)
        if not hasattr(D, "shape"):
            raise RuntimeError("distance matrix D has no shape")
        if D.shape != (S, C):
            raise RuntimeError(f"D shape mismatch: got {D.shape}, expected {(S, C)}")

        C_init = graph_utils.distances_to_confidence(D, global_thresh)      # (S x C)
        _, anchor_mask, _ = graph_utils.make_anchor_onehots(C_init, anchor_idx_per_comp)
        if anchor_mask is None:
            raise RuntimeError("anchor_mask is None")
        am = np.asarray(anchor_mask)
        if am.ndim != 2 or am.shape != (S, C):
            raise RuntimeError(f"anchor_mask bad shape {am.shape}, expected {(S, C)}")

        anchors: List[Dict[str, Any]] = []
        for col, comp_name in enumerate(instance_names):
            rows = np.where(am[:, col] == 1)[0]
            if rows.size == 0:
                sidx = int(np.argmax(C_init[:, col]))  # fallback per col
            else:
                sidx = int(rows[0])
            anchors.append({
                "cuboidId": comp_name,
                "cuboidName": comp_name,
                "strokeIndex": sidx,
            })
        return anchors

    except Exception as e:
        # 4) Fallback: nearest stroke to cuboid center (centroid distance)
        print(f"[anchor] graph_utils path failed: {e} — using centroid fallback")
        # Create centers
        centers: List[List[float]] = []
        for k, comp in enumerate(components):
            if isinstance(comp, dict) and "origin" in comp and "size" in comp:
                ox, oy, oz = comp["origin"]
                L, W, H = comp["size"]
                centers.append([ox + L/2.0, oy + W/2.0, oz + H/2.0])
            else:
                name = instance_names[k]
                inst = getattr(exe, "instances", {}).get(name)
                if inst is None:
                    centers.append([0.0, 0.0, 0.0])
                else:
                    o = inst.T[:3, 3].astype(float)
                    L, W, H = float(inst.spec.l), float(inst.spec.w), float(inst.spec.h)
                    centers.append([float(o[0] + L/2.0), float(o[1] + W/2.0), float(o[2] + H/2.0)])

        # Stroke centroids
        stroke_centroids = []
        for pts in strokes:
            pts_arr = np.asarray(pts, dtype=float)
            if pts_arr.ndim != 2 or pts_arr.shape[1] != 3:
                stroke_centroids.append(np.array([0.0, 0.0, 0.0], dtype=float))
            else:
                stroke_centroids.append(pts_arr.mean(axis=0))
        stroke_centroids = np.vstack(stroke_centroids)  # (S,3)

        anchors: List[Dict[str, Any]] = []
        for col, comp_name in enumerate(instance_names):
            ctr = np.asarray(centers[col], dtype=float)
            dists = np.linalg.norm(stroke_centroids - ctr[None, :], axis=1)
            sidx = int(np.argmin(dists))
            anchors.append({
                "cuboidId": comp_name,
                "cuboidName": comp_name,
                "strokeIndex": sidx,
            })
        return anchors


# ============================================================================
# Public API: default IR only
# ============================================================================
def execute_default_to_cuboids(
    *,
    use_offsets: bool = False,
    use_scales: bool = False,
) -> Dict[str, Any]:
    """
    Execute default IR and return { "cuboids": [...], "anchors": [...] }.
    Anchors are non-fatal: failures yield an empty list and a warning.
    """
    if not DEFAULT_IR.exists():
        raise FileNotFoundError(f"Default IR not found: {DEFAULT_IR}")

    exe = rescale_and_execute(INPUT_DIR, DEFAULT_IR)
    cuboids = _executor_to_cuboids(exe, use_offsets=use_offsets, use_scales=use_scales)

    anchors: List[Dict[str, Any]] = []
    try:
        anchors = _compute_anchors_for_executor(
            exe, use_offsets=use_offsets, use_scales=use_scales
        )
    except Exception as e:
        print(f"[warn] anchor computation failed: {e}")

    # Debug: see anchors server-side
    print("[execute_default_to_cuboids] anchors:", len(anchors))
    return {"cuboids": cuboids, "anchors": anchors}
