# stroke-viewer/backend/load_program_main.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os
from tempfile import NamedTemporaryFile

# ---- Public API of this module ------------------------------------------------

__all__ = [
    "INPUT_DIR",
    "UI_DIST",
    "load_strokes_payload",
    "execute_default_to_cuboids",
    "execute_program_json_to_cuboids",
]

# ---- Paths (centralized here) ------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]       # .../stroke-viewer
INPUT_DIR = ROOT.parent / "input"                # sibling input/
UI_DIST = ROOT / "dist"                          # vite build output

DEFAULT_IR = INPUT_DIR / "sketch_program_ir_editted.json"

# ---- Imports for your executor ------------------------------------------------
# Keep all executor details private to this module
from backend.shape_optimizer import rescale_and_execute  # your code

# ---- Strokes loader & normalization ------------------------------------------

def _is_point(p: Any) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 3 and all(isinstance(x, (int, float)) for x in p)

def _flatten_polylines(raw: Any) -> List[List[List[float]]]:
    """
    Accept many shapes:
      - [[x,y,z], ...]                       -> single polyline
      - [[[x,y,z],...], [[x,y,z],...], ...]  -> many polylines
      - nested mixes -> flatten
    Returns a list of polylines with >= 2 points each.
    """
    out: List[List[List[float]]] = []
    if not isinstance(raw, list):
        return out
    if all(_is_point(p) for p in raw):
        if len(raw) >= 2:
            out.append([[float(a), float(b), float(c)] for a,b,c in raw])
        return out
    for it in raw:
        if isinstance(it, list):
            if all(_is_point(p) for p in it):
                if len(it) >= 2:
                    out.append([[float(a), float(b), float(c)] for a,b,c in it])
            else:
                out.extend(_flatten_polylines(it))
    return out

def _norm_bundle(bundle: Any) -> List[Dict[str, Any]]:
    return [{"points": pl} for pl in _flatten_polylines(bundle)]

def load_strokes_payload() -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads input/stroke_lines.json and returns the UI payload:
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

# ---- Offsets / scales (match your plot_program_only semantics) ---------------

def _load_offsets_scales(use_offsets: bool, use_scales: bool) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    offsets: Dict[str, List[float]] = {}
    scales: Dict[str, List[float]] = {}
    if use_offsets:
        tf = INPUT_DIR / "fit_translations.json"
        if tf.exists():
            try:
                data = json.loads(tf.read_text())
                offsets = data.get("offsets_xyz", {}) or {}
            except Exception as e:
                print(f"[warn] read {tf} failed: {e}")
    if use_scales:
        sf = INPUT_DIR / "fit_scales.json"
        if sf.exists():
            try:
                data = json.loads(sf.read_text())
                scales = data.get("scales_lwh", {}) or {}
            except Exception as e:
                print(f"[warn] read {sf} failed: {e}")
    return offsets, scales

# ---- Executor -> cuboids (origin -> center conversion) -----------------------

def _executor_to_cuboids(executor: Any, *, use_offsets: bool, use_scales: bool) -> List[Dict[str, Any]]:
    """
    Mirror plot_program_only:
      for name, inst in executor.instances.items():
        if name=='bbox': skip
        o = inst.T[:3,3]      # origin (min corner)
        s = [inst.spec.l, inst.spec.w, inst.spec.h]  # size
        apply optional offsets/scales
        center = o + s/2
    No rotation in your pyplot plotter; keep rotationEuler=None.
    """
    offsets, scales = _load_offsets_scales(use_offsets, use_scales)
    out: List[Dict[str, Any]] = []

    instances = getattr(executor, "instances", {})
    for name, inst in instances.items():
        if name == "bbox":
            continue

        o = inst.T[:3, 3].astype(float)
        s = [float(inst.spec.l), float(inst.spec.w), float(inst.spec.h)]

        if use_offsets and name in offsets:
            off = offsets[name]
            if isinstance(off, (list, tuple)) and len(off) == 3:
                o = o + [float(off[0]), float(off[1]), float(off[2])]

        if use_scales and name in scales:
            sc = scales[name]
            if isinstance(sc, (list, tuple)) and len(sc) == 3:
                s = [s[0]*float(sc[0]), s[1]*float(sc[1]), s[2]*float(sc[2])]

        center = [float(o[0] + s[0]/2), float(o[1] + s[1]/2), float(o[2] + s[2]/2)]
        out.append({
            "id": name,
            "name": name,
            "center": center,
            "size": s,
            "rotationEuler": None,   # axis-aligned
        })
    return out

# ---- Public: execute default IR ---------------------------------------------

def execute_default_to_cuboids(*, use_offsets: bool = False, use_scales: bool = False) -> Dict[str, Any]:
    if not DEFAULT_IR.exists():
        raise FileNotFoundError(f"Default IR not found: {DEFAULT_IR}")
    exe = rescale_and_execute(INPUT_DIR, DEFAULT_IR)
    cuboids = _executor_to_cuboids(exe, use_offsets=use_offsets, use_scales=use_scales)
    return {"cuboids": cuboids}

# ---- Public: execute arbitrary program JSON ---------------------------------

def execute_program_json_to_cuboids(program_json: Dict[str, Any], *, use_offsets: bool = False, use_scales: bool = False) -> Dict[str, Any]:
    # Write to INPUT_DIR to keep any relative includes stable
    with NamedTemporaryFile("w", delete=False, suffix=".json", dir=str(INPUT_DIR)) as tmp:
        json.dump(program_json, tmp)
        tmp_path = Path(tmp.name)
    try:
        exe = rescale_and_execute(INPUT_DIR, tmp_path)
        cuboids = _executor_to_cuboids(exe, use_offsets=use_offsets, use_scales=use_scales)
        return {"cuboids": cuboids}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
