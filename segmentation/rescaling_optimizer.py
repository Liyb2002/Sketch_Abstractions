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

from __future__ import annotations
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
    print(f"\n===== BBOX DIAGNOSTICS ({tag}) =====")
    print(f"Strokes  min: {fmt(strokes_min)}   max: {fmt(strokes_max)}   size: {fmt(Ls)}")
    print(f"Geometry min: {fmt(geom_min)}     max: {fmt(geom_max)}     size: {fmt(Lg)}")
    dmin = (geom_min[0]-strokes_min[0], geom_min[1]-strokes_min[1], geom_min[2]-strokes_min[2])
    dmax = (geom_max[0]-strokes_max[0], geom_max[1]-strokes_max[1], geom_max[2]-strokes_max[2])
    print(f"Δmin (geom - strokes): {fmt(dmin)}")
    print(f"Δmax (geom - strokes): {fmt(dmax)}")
    print("====================================\n")


# ---------- Public API ----------
def rescale_and_execute(input_dir: Path) -> Executor:
    """
    - Loads IR + info
    - Executes current IR, prints diagnostics
    - Computes dz to align geometry Z to strokes Z (min-align by default; uses max if smaller correction)
    - Edits IR in place (overwrites sketch_program_ir.json)
    - Returns a fresh Executor built from the updated IR
    """
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

    # Decide Z shift: prefer aligning mins; if max correction is smaller magnitude, use that
    dz_min = strokes_min[2] - geom_min[2]
    dz_max = strokes_max[2] - geom_max[2]
    dz = dz_min if abs(dz_min) <= abs(dz_max) else dz_max
    print(f"Proposed Z shift dz = {dz:.6f}")

    if abs(dz) >= 1e-9:
        _shift_ir_bbox_z_inplace(ir, dz)
        ir_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
        print(f"🧩 Overwrote IR with Z-shift at: {ir_path}")
    else:
        print("dz negligible; IR not modified.")

    # Re-execute updated IR and print post diagnostics
    ir_after = json.loads(ir_path.read_text(encoding="utf-8"))
    exe_after = Executor(ir_after)
    geom_min2, geom_max2 = _geom_aabb_from_executor(exe_after)
    _print_diagnostics("after shift", strokes_min, strokes_max, geom_min2, geom_max2)

    return exe_after
