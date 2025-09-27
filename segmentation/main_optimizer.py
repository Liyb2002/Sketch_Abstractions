#!/usr/bin/env python3
"""
main_optimizer.py

Diagnostics + deterministic vertical alignment:
  1) Execute current IR -> export STL
  2) Print strokes vs geometry min/max
  3) Compute dz and SHIFT the program.bblock.min.z/max.z
  4) OVERWRITE ../input/sketch_program_ir.json with the shifted IR
  5) Re-execute -> export STL and print diagnostics
  6) Visualize strokes + program (before and after)

Fixed I/O (no args):
  Reads:
    ../input/sketch_program_ir.json
    ../input/info.json  (expects bbox{x_min,x_max,y_min,y_max,z_min,z_max})
    ../input/perturbed_feature_lines.json + feature_lines.json (for plotting)
  Writes:
    ../input/sketch_model.stl                (before)
    ../input/sketch_program_ir.json          (OVERWRITTEN with shifted IR)
    ../input/sketch_model.stl                (after; same filename)
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from program_executor import Executor
from shape_optimizer import (
    load_perturbed_feature_lines,
    plot_strokes_and_program,
)


# ---------- AABB helpers ----------
def _geom_aabb_from_executor(exe: Executor):
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


def _strokes_aabb_from_info(info: dict):
    """Extract stroke bbox from info.json (expects bbox with x_min..z_max)."""
    b = info.get("bbox")
    if not b:
        raise SystemExit("info.json missing 'bbox' with x_min..z_max")
    mins = (float(b["x_min"]), float(b["y_min"]), float(b["z_min"]))
    maxs = (float(b["x_max"]), float(b["y_max"]), float(b["z_max"]))
    return mins, maxs


# ---------- IR editing (Z shift via bbox min/max only) ----------
def _ensure_bbox_minmax_inplace(ir: dict):
    """Ensure program.bblock has min/max; if absent, assume min=(0,0,0), max=(l,w,h)."""
    bb = ir["program"]["bblock"]
    if "min" not in bb or "max" not in bb:
        L, W, H = float(bb["l"]), float(bb["w"]), float(bb["h"])
        bb["min"] = [0.0, 0.0, 0.0]
        bb["max"] = [L, W, H]


def _shift_ir_bbox_z_inplace(ir: dict, dz: float):
    """Translate the entire program in Z by shifting bbox min/max.z in place."""
    _ensure_bbox_minmax_inplace(ir)
    bb = ir["program"]["bblock"]
    bb["min"][2] = float(bb["min"][2]) + dz
    bb["max"][2] = float(bb["max"][2]) + dz
    # l/w/h unchanged (size preserved)


def _print_diagnostics(tag: str, strokes_min, strokes_max, geom_min, geom_max):
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


def run_once():
    input_dir = Path.cwd().parent / "input"
    ir_path   = input_dir / "sketch_program_ir.json"
    info_path = input_dir / "info.json"
    out_stl   = input_dir / "sketch_model.stl"

    if not ir_path.exists():
        raise SystemExit(f"IR not found: {ir_path}")
    if not info_path.exists():
        raise SystemExit(f"info.json not found: {info_path}")

    # Load IR & strokes info
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    sample_points, feature_lines = load_perturbed_feature_lines(input_dir)

    # ---------- BEFORE SHIFT ----------
    exe = Executor(ir)
    mesh = exe.to_trimesh()
    mesh.export(out_stl)
    print(f"✅ Wrote {out_stl}  (faces: {len(mesh.faces)}, verts: {len(mesh.vertices)})")

    strokes_min, strokes_max = _strokes_aabb_from_info(info)
    geom_min, geom_max = _geom_aabb_from_executor(exe)
    _print_diagnostics("before shift", strokes_min, strokes_max, geom_min, geom_max)

    # Visualize BEFORE
    print("📈 Plotting BEFORE shift ...")
    plot_strokes_and_program(exe, sample_points, feature_lines)

    # ---------- COMPUTE & APPLY Z SHIFT ----------
    dz_min = strokes_min[2] - geom_min[2]
    dz_max = strokes_max[2] - geom_max[2]
    # Default to align mins; if max correction is smaller magnitude, use that instead
    dz = dz_min if abs(dz_min) <= abs(dz_max) else dz_max
    strategy = "align geom.min.z → strokes.min.z" if dz == dz_min else "align geom.max.z → strokes.max.z"
    print(f"Proposed Z shift dz = {dz:.6f} ({strategy})")

    if abs(dz) >= 1e-9:
        # Shift IR IN PLACE and OVERWRITE the canonical IR file
        _shift_ir_bbox_z_inplace(ir, dz)
        ir_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
        print(f"🧩 Overwrote IR with Z-shift at: {ir_path}")
    else:
        print("dz is negligible; IR not modified.")

    # ---------- AFTER SHIFT ----------
    # Re-execute from the overwritten IR on disk (for certainty)
    ir_after = json.loads(ir_path.read_text(encoding="utf-8"))
    exe2 = Executor(ir_after)
    mesh2 = exe2.to_trimesh()
    mesh2.export(out_stl)  # overwrite STL with the aligned version
    print(f"✅ Rewrote {out_stl}  (faces: {len(mesh2.faces)}, verts: {len(mesh2.vertices)})")

    geom_min2, geom_max2 = _geom_aabb_from_executor(exe2)
    _print_diagnostics("after shift", strokes_min, strokes_max, geom_min2, geom_max2)

    # Visualize AFTER
    print("📈 Plotting AFTER shift ...")
    plot_strokes_and_program(exe2, sample_points, feature_lines)


if __name__ == "__main__":
    run_once()
