#!/usr/bin/env python3
"""
main_optimizer.py

Step 1: Execute the ShapeAssembly-like IR (cuboids + attach/reflect/translate)
        and export the resulting mesh to STL in ../input/.

Step 2: Load pre-sampled 3D strokes from ../input/perturbed_feature_lines.json.
        (No optimization yet—this just wires in the data pipeline.)

Usage:
  python main_optimizer.py
"""

from __future__ import annotations
from pathlib import Path
import json

from program_executor import Executor
from shape_optimizer import load_perturbed_feature_lines


def run_once():
    input_dir = Path.cwd().parent / "input"
    ir_path = input_dir / "sketch_program_ir.json"
    out_stl = input_dir / "sketch_model.stl"

    if not ir_path.exists():
        raise SystemExit(f"IR not found: {ir_path}")

    # ---- Step 1: Execute current program
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    exe = Executor(ir)

    mesh = exe.to_trimesh()
    mesh.export(out_stl)
    print(f"✅ Wrote {out_stl}  (faces: {len(mesh.faces)}, verts: {len(mesh.vertices)})")

    # ---- Step 2: Load pre-sampled 3D strokes
    strokes = load_perturbed_feature_lines(input_dir)


if __name__ == "__main__":
    run_once()
