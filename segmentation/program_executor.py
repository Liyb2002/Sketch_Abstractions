#!/usr/bin/env python3
"""
execute_shapeassembly.py
Execute a ShapeAssembly-like JSON program (cuboids + attach/reflect/translate)
and export a combined mesh to STL.

Inputs:
  - INPUT_DIR/sketch_program_ir.json  (from previous stage)

Outputs (written into INPUT_DIR):
  - INPUT_DIR/sketch_model.stl

Assumptions:
  - All cuboids are axis-aligned with the bbox frame.
  - 'attach' aligns points (no rotation).
  - 'reflect' mirrors across bbox center plane for the given axis.
  - 'translate' makes n extra copies spaced to reach d * bbox_axis_length offset.
  - 'squeeze' is a no-op placeholder (safe to leave present in IR).

Install:
  pip install numpy trimesh
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

# ---------- Config ----------
INPUT_DIR = Path.cwd().parent / "input"
IN_IR  = INPUT_DIR / "sketch_program_ir.json"
OUT_STL = INPUT_DIR / "sketch_model.stl"

# ---------- Data Types ----------
@dataclass
class CuboidSpec:
    name: str
    l: float
    w: float
    h: float
    aligned: bool = True

@dataclass
class Instance:
    spec: CuboidSpec
    T: np.ndarray  # 4x4 world transform (we use translation only here)

# ---------- Helpers ----------
def vec(x, y, z) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)

def make_T(translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = translation
    return T

def center_from_T(spec: CuboidSpec, T: np.ndarray) -> np.ndarray:
    # center = origin + half extents
    return T[:3, 3] + vec(spec.l, spec.w, spec.h) * 0.5

def T_from_center(spec: CuboidSpec, center: np.ndarray) -> np.ndarray:
    # origin = center - half extents
    origin = center - vec(spec.l, spec.w, spec.h) * 0.5
    return make_T(origin)

# ---------- Executor ----------
class Executor:
    def __init__(self, ir: Dict):
        self.P = ir["program"]
        bb = self.P["bblock"]
        self.bbox = CuboidSpec("bbox", float(bb["l"]), float(bb["w"]), float(bb["h"]), bool(bb["aligned"]))
        # world frame: bbox origin at (0,0,0)
        self.instances: Dict[str, Instance] = {"bbox": Instance(self.bbox, make_T(vec(0,0,0)))}
        # declare cuboid specs
        self.specs: Dict[str, CuboidSpec] = {"bbox": self.bbox}
        for c in self.P.get("cuboids", []):
            name = str(c["var"])
            self.specs[name] = CuboidSpec(
                name=name,
                l=float(c["l"]), w=float(c["w"]), h=float(c["h"]),
                aligned=bool(c.get("aligned", True))
            )
            # not placed yet
        # apply statements
        self._apply_attaches()
        self._apply_reflects()
        self._apply_translates()
        # optional squeeze (no-op placeholder)
        # self._apply_squeezes()

    # ---- Attach ----
    def _apply_attaches(self):
        """
        attach(a, b, x1,y1,z1, x2,y2,z2)
        Place the un-grounded one so that:
          point_a_local = (x1 * a.l, y1 * a.w, z1 * a.h)
          point_b_world = T_b * (x2 * b.l, y2 * b.w, z2 * b.h)
        and T_a positions point_a_local exactly at point_b_world.
        """
        grounded = set(self.instances.keys())  # includes 'bbox'
        for a in self.P.get("attach", []):
            a_name, b_name = str(a["a"]), str(a["b"])
            x1,y1,z1 = float(a["x1"]), float(a["y1"]), float(a["z1"])
            x2,y2,z2 = float(a["x2"]), float(a["y2"]), float(a["z2"])

            # ensure specs exist
            if a_name not in self.specs or b_name not in self.specs:
                raise ValueError(f"attach refers unknown cuboid: {a_name} or {b_name}")

            # ensure at least one is grounded (as promised by IR validation)
            if not ((a_name in grounded) or (b_name in grounded)):
                raise ValueError(f"attach not grounded: {a_name}<->{b_name}")

            # fetch or place
            if b_name not in self.instances:
                # place b via current a (if a grounded)
                if a_name not in self.instances:
                    raise ValueError("Neither side placed—unexpected under grounded-order rule.")
                inst_a = self.instances[a_name]
                spec_b = self.specs[b_name]
                p_world = self._point_in_world(inst_a, self.specs[a_name], x1,y1,z1)
                p_b_local = vec(x2*spec_b.l, y2*spec_b.w, z2*spec_b.h)
                T_b = make_T(p_world - p_b_local)
                self.instances[b_name] = Instance(spec_b, T_b)
                grounded.add(b_name)
            else:
                # b exists; place a
                inst_b = self.instances[b_name]
                spec_a = self.specs[a_name]
                p_world = self._point_in_world(inst_b, self.specs[b_name], x2,y2,z2)
                p_a_local = vec(x1*spec_a.l, y1*spec_a.w, z1*spec_a.h)
                T_a = make_T(p_world - p_a_local)
                self.instances[a_name] = Instance(spec_a, T_a)
                grounded.add(a_name)

    def _point_in_world(self, inst: Instance, spec: CuboidSpec, x: float, y: float, z: float) -> np.ndarray:
        local = vec(x*spec.l, y*spec.w, z*spec.h)
        # only translation in T, so it's simply:
        return inst.T[:3,3] + local

    # ---- Reflect ----
    def _apply_reflects(self):
        """
        reflect(c, axis) : make a mirrored copy across the bbox center plane on that axis.
        New instance name auto-suffixed: e.g., part -> part_R1, part_R2, ...
        """
        suffix_count: Dict[str, int] = {}
        for r in self.P.get("reflect", []):
            src = str(r["c"])
            axis = str(r["axis"]).upper()
            if src not in self.instances:
                # if reflect is listed before attachment grounded it, skip gracefully
                # (you can prepass attaches in IR if needed)
                continue
            src_inst = self.instances[src]
            spec = src_inst.spec
            center = center_from_T(spec, src_inst.T)
            # bbox center:
            bb_center = vec(self.bbox.l, self.bbox.w, self.bbox.h) * 0.5

            mirrored_center = center.copy()
            if axis == "X":
                mirrored_center[0] = 2*bb_center[0] - center[0]
            elif axis == "Y":
                mirrored_center[1] = 2*bb_center[1] - center[1]
            elif axis == "Z":
                mirrored_center[2] = 2*bb_center[2] - center[2]
            else:
                raise ValueError(f"Bad axis for reflect: {axis}")

            T_new = T_from_center(spec, mirrored_center)

            n = suffix_count.get(src, 0) + 1
            suffix_count[src] = n
            new_name = f"{src}_R{n}"
            self.instances[new_name] = Instance(spec, T_new)

    # ---- Translate ----
    def _apply_translates(self):
        """
        translate(c, axis, n, d)
        Create n additional copies of c along axis so that the last copy's center
        is at offset d * bbox_axis_length from the original center.
        """
        suffix_count: Dict[str, int] = {}
        axis_vecs = {"X": vec(1,0,0), "Y": vec(0,1,0), "Z": vec(0,0,1)}
        axis_len = {"X": self.bbox.l, "Y": self.bbox.w, "Z": self.bbox.h}

        for t in self.P.get("translate", []):
            src = str(t["c"])
            axis = str(t["axis"]).upper()
            n = int(t["n"])
            d = float(t["d"])

            if src not in self.instances:
                continue  # not yet placed; safe to skip

            base = self.instances[src]
            direction = axis_vecs[axis]
            total_offset_world = direction * (d * axis_len[axis])
            if n <= 0:
                continue
            step = total_offset_world / float(n)

            for i in range(1, n+1):
                T_i = make_T(base.T[:3,3] + step * i)
                cnt = suffix_count.get(src, 0) + 1
                suffix_count[src] = cnt
                name_i = f"{src}_T{cnt}"
                self.instances[name_i] = Instance(base.spec, T_i)

    # ---- Mesh build ----
    def to_trimesh(self) -> trimesh.Trimesh:
        meshes: List[trimesh.Trimesh] = []
        for name, inst in self.instances.items():
            # skip bbox if you don't want it in the export; include it if you do
            if name == "bbox":
                # Comment out next line to EXCLUDE bbox from the model
                continue
            # trimesh box is centered at origin by default; we want origin at min corner
            # Trick: create box centered at half extents, or make and translate.
            box = trimesh.creation.box(extents=(inst.spec.l, inst.spec.w, inst.spec.h))
            # trimesh box is centered at origin; we need to move it so its min corner is at origin
            # box.bounds -> (min, max) around origin; we shift by +half extents
            half = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float) * 0.5
            box.apply_translation(half)
            # now place via instance transform (translation only in our T)
            box.apply_translation(inst.T[:3,3])
            meshes.append(box)

        if not meshes:
            # Export an empty bbox-sized point if nothing else; avoids failure
            return trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=int))

        return trimesh.util.concatenate(meshes)

# ---------- Main ----------
def main():
    if not IN_IR.exists():
        raise SystemExit(f"IR not found: {IN_IR}")

    ir = json.loads(IN_IR.read_text(encoding="utf-8"))
    exe = Executor(ir)
    mesh = exe.to_trimesh()

    # Ensure watertight? (not necessary for a union of boxes; STL is fine with overlaps)
    mesh.export(OUT_STL)
    print(f"✅ Wrote {OUT_STL}  (faces: {len(mesh.faces)}, verts: {len(mesh.vertices)})")

if __name__ == "__main__":
    main()
