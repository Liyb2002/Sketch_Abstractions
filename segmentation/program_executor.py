#!/usr/bin/env python3
"""
program_executor.py

Execute a ShapeAssembly-like JSON program (cuboids + attach/reflect/translate)
and export a combined mesh to STL. Also exposes analytic primitives for downstream
optimization (edges/faces per cuboid).

Inputs (by default, in --input-dir):
  - sketch_program_ir.json

Outputs:
  - sketch_model.stl  (unless --no-export)

Usage:
  python program_executor.py --input-dir ./input
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
import argparse


# ---------- Data Types ----------
@dataclass
class CuboidSpec:
    name: str
    l: float
    w: float
    h: float
    aligned: bool = True  # reserved for future rotations

@dataclass
class Instance:
    spec: CuboidSpec
    T: np.ndarray  # 4x4 world transform (translation only in this executor)

@dataclass
class CuboidGeom:
    """Analytic geometry for a placed, axis-aligned cuboid."""
    name: str
    origin: np.ndarray  # min corner (x0,y0,z0)
    size: np.ndarray    # (l,w,h)

    def edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        l, w, h = self.size
        x0, y0, z0 = self.origin
        x1, y1, z1 = x0 + l, y0 + w, z0 + h
        vs = [
            np.array([x0,y0,z0]), np.array([x1,y0,z0]),
            np.array([x0,y1,z0]), np.array([x1,y1,z0]),
            np.array([x0,y0,z1]), np.array([x1,y0,z1]),
            np.array([x0,y1,z1]), np.array([x1,y1,z1]),
        ]
        E = []
        # bottom rectangle
        E += [(vs[0],vs[1]), (vs[1],vs[3]), (vs[3],vs[2]), (vs[2],vs[0])]
        # top rectangle
        E += [(vs[4],vs[5]), (vs[5],vs[7]), (vs[7],vs[6]), (vs[6],vs[4])]
        # verticals
        E += [(vs[0],vs[4]), (vs[1],vs[5]), (vs[2],vs[6]), (vs[3],vs[7])]
        return E

    def faces(self) -> List[Tuple[np.ndarray, float, Tuple[np.ndarray,np.ndarray]]]:
        """
        Faces as (normal n, plane offset d, (center, half_size)),
        with plane n·x = d and half_size the two in-plane half-lengths.
        Normals are one of ±ex, ±ey, ±ez.
        """
        l, w, h = self.size
        x0, y0, z0 = self.origin
        x1, y1, z1 = x0 + l, y0 + w, z0 + h
        faces = []
        # X- planes
        faces.append((np.array([ 1,0,0],float),  x0, (np.array([x0, (y0+y1)/2, (z0+z1)/2]), np.array([w/2, h/2]))))
        faces.append((np.array([-1,0,0],float), -x1, (np.array([x1, (y0+y1)/2, (z0+z1)/2]), np.array([w/2, h/2]))))
        # Y- planes
        faces.append((np.array([0, 1,0],float),  y0, (np.array([(x0+x1)/2, y0, (z0+z1)/2]), np.array([l/2, h/2]))))
        faces.append((np.array([0,-1,0],float), -y1, (np.array([(x0+x1)/2, y1, (z0+z1)/2]), np.array([l/2, h/2]))))
        # Z- planes
        faces.append((np.array([0,0, 1],float),  z0, (np.array([(x0+x1)/2, (y0+y1)/2, z0]), np.array([l/2, w/2]))))
        faces.append((np.array([0,0,-1],float), -z1, (np.array([(x0+x1)/2, (y0+y1)/2, z1]), np.array([l/2, w/2]))))
        return faces


# ---------- Helpers ----------
def vec(x, y, z) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)

def make_T(translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = translation
    return T

def center_from_T(spec: CuboidSpec, T: np.ndarray) -> np.ndarray:
    return T[:3, 3] + vec(spec.l, spec.w, spec.h) * 0.5

def T_from_center(spec: CuboidSpec, center: np.ndarray) -> np.ndarray:
    origin = center - vec(spec.l, spec.w, spec.h) * 0.5
    return make_T(origin)




# ---------- Executor ----------
class Executor:
    """
    Minimal interpreter: attach, reflect, translate.
    All cuboids are axis-aligned; transforms are translations only.

    Supports extended bbox grammar:
      - program.bblock has required l,w,h
      - optionally also has min/max; if so, bbox is placed at min, size = max-min
        (l,w,h must agree with max-min).
      - if min/max absent, bbox min is assumed at (0,0,0).
    """

    def __init__(self, ir: Dict):
        self.P = ir["program"]
        bb = self.P["bblock"]

        # ---- Read bbox with optional min/max
        if "min" in bb and "max" in bb:
            bb_min = np.array(bb["min"], dtype=float).reshape(3)
            bb_max = np.array(bb["max"], dtype=float).reshape(3)
            LWH   = (bb_max - bb_min).astype(float)
            L, W, H = float(LWH[0]), float(LWH[1]), float(LWH[2])
            origin_world = bb_min
        else:
            L, W, H = float(bb["l"]), float(bb["w"]), float(bb["h"])
            origin_world = vec(0, 0, 0)

        self.bbox = CuboidSpec("bbox", L, W, H, bool(bb.get("aligned", True)))

        # World frame: bbox min is at origin_world (NOT implicitly 0)
        self.instances: Dict[str, Instance] = {"bbox": Instance(self.bbox, make_T(origin_world))}

        # declare cuboid specs
        self.specs: Dict[str, CuboidSpec] = {"bbox": self.bbox}
        for c in self.P.get("cuboids", []):
            name = str(c["var"])
            self.specs[name] = CuboidSpec(
                name=name,
                l=float(c["l"]), w=float(c["w"]), h=float(c["h"]),
                aligned=bool(c.get("aligned", True))
            )

        # apply statements in robust order
        self._apply_attaches()    # multi-pass until fixed point
        self._apply_reflects(origin_world)  # <-- pass bbox origin for world-center reflection
        self._apply_translates()

    # ---- Attach (robust to IR ordering via multi-pass) ----
    def _apply_attaches(self):
        grounded = set(self.instances.keys())  # includes 'bbox'
        pending = list(self.P.get("attach", []))
        last_len = None
        while pending and last_len != len(pending):
            last_len = len(pending)
            next_pending = []
            for a in pending:
                a_name, b_name = str(a["a"]), str(a["b"])
                x1,y1,z1 = float(a["x1"]), float(a["y1"]), float(a["z1"])
                x2,y2,z2 = float(a["x2"]), float(a["y2"]), float(a["z2"])
                if a_name not in self.specs or b_name not in self.specs:
                    raise ValueError(f"attach refers unknown cuboid: {a_name} or {b_name}")
                if (a_name in grounded) ^ (b_name in grounded):
                    if b_name in grounded:
                        inst_b = self.instances[b_name]
                        spec_a = self.specs[a_name]
                        p_world = self._point_in_world(inst_b, self.specs[b_name], x2,y2,z2)
                        p_a_local = vec(x1*spec_a.l, y1*spec_a.w, z1*spec_a.h)
                        self.instances[a_name] = Instance(spec_a, make_T(p_world - p_a_local))
                        grounded.add(a_name)
                    else:
                        inst_a = self.instances[a_name]
                        spec_b = self.specs[b_name]
                        p_world = self._point_in_world(inst_a, self.specs[a_name], x1,y1,z1)
                        p_b_local = vec(x2*spec_b.l, y2*spec_b.w, z2*spec_b.h)
                        self.instances[b_name] = Instance(spec_b, make_T(p_world - p_b_local))
                        grounded.add(b_name)
                else:
                    next_pending.append(a)
            pending = next_pending
        if pending:
            raise ValueError(f"Unresolved attaches (cyclic or ungrounded): {pending}")

    def _point_in_world(self, inst: Instance, spec: CuboidSpec, x: float, y: float, z: float) -> np.ndarray:
        local = vec(x*spec.l, y*spec.w, z*spec.h)
        return inst.T[:3,3] + local

    # ---- Reflect ----
    def _apply_reflects(self, bbox_min_world: np.ndarray):
        """
        reflect(c, axis): mirrored copy across the **world** bbox center plane on that axis.
        New instance name auto-suffixed: part -> part_R1, part_R2, ...
        """
        suffix_count: Dict[str, int] = {}
        bb_half = vec(self.bbox.l, self.bbox.w, self.bbox.h) * 0.5
        bb_center_world = bbox_min_world + bb_half

        for r in self.P.get("reflect", []):
            src = str(r["c"])
            axis = str(r["axis"]).upper()
            if src not in self.instances:
                continue
            src_inst = self.instances[src]
            spec = src_inst.spec
            center = center_from_T(spec, src_inst.T)

            mirrored_center = center.copy()
            if axis == "X":
                mirrored_center[0] = 2*bb_center_world[0] - center[0]
            elif axis == "Y":
                mirrored_center[1] = 2*bb_center_world[1] - center[1]
            elif axis == "Z":
                mirrored_center[2] = 2*bb_center_world[2] - center[2]
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
                continue
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
            if name == "bbox":
                continue  # exclude bbox from export
            box = trimesh.creation.box(extents=(inst.spec.l, inst.spec.w, inst.spec.h))
            half = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float) * 0.5
            box.apply_translation(half)          # move min corner to origin
            box.apply_translation(inst.T[:3,3])  # place
            meshes.append(box)

        if not meshes:
            return trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=int))
        return trimesh.util.concatenate(meshes)

    # ---- Primitive extraction for fitting ----
    def primitives(self) -> List[CuboidGeom]:
        prims = []
        for name, inst in self.instances.items():
            if name == "bbox":
                continue
            o = inst.T[:3,3].copy()  # min corner by construction
            s = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float)
            prims.append(CuboidGeom(name=name, origin=o, size=s))
        return prims


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("./input"))
    parser.add_argument("--no-export", action="store_true", help="Skip STL export")
    args = parser.parse_args()

    in_ir = args.input_dir / "sketch_program_ir.json"
    out_stl = args.input_dir / "sketch_model.stl"

    if not in_ir.exists():
        raise SystemExit(f"IR not found: {in_ir}")

    ir = json.loads(in_ir.read_text(encoding="utf-8"))
    exe = Executor(ir)

    if not args.no_export:
        mesh = exe.to_trimesh()
        mesh.export(out_stl)
        print(f"Wrote {out_stl}  (faces: {len(mesh.faces)}, verts: {len(mesh.vertices)})")

    prims = exe.primitives()
    print(f"Placed cuboids (excluding bbox): {len(prims)}")
    if prims:
        p0 = prims[0]
        print(f"  First cuboid '{p0.name}' origin={p0.origin.tolist()} size={p0.size.tolist()}")




if __name__ == "__main__":
    main()
