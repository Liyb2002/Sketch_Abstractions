#!/usr/bin/env python3
"""
program_executor.py  (Z-up)

Execute a ShapeAssembly-like JSON program (cuboids + attach/reflect/translate)
and export a combined mesh to STL. Also exposes analytic primitives for downstream
optimization (edges/faces per cuboid).

Coordinate system & semantics
-----------------------------
- Z is UP. All cuboids are axis-aligned in world space (no rotations in this executor).
- Each attach entry treats (x,y,z) in [0,1]^3 as *fractions from the cuboid's min corner*.
  Example: (x=0.0,y=0.0,z=0.0) is the min corner; (1,1,1) is the max corner; (0.5,0.5,0.0) is
  the center of the bottom face. With Z-up, decreasing Z moves things down.

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
@dataclass(frozen=True)
class CuboidSpec:
    name: str
    l: float
    w: float
    h: float
    aligned: bool = True  # reserved for future rotations

@dataclass
class Instance:
    spec: CuboidSpec
    # World transform for the cuboid's *min corner*. This executor stores only translation.
    T: np.ndarray  # 4x4

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

def _assert_positive_size(s: CuboidSpec):
    if not (s.l > 0 and s.w > 0 and s.h > 0):
        raise ValueError(f"Non-positive size in cuboid '{s.name}': (l={s.l}, w={s.w}, h={s.h})")


# ---------- Executor ----------
class Executor:
    """
    Minimal interpreter: attach, reflect, translate.
    All cuboids are axis-aligned; transforms are translations only.

    Supports extended bbox grammar:
      - program.bblock has required l,w,h
      - optionally also has min/max; if so, bbox is placed at min, size = max-min
        (l,w,h must agree with max-min within a small tolerance).
      - if min/max absent, bbox min is assumed at (0,0,0).
    """

    def __init__(self, ir: Dict, *, attach_max_passes: int = 64, tol: float = 1e-6):
        self.P = ir["program"]
        self.tol = float(tol)

        # ---- Read bbox with optional min/max
        bb = self.P["bblock"]
        if "min" in bb and "max" in bb:
            bb_min = np.array(bb["min"], dtype=float).reshape(3)
            bb_max = np.array(bb["max"], dtype=float).reshape(3)
            LWH = (bb_max - bb_min).astype(float)
            L, W, H = map(float, LWH)
            # if l,w,h present, validate
            if all(k in bb for k in ("l","w","h")):
                exp = np.array([bb["l"], bb["w"], bb["h"]], dtype=float)
                if not np.allclose(exp, LWH, rtol=1e-5, atol=1e-8):
                    raise ValueError(f"bbox l,w,h do not match max-min: given {exp} vs {LWH}")
            origin_world = bb_min
        else:
            L, W, H = float(bb["l"]), float(bb["w"]), float(bb["h"])
            origin_world = vec(0, 0, 0)

        self.bbox = CuboidSpec("bbox", L, W, H, bool(bb.get("aligned", True)))
        _assert_positive_size(self.bbox)

        # World frame: bbox min is at origin_world
        self.instances: Dict[str, Instance] = {"bbox": Instance(self.bbox, make_T(origin_world))}

        # declare cuboid specs
        self.specs: Dict[str, CuboidSpec] = {"bbox": self.bbox}
        for c in self.P.get("cuboids", []):
            name = str(c["var"])
            spec = CuboidSpec(
                name=name,
                l=float(c["l"]), w=float(c["w"]), h=float(c["h"]),
                aligned=bool(c.get("aligned", True))
            )
            _assert_positive_size(spec)
            if name in self.specs:
                raise ValueError(f"Duplicate cuboid spec name: {name}")
            self.specs[name] = spec

        # apply statements in robust order
        self._apply_attaches(max_passes=attach_max_passes)  # multi-pass until fixed point
        self._apply_reflects(origin_world)                  # reflect across world bbox center
        self._apply_translates()                            # Step-3-compatible translate

    # ---- Attach (robust to IR ordering via multi-pass) ----
    def _apply_attaches(self, *, max_passes: int):
        grounded = set(self.instances.keys())  # includes 'bbox'
        pending = list(self.P.get("attach", []))

        for _pass in range(max_passes):
            if not pending:
                break
            next_pending = []
            progressed = False

            for a in pending:
                a_name, b_name = str(a["a"]), str(a["b"])
                x1,y1,z1 = float(a["x1"]), float(a["y1"]), float(a["z1"])
                x2,y2,z2 = float(a["x2"]), float(a["y2"]), float(a["z2"])

                if a_name not in self.specs or b_name not in self.specs:
                    raise ValueError(f"attach refers to unknown cuboid: '{a_name}' or '{b_name}'")

                # One side grounded?
                a_g, b_g = a_name in grounded, b_name in grounded
                if a_g ^ b_g:
                    if b_g:
                        inst_b = self.instances[b_name]
                        spec_a = self.specs[a_name]
                        p_world = self._point_in_world(inst_b, inst_b.spec, x2,y2,z2)
                        p_a_local = vec(x1*spec_a.l, y1*spec_a.w, z1*spec_a.h)
                        self.instances[a_name] = Instance(spec_a, make_T(p_world - p_a_local))
                        grounded.add(a_name)
                    else:
                        inst_a = self.instances[a_name]
                        spec_b = self.specs[b_name]
                        p_world = self._point_in_world(inst_a, inst_a.spec, x1,y1,z1)
                        p_b_local = vec(x2*spec_b.l, y2*spec_b.w, z2*spec_b.h)
                        self.instances[b_name] = Instance(spec_b, make_T(p_world - p_b_local))
                        grounded.add(b_name)
                    progressed = True
                else:
                    next_pending.append(a)

            pending = next_pending
            if not progressed:
                break

        if pending:
            names = [(str(a["a"]), str(a["b"])) for a in pending]
            raise ValueError(f"Unresolved attaches after {max_passes} passes (cyclic or ungrounded): {names}")

    def _point_in_world(self, inst: Instance, spec_of_inst: CuboidSpec, x: float, y: float, z: float) -> np.ndarray:
        """Return world-space point at local fractional coords (x,y,z) on 'inst' cuboid."""
        local = vec(x*spec_of_inst.l, y*spec_of_inst.w, z*spec_of_inst.h)
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
            if axis not in ("X","Y","Z"):
                raise ValueError(f"Bad axis for reflect: {axis}")

            src_inst = self.instances[src]
            spec = src_inst.spec
            center = center_from_T(spec, src_inst.T)

            mirrored_center = center.copy()
            ax = {"X":0,"Y":1,"Z":2}[axis]
            mirrored_center[ax] = 2*bb_center_world[ax] - center[ax]

            T_new = T_from_center(spec, mirrored_center)
            n = suffix_count.get(src, 0) + 1
            suffix_count[src] = n
            new_name = f"{src}_R{n}"
            self.instances[new_name] = Instance(spec, T_new)

    # ---- Translate (Step-3 semantics) ----
    def _apply_translates(self):
        """
        translate(c, axis, n, d)

        Step-3 convention:
          - n = total positions INCLUDING the prototype (n>=1).
          - d = per-step spacing in bbox-normalized units along the axis.

        Executor behavior:
          - If n<=1: no copies.
          - Otherwise create (n-1) copies at offsets i*d*axis_len, i=1..(n-1).
        """
        suffix_count: Dict[str, int] = {}
        axis_vecs = {"X": vec(1,0,0), "Y": vec(0,1,0), "Z": vec(0,0,1)}
        axis_len  = {"X": self.bbox.l, "Y": self.bbox.w, "Z": self.bbox.h}

        for t in self.P.get("translate", []):
            src  = str(t["c"])
            axis = str(t["axis"]).upper()
            n    = int(t["n"])
            d    = float(t["d"])

            if src not in self.instances:
                continue
            if axis not in axis_vecs:
                raise ValueError(f"Bad axis for translate: {axis}")
            if n <= 1:
                continue  # nothing to do

            base = self.instances[src]
            step_world = axis_vecs[axis] * (d * axis_len[axis])  # per-step offset in world units

            for i in range(1, n):  # make n-1 copies
                T_i = make_T(base.T[:3,3] + step_world * i)
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
            # Create a box with extents and move it so its min corner is at (0,0,0),
            # then translate to the instance origin.
            box = trimesh.creation.box(extents=(inst.spec.l, inst.spec.w, inst.spec.h))
            half = np.array([inst.spec.l, inst.spec.w, inst.spec.h], dtype=float) * 0.5
            box.apply_translation(half)          # trimesh box is centered; move to min-corner frame
            box.apply_translation(inst.T[:3,3])  # place
            meshes.append(box)

        if not meshes:
            return trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=int))
        return trimesh.util.concatenate(meshes)

    # ---- Primitive extraction for fitting ----
    def primitives(self) -> List[CuboidGeom]:
        """
        Return analytic cuboid primitives for all placed parts (excluding 'bbox').
        Each primitive exposes min-corner origin and (l,w,h) size.
        """
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
        print(f"  First cuboid '{p0.name}' origin={np.round(p0.origin,6).tolist()} size={np.round(p0.size,6).tolist()}")

if __name__ == "__main__":
    main()
