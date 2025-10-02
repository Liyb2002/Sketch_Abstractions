#!/usr/bin/env python3
"""
program_update.py  — parent-preserving edit

Update one cuboid's size and placement while preserving its original parent attach:
  - Keep the cuboid-side fractional anchor (vA) as in the IR.
  - Recompute the parent-side fractional anchor (vB) so the cuboid's min corner
    lands at the requested new_origin given the new_size.
  - Do NOT remove other attaches (children keep working).
  - If no suitable parent attach exists or the graph becomes ungrounded,
    auto-ground missing components to bbox using old placements (rare safeguard).

Writes a sibling file with "_new" suffix.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Set
import json
import numpy as np


# ---------- helpers ----------
def _bbox_info(P: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    bb = P["bblock"]
    if "min" in bb and "max" in bb:
        bb_min = np.array(bb["min"], float).reshape(3)
        bb_max = np.array(bb["max"], float).reshape(3)
        size = (bb_max - bb_min).astype(float)
        return bb_min, size
    bb_min = np.zeros(3, dtype=float)
    size = np.array([bb["l"], bb["w"], bb["h"]], dtype=float)
    return bb_min, size

def _spec_size(P: Dict[str, Any], name: str) -> np.ndarray:
    if name == "bbox":
        _, sz = _bbox_info(P)
        return sz
    for c in P.get("cuboids", []):
        if str(c["var"]) == name:
            return np.array([float(c["l"]), float(c["w"]), float(c["h"])], dtype=float)
    raise KeyError(f"Unknown spec: {name}")

def _reachable_from_bbox(P: Dict[str, Any]) -> Set[str]:
    nodes = {"bbox"} | {str(c["var"]) for c in P.get("cuboids", [])}
    nbrs: Dict[str, Set[str]] = {n: set() for n in nodes}
    for a in P.get("attach", []):
        A, B = str(a["a"]), str(a["b"])
        if A in nbrs: nbrs[A].add(B)
        if B in nbrs: nbrs[B].add(A)
    seen, stack = set(), ["bbox"]
    while stack:
        u = stack.pop()
        if u in seen: continue
        seen.add(u)
        stack.extend(nbrs.get(u, ()))
    return seen

def _find_parent_attach(P: Dict[str, Any], target: str, reachable: Set[str]) -> Optional[Tuple[int, str, str, np.ndarray]]:
    """
    Return (attach_index, parent_name, side, vA) where:
      - if side == 'a_is_target': vA = (x1,y1,z1) and parent=B
      - if side == 'b_is_target': vA = (x2,y2,z2) and parent=A
    Prefer parents that are reachable from bbox. Fall back to any attach involving target.
    """
    candidates: list[Tuple[int, str, str, np.ndarray, bool]] = []
    for idx, a in enumerate(P.get("attach", [])):
        A, B = str(a["a"]), str(a["b"])
        if A == target and B != target:
            parent = B
            vA = np.array([a["x1"], a["y1"], a["z1"]], dtype=float)
            candidates.append((idx, parent, "a_is_target", vA, parent in reachable))
        elif B == target and A != target:
            parent = A
            vA = np.array([a["x2"], a["y2"], a["z2"]], dtype=float)
            candidates.append((idx, parent, "b_is_target", vA, parent in reachable))
    if not candidates:
        return None
    # prefer reachable parent
    for (idx, parent, side, vA, good) in candidates:
        if good:
            return (idx, parent, side, vA)
    # otherwise, just take the first
    idx, parent, side, vA, _ = candidates[0]
    return (idx, parent, side, vA)

def _frac_on_spec(bb_min: np.ndarray, bb_size: np.ndarray, world_point: np.ndarray) -> np.ndarray:
    return (np.asarray(world_point, float) - bb_min) / np.maximum(bb_size, 1e-12)


# ---------- main ----------
def write_updated_program(
    ir_path: Path,
    cuboid_name: str,
    new_origin: np.ndarray,   # (3,) desired min corner of the edited cuboid (world)
    new_size:   np.ndarray,   # (3,) new (l,w,h) for the edited cuboid
    old_min_corners: Dict[str, np.ndarray] | None = None,  # prototype -> old world min corner (from old executor)
) -> Path:
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    P = ir["program"]

    # 1) Update cuboid's size in specs
    updated = False
    for c in P.get("cuboids", []):
        if str(c["var"]) == cuboid_name:
            c["l"], c["w"], c["h"] = map(float, new_size.tolist())
            updated = True
            break
    if not updated:
        raise ValueError(f"Cuboid '{cuboid_name}' not found in IR.")

    # 2) Preserve parent attach: recompute parent-side fractions, keep cuboid-side fractions
    #    a) find a parent attach in the ORIGINAL graph (before we touch attaches)
    reachable0 = _reachable_from_bbox(P)
    pa = _find_parent_attach(P, cuboid_name, reachable0)

    if pa is not None:
        attach_idx, parent_name, side, vA = pa  # keep vA as-is
        # Get parent's world origin and size
        if old_min_corners is None or parent_name not in old_min_corners:
            # if missing, default to bbox min (harmless if parent is bbox)
            if parent_name == "bbox":
                parent_o, parent_s = _bbox_info(P)
            else:
                parent_o = np.zeros(3, dtype=float)
                parent_s = _spec_size(P, parent_name)
        else:
            parent_o = np.asarray(old_min_corners[parent_name], dtype=float)
            parent_s = _spec_size(P, parent_name)

        # Solve for the parent-side fractions vB
        oA = np.asarray(new_origin, float)
        sA = np.asarray(new_size, float)
        vB = (oA + sA * vA - parent_o) / np.maximum(parent_s, 1e-12)

        # Write back into that attach entry (only the parent-side)
        a = P["attach"][attach_idx]
        if side == "a_is_target":
            # attach: a=target (keep x1,y1,z1), b=parent (overwrite x2,y2,z2)
            a["x2"], a["y2"], a["z2"] = map(float, vB.tolist())
        else:
            # attach: b=target (keep x2,y2,z2), a=parent (overwrite x1,y1,z1)
            a["x1"], a["y1"], a["z1"] = map(float, vB.tolist())
        # DO NOT modify other attaches — children keep their relations.

    else:
        # No parent attach found (rare): fall back to a direct anchor to bbox at new_origin (min corner)
        bb_min, bb_size = _bbox_info(P)
        frac = _frac_on_spec(bb_min, bb_size, new_origin)
        P.setdefault("attach", []).append({
            "a": cuboid_name, "b": "bbox",
            "x1": 0.0, "y1": 0.0, "z1": 0.0,
            "x2": float(frac[0]), "y2": float(frac[1]), "z2": float(frac[2]),
        })

    # 3) Safety net: ensure all prototypes are still reachable from bbox; if not, auto-ground
    reachable = _reachable_from_bbox(P)
    nodes = {"bbox"} | {str(c["var"]) for c in P.get("cuboids", [])}
    if old_min_corners is None:
        old_min_corners = {}
    bb_min, bb_size = _bbox_info(P)

    for name in nodes:
        if name == "bbox":
            continue
        if name not in reachable:
            world_min = old_min_corners.get(name, np.zeros(3, dtype=float))
            fx, fy, fz = _frac_on_spec(bb_min, bb_size, world_min).tolist()
            P.setdefault("attach", []).append({
                "a": name, "b": "bbox",
                "x1": 0.0, "y1": 0.0, "z1": 0.0,
                "x2": fx,  "y2": fy,  "z2": fz,
            })

    # 4) Save _new.json
    new_path = ir_path.with_name(ir_path.stem + "_new.json")
    new_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
    return new_path
