#!/usr/bin/env python3
"""
program_update.py

Update a specific cuboid's size and world placement by:
  - setting its (l, w, h) to new_size
  - replacing any existing attaches that reference it with a single attach to bbox
    that places its MIN CORNER at new_origin (world).
Writes a sibling file with "_new" suffix.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np


def _bbox_info(P: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return bbox min (3,) and size (3,) from program JSON."""
    bb = P["bblock"]
    if "min" in bb and "max" in bb:
        bb_min = np.array(bb["min"], float).reshape(3)
        bb_max = np.array(bb["max"], float).reshape(3)
        size = (bb_max - bb_min).astype(float)
        return bb_min, size
    # default: bbox min at (0,0,0)
    bb_min = np.zeros(3, dtype=float)
    size = np.array([bb["l"], bb["w"], bb["h"]], dtype=float)
    return bb_min, size


def write_updated_program(
    ir_path: Path,
    cuboid_name: str,
    new_origin: np.ndarray,   # (3,) min corner in world coords
    new_size:   np.ndarray,   # (3,) (l,w,h)
) -> Path:
    """Load IR, update named cuboid size + attach to bbox at new_origin, save *_new.json."""
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    P = ir["program"]

    # ---- 1) Update the cuboid's size in specs ----
    found = False
    for c in P.get("cuboids", []):
        if str(c["var"]) == cuboid_name:
            c["l"], c["w"], c["h"] = float(new_size[0]), float(new_size[1]), float(new_size[2])
            found = True
            break
    if not found:
        raise ValueError(f"Cuboid '{cuboid_name}' not found in IR.")

    # ---- 2) Compute bbox-relative attach fractions to land MIN corner at new_origin ----
    bb_min, bb_size = _bbox_info(P)
    # fractions from bbox min corner (in [0,1] if inside bbox; can be outside)
    frac = (np.asarray(new_origin, float) - bb_min) / np.maximum(bb_size, 1e-12)
    x2, y2, z2 = map(float, frac)   # where on bbox we want to land the min corner

    # For the cuboid's own anchor, use its min corner: (x1,y1,z1) = (0,0,0)
    x1 = y1 = z1 = 0.0

    # ---- 3) Remove any existing attaches involving this cuboid ----
    attaches = P.get("attach", [])
    attaches = [
        a for a in attaches
        if str(a.get("a")) != cuboid_name and str(a.get("b")) != cuboid_name
    ]

    # ---- 4) Add a single attach from cuboid -> bbox to place its min corner ----
    attaches.append({
        "a": cuboid_name, "b": "bbox",
        "x1": x1, "y1": y1, "z1": z1,
        "x2": x2, "y2": y2, "z2": z2
    })
    P["attach"] = attaches

    # ---- 5) Save new IR beside the old one ----
    new_path = ir_path.with_name(ir_path.stem + "_new.json")
    new_json = json.dumps(ir, indent=2)
    new_path.write_text(new_json, encoding="utf-8")
    return new_path
