#!/usr/bin/env python3
"""
initial_program.py  (deterministic Step-3, strict Z-up with NO z-anchor rewrites)

One-shot IR builder (Step1 + Step3 + Deterministic scaling using strokes AABB):
- INPUT_DIR = Path.cwd().parent / "input"
- Uses:
    INPUT_DIR/sketch_narrative.json
    INPUT_DIR/sketch_components.json
    INPUT_DIR/stroke_lines.json   <-- used to set bblock.min/max after Step-3
- Produces:
    INPUT_DIR/sketch_program_ir_instanced.json   (raw Step-3, before deterministic scaling)
    INPUT_DIR/sketch_program_ir.json             (final, deterministically scaled with min/max)

Pipeline:
1) Step-1 (LLM): Draft minimal ShapeAssembly IR from narrative+components. (Prompt 1 unchanged)
2) Print the Step-1 program (DSL text) to screen.
3) Step-3 (deterministic): Translate-only instancing (deduplicate repeated parts with translate arrays).
4) Deterministic scaling (code): Set program.bblock.min/max to the strokes' AABB; rescale all cuboids to the new bbox size.

Conventions:
- Coordinate system is Z-up. The top of the object is the largest z.
- No z-anchor rewriting is performed. We preserve whatever Step-1 produced.
"""

from __future__ import annotations
import os, json, re, argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI  # used for Step-1


from program_executor import Executor


# ---------- Config ----------
INPUT_DIR = Path.cwd().parent / "input"
MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")  # chat model; no images used here

# ---------- Utilities ----------
FACES = {"right","left","top","bot","front","back"}
AXES  = {"X","Y","Z"}

def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        return json.loads(m.group(1))
    raise RuntimeError("Could not parse JSON from model output.")

def _ident(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s.strip())
    if not s or s[0].isdigit(): s = "x_" + s
    return s[:48]

def _num(x: Any) -> str:
    if isinstance(x, float):
        s = f"{x:.3f}".rstrip("0").rstrip(".")
        return s or "0"
    return str(x)

def emit_shapeassembly(ir: Dict[str, Any]) -> str:
    """
    DSL printer (unchanged): the printed DSL keeps
      bbox = Cuboid(l, w, h, aligned)
    Location (min/max) lives in JSON only.
    """
    P = ir["program"]
    name = _ident(P["name"])
    def emit_block(prog: Dict[str, Any], header: str) -> List[str]:
        L: List[str] = []
        L.append(f"def {header}:")
        bb = prog["bblock"]
        L.append(f"bbox = Cuboid({_num(float(bb['l']))}, {_num(float(bb['w']))}, {_num(float(bb['h']))}, {str(bool(bb['aligned']))})")
        for c in prog.get("cuboids", []):
            L.append(f"{_ident(c['var'])} = Cuboid({_num(float(c['l']))}, {_num(float(c['w']))}, {_num(float(c['h']))}, {str(bool(c.get('aligned', True)))})")
        for a in prog.get("attach", []):
            L.append(f"attach({_ident(a['a'])}, {_ident(a['b'])}, "
                     f"{_num(float(a['x1']))}, {_num(float(a['y1']))}, {_num(float(a['z1']))}, "
                     f"{_num(float(a['x2']))}, {_num(float(a['y2']))}, {_num(float(a['z2']))})")
        for s in prog.get("squeeze", []):
            L.append(f"squeeze({_ident(s['a'])}, {_ident(s['b'])}, {_ident(s['c'])}, "
                     f"{s['face']}, {_num(float(s['u']))}, {_num(float(s['v']))})")
        for r in prog.get("reflect", []):
            L.append(f"reflect({_ident(r['c'])}, {r['axis']})")
        for t in prog.get("translate", []):
            L.append(f"translate({_ident(t['c'])}, {t['axis']}, {int(t['n'])}, {_num(float(t['d']))})")
        return L
    lines: List[str] = []
    lines.extend(emit_block(P, f"{name}()"))
    lines.append("")
    for sub in P.get("subroutines", []):
        sig = sub["sig"]
        header = f"{_ident(sub['name'])}({_num(float(sig['l']))}, {_num(float(sig['w']))}, {_num(float(sig['h']))}, {str(bool(sig['aligned']))})"
        lines.extend(emit_block(sub, header))
        lines.append("")
    return "\n".join(lines)

def validate_ir(ir: Dict[str, Any]) -> None:
    if "program" not in ir: raise ValueError("IR missing 'program'")
    P = ir["program"]
    for key in ("name","bblock"):
        if key not in P: raise ValueError(f"program missing '{key}'")
    bb = P["bblock"]
    for k in ("l","w","h","aligned"):
        if k not in bb: raise ValueError(f"bblock missing '{k}'")
    # optional min/max; if present, they must agree with l,w,h
    if "min" in bb and "max" in bb:
        mn, mx = bb["min"], bb["max"]
        if not (isinstance(mn,(list,tuple)) and len(mn)==3 and isinstance(mx,(list,tuple)) and len(mx)==3):
            raise ValueError("bblock.min/max must be 3-element lists")
        size = [float(mx[i]-mn[i]) for i in range(3)]
        if abs(size[0]-float(bb["l"])) > 1e-5 or abs(size[1]-float(bb["w"])) > 1e-5 or abs(size[2]-float(bb["h"])) > 1e-5:
            raise ValueError("bblock.l/w/h disagree with (max-min)")
    declared = {"bbox"}
    cuboids = P.get("cuboids", [])
    if not isinstance(cuboids, list): raise ValueError("'cuboids' must be a list")
    for c in cuboids:
        for k in ("var","l","w","h","aligned"):
            if k not in c: raise ValueError("each cuboid needs var,l,w,h,aligned")
        if c["var"] in declared: raise ValueError(f"duplicate name '{c['var']}'")
        declared.add(c["var"])
    attaches = P.get("attach", [])
    grounded = {"bbox"}
    for a in attaches:
        for k in ("a","b","x1","y1","z1","x2","y2","z2"):
            if k not in a: raise ValueError("attach missing a key")
        if a["a"] not in declared or a["b"] not in declared:
            raise ValueError("attach refers to unknown cuboid")
        if not (a["a"] in grounded or a["b"] in grounded):
            raise ValueError(f"attach not grounded: {a}")
        grounded.add(a["a"]); grounded.add(a["b"])
        for k in ("x1","y1","z1","x2","y2","z2"):
            v = a[k]
            if not (isinstance(v,(int,float)) and -1e-6 <= v <= 1.0+1e-6):
                raise ValueError(f"attach coord '{k}' out of [0,1]: {v}")
    for s in P.get("squeeze", []):
        for k in ("a","b","c","face","u","v"):
            if k not in s: raise ValueError("squeeze missing a key")
        if s["face"] not in FACES: raise ValueError(f"bad face '{s['face']}'")
    for r in P.get("reflect", []):
        for k in ("c","axis"):
            if k not in r: raise ValueError("reflect missing a key")
        if r["axis"] not in AXES: raise ValueError("reflect axis must be X|Y|Z")
    for t in P.get("translate", []):
        for k in ("c","axis","n","d"):
            if k not in t: raise ValueError("translate missing a key")
        if t["axis"] not in AXES: raise ValueError("translate axis must be X|Y|Z")
        if int(t["n"]) < 1: raise ValueError("translate n must be >=1")
        float(t["d"])

def nearly_equal(a: float, b: float, tol: float=1e-6) -> bool:
    return abs(a-b) <= max(tol, 1e-6*(abs(a)+abs(b)+1.0))

def check_bbox_size_matches(ir: Dict[str,Any], LWH: Tuple[float,float,float]) -> Optional[str]:
    L,W,H = LWH
    try:
        bb = ir["program"]["bblock"]
        mism = []
        if not nearly_equal(float(bb["l"]), L): mism.append(f"bblock.l {bb['l']} != {L}")
        if not nearly_equal(float(bb["w"]), W): mism.append(f"bblock.w {bb['w']} != {W}")
        if not nearly_equal(float(bb["h"]), H): mism.append(f"bblock.h {bb['h']} != {H}")
        return None if not mism else "; ".join(mism)
    except Exception as e:
        return f"IR missing/invalid bblock: {e}"

def check_bbox_minmax_matches(ir: Dict[str,Any], mn: Tuple[float,float,float], mx: Tuple[float,float,float]) -> Optional[str]:
    bb = ir["program"]["bblock"]
    if "min" not in bb or "max" not in bb:
        return "bblock.min/max missing"
    err = []
    for i,k in enumerate("xyz"):
        if abs(float(bb["min"][i]) - float(mn[i])) > 1e-6: err.append(f"min.{k} {bb['min'][i]} != {mn[i]}")
        if abs(float(bb["max"][i]) - float(mx[i])) > 1e-6: err.append(f"max.{k} {bb['max'][i]} != {mx[i]}")
    return ", ".join(err) if err else None

# ---------- Strokes AABB ----------
def strokes_aabb(input_dir: Path) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Reads stroke_lines.json and returns (min, max) over all sampled stroke points.
    Expected structure: {"perturbed_feature_lines": List[List[[x,y,z], ...]], ...}
    """
    sp_path = input_dir / "stroke_lines.json"
    if not sp_path.exists():
        raise SystemExit(f"Missing {sp_path} — required for deterministic scaling.")
    
    # Load the combined stroke data
    data = json.loads(sp_path.read_text(encoding="utf-8"))
    perturbed_feature_lines = data.get("perturbed_feature_lines")
    if not isinstance(perturbed_feature_lines, list):
        raise SystemExit("stroke_lines.json must contain a list under 'perturbed_feature_lines'")
    
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    valid_pts = 0

    # Traverse all [x, y, z] points
    for s in perturbed_feature_lines:
        if not isinstance(s, list):
            continue
        for p in s:
            if not (isinstance(p, (list, tuple)) and len(p) == 3):
                continue
            valid_pts += 1
            for i in range(3):
                v = float(p[i])
                if v < mins[i]: mins[i] = v
                if v > maxs[i]: maxs[i] = v

    if valid_pts == 0:
        raise SystemExit("No valid [x,y,z] points found in 'perturbed_feature_lines' within stroke_lines.json")
    
    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])

# ---------- Prompts (Step-1 only) ----------

PROMPT_STEP1 = """
System instructions (comply strictly):
- ROLE: You are a ShapeAssembly compiler. Think internally; DO NOT reveal chain-of-thought.
- OUTPUT: Return ONLY a single JSON object that matches the schema below. No prose, no code fences.

Global constraints:
- Coordinate system is **Z-up**: x=l (length), y=w (width), z=h (height). The top of the object has the largest z.
- Use only these ops in Step-1: bbox + Cuboid(l,w,h,aligned), attach, squeeze. (Do NOT use reflect or translate.)
- Normalized coordinates: All coordinates (attach/squeeze) are in [0,1] in each local axis; z=0 is the bottom face, z=1 is the top face.
- Grounding: The very first attach must involve 'bbox'. After an attach, both endpoints are grounded.
- Keep the program minimal but valid; avoid overlaps when possible; keep 'aligned' = true unless contradicted.
- If scene size is unknown, set bbox l=w=h=1.0.
- Make sure no components should overlap in 3D space.

Authoring rules for attaches (exact ShapeAssembly, point-to-point):
- You always specify the child’s anchor (x1,y1,z1) and the parent’s anchor (x2,y2,z2) as fractions from each part’s **min corner**.
- When attaching the child A to the **BOTTOM** face of parent B (B’s z2 near 0), set **A.z1 = 1.0** (A’s TOP) so A extends downward from the joint.
- When attaching A to the **TOP** face of B (B’s z2 near 1), set **A.z1 = 0.0** (A’s BOTTOM).
- When attaching A to the **LEFT** face of B (B’s x2 near 0), set **A.x1 = 1.0** (A’s RIGHT).
- When attaching A to the **RIGHT** face of B (B’s x2 near 1), set **A.x1 = 0.0** (A’s LEFT).
- When attaching A to the **FRONT** face of B (B’s y2 near 0), set **A.y1 = 1.0** (A’s BACK).
- When attaching A to the **BACK** face of B (B’s y2 near 1), set **A.y1 = 0.0** (A’s FRONT).
- “Near” can be interpreted as ≤ 0.2 from the respective face unless otherwise needed to avoid overlaps.
- Do not invent rotations or flips; this is pure point-to-point ShapeAssembly semantics.
- Unless intentionally offset, attach by the child's top-center on the contact face:
  set (x1,y1) = (0.5,0.5) when attaching under a parent face.
- For repeated symmetric parts, choose parent anchor fractions that are uniformly spaced
  across the parent extent so that translate can be inferred (e.g., 0.15 and 0.85 for 2 items).

About the provided `components` JSON (authoritative; may describe ANY object, not only chairs):
- It can be one of:
  1) ["partA","partB","leg","leg", ...]                  # list of names (duplicates imply count)
  2) {"partA":2, "hinge":3, ...}                         # dict name -> count
  3) [{"name":"partA","count":2}, ...]                   # list of objects
- You MUST expand this **exactly** into instances. The final program MUST contain **exactly one Cuboid per instance**:
  * For every component with count N, create N instances (no more, no less).
  * Variable naming: sanitize the base name into an identifier, then append 1-based indices: e.g., `hinge1`, `hinge2`, ..., even if N=1.
  * Do NOT invent extra parts; do NOT omit parts; do NOT merge repeated parts; do NOT reuse one Cuboid for multiple instances.

Geometry guidance (generic):
- Choose reasonable (l,w,h) per instance (instances of the same base name should generally share sizes).
- Attach every cuboid to 'bbox' using normalized coordinates. Respect Z-up: contacts that rest on top of the bbox use z=1 on the bbox face and z=0 on the part’s bottom face.

Required JSON shape (use exactly these keys; arrays may be empty):
{
  "program": {
    "name": "Program1",
    "bblock": { "l": 1.0, "w": 1.0, "h": 1.0, "aligned": true },
    "cuboids": [ { "var":"part1","l":0.5,"w":0.5,"h":0.5,"aligned": true }, ... ],
    "attach":  [ { "a":"part1","b":"bbox","x1":0.5,"y1":0.5,"z1":0.0,"x2":0.5,"y2":0.5,"z2":1.0 }, ... ],
    "squeeze": [],
    "reflect": [],
    "translate": [],
    "subroutines": []
  }
}

Return ONLY the JSON object.
""".strip()



PROMPT_REPAIR1 = """
Your Step-1 JSON IR failed validation:

{errors}

Please return a corrected JSON IR that:
- Uses ONLY the same keys and schema.
- Satisfies grounded attachment order.
- Keeps all coordinates in [0,1].
- Is minimal (few cuboids/ops), consistent with the narrative & components provided.
Return ONLY the JSON object (no prose).
""".strip()

# ---------- Step-1 model calls ----------
def call_step1(narrative: str, components: List[str]) -> Dict[str, Any]:
    client = OpenAI()
    content = [
        {"type": "text", "text": PROMPT_STEP1},
        {"type": "text", "text": "Components: " + json.dumps(components, ensure_ascii=False)},
        {"type": "text", "text": "Narrative: " + narrative},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content": content}],
        temperature=0.2,
        max_tokens=2000,
    )
    return _extract_json(resp.choices[0].message.content or "")

def repair_step1(bad_ir: Dict[str,Any], errors: str, narrative: str, components: List[str]) -> Dict[str,Any]:
    client = OpenAI()
    content = [
        {"type": "text", "text": PROMPT_REPAIR1.replace("{errors}", errors)},
        {"type": "text", "text": "Original (invalid) JSON:"},
        {"type": "text", "text": json.dumps(bad_ir, ensure_ascii=False)},
        {"type": "text", "text": "Components: " + json.dumps(components, ensure_ascii=False)},
        {"type": "text", "text": "Narrative: " + narrative},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content": content}],
        temperature=0.1,
        max_tokens=2000,
    )
    return _extract_json(resp.choices[0].message.content or "")

# ---------- Deterministic Step-3: translate-only instancing ----------

_TRAILING_DIGITS = re.compile(r"(\d+)$")

def _basename(var: str) -> str:
    m = _TRAILING_DIGITS.search(var)
    return var[:m.start()] if m else var

def _has_trailing_digits(var: str) -> bool:
    return _TRAILING_DIGITS.search(var) is not None


def _is_uniform_progression(vals: List[float], tol: float=1e-6) -> Tuple[bool, float]:
    """
    Given sorted unique vals (should include 0 for the prototype), check equal spacing.
    Returns (ok, step).
    """
    if len(vals) <= 1:
        return True, 0.0
    diffs = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
    step = sum(diffs) / len(diffs)
    if step <= tol:  # degenerate
        return False, 0.0
    for d in diffs:
        if abs(d - step) > tol:
            return False, 0.0
    return True, step

def _parent_anchor_for(var: str, attach_list: List[Dict[str,Any]]) -> Optional[Tuple[str, Tuple[float,float,float]]]:
    """
    Return (parent_name, (x_parent,y_parent,z_parent)) from the FIRST attach that involves `var`.
    If var is on 'a' side, parent is 'b' and we return (x2,y2,z2).
    If var is on 'b' side, parent is 'a' and we return (x1,y1,z1).
    """
    for a in attach_list:
        if a["a"] == var:
            return a["b"], (float(a["x2"]), float(a["y2"]), float(a["z2"]))
        if a["b"] == var:
            return a["a"], (float(a["x1"]), float(a["y1"]), float(a["z1"]))
    return None




def deterministic_step3_translate(ir_in: Dict[str, Any], tol: float = 1e-6) -> Dict[str, Any]:
    """
    Deterministic translate-only instancing (program-agnostic, robust).

    FIX: Parent-side anchor steps are in the parent's normalized units. We convert them
    to bbox-normalized distances before emitting translate ops so they remain correct
    after deterministic scaling.
    """
    import json as _json
    import re as _re

    def _has_trailing_digits(var: str) -> bool:
        return _re.search(r"(\d+)$", var) is not None

    def _basename(var: str) -> str:
        m = _re.search(r"(\d+)$", var)
        return var[:m.start()] if m else var

    def _is_uniform_progression(vals: List[float], tol: float = 1e-6) -> Tuple[bool, float]:
        if len(vals) <= 1:
            return True, 0.0
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        step = sum(diffs) / len(diffs)
        if step <= tol:
            return False, 0.0
        return all(abs(d - step) <= tol for d in diffs), step

    def _parent_anchor_for(var: str, attach_list: List[Dict[str, Any]]) -> Optional[Tuple[str, Tuple[float, float, float]]]:
        for a in attach_list:
            if a["a"] == var:
                return a["b"], (float(a["x2"]), float(a["y2"]), float(a["z2"]))
            if a["b"] == var:
                return a["a"], (float(a["x1"]), float(a["y1"]), float(a["z1"]))
        return None

    def _round_frac3(t: Tuple[float, float, float], nd: int = 6) -> Tuple[float, float, float]:
        return (round(t[0], nd), round(t[1], nd), round(t[2], nd))

    # deep copy IR
    ir = _json.loads(_json.dumps(ir_in))
    P = ir["program"]
    cuboids: List[Dict[str, Any]] = P.get("cuboids", [])
    attaches: List[Dict[str, Any]] = P.get("attach", [])
    P.setdefault("translate", [])
    translates_existing: List[Dict[str, Any]] = P["translate"]

    if not cuboids or not attaches:
        return ir

    # sizes per var (for parent->bbox unit conversion)
    size_by_var_lwh: Dict[str, Tuple[float, float, float]] = {
        c["var"]: (float(c["l"]), float(c["w"]), float(c["h"])) for c in cuboids
    }
    bb = P["bblock"]
    bbox_size = (float(bb["l"]), float(bb["w"]), float(bb["h"]))
    size_by_var_lwh["bbox"] = bbox_size

    # group by basename + identical (l,w,h,aligned) and has trailing digits
    size_with_align: Dict[str, Tuple[float, float, float, bool]] = {
        c["var"]: (float(c["l"]), float(c["w"]), float(c["h"]), bool(c.get("aligned", True)))
        for c in cuboids
    }
    groups: Dict[Tuple[str, Tuple[float, float, float, bool]], List[str]] = {}
    for c in cuboids:
        var = c["var"]
        if not _has_trailing_digits(var):
            continue
        key = (_basename(var), size_with_align[var])
        groups.setdefault(key, []).append(var)

    if not groups:
        return ir

    removed_vars_total: List[str] = []
    new_translates: List[Dict[str, Any]] = []

    for (base, _size_wa), members in groups.items():
        if len(members) < 2:
            continue

        # collect first-parent anchors; require common parent
        parent_by_member: Dict[str, str] = {}
        anchor_by_member: Dict[str, Tuple[float, float, float]] = {}
        common_parent: Optional[str] = None
        ok_group = True

        for v in members:
            pa = _parent_anchor_for(v, attaches)
            if pa is None:
                ok_group = False
                break
            parent_name, parent_frac = pa
            parent_by_member[v] = parent_name
            anchor_by_member[v] = _round_frac3(parent_frac, nd=6)
            if common_parent is None:
                common_parent = parent_name
            elif parent_name != common_parent:
                ok_group = False
                break

        if not ok_group or common_parent is None:
            continue

        # choose prototype = lexicographically minimal (x,y,z) in parent fractions
        proto = min(members, key=lambda v: anchor_by_member[v])
        axp, ayp, azp = anchor_by_member[proto]

        # deltas from prototype (still in PARENT FRACTIONS)
        deltas = {
            v: (anchor_by_member[v][0] - axp,
                anchor_by_member[v][1] - ayp,
                anchor_by_member[v][2] - azp)
            for v in members
        }

        xs = sorted(set(round(d[0], 6) for d in deltas.values()))
        ys = sorted(set(round(d[1], 6) for d in deltas.values()))
        zs = sorted(set(round(d[2], 6) for d in deltas.values()))

        okx, dx_parent = _is_uniform_progression(xs, tol)
        oky, dy_parent = _is_uniform_progression(ys, tol)
        okz, dz_parent = _is_uniform_progression(zs, tol)
        if not (okx and oky and okz):
            continue

        # full cartesian coverage check
        expected = len(xs) * len(ys) * len(zs)
        if expected != len(members):
            continue
        grid = {(x, y, z) for x in xs for y in ys for z in zs}
        if any((round(dxv, 6), round(dyv, 6), round(dzv, 6)) not in grid for (dxv, dyv, dzv) in deltas.values()):
            continue

        # --- unit conversion: parent fractions -> bbox-normalized distances ---
        Lp, Wp, Hp = size_by_var_lwh.get(common_parent, bbox_size)
        Lbb, Wbb, Hbb = bbox_size

        def _norm_step(frac_step: float, parent_axis_size: float, bbox_axis_size: float) -> float:
            if abs(frac_step) <= tol or parent_axis_size <= tol or bbox_axis_size <= tol:
                return 0.0
            return (frac_step * parent_axis_size) / bbox_axis_size

        dX = _norm_step(dx_parent, Lp, Lbb)
        dY = _norm_step(dy_parent, Wp, Wbb)
        dZ = _norm_step(dz_parent, Hp, Hbb)

        # emit translates (n includes prototype)
        if len(xs) > 1 and abs(dX) > tol:
            new_translates.append({"c": proto, "axis": "X", "n": len(xs), "d": float(dX)})
        if len(ys) > 1 and abs(dY) > tol:
            new_translates.append({"c": proto, "axis": "Y", "n": len(ys), "d": float(dY)})
        if len(zs) > 1 and abs(dZ) > tol:
            new_translates.append({"c": proto, "axis": "Z", "n": len(zs), "d": float(dZ)})

        # remove all but prototype
        removed_vars_total.extend([v for v in members if v != proto])

    if not removed_vars_total and not new_translates:
        return ir

    removed_set = set(removed_vars_total)

    # prune geometry & ops that reference removed vars
    P["cuboids"] = [c for c in P.get("cuboids", []) if c["var"] not in removed_set]
    P["attach"] = [a for a in attaches if a["a"] not in removed_set and a["b"] not in removed_set]
    if "squeeze" in P:
        P["squeeze"] = [s for s in P["squeeze"] if s["a"] not in removed_set and s["b"] not in removed_set and s["c"] not in removed_set]
    if "reflect" in P:
        P["reflect"] = [r for r in P["reflect"] if r["c"] not in removed_set]
    if "translate" in P:
        P["translate"] = [t for t in translates_existing if t["c"] not in removed_set]
    else:
        P["translate"] = []

    # append our new arrays last (deterministic order)
    P["translate"].extend(new_translates)
    return ir


# ---------- Deterministic scaling (after Step-3) ----------
def deterministic_scale_to_strokes_bbox(ir_in: Dict[str, Any], input_dir: Path) -> Dict[str, Any]:
    """
    Deterministically scale the program so the *executed geometry spans* (L,W,H)
    match the strokes' AABB spans. Then set program.bblock.{min,max,l,w,h} to the
    strokes' bbox. Prints a side-by-side report.

    Notes
    -----
    - Current spans are computed from the *executed cuboids* (like your
      _cuboids_height_from_executor logic, generalized to X/Y/Z).
    - Executor is loaded dynamically from a sibling 'program_executor.py'.
    - We only scale sizes (l,w,h) in IR; we don't rewrite anchors/translates.
    - After scaling + declaring bbox to strokes AABB, we re-execute for the final report.
    """
    import json as _json
    import math as _math
    import importlib.util as _importlib_util
    from pathlib import Path as _Path

    # ------------------------ sub-functions (scoped) ------------------------

    def _load_executor_from_neighbor():
        """Load program_executor.Executor from the same folder as this file."""
        here = _Path(__file__).parent
        mod_path = here / "program_executor.py"
        if not mod_path.exists():
            raise RuntimeError(f"program_executor.py not found next to {__file__}")
        spec = _importlib_util.spec_from_file_location("program_executor", str(mod_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load spec for program_executor.py")
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        if not hasattr(module, "Executor"):
            raise RuntimeError("program_executor.py does not define Executor")
        return module.Executor

    def _executed_cuboids_aabb(exe) -> Tuple[Tuple[float,float,float], Tuple[float,float,float], int]:
        """
        AABB over executed **cuboid primitives** (exclude bbox).
        Returns (min, max, count). If no cuboids found, returns (inf,-inf,0) for detection.
        """
        prims = []
        try:
            prims = exe.primitives()
        except Exception:
            prims = []

        if not prims:
            return ( _math.inf, _math.inf, _math.inf ), ( -_math.inf, -_math.inf, -_math.inf ), 0

        xmins, ymins, zmins = [], [], []
        xmaxs, ymaxs, zmaxs = [], [], []
        for p in prims:
            # p.origin = min corner; p.size = (l,w,h)
            x0, y0, z0 = float(p.origin[0]), float(p.origin[1]), float(p.origin[2])
            l,  w,  h  = float(p.size[0]),   float(p.size[1]),   float(p.size[2])
            x1, y1, z1 = x0 + l, y0 + w, z0 + h
            xmins.append(x0); ymins.append(y0); zmins.append(z0)
            xmaxs.append(x1); ymaxs.append(y1); zmaxs.append(z1)
        mn = (min(xmins), min(ymins), min(zmins))
        mx = (max(xmaxs), max(ymaxs), max(zmaxs))
        return mn, mx, len(prims)

    def _overall_geom_aabb_fallback(exe) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
        """
        Fallback to overall geometry AABB if no cuboid prims were found.
        Uses instances if available; otherwise returns bbox-sized degenerate.
        """
        # Try instances on the executor (program_executor.Executor keeps instances dict)
        if hasattr(exe, "instances") and isinstance(exe.instances, dict):
            xmins, ymins, zmins = [], [], []
            xmaxs, ymaxs, zmaxs = [], [], []
            for name, inst in exe.instances.items():
                if name == "bbox":
                    continue
                try:
                    l, w, h = float(inst.spec.l), float(inst.spec.w), float(inst.spec.h)
                    ox, oy, oz = float(inst.T[0,3]), float(inst.T[1,3]), float(inst.T[2,3])
                    x0, y0, z0 = ox, oy, oz
                    x1, y1, z1 = ox + l, oy + w, oz + h
                    xmins.append(x0); ymins.append(y0); zmins.append(z0)
                    xmaxs.append(x1); ymaxs.append(y1); zmaxs.append(z1)
                except Exception:
                    pass
            if xmins:
                return (min(xmins), min(ymins), min(zmins)), (max(xmaxs), max(ymaxs), max(zmaxs))

        # As a last resort, use bbox itself (no placed parts)
        if hasattr(exe, "bbox"):
            # bbox min is in instances["bbox"].T
            try:
                bb_inst = exe.instances["bbox"]
                l, w, h = float(bb_inst.spec.l), float(bb_inst.spec.w), float(bb_inst.spec.h)
                ox, oy, oz = float(bb_inst.T[0,3]), float(bb_inst.T[1,3]), float(bb_inst.T[2,3])
                return (ox, oy, oz), (ox + l, oy + w, oz + h)
            except Exception:
                pass
        # If truly nothing, return a degenerate zero box at origin
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    def _spans_from_aabb(mn, mx) -> Tuple[float,float,float]:
        return (float(mx[0]-mn[0]), float(mx[1]-mn[1]), float(mx[2]-mn[2]))

    def _safe_div(n: float, d: float) -> float:
        # If current span is ~0, scale factor should be 0 when target is 0, else a big stretch.
        if abs(d) < 1e-12:
            return 0.0 if abs(n) < 1e-12 else (n / 1e-12)
        return n / d

    def _scale_all_cuboids_inplace(ir_obj: Dict[str,Any], sx: float, sy: float, sz: float) -> None:
        for c in ir_obj["program"].get("cuboids", []):
            c["l"] = float(c["l"]) * sx
            c["w"] = float(c["w"]) * sy
            c["h"] = float(c["h"]) * sz
        for sub in ir_obj["program"].get("subroutines", []):
            sig = sub.get("sig", {})
            if {"l","w","h"}.issubset(sig):
                sig["l"] = float(sig["l"]) * sx
                sig["w"] = float(sig["w"]) * sy
                sig["h"] = float(sig["h"]) * sz
            for c in sub.get("cuboids", []):
                c["l"] = float(c["l"]) * sx
                c["w"] = float(c["w"]) * sy
                c["h"] = float(c["h"]) * sz

    def _fmt3(v) -> str:
        return f"({float(v[0]):.6f}, {float(v[1]):.6f}, {float(v[2]):.6f})"

    # ------------------------ core logic ------------------------

    # Deep copy IR to avoid mutating caller
    ir = _json.loads(_json.dumps(ir_in))

    # Target spans from strokes
    tgt_min, tgt_max = strokes_aabb(input_dir)
    tgt_L, tgt_W, tgt_H = _spans_from_aabb(tgt_min, tgt_max)

    # Execute *current* IR and measure executed spans from cuboids
    Executor = _load_executor_from_neighbor()
    exe_before = Executor(ir)
    cub_min, cub_max, cub_ct = _executed_cuboids_aabb(exe_before)
    if cub_ct == 0 or _math.isinf(cub_min[0]):
        # fallback to overall geometry if no cuboid prims reported
        cub_min, cub_max = _overall_geom_aabb_fallback(exe_before)
    cur_L, cur_W, cur_H = _spans_from_aabb(cub_min, cub_max)

    # Compute per-axis scale based on *executed* spans (this is the key!)
    sx = _safe_div(tgt_L, cur_L)
    sy = _safe_div(tgt_W, cur_W)
    sz = _safe_div(tgt_H, cur_H)

    # Scale all cuboids
    _scale_all_cuboids_inplace(ir, sx, sy, sz)

    # Declare bbox exactly as strokes AABB
    bb = ir["program"]["bblock"]
    bb["l"], bb["w"], bb["h"] = float(tgt_L), float(tgt_W), float(tgt_H)
    bb["min"] = [float(tgt_min[0]), float(tgt_min[1]), float(tgt_min[2])]
    bb["max"] = [float(tgt_max[0]), float(tgt_max[1]), float(tgt_max[2])]
    if "origin" in bb:
        del bb["origin"]

    # Re-execute after scaling for a final report
    exe_after = Executor(ir)
    aft_min, aft_max, aft_ct = _executed_cuboids_aabb(exe_after)
    if aft_ct == 0 or _math.isinf(aft_min[0]):
        aft_min, aft_max = _overall_geom_aabb_fallback(exe_after)
    aft_L, aft_W, aft_H = _spans_from_aabb(aft_min, aft_max)

    # ------------------------ print report ------------------------
    # print("\n[Deterministic-Scale] Target (strokes) bbox:")
    # print(f"  min  = {_fmt3(tgt_min)}")
    # print(f"  max  = {_fmt3(tgt_max)}")
    # print(f"  size = (L={tgt_L:.6f}, W={tgt_W:.6f}, H={tgt_H:.6f})")

    # print("[Deterministic-Scale] Executed spans BEFORE scaling (cuboids):")
    # print(f"  min  = {_fmt3(cub_min)}")
    # print(f"  max  = {_fmt3(cub_max)}")
    # print(f"  size = (L={cur_L:.6f}, W={cur_W:.6f}, H={cur_H:.6f})")

    # print("[Deterministic-Scale] Scale factors:")
    # print(f"  sx={sx:.6f}, sy={sy:.6f}, sz={sz:.6f}")

    # print("[Deterministic-Scale] Program (declared) bbox AFTER scaling:")
    # print(f"  min  = {_fmt3(bb['min'])}")
    # print(f"  max  = {_fmt3(bb['max'])}")
    # print(f"  size = (L={bb['l']:.6f}, W={bb['w']:.6f}, H={bb['h']:.6f})")

    # print("[Deterministic-Scale] Executed spans AFTER scaling (cuboids):")
    # print(f"  min  = {_fmt3(aft_min)}")
    # print(f"  max  = {_fmt3(aft_max)}")
    # print(f"  size = (L={aft_L:.6f}, W={aft_W:.6f}, H={aft_H:.6f})\n")

    return ir

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY")

    narrative_path = INPUT_DIR / "sketch_narrative.json"
    components_path = INPUT_DIR / "sketch_components.json"
    strokes_path = INPUT_DIR / "stroke_lines.json"

    if not narrative_path.exists() or not components_path.exists():
        raise SystemExit("Missing sketch_narrative.json or sketch_components.json in INPUT_DIR.")
    if not strokes_path.exists():
        raise SystemExit("Missing stroke_lines.json in INPUT_DIR.")

    narrative = json.loads(narrative_path.read_text(encoding="utf-8"))["narrative"]
    components = json.loads(components_path.read_text(encoding="utf-8"))["components"]

    # --- Step 1: initial IR (LLM) ---
    ir1 = call_step1(narrative, components)
    try:
        validate_ir(ir1)
    except Exception as e:
        ir1 = repair_step1(ir1, str(e), narrative, components)
        validate_ir(ir1)

    # Print Step 1 program to screen
    print("\n===== Step 1 ShapeAssembly Program (from narrative+components) =====\n")
    print(emit_shapeassembly(ir1))
    print("\n====================================================================\n")

    # --- Step 3: deterministic translate-only instancing ---
    ir3 = deterministic_step3_translate(ir1)
    validate_ir(ir3)

    # --- Deterministic scaling AFTER Step-3 using strokes AABB ---
    ir_final = deterministic_scale_to_strokes_bbox(ir3, INPUT_DIR)

    # Strict validation against strokes AABB (Z-up)
    try:
        validate_ir(ir_final)
        mn, mx = strokes_aabb(INPUT_DIR)
        size = (mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])
        err_sz = check_bbox_size_matches(ir_final, size)
        err_mm = check_bbox_minmax_matches(ir_final, mn, mx)
        if err_sz or err_mm:
            raise SystemExit(f"BBox mismatch after deterministic scaling: size[{err_sz}] minmax[{err_mm}]")
    except Exception as e2:
        raise SystemExit(f"Deterministic scaling validation error: {e2}")

    # Save FINAL scaled IR (with min/max)
    out_ir = INPUT_DIR / "sketch_program_ir.json"
    out_ir.write_text(json.dumps(ir_final, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_ir}")

if __name__ == "__main__":
    main()
