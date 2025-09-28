#!/usr/bin/env python3
"""
initial_program.py  (deterministic Step-3, strict Z-up with NO z-anchor rewrites)

One-shot IR builder (Step1 + Step3 + Deterministic scaling using strokes AABB):
- INPUT_DIR = Path.cwd().parent / "input"
- Uses:
    INPUT_DIR/sketch_narrative.json
    INPUT_DIR/sketch_components.json
    INPUT_DIR/perturbed_feature_lines.json   <-- used to set bblock.min/max after Step-3
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
def strokes_aabb(input_dir: Path) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """
    Reads perturbed_feature_lines.json and returns (min, max) over all sampled stroke points.
    Expected structure: List[ List[[x,y,z], ...] ] (one list per stroke).
    """
    sp_path = input_dir / "perturbed_feature_lines.json"
    if not sp_path.exists():
        raise SystemExit(f"Missing {sp_path} — required for deterministic scaling.")
    data = json.loads(sp_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("perturbed_feature_lines.json must be a list of strokes")
    mins = [float("inf")]*3; maxs = [float("-inf")]*3
    valid_pts = 0
    for s in data:
        if not isinstance(s, list): continue
        for p in s:
            if not (isinstance(p, (list,tuple)) and len(p)==3): continue
            valid_pts += 1
            for i in range(3):
                v = float(p[i])
                if v < mins[i]: mins[i]=v
                if v > maxs[i]: maxs[i]=v
    if valid_pts == 0:
        raise SystemExit("No valid [x,y,z] points found in perturbed_feature_lines.json")
    return (mins[0],mins[1],mins[2]), (maxs[0],maxs[1],maxs[2])

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

def _anchor_to_bbox(attach_list: List[Dict[str,Any]], var: str) -> Optional[Tuple[float,float,float]]:
    """
    Return the normalized (x,y,z) anchor of `var` relative to bbox from an attach involving bbox.
    If multiple attaches to bbox exist, prefer the first. If none, return None.
    """
    for a in attach_list:
        a_name, b_name = str(a["a"]), str(a["b"])
        if a_name == var and b_name == "bbox":
            return (float(a["x1"]), float(a["y1"]), float(a["z1"]))
        if a_name == "bbox" and b_name == var:
            return (float(a["x2"]), float(a["y2"]), float(a["z2"]))
    return None

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

def deterministic_step3_translate(ir_in: Dict[str, Any], tol: float=1e-6) -> Dict[str, Any]:
    """
    Deterministic translate-only instancing:
      - Find groups like baseName + digits (leg1..leg4) with identical (l,w,h,aligned).
      - Require each member to have an anchor attach to bbox.
      - If members fall on a uniform grid along X/Y/Z:
          * Keep one prototype at the minimal (x,y,z)
          * Remove others and their attaches
          * Emit translate ops on axes with n>1 and d>0
      - If any condition fails, leave the group as-is.
    """
    ir = json.loads(json.dumps(ir_in))  # deep copy
    P = ir["program"]
    cuboids: List[Dict[str,Any]] = P.get("cuboids", [])
    attaches: List[Dict[str,Any]] = P.get("attach", [])
    translates: List[Dict[str,Any]] = P.setdefault("translate", [])

    # index sizes
    size_by_var = {
        c["var"]: (float(c["l"]), float(c["w"]), float(c["h"]), bool(c.get("aligned", True)))
        for c in cuboids
    }

    # group candidates: basename + identical size + has trailing digits
    groups: Dict[Tuple[str,Tuple[float,float,float,bool]], List[str]] = {}
    for c in cuboids:
        var = c["var"]
        if not _has_trailing_digits(var):
            continue
        key = (_basename(var), size_by_var[var])
        groups.setdefault(key, []).append(var)

    if not groups:
        return ir  # nothing to do

    cuboid_by_var = {c["var"]: c for c in cuboids}

    removed_vars_total: List[str] = []
    new_translates: List[Dict[str,Any]] = []

    for (base, size), members in groups.items():
        if len(members) < 2:
            continue

        # collect anchors
        anchors: Dict[str, Tuple[float,float,float]] = {}
        ok_group = True
        for v in members:
            anch = _anchor_to_bbox(attaches, v)
            if anch is None:
                ok_group = False
                break
            anchors[v] = anch
        if not ok_group:
            continue

        # choose prototype at minimal (x,y,z) lexicographically
        proto = min(members, key=lambda v: anchors[v])
        axp, ayp, azp = anchors[proto]

        # compute deltas from proto
        deltas = {v: (anchors[v][0]-axp, anchors[v][1]-ayp, anchors[v][2]-azp) for v in members}

        # unique values along each axis (should include 0)
        xs = sorted(set(round(d[0], 10) for d in deltas.values()))
        ys = sorted(set(round(d[1], 10) for d in deltas.values()))
        zs = sorted(set(round(d[2], 10) for d in deltas.values()))

        # verify uniform progressions
        okx, dx = _is_uniform_progression(xs, tol)
        oky, dy = _is_uniform_progression(ys, tol)
        okz, dz = _is_uniform_progression(zs, tol)
        if not (okx and oky and okz):
            continue

        # verify full cartesian grid coverage
        expected_count = len(xs) * len(ys) * len(zs)
        if expected_count != len(members):
            continue

        grid = {(x, y, z) for x in xs for y in ys for z in zs}
        if any((round(dxv,10), round(dyv,10), round(dzv,10)) not in grid for (dxv,dyv,dzv) in deltas.values()):
            continue

        # build translate ops (n includes the prototype)
        if len(xs) > 1 and dx > tol:
            new_translates.append({"c": proto, "axis": "X", "n": len(xs), "d": float(dx)})
        if len(ys) > 1 and dy > tol:
            new_translates.append({"c": proto, "axis": "Y", "n": len(ys), "d": float(dy)})
        if len(zs) > 1 and dz > tol:
            new_translates.append({"c": proto, "axis": "Z", "n": len(zs), "d": float(dz)})

        # mark all but the prototype for removal
        to_remove = [v for v in members if v != proto]
        removed_vars_total.extend(to_remove)

    if not removed_vars_total and not new_translates:
        return ir

    # 1) remove cuboids for removed vars
    P["cuboids"] = [c for c in cuboids if c["var"] not in set(removed_vars_total)]

    # 2) remove attaches that reference removed vars
    P["attach"] = [a for a in attaches if a["a"] not in removed_vars_total and a["b"] not in removed_vars_total]

    # 3) keep unrelated squeeze/reflect; drop ones that reference removed vars
    if "squeeze" in P:
        P["squeeze"] = [s for s in P["squeeze"] if s["a"] not in removed_vars_total
                        and s["b"] not in removed_vars_total and s["c"] not in removed_vars_total]
    if "reflect" in P:
        P["reflect"] = [r for r in P["reflect"] if r["c"] not in removed_vars_total]
    if "translate" in P:
        P["translate"] = [t for t in P["translate"] if t["c"] not in removed_vars_total]
    else:
        P["translate"] = []

    # 4) append new translate arrays
    P["translate"].extend(new_translates)

    return ir

# ---------- Deterministic scaling (after Step-3) ----------
def deterministic_scale_to_strokes_bbox(ir_in: Dict[str, Any], input_dir: Path) -> Dict[str, Any]:
    """
    Deterministically edit program.bblock to match the strokes' AABB, and rescale all cuboids:
      - Compute (min,max) from perturbed_feature_lines.json.
      - Set bblock.min = min, bblock.max = max, and bblock.l/w/h = max - min.
      - Rescale each cuboid's (l,w,h) by per-axis ratios new_size / old_size.
      - Leave attach/squeeze/reflect/translate untouched (they are normalized).
    """
    ir = json.loads(json.dumps(ir_in))  # deep copy
    mn, mx = strokes_aabb(input_dir)
    L, W, H = (mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])

    bb = ir["program"]["bblock"]
    L0, W0, H0 = float(bb["l"]), float(bb["w"]), float(bb["h"])

    def sdiv(n, d): return (n/d) if abs(d) > 1e-12 else 1.0
    sx, sy, sz = sdiv(L, L0), sdiv(W, W0), sdiv(H, H0)

    # Write new bbox grammar: size + min/max
    bb["l"], bb["w"], bb["h"] = float(L), float(W), float(H)
    bb["min"] = [float(mn[0]), float(mn[1]), float(mn[2])]
    bb["max"] = [float(mx[0]), float(mx[1]), float(mx[2])]
    if "origin" in bb:
        del bb["origin"]

    # Rescale all cuboids
    for c in ir["program"].get("cuboids", []):
        c["l"] = float(c["l"]) * sx
        c["w"] = float(c["w"]) * sy
        c["h"] = float(c["h"]) * sz

    return ir

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY")

    narrative_path = INPUT_DIR / "sketch_narrative.json"
    components_path = INPUT_DIR / "sketch_components.json"
    strokes_path = INPUT_DIR / "perturbed_feature_lines.json"

    if not narrative_path.exists() or not components_path.exists():
        raise SystemExit("Missing sketch_narrative.json or sketch_components.json in INPUT_DIR.")
    if not strokes_path.exists():
        raise SystemExit("Missing perturbed_feature_lines.json in INPUT_DIR.")

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

    # Save raw Step-3 output (before deterministic scaling)
    out_ir3 = INPUT_DIR / "sketch_program_ir_instanced.json"
    out_ir3.write_text(json.dumps(ir3, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_ir3}")

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
