#!/usr/bin/env python3
"""
initial_program.py

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
3) Step-3 (LLM): Translate-only instancing (deduplicate repeated parts with translate arrays).
4) Deterministic scaling (code): Set program.bblock.min/max to the strokes' AABB; rescale all cuboids to the new bbox size.
"""

from __future__ import annotations
import os, json, re, argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI

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

# ---------- Prompts ----------
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
- Strict non-overlap requirement: The axis-aligned bounding boxes of all cuboids must be pairwise disjoint within the bbox’s normalized space. No touching or intersection is allowed.
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

PROMPT_STEP3_TRANSLATE = """
You are a ShapeAssembly compiler/editor.

INPUT:
- step2_IR: a VALID ShapeAssembly JSON IR (keys: program -> name, bblock, cuboids, attach, squeeze, reflect, translate, subroutines).
- The program may contain repeated parts like leg1, leg2, leg3, leg4 that are identical except location.

TASK (Translate-Only Instancing):
1) Detect groups of cuboids that share a base name with trailing digits (e.g., leg1..leg4, arm1..arm2) AND have identical (l,w,h,aligned).
2) For each group:
   a) Choose ONE member as the prototype (the one placed at the "first" corner/position is fine). Keep its 'attach' as-is.
   b) REMOVE the other group members from 'cuboids' AND remove their 'attach' entries.
   c) Recreate the removed members **only** using 'translate' operations applied to the prototype.
      - Use ONLY keys: {"c","axis","n","d"} with axis in {"X","Y","Z"}, n >= 2 for arrays, and d in [0,1].
      - Compute 'd' from normalized coordinate deltas between member placements and the prototype.
      - If the group forms a 1x2 or 2x2 (or NxM) grid, emit minimal arrays:
        * Example: two columns -> {"c":"leg","axis":"X","n":2,"d":dx}
        * Then two rows -> {"c":"leg","axis":"Y","n":2,"d":dy}
      - DO NOT use 'reflect' for this task.
3) Do NOT modify program.bblock nor any existing normalized attach coordinates (besides deleting the duplicates).
4) Preserve all unrelated parts and ops. Keep ordering minimal and stable.
5) Return ONLY a single JSON object with the same top-level schema. No prose.

Hard constraints:
- Allowed keys under program: name, bblock, cuboids, attach, squeeze, reflect, translate, subroutines. (You may leave reflect/squeeze/subroutines empty.)
- Every number must be valid JSON; coordinates must remain normalized in [0,1].
- Use translate-only for instancing.

Return ONLY the final JSON IR (no code fences).
""".strip()

PROMPT_STEP3_REPAIR = """
Your Step-3 translate-only JSON IR failed validation:

{errors}

Please return a corrected JSON IR that:
- Keeps only translate (no reflect) for instancing.
- Preserves the existing bblock and normalized coordinates.
- Uses the minimal number of translate arrays to recreate the removed instances.
Return ONLY the JSON object (no prose).
""".strip()

# ---------- Model calls ----------
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

def call_step3_translate(step2_ir: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI()
    content = [
        {"type": "text", "text": PROMPT_STEP3_TRANSLATE},
        {"type": "text", "text": "step2_IR:"},
        {"type": "text", "text": json.dumps(step2_ir, separators=(',', ':'))[:24000]},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=3000,
    )
    return _extract_json(resp.choices[0].message.content or "")

def repair_step3_translate(bad_ir: Dict[str, Any], errors: str) -> Dict[str, Any]:
    client = OpenAI()
    content = [
        {"type": "text", "text": PROMPT_STEP3_REPAIR.replace("{errors}", errors)},
        {"type": "text", "text": "Original (invalid) JSON:"},
        {"type": "text", "text": json.dumps(bad_ir, separators=(',', ':'))[:24000]},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.05,
        max_tokens=3000,
    )
    return _extract_json(resp.choices[0].message.content or "")

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
    # (Optional) remove legacy 'origin' if present
    if "origin" in bb:
        del bb["origin"]

    # Rescale all cuboids
    for c in ir["program"].get("cuboids", []):
        c["l"] = float(c["l"]) * sx
        c["w"] = float(c["w"]) * sy
        c["h"] = float(c["h"]) * sz

    return ir


def rewrite_z_attachments_to_contain(ir_in: Dict[str, Any], eps: float = 1e-6) -> Dict[str, Any]:
    """
    Ensure parts attached to the bbox are contained in Z by flipping local anchor Z
    when the attach maps the part's bottom to the bbox top (and vice versa).
    Rules (only for attaches involving 'bbox'):
      - If a->part, b->bbox and (z1≈0, z2≈1): set z1 := 1 (top of part to bbox top).
      - If a->part, b->bbox and (z1≈1, z2≈0): set z1 := 0 (bottom to bbox bottom).
      - If a->bbox, b->part and (z1≈1, z2≈0): set z2 := 1 (part’s top to bbox top).
      - If a->bbox, b->part and (z1≈0, z2≈1): set z2 := 0 (part’s bottom to bbox bottom).
    Leaves X/Y as-is; non-bbox attaches untouched.
    """
    ir = json.loads(json.dumps(ir_in))  # deep copy
    n_fix = 0
    for a in ir["program"].get("attach", []):
        a_name, b_name = str(a["a"]), str(a["b"])
        z1 = float(a["z1"]); z2 = float(a["z2"])

        # part -> bbox
        if b_name == "bbox" and a_name != "bbox":
            if z2 > 1.0 - eps and z1 < eps:   # bottom->top (pushes part above bbox)
                a["z1"] = 1.0; n_fix += 1      # top->top (contained)
            elif z2 < eps and z1 > 1.0 - eps:  # top->bottom (pushes part below bbox)
                a["z1"] = 0.0; n_fix += 1      # bottom->bottom (contained)

        # bbox -> part (less common, but handle symmetrically)
        elif a_name == "bbox" and b_name != "bbox":
            if z1 > 1.0 - eps and z2 < eps:    # bbox top to part bottom
                a["z2"] = 1.0; n_fix += 1      # bbox top to part top
            elif z1 < eps and z2 > 1.0 - eps:  # bbox bottom to part top
                a["z2"] = 0.0; n_fix += 1      # bbox bottom to part bottom

        # (optional) clamp safety
        a["z1"] = max(0.0, min(1.0, float(a["z1"])))
        a["z2"] = max(0.0, min(1.0, float(a["z2"])))

    print(f"🔧 Z-containment rewrites applied: {n_fix}")
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

    # --- Step 1: initial IR ---
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

    # --- Step 3: translate-only instancing via API (on Step-1 IR) ---
    ir3 = call_step3_translate(ir1)
    try:
        validate_ir(ir3)
    except Exception as e3:
        ir3 = repair_step3_translate(ir3, str(e3))
        validate_ir(ir3)

    # Save raw Step-3 output (before deterministic scaling)
    out_ir3 = INPUT_DIR / "sketch_program_ir_instanced.json"
    out_ir3.write_text(json.dumps(ir3, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_ir3}")

    # --- Deterministic scaling AFTER Step-3 using strokes AABB ---
    ir_final = deterministic_scale_to_strokes_bbox(ir3, INPUT_DIR)

    # --- Automatic Z containment rewrite (no human factor) ---
    ir_final = rewrite_z_attachments_to_contain(ir_final)

    # Validate and strict checks against strokes AABB
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
