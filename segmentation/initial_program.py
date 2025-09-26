#!/usr/bin/env python3
"""
One-shot IR builder (Step1 + Step2 combined, prints Step1 program):
- INPUT_DIR = Path.cwd().parent / "input"
- Uses:
    INPUT_DIR/sketch_narrative.json
    INPUT_DIR/sketch_components.json
    INPUT_DIR/info.json
- Produces ONLY:
    INPUT_DIR/sketch_program_ir.json

It:
1) Drafts a minimal ShapeAssembly IR (JSON) from narrative+components.
2) Prints the Step-1 program (DSL text) to screen.
3) Reads REAL bbox from info.json (supports keys: size{x,y,z}, bbox{x_min,...}, meta.bbox_scene, bbox_scene, bblock).
4) Rescales IR to that bbox, validates, repairs once if needed.
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
        for ref in (s["a"], s["b"], s["c"]):
            if ref not in declared: raise ValueError(f"squeeze refers to unknown '{ref}'")
        if not (0.0 <= float(s["u"]) <= 1.0 and 0.0 <= float(s["v"]) <= 1.0):
            raise ValueError("squeeze (u,v) must be in [0,1]")
    for r in P.get("reflect", []):
        for k in ("c","axis"):
            if k not in r: raise ValueError("reflect missing a key")
        if r["c"] not in declared: raise ValueError("reflect unknown cuboid")
        if r["axis"] not in AXES: raise ValueError("reflect axis must be X|Y|Z")
    for t in P.get("translate", []):
        for k in ("c","axis","n","d"):
            if k not in t: raise ValueError("translate missing a key")
        if t["c"] not in declared: raise ValueError("translate unknown cuboid")
        if t["axis"] not in AXES: raise ValueError("translate axis must be X|Y|Z")
        if int(t["n"]) < 1: raise ValueError("translate n must be >=1")
        float(t["d"])

# ---------- BBox helpers (robust to your info.json) ----------
def _maybe(val: Any, *keys: str) -> Optional[Any]:
    cur = val
    for k in keys:
        if not isinstance(cur, dict): return None
        if k not in cur: return None
        cur = cur[k]
    return cur

def bbox_from_info(info: Dict[str, Any]) -> Tuple[float,float,float]:
    """
    Accepts any of:
      - info['size'] = {'x','y','z'}                    -> (l,w,h) = (x,y,z)
      - info['bbox'] = {x_min,x_max,y_min,y_max,...}    -> diffs
      - info.meta.bbox_scene or info.bbox_scene as {l,w,h} or {min,max}
      - info.bblock = {l,w,h}
    """
    # 1) Preferred: explicit size
    s = info.get("size")
    if isinstance(s, dict) and all(k in s for k in ("x","y","z")):
        L, W, H = float(s["x"]), float(s["y"]), float(s["z"])
        return abs(L), abs(W), abs(H)
    # 2) bbox with mins/maxes
    b = info.get("bbox")
    if isinstance(b, dict) and all(k in b for k in ("x_min","x_max","y_min","y_max","z_min","z_max")):
        L = float(b["x_max"] - b["x_min"])
        W = float(b["y_max"] - b["y_min"])
        H = float(b["z_max"] - b["z_min"])
        return abs(L), abs(W), abs(H)
    # 3) meta.bbox_scene
    bs = _maybe(info, "meta", "bbox_scene")
    if isinstance(bs, dict):
        if all(k in bs for k in ("l","w","h")):
            return float(bs["l"]), float(bs["w"]), float(bs["h"])
        if all(k in bs for k in ("min","max")):
            mn, mx = bs["min"], bs["max"]
            if isinstance(mn,(list,tuple)) and isinstance(mx,(list,tuple)) and len(mn)==3 and len(mx)==3:
                return float(mx[0]-mn[0]), float(mx[1]-mn[1]), float(mx[2]-mn[2])
    # 4) bbox_scene
    bs2 = info.get("bbox_scene")
    if isinstance(bs2, dict):
        if all(k in bs2 for k in ("l","w","h")):
            return float(bs2["l"]), float(bs2["w"]), float(bs2["h"])
        if all(k in bs2 for k in ("min","max")):
            mn, mx = bs2["min"], bs2["max"]
            if isinstance(mn,(list,tuple)) and isinstance(mx,(list,tuple)) and len(mn)==3 and len(mx)==3:
                return float(mx[0]-mn[0]), float(mx[1]-mn[1]), float(mx[2]-mn[2])
    # 5) bblock
    bb = info.get("bblock")
    if isinstance(bb, dict) and all(k in bb for k in ("l","w","h")):
        return float(bb["l"]), float(bb["w"]), float(bb["h"])
    raise ValueError("Could not find bbox in info.json (looked for size{x,y,z}, bbox{x_min..}, meta.bbox_scene, bbox_scene, bblock).")

def nearly_equal(a: float, b: float, tol: float=1e-6) -> bool:
    return abs(a-b) <= max(tol, 1e-6*(abs(a)+abs(b)+1.0))

def check_bbox_matches(ir: Dict[str,Any], bbox_lwh: Tuple[float,float,float]) -> Optional[str]:
    L,W,H = bbox_lwh
    try:
        bb = ir["program"]["bblock"]
        mism = []
        if not nearly_equal(float(bb["l"]), L): mism.append(f"bblock.l {bb['l']} != {L}")
        if not nearly_equal(float(bb["w"]), W): mism.append(f"bblock.w {bb['w']} != {W}")
        if not nearly_equal(float(bb["h"]), H): mism.append(f"bblock.h {bb['h']} != {H}")
        return None if not mism else "; ".join(mism)
    except Exception as e:
        return f"IR missing/invalid bblock: {e}"

# ---------- Prompts ----------
PROMPT_STEP1 = """
System instructions (comply strictly):
- You are a ShapeAssembly compiler. Think internally; DO NOT reveal your reasoning.
- Output ONLY a JSON object matching the schema below. No prose, no code blocks.
- Use only: bbox + Cuboid(l,w,h,aligned), attach, squeeze. (Do NOT use reflect or translate in Step-1.)
- Enforce grounded order: the first attach must involve 'bbox'. After an attach, both endpoints are grounded.
- All coordinates must be in [0,1]. Keep the program minimal: as few parts/ops as possible consistent with the components and their counts.
- If info is missing, set bbox to l=w=h=1.0 and use simple centered attaches.

About `components` (input you will receive as JSON):
- It may be one of:
  1) ["seat","backrest","leg","leg","leg","leg"]                       # list of names (duplicates imply count)
  2) {"seat":1, "backrest":1, "leg":4, "arm":2}                        # dict name -> count
  3) [{"name":"seat","count":1},{"name":"leg","count":4}, ...]         # list of objects
- Expand this into **instances**. For any component with count > 1, create distinct instances and unique variable names by appending 1-based indices, e.g., leg1, leg2, leg3, leg4.
- Each component instance MUST have its own cuboid (one cuboid per instance). Do NOT merge instances or use translate arrays here.

Minimal geometry guidance:
- Choose reasonable (l,w,h) per component instance; keep 'aligned' = true unless contradicted.
- Attach every cuboid to 'bbox' using normalized coordinates in [0,1]. Keep placements simple and distinct enough to avoid overlaps when possible.
- Prefer keeping sizes identical for instances of the same component name unless clearly unreasonable.

JSON shape (use exactly these keys; add instances into 'cuboids' and corresponding 'attach'):
{
  "program": {
    "name": "Program1",
    "bblock": { "l": 1.0, "w": 1.0, "h": 1.0, "aligned": true },
    "cuboids": [ { "var":"seat1","l":0.6,"w":0.6,"h":0.1,"aligned": true }, { "var":"leg1","l":0.1,"w":0.1,"h":0.5,"aligned": true }, ... ],
    "attach":  [ { "a":"seat1","b":"bbox","x1":0.5,"y1":0.5,"z1":1.0,"x2":0.5,"y2":0.5,"z2":1.0 }, { "a":"leg1","b":"bbox", ... }, ... ],
    "squeeze": [],
    "reflect": [],
    "translate": [],
    "subroutines": []
  }
}

Return ONLY the JSON object.
""".strip()



PROMPT_STEP2 = """
You are a ShapeAssembly compiler/editor.

TASK:
Given:
  (A) A first-pass ShapeAssembly JSON IR ("step1_IR") with a placeholder bbox (often 1,1,1).
  (B) An info.json containing the REAL scene bbox ("bbox_scene").

Do TWO things and return ONLY a corrected JSON IR:
1) Read the real bbox (l,w,h) from info.json. It may appear as:
     - size = { "x","y","z" }                             (map to l,w,h)
     - bbox = { "x_min","x_max","y_min","y_max","z_min","z_max" } -> diffs
     - meta.bbox_scene or bbox_scene = { "l","w","h" } or { "min":[...], "max":[...] }
     - bblock = { "l","w","h" }
2) Edit the ShapeAssembly IR so that:
   - program.bblock.l,w,h equal the REAL bbox (keep aligned=true unless contradicted).
   - Each cuboid's (l,w,h) is rescaled axiswise by factors (L_new/L_old, W_new/W_old, H_new/H_old)
     computed from the OLD bblock in step1_IR.
   - Do NOT change any 'attach' coordinates (they are already normalized).
   - Do NOT change reflect/translate unless necessary for validity.
   - Preserve names and ordering.

Return ONLY the final JSON IR. No prose, no code fences.
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

PROMPT_REPAIR2 = """
Your updated JSON IR failed validation or bbox check.

Validation errors (if any):
{errors}

BBox mismatch (if any):
{bbox_error}

Please return a corrected JSON IR that:
- Sets program.bblock.l,w,h EXACTLY to the real bbox: ({L},{W},{H})
- Rescales each cuboid's (l,w,h) by axiswise factors (L/L0, W/W0, H/H0) from the OLD bblock you used.
- Keeps attach coordinates unchanged in [0,1]; avoid adding/removing parts unless essential.
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

def call_step2(step1_ir: Dict[str,Any], info: Dict[str,Any]) -> Dict[str,Any]:
    client = OpenAI()
    content = [
        {"type": "text", "text": PROMPT_STEP2},
        {"type": "text", "text": "step1_IR:"},
        {"type": "text", "text": json.dumps(step1_ir, separators=(',',':'))[:24000]},
        {"type": "text", "text": "info.json:"},
        {"type": "text", "text": json.dumps(info, separators=(',',':'))[:24000]},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content": content}],
        temperature=0.1,
        max_tokens=3000,
    )
    return _extract_json(resp.choices[0].message.content or "")

def repair_step2(bad_ir: Dict[str,Any], info: Dict[str,Any],
                 errors: str, bbox_err: str,
                 L: float, W: float, H: float) -> Dict[str,Any]:
    client = OpenAI()
    prompt = (PROMPT_REPAIR2
              .replace("{errors}", errors or "None")
              .replace("{bbox_error}", bbox_err or "None")
              .replace("{L}", str(L)).replace("{W}", str(W)).replace("{H}", str(H)))
    content = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": "Previous (invalid) JSON IR:"},
        {"type": "text", "text": json.dumps(bad_ir, separators=(',',':'))[:24000]},
        {"type": "text", "text": "info.json:"},
        {"type": "text", "text": json.dumps(info, separators=(',',':'))[:24000]},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content": content}],
        temperature=0.05,
        max_tokens=3000,
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

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY")

    narrative_path = INPUT_DIR / "sketch_narrative.json"
    components_path = INPUT_DIR / "sketch_components.json"
    info_path = INPUT_DIR / "info.json"

    if not narrative_path.exists() or not components_path.exists():
        raise SystemExit("Missing sketch_narrative.json or sketch_components.json in INPUT_DIR.")
    if not info_path.exists():
        raise SystemExit("Missing info.json in INPUT_DIR.")

    narrative = json.loads(narrative_path.read_text(encoding="utf-8"))["narrative"]
    components = json.loads(components_path.read_text(encoding="utf-8"))["components"]
    info = json.loads(info_path.read_text(encoding="utf-8"))

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

    # --- Step 2: refine using real bbox from info.json ---
    L, W, H = bbox_from_info(info)  # robust to your info.json structure
    ir2 = call_step2(ir1, info)
    bbox_err = check_bbox_matches(ir2, (L,W,H))
    try:
        validate_ir(ir2)
        if bbox_err:
            raise ValueError(bbox_err)
    except Exception as e2:
        ir2 = repair_step2(ir2, info, str(e2), bbox_err or "", L, W, H)
        validate_ir(ir2)
        bbox_err2 = check_bbox_matches(ir2, (L,W,H))
        if bbox_err2:
            raise SystemExit(f"BBox mismatch after repair: {bbox_err2}")

    # --- Step 3: translate-only instancing via API ---
    ir3 = call_step3_translate(ir2)
    try:
        validate_ir(ir3)
    except Exception as e3:
        ir3 = repair_step3_translate(ir3, str(e3))
        validate_ir(ir3)

    # Optional: print the Step-3 program
    print("\n===== Step 3 ShapeAssembly Program (translate-only instancing) =====\n")
    print(emit_shapeassembly(ir3))
    print("\n=====================================================================\n")

    # --- Save BOTH (step2 and instanced) ---
    out_ir = INPUT_DIR / "sketch_program_ir.json"
    out_ir.write_text(json.dumps(ir2, indent=2), encoding="utf-8")
    out_ir3 = INPUT_DIR / "sketch_program_ir_instanced.json"
    out_ir3.write_text(json.dumps(ir3, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_ir}")
    print(f"✅ Wrote {out_ir3}")



if __name__ == "__main__":
    main()
