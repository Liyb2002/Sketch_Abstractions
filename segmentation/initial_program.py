#!/usr/bin/env python3
"""
Stage 2 (ShapeAssembly-faithful):
- Inputs (from INPUT_DIR):
  * one or more .png images (any perspectives)
  * sketch_narrative.json
  * sketch_components.json
  * info.json  (compact 3D info: bbox + a few bounding boxes, optional symmetry hints)
- Output:
  * sketch_program_ir.json  (ShapeAssembly IR)
  * sketch_program.sa       (valid ShapeAssembly program text, faithful to the paper)

Requirements:
  pip install openai>=1.40
  export OPENAI_API_KEY=...

Notes:
  * We use chat.completions with image_url (base64 data URL) for vision.
  * The IR and emitter stick to the paper primitives:
      - bbox + Cuboid(l, w, h, aligned)
      - attach(c1, c2, x1,y1,z1, x2,y2,z2)
      - squeeze(c1, c2, c3, face, u, v)    where face∈{right,left,top,bot,front,back}
      - reflect(c, axis)                    axis∈{X,Y,Z} (of the bbox)
      - translate(c, axis, n, d)            n additional members along axis, ending distance d
    Execution is **imperative** and we enforce **grounded order** for attaches, as in the paper.
"""

from __future__ import annotations
import os, json, base64, mimetypes, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI

# ----------------- Configuration -----------------
INPUT_DIR = Path.cwd().parent / "input"
OUT_IR   = Path("sketch_program_ir.json")
OUT_SA   = Path("sketch_program.sa")
MODEL    = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

# ----------------- I/O helpers -----------------
def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64  = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

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
    raise RuntimeError(f"Could not parse JSON from model output:\n{text}")

def _ident(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s.strip())
    if not s: s = "x"
    if s[0].isdigit(): s = "x_" + s
    return s[:48]

def _num(x: Any) -> str:
    if isinstance(x, float):
        s = f"{x:.3f}"
        s = s.rstrip("0").rstrip(".")
        return s if s else "0"
    return str(x)

# ----------------- IR validation (faithful to paper) -----------------
FACES = {"right","left","top","bot","front","back"}
AXES  = {"X","Y","Z"}

def validate_ir(ir: Dict[str, Any]) -> None:
    """
    Expected IR (single-root program; optional subroutines allowed but not required):

    {
      "program": {
        "name": "Program1",
        "bblock": { "l":..., "w":..., "h":..., "aligned": true },
        "cuboids": [
          {"var":"cube0","l":...,"w":...,"h":...,"aligned": true}
        ],
        "attach": [
          {"a":"cube0","b":"bbox",
           "x1":0.5,"y1":0.0,"z1":0.5,"x2":0.5,"y2":0.0,"z2":0.5}
        ],
        "squeeze": [
          {"a":"cubeA","b":"bbox","c":"cubeB","face":"top","u":0.5,"v":0.05}
        ],
        "reflect": [
          {"c":"cubeX","axis":"X"}
        ],
        "translate": [
          {"c":"cubeY","axis":"Y","n":2,"d":0.5}
        ],
        "subroutines": [
          { "name":"Sub", "sig":{"l":0.8,"w":0.5,"h":0.2,"aligned": True},
            "cuboids":[...], "attach":[...], "squeeze":[...], "reflect":[...], "translate":[...] }
        ]
      }
    }
    """
    if "program" not in ir: raise ValueError("IR missing 'program'")
    P = ir["program"]
    for key in ("name","bblock"):
        if key not in P: raise ValueError(f"program missing '{key}'")

    # bbox
    bb = P["bblock"]
    for k in ("l","w","h","aligned"):
        if k not in bb: raise ValueError(f"bblock missing '{k}'")

    # collect declared symbols
    declared = {"bbox"}
    cuboids = P.get("cuboids", [])
    if not isinstance(cuboids, list): raise ValueError("'cuboids' must be a list")
    for c in cuboids:
        for k in ("var","l","w","h","aligned"):
            if k not in c: raise ValueError("each cuboid needs var,l,w,h,aligned")
        v = c["var"]
        if v in declared: raise ValueError(f"duplicate name '{v}'")
        declared.add(v)

    # grounded-order attach
    attaches = P.get("attach", [])
    grounded = {"bbox"}  # bbox is grounded initially (paper)
    for a in attaches:
        for k in ("a","b","x1","y1","z1","x2","y2","z2"):
            if k not in a: raise ValueError("attach missing a key")
        if a["a"] not in declared or a["b"] not in declared:
            raise ValueError("attach refers to unknown cuboid")
        # enforce grounded order: at least one endpoint must already be grounded
        if not (a["a"] in grounded or a["b"] in grounded):
            raise ValueError(f"attach not grounded: {a}")
        # after attach, both become grounded
        grounded.add(a["a"]); grounded.add(a["b"])
        # coords should be in [0,1]
        for k in ("x1","y1","z1","x2","y2","z2"):
            v = a[k]
            if not (isinstance(v,(int,float)) and 0.0-1e-6 <= v <= 1.0+1e-6):
                raise ValueError(f"attach coord '{k}' out of [0,1]: {v}")

    # squeeze
    for s in P.get("squeeze", []):
        for k in ("a","b","c","face","u","v"):
            if k not in s: raise ValueError("squeeze missing a key")
        if s["face"] not in FACES: raise ValueError(f"bad face '{s['face']}'")
        for ref in (s["a"], s["b"], s["c"]):
            if ref not in declared:
                raise ValueError(f"squeeze refers to unknown '{ref}'")
        if not (0.0 <= float(s["u"]) <= 1.0 and 0.0 <= float(s["v"]) <= 1.0):
            raise ValueError("squeeze (u,v) must be in [0,1]")

    # reflect
    for r in P.get("reflect", []):
        for k in ("c","axis"):
            if k not in r: raise ValueError("reflect missing a key")
        if r["c"] not in declared: raise ValueError("reflect unknown cuboid")
        if r["axis"] not in AXES: raise ValueError("reflect axis must be X|Y|Z")

    # translate
    for t in P.get("translate", []):
        for k in ("c","axis","n","d"):
            if k not in t: raise ValueError("translate missing a key")
        if t["c"] not in declared: raise ValueError("translate unknown cuboid")
        if t["axis"] not in AXES: raise ValueError("translate axis must be X|Y|Z")
        if int(t["n"]) < 1: raise ValueError("translate n must be >=1")
        float(t["d"])  # just check castable

    # optional subroutines (hierarchy)
    for sub in P.get("subroutines", []):
        if "name" not in sub or "sig" not in sub:
            raise ValueError("subroutine needs name and sig")
        for k in ("l","w","h","aligned"):
            if k not in sub["sig"]:
                raise ValueError("subroutine sig missing a key")

# ----------------- Emitter: IR -> ShapeAssembly text -----------------
def emit_shapeassembly(ir: Dict[str, Any]) -> str:
    P = ir["program"]
    name = _ident(P["name"])

    def emit_block(prog: Dict[str, Any], header: str) -> List[str]:
        L: List[str] = []
        L.append(f"def {header}:")
        bb = prog["bblock"]
        L.append(f"bbox = Cuboid({_num(bb['l'])}, {_num(bb['w'])}, {_num(bb['h'])}, {str(bool(bb['aligned']))})")
        # Cuboids
        for c in prog.get("cuboids", []):
            L.append(f"{_ident(c['var'])} = Cuboid({_num(c['l'])}, {_num(c['w'])}, {_num(c['h'])}, {str(bool(c['aligned']))})")
        # Attaches
        for a in prog.get("attach", []):
            L.append(f"attach({_ident(a['a'])}, {_ident(a['b'])}, "
                     f"{_num(a['x1'])}, {_num(a['y1'])}, {_num(a['z1'])}, "
                     f"{_num(a['x2'])}, {_num(a['y2'])}, {_num(a['z2'])})")
        # Squeezes
        for s in prog.get("squeeze", []):
            L.append(f"squeeze({_ident(s['a'])}, {_ident(s['b'])}, {_ident(s['c'])}, "
                     f"{s['face']}, {_num(s['u'])}, {_num(s['v'])})")
        # Reflects
        for r in prog.get("reflect", []):
            L.append(f"reflect({_ident(r['c'])}, {r['axis']})")
        # Translates
        for t in prog.get("translate", []):
            L.append(f"translate({_ident(t['c'])}, {t['axis']}, {int(t['n'])}, {_num(t['d'])})")
        return L

    lines: List[str] = []
    # Root program
    lines.extend(emit_block(P, f"{name}()"))
    lines.append("")  # blank

    # Optional subroutines
    for sub in P.get("subroutines", []):
        sig = sub["sig"]
        header = f"{_ident(sub['name'])}({ _num(sig['l']) }, { _num(sig['w']) }, { _num(sig['h']) }, { str(bool(sig['aligned'])) })"
        lines.extend(emit_block(sub, header))
        lines.append("")

    return "\n".join(lines)

# ----------------- Prompt (asks ONLY for IR) -----------------
PROMPT = """
You will write a STRICT JSON IR for a ShapeAssembly program (as defined in the original paper).
Rules you MUST follow (no exceptions):
- Output ONLY JSON (no code). Use the exact keys shown below.
- Match ShapeAssembly primitives and semantics:
  * bbox + Cuboid(l, w, h, aligned:bool)
  * attach(c1, c2, x1,y1,z1, x2,y2,z2)  with all coords in [0,1]
  * squeeze(c1, c2, c3, face, u, v)     face ∈ {right,left,top,bot,front,back}; u,v∈[0,1]
  * reflect(c, axis)                    axis ∈ {X,Y,Z} (w.r.t. bbox)
  * translate(c, axis, n, d)            n≥1 additional members along axis, ends distance d away
- Enforce grounded attachment order: the first attachment must connect a cuboid to bbox; after an attachment, the newly attached cuboid is grounded.
- Use neutral, category-agnostic names (cube0, cube1, ...), unless info.json provides names.
- Prefer symmetry with reflect/translate when boxes indicate pairs or regular spacing.
- Keep counts small and faithful to the provided few bounding boxes (do NOT invent hundreds of parts).

JSON shape (example structure—use same keys):
{
  "program": {
    "name": "Program1",
    "bblock": { "l": 1.0, "w": 1.0, "h": 1.0, "aligned": true },
    "cuboids": [ { "var":"cube0","l":0.8,"w":0.5,"h":0.2,"aligned": true } ],
    "attach":  [ { "a":"cube0","b":"bbox","x1":0.5,"y1":0.0,"z1":0.5,"x2":0.5,"y2":0.0,"z2":0.5 } ],
    "squeeze": [ { "a":"cubeA","b":"bbox","c":"cubeB","face":"top","u":0.5,"v":0.05 } ],
    "reflect": [ { "c":"cubeX","axis":"X" } ],
    "translate":[ { "c":"cubeY","axis":"Y","n":2,"d":0.5 } ],
    "subroutines": []
  }
}
Return ONLY that JSON object. If unsure about a macro, omit it (do not guess).
"""

# ----------------- Call the model -----------------
def call_model(images: List[str], narrative: str, components: List[str], info: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI()
    content: List[Dict[str, Any]] = [
        {"type":"text","text": PROMPT.strip()},
        {"type":"text","text": "Components: " + json.dumps(components, ensure_ascii=False)},
        {"type":"text","text": "Narrative: " + narrative},
        {"type":"text","text": "info.json: " + json.dumps(info, separators=(',',':'))[:12000]}
    ]
    for url in images[:6]:
        content.append({"type":"image_url","image_url":{"url":url}})
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content": content}],
        temperature=0.2,
        max_tokens=3000
    )
    return _extract_json(resp.choices[0].message.content or "")

# ----------------- Orchestration -----------------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY")

    # Collect inputs
    pngs = sorted(INPUT_DIR.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No .png found under {INPUT_DIR}")
    images = [_data_url(p) for p in pngs]

    narrative = json.loads((INPUT_DIR/"sketch_narrative.json").read_text(encoding="utf-8"))["narrative"]
    components = json.loads((INPUT_DIR/"sketch_components.json").read_text(encoding="utf-8"))["components"]
    info = json.loads((INPUT_DIR/"info.json").read_text(encoding="utf-8"))

    # Model -> IR
    ir = call_model(images, narrative, components, info)

    # Validate and emit
    validate_ir(ir)
    sa = emit_shapeassembly(ir)

    # Save
    OUT_IR.write_text(json.dumps(ir, indent=2), encoding="utf-8")
    OUT_SA.write_text(sa, encoding="utf-8")
    print(f"✅ Wrote {OUT_IR.resolve()}")
    print(f"✅ Wrote {OUT_SA.resolve()}")
    print("\n--- ShapeAssembly program ---\n")
    print(sa)

if __name__ == "__main__":
    main()
