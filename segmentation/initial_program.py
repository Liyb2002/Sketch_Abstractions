#!/usr/bin/env python3
"""
Stage 2 (ShapeAssembly-faithful, saved next to inputs):
- Reads from INPUT_DIR = Path.cwd().parent / "input"
- Produces:
    INPUT_DIR/sketch_program_ir.json
    INPUT_DIR/sketch_program.py     (default; use --ext to change)
"""

from __future__ import annotations
import os, json, base64, mimetypes, re, argparse
from pathlib import Path
from typing import Any, Dict, List
from openai import OpenAI

# ---------- Config ----------
INPUT_DIR = Path.cwd().parent / "input"
MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

# ---------- I/O helpers ----------
def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{path.read_bytes().decode('latin1')}"  # will fix below

# Use a safe base64 builder to avoid encoding issues
import base64
def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
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

# ---------- IR validation (faithful to ShapeAssembly) ----------
FACES = {"right","left","top","bot","front","back"}
AXES  = {"X","Y","Z"}

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

def emit_shapeassembly(ir: Dict[str, Any]) -> str:
    P = ir["program"]
    name = _ident(P["name"])

    def emit_block(prog: Dict[str, Any], header: str) -> List[str]:
        L: List[str] = []
        L.append(f"def {header}:")
        bb = prog["bblock"]
        L.append(f"bbox = Cuboid({_num(bb['l'])}, {_num(bb['w'])}, {_num(bb['h'])}, {str(bool(bb['aligned']))})")
        for c in prog.get("cuboids", []):
            L.append(f"{_ident(c['var'])} = Cuboid({_num(c['l'])}, {_num(c['w'])}, {_num(c['h'])}, {str(bool(c['aligned']))})")
        for a in prog.get("attach", []):
            L.append(f"attach({_ident(a['a'])}, {_ident(a['b'])}, "
                     f"{_num(a['x1'])}, {_num(a['y1'])}, {_num(a['z1'])}, "
                     f"{_num(a['x2'])}, {_num(a['y2'])}, {_num(a['z2'])})")
        for s in prog.get("squeeze", []):
            L.append(f"squeeze({_ident(s['a'])}, {_ident(s['b'])}, {_ident(s['c'])}, "
                     f"{s['face']}, {_num(s['u'])}, {_num(s['v'])})")
        for r in prog.get("reflect", []):
            L.append(f"reflect({_ident(r['c'])}, {r['axis']})")
        for t in prog.get("translate", []):
            L.append(f"translate({_ident(t['c'])}, {t['axis']}, {int(t['n'])}, {_num(t['d'])})")
        return L

    lines: List[str] = []
    lines.extend(emit_block(P, f"{name}()"))
    lines.append("")
    for sub in P.get("subroutines", []):
        sig = sub["sig"]
        header = f"{_ident(sub['name'])}({ _num(sig['l']) }, { _num(sig['w']) }, { _num(sig['h']) }, { str(bool(sig['aligned'])) })"
        lines.extend(emit_block(sub, header))
        lines.append("")
    return "\n".join(lines)

PROMPT = """
You will write a STRICT JSON IR for a ShapeAssembly program (as in the original paper).
Rules:
- Output ONLY JSON with the exact keys shown.
- Use primitives/ops: bbox + Cuboid(l,w,h,aligned), attach, squeeze, reflect, translate.
- Enforce grounded attachment order (first attach must involve bbox).
- Coordinates u,v,x*,y*,z* ∈ [0,1].
- Prefer symmetry with reflect/translate if suggested by boxes.
- Keep part count faithful to provided few bounding boxes.

Example JSON shape:
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
Return ONLY the JSON object.
""".strip()

def call_model(images: List[str], narrative: str, components: List[str], info: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI()
    content: List[Dict[str, Any]] = [
        {"type":"text","text": PROMPT},
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default=".py", help="output program file extension (e.g., .py, .sa, .txt)")
    args = ap.parse_args()
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY")

    pngs = sorted(INPUT_DIR.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No .png found under {INPUT_DIR}")
    images = [_data_url(p) for p in pngs]

    narrative = json.loads((INPUT_DIR/"sketch_narrative.json").read_text(encoding="utf-8"))["narrative"]
    components = json.loads((INPUT_DIR/"sketch_components.json").read_text(encoding="utf-8"))["components"]
    info = json.loads((INPUT_DIR/"info.json").read_text(encoding="utf-8"))

    ir = call_model(images, narrative, components, info)
    validate_ir(ir)
    program_text = emit_shapeassembly(ir)

    out_ir  = INPUT_DIR / "sketch_program_ir.json"
    out_ir.write_text(json.dumps(ir, indent=2), encoding="utf-8")

    print(f"✅ Wrote {out_ir}")
    print("\n--- Program preview ---\n")
    print(program_text)

if __name__ == "__main__":
    main()
