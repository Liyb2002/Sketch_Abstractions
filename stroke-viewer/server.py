# stroke-viewer/server.py
from pathlib import Path
from typing import Any, Dict, List, Optional
import json, os
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- paths ---
ROOT = Path(__file__).parent.resolve()         # .../stroke-viewer
UI_DIST = ROOT / "dist"
INPUT_DIR = ROOT.parent / "input"
DEFAULT_IR = INPUT_DIR / "sketch_program_ir_editted.json"

# --- import your executor ---
from backend.load_program_main import rescale_and_execute  # adjust if path differs

# ============= CREATE APP FIRST =============
app = FastAPI(title="Stroke Viewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
# ============================================

# --- models ---
class Polyline(BaseModel):
    points: List[List[float]]

class StrokePayload(BaseModel):
    perturbed_feature_lines: List[Polyline]
    perturbed_construction_lines: List[Polyline]
    feature_lines: List[Polyline]

class Cuboid(BaseModel):
    id: str
    name: Optional[str] = None
    center: List[float]
    size: List[float]
    rotationEuler: Optional[List[float]] = None

class ExecuteResponse(BaseModel):
    cuboids: List[Cuboid]

# --- strokes endpoint (unchanged example) ---
@app.get("/api/strokes", response_model=StrokePayload)
def get_strokes():
    path = INPUT_DIR / "stroke_lines.json"
    data = json.loads(path.read_text())
    def norm(bundle):
        def is_pt(p): return isinstance(p, (list,tuple)) and len(p)==3
        def flat(raw):
            if not isinstance(raw, list): return []
            if all(is_pt(p) for p in raw): return [raw] if len(raw)>=2 else []
            out=[]; 
            for it in raw:
                if isinstance(it, list):
                    if all(is_pt(p) for p in it): 
                        if len(it)>=2: out.append(it)
                    else:
                        out.extend(flat(it))
            return out
        return [Polyline(points=[[float(a),float(b),float(c)] for a,b,c in pl]) for pl in flat(bundle)]
    return {
        "perturbed_feature_lines": norm(data.get("perturbed_feature_lines", [])),
        "perturbed_construction_lines": norm(data.get("perturbed_construction_lines", [])),
        "feature_lines": norm(data.get("feature_lines", [])),
    }

# --- helpers to mirror plot_program_only semantics ---
def _load_offsets_scales(use_offsets: bool, use_scales: bool):
    offsets, scales = {}, {}
    if use_offsets:
        tf = INPUT_DIR / "fit_translations.json"
        if tf.exists():
            try: offsets = json.loads(tf.read_text()).get("offsets_xyz", {}) or {}
            except Exception as e: print(f"[warn] read {tf} failed: {e}")
    if use_scales:
        sf = INPUT_DIR / "fit_scales.json"
        if sf.exists():
            try: scales = json.loads(sf.read_text()).get("scales_lwh", {}) or {}
            except Exception as e: print(f"[warn] read {sf} failed: {e}")
    return offsets, scales

def _executor_to_cuboids(executor: Any, *, use_offsets=False, use_scales=False) -> List[Dict[str, Any]]:
    offsets, scales = _load_offsets_scales(use_offsets, use_scales)
    out: List[Dict[str, Any]] = []
    for name, inst in getattr(executor, "instances", {}).items():
        if name == "bbox": continue
        o = inst.T[:3, 3].astype(float)                       # min corner (origin)
        s = [float(inst.spec.l), float(inst.spec.w), float(inst.spec.h)]
        if use_offsets and name in offsets:
            off = offsets[name];  o = o + [float(off[0]), float(off[1]), float(off[2])]
        if use_scales and name in scales:
            sc = scales[name];    s = [s[0]*float(sc[0]), s[1]*float(sc[1]), s[2]*float(sc[2])]
        center = [float(o[0]+s[0]/2), float(o[1]+s[1]/2), float(o[2]+s[2]/2)]
        out.append({"id": name, "name": name, "center": center, "size": s, "rotationEuler": None})
    return out

# --- program execution endpoints (these need 'app' defined already) ---
@app.post("/api/execute-default", response_model=ExecuteResponse)
def api_execute_default(use_offsets: bool = False, use_scales: bool = False):
    if not DEFAULT_IR.exists():
        raise HTTPException(status_code=404, detail=f"Default IR not found: {DEFAULT_IR}")
    exe = rescale_and_execute(INPUT_DIR, DEFAULT_IR)
    cuboids = _executor_to_cuboids(exe, use_offsets=use_offsets, use_scales=use_scales)
    return {"cuboids": [Cuboid(**c) for c in cuboids]}

@app.post("/api/execute", response_model=ExecuteResponse)
def api_execute(program: dict = Body(...), use_offsets: bool = False, use_scales: bool = False):
    with NamedTemporaryFile("w", delete=False, suffix=".json", dir=str(INPUT_DIR)) as tmp:
        json.dump(program, tmp); tmp_path = Path(tmp.name)
    try:
        exe = rescale_and_execute(INPUT_DIR, tmp_path)
        cuboids = _executor_to_cuboids(exe, use_offsets=use_offsets, use_scales=use_scales)
        return {"cuboids": [Cuboid(**c) for c in cuboids]}
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

# --- static UI at '/' (after app exists is fine) ---
if not UI_DIST.exists():
    print(f"⚠️  Build not found at {UI_DIST}. Run `npm run build` in stroke-viewer/")
app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
