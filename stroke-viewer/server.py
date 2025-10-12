# stroke-viewer/server.py
from pathlib import Path
from typing import List, Any, Dict
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Paths (match your layout) ---
ROOT = Path(__file__).parent.resolve()     # .../stroke-viewer
UI_DIST = ROOT / "dist"                    # Vite build output
INPUT_DIR = ROOT.parent / "input"          # sibling folder
STROKE_FILE = INPUT_DIR / "stroke_lines.json"

# --- Models ---
class Polyline(BaseModel):
    points: List[List[float]]

class StrokePayload(BaseModel):
    perturbed_feature_lines: List[Polyline]
    perturbed_construction_lines: List[Polyline]
    feature_lines: List[Polyline]

# --- Loader (your function) ---
def load_perturbed_feature_lines(input_dir: Path):
    path_stroke_lines = input_dir / "stroke_lines.json"
    with open(path_stroke_lines, "r", encoding="utf-8") as f:
        stroke_lines = json.load(f)

    perturbed_feature_lines = stroke_lines.get("perturbed_feature_lines", [])
    perturbed_construction_lines = stroke_lines.get("perturbed_construction_lines", [])
    feature_lines = stroke_lines.get("feature_lines", [])

    print(
        f"📄 Loaded {len(perturbed_feature_lines)} perturbed feature lines, "
        f"{len(perturbed_construction_lines)} perturbed construction lines, "
        f"and {len(feature_lines)} feature lines"
    )
    return perturbed_feature_lines, perturbed_construction_lines, feature_lines

# --- Normalize to polylines ---
def _is_point(p: Any) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 3 and all(isinstance(x, (int, float)) for x in p)

def _flatten_to_polylines(raw: Any) -> List[List[List[float]]]:
    polys: List[List[List[float]]] = []
    if not isinstance(raw, list):
        return polys
    if all(_is_point(p) for p in raw):
        if len(raw) >= 2:
            polys.append([[float(a), float(b), float(c)] for a, b, c in raw])
        return polys
    for item in raw:
        if isinstance(item, list):
            if all(_is_point(p) for p in item):
                if len(item) >= 2:
                    polys.append([[float(a), float(b), float(c)] for a, b, c in item])
            else:
                polys.extend(_flatten_to_polylines(item))
    return polys

def normalize_bundle(bundle: Any) -> List[Dict[str, Any]]:
    return [{"points": pl} for pl in _flatten_to_polylines(bundle) if len(pl) >= 2]

# --- App ---
app = FastAPI(title="Stroke Viewer")

# If you will serve UI + API from same origin, CORS not needed — harmless if left on.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/strokes", response_model=StrokePayload)
def get_strokes():
    pf, pc, fl = load_perturbed_feature_lines(INPUT_DIR)
    return StrokePayload(
        perturbed_feature_lines=[Polyline(**pl) for pl in normalize_bundle(pf)],
        perturbed_construction_lines=[Polyline(**pl) for pl in normalize_bundle(pc)],
        feature_lines=[Polyline(**pl) for pl in normalize_bundle(fl)],
    )

# Serve the built React app at /
if not UI_DIST.exists():
    print(f"⚠️  Build not found at {UI_DIST}. Run `npm run build` in stroke-viewer/")
app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
