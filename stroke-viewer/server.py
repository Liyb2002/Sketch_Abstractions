# stroke-viewer/server.py
from __future__ import annotations
from typing import List, Optional
import traceback
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.load_program_main import (
    UI_DIST,
    INPUT_DIR,
    load_strokes_payload,
    execute_default_to_cuboids,
)

# App setup
app = FastAPI(title="Stroke Viewer (default IR only)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Schemas
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

class Anchor(BaseModel):
    cuboidId: str
    cuboidName: Optional[str] = None
    strokeIndex: int

class ExecuteResponse(BaseModel):
    cuboids: List[Cuboid]
    anchors: List[Anchor]

# Save-anchors request/response
class SaveAnchorItem(BaseModel):
    cuboidId: str
    strokeIndex: int

class SaveAnchorsRequest(BaseModel):
    anchors: List[SaveAnchorItem]

class SaveAnchorsResponse(BaseModel):
    ok: bool
    saved: int
    path: str

# Routes
@app.get("/api/strokes", response_model=StrokePayload)
def api_strokes():
    return load_strokes_payload()

@app.post("/api/execute-default", response_model=ExecuteResponse)
def api_execute_default(use_offsets: bool = False, use_scales: bool = False):
    try:
        return execute_default_to_cuboids(use_offsets=use_offsets, use_scales=use_scales)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=tb)

@app.post("/api/save-anchors", response_model=SaveAnchorsResponse)
def api_save_anchors(req: SaveAnchorsRequest):
    try:
        out_path = INPUT_DIR / "anchor_strokes.json"
        payload = {
            "anchors": [
                {"cuboidId": a.cuboidId, "strokeIndex": a.strokeIndex}
                for a in req.anchors
            ]
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return SaveAnchorsResponse(ok=True, saved=len(req.anchors), path=str(out_path))
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=tb)

# Static UI
if not UI_DIST.exists():
    print(f"⚠️  UI build not found at {UI_DIST}. Run `npm run build` in stroke-viewer/")
app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
