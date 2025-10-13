# stroke-viewer/server.py
from __future__ import annotations
from typing import List, Optional
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.load_program_main import (
    UI_DIST,
    load_strokes_payload,
    execute_default_to_cuboids,   # <-- only this remains
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
    cuboidName: str
    strokeIndex: int
    # strokePoints optional—UI pulls by index

class ExecuteResponse(BaseModel):
    cuboids: List[Cuboid]
    anchors: List[Anchor]

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

# Static UI
if not UI_DIST.exists():
    print(f"⚠️  UI build not found at {UI_DIST}. Run `npm run build` in stroke-viewer/")
app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
