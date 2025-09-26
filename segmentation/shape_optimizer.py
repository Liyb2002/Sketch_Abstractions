#!/usr/bin/env python3
"""
shape_optimizer.py

Step 2: Load the pre-sampled 3D strokes from
../input/perturbed_feature_lines.json.

Usage:
  Imported into main_optimizer.py
"""

from __future__ import annotations
from pathlib import Path
import json


def load_perturbed_feature_lines(input_dir: Path):
    path = input_dir / "perturbed_feature_lines.json"
    if not path.exists():
        raise SystemExit(f"perturbed_feature_lines.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        strokes = json.load(f)

    print(f"📄 Loaded {len(strokes)} strokes from {path}")
    return strokes
