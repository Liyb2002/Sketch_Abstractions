#!/usr/bin/env python3
"""
Stage 1 (two calls): Image -> narrative, components
- Reads ./sketch.png
- CALL #1: get a concise narrative (JSON: {"narrative": "..."}).
- CALL #2: get components only (JSON: {"components": ["seat","backrest",...]}).
- Writes ./sketch_narrative.json and ./sketch_components.json

Notes:
- This version avoids `response_format=...` to be compatible across OpenAI SDK variants.
- It enforces JSON via prompt instructions and validates the result in Python.
"""

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

# ---- Config ----
MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")  # any vision-enabled model
INPUT_IMAGE = Path("sketch.png")
OUT_NARR = Path("sketch_narrative.json")
OUT_COMP = Path("sketch_components.json")

PROMPT_NARRATIVE = (
    "You are an expert industrial-design sketch analyst. "
    "Produce exactly this JSON object:\n"
    '{ "narrative": "<2–4 concise sentences focusing ONLY on geometry and structure: '
    'name the main components, their shapes (e.g., rectangular seat slab, curved back band, '
    'tapered cylindrical legs), how they connect (join type, alignment, offsets), '
    'dominant axes and symmetry, key proportions (e.g., back height vs seat depth), and any '
    'interlocks or overlaps. Do NOT mention that it is a sketch, 3D, an overview, or minimalistic.>'
    '" }\n'
    "STRICT: Output only a single JSON object. No extra text."
)

PROMPT_COMPONENTS = (
    "From this sketch image, output exactly this JSON object:\n"
    '{ "components": ["<component 1>", "<component 2>", "..."] }\n'
    "List only the major physical parts the object is built from (e.g., seat slab, back band, "
    "front legs, rear legs, stretchers, arm rails). Do NOT include meta terms like 'sketch' or '3D'. "
    "STRICT: Output only a single JSON object. No extra text."
)

# ---- Helpers ----
def image_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _extract_output_text(resp: Any) -> str:
    """
    Works across SDK shapes:
    - Prefer resp.output_text when present
    - Otherwise walk resp.output[*].content[*].text
    """
    text = getattr(resp, "output_text", None)
    if text:
        return text

    chunks: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") == "output_text":
                chunks.append(getattr(c, "text", ""))
    return "\n".join(chunks).strip()

def _coerce_json(text: str) -> Dict[str, Any]:
    """
    Try json.loads; if the model added stray text, extract the first {...} block.
    """
    try:
        return json.loads(text)
    except Exception:
        # Try to grab the first JSON object in the text
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError(f"Model did not return valid JSON.\nRaw text:\n{text}")

def _validate_narrative(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict) or "narrative" not in obj or not isinstance(obj["narrative"], str):
        raise ValueError("Expected JSON object with a 'narrative' string.")
    if len(obj["narrative"].strip()) < 10:
        raise ValueError("Narrative seems too short (<10 chars).")
    # Keep only the required field (be strict)
    return {"narrative": obj["narrative"].strip()}

def _validate_components(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict) or "components" not in obj or not isinstance(obj["components"], list):
        raise ValueError("Expected JSON object with a 'components' array.")
    comps = [str(x).strip() for x in obj["components"] if str(x).strip()]
    if not comps:
        raise ValueError("Components list is empty.")
    return {"components": comps}

def call_responses(client: OpenAI, model: str, prompt: str, img_data_url: str) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": img_data_url},
            ],
        }],
        temperature=0.2,
        max_output_tokens=600,
    )

    text = _extract_output_text(resp)
    if not text:
        raise RuntimeError("Empty response from model.")
    return _coerce_json(text)

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    img_url = image_to_data_url(INPUT_IMAGE)
    client = OpenAI()  # uses OPENAI_API_KEY from env

    # ---- CALL #1: Narrative-only
    narr_obj = call_responses(client, MODEL, PROMPT_NARRATIVE, img_url)
    narr_obj = _validate_narrative(narr_obj)
    OUT_NARR.write_text(json.dumps(narr_obj, indent=2), encoding="utf-8")
    print(f"✅ Wrote {OUT_NARR.resolve()}")
    print(json.dumps(narr_obj, indent=2))

    # ---- CALL #2: Components-only
    comps_obj = call_responses(client, MODEL, PROMPT_COMPONENTS, img_url)
    comps_obj = _validate_components(comps_obj)
    OUT_COMP.write_text(json.dumps(comps_obj, indent=2), encoding="utf-8")
    print(f"✅ Wrote {OUT_COMP.resolve()}")
    print(json.dumps(comps_obj, indent=2))

if __name__ == "__main__":
    main()
