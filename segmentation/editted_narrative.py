#!/usr/bin/env python3
"""
Stage 2 (single call): Image + user-edited components -> narrative only

Inputs:
- ../input/*.png
- ../input/sketch_components.json   (assumed correct; either {"components":[...]} or ["..."])

Output:
- ../input/sketch_narrative.json
    { "narrative": "..." }

Guarantees:
- One API call to generate the narrative.
- The final narrative includes EVERY component term from sketch_components.json.
  (Prompt enforces this; validator appends a compact inclusion clause if any term is missing.)
"""

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

# ---------------- Config ----------------
MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
# CHANGE: use ../input folder for all inputs
INPUT_DIR = Path.cwd().parent / "input"
INPUT_COMPONENTS = INPUT_DIR / "sketch_components.json"
# CHANGE: save outputs into the same folder as the images
OUT_NARR = INPUT_DIR / "sketch_narrative.json"

PROMPT_NARRATIVE_ONLY = (
    "You are an expert industrial-design sketch analyst.\n"
    "The provided component list is CORRECT and AUTHORITATIVE.\n"
    "Task: Using the image and the component list, output EXACTLY this JSON object:\n"
    '{ "narrative": "<2–4 concise sentences that ONLY mention the provided components by name; '
    'state their broad shapes (e.g., slab, band, tube), how they connect (which attaches to which), '
    'dominant axes/symmetries, and key proportions/ordering (e.g., back height vs seat depth). '
    'STRICT NAMING: Use EVERY provided component name verbatim at least once, and use NO other component names. '
    'Avoid meta-remarks (do NOT say it is a sketch/3D/overview). '
    'If something is not visible, phrase conservatively but do not invent components.>" }\n'
    "STRICT: Output only a single JSON object. No extra text."
)

# ---------------- Helpers ----------------
def image_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _extract_output_text(resp: Any) -> str:
    """Works across Responses SDK shapes."""
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
    Try json.loads; if that fails, strip code fences and extract the first
    top-level {...} block using a brace counter (string-aware).
    """
    def _strip_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = "\n".join(s.splitlines()[1:])
        if s.endswith("```"):
            s = "\n".join(s.splitlines()[:-1])
        return s.strip()

    s = _strip_fences(text)
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    if start == -1:
        raise RuntimeError(f"Model did not return valid JSON.\nRaw text:\n{text}")

    in_str = False
    esc = False
    depth = 0
    end = None

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

    if end is None:
        raise RuntimeError(f"Could not find balanced JSON object.\nRaw text:\n{text}")

    return json.loads(s[start:end])

def _load_components(p: Path) -> List[str]:
    if not p.exists():
        raise FileNotFoundError(
            f"Components file not found: {p}\n"
            'Expected either: {"components":[...]} or just a JSON array [ ... ].'
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    comps = data["components"] if isinstance(data, dict) and "components" in data else data
    if not isinstance(comps, list):
        raise ValueError("Components must be a JSON array or an object with 'components' array.")
    comps = [str(x).strip() for x in comps if str(x).strip()]
    if not comps:
        raise ValueError("Components list is empty.")
    return comps

def _validate_narrative(obj: Dict[str, Any]) -> str:
    if not isinstance(obj, dict) or "narrative" not in obj or not isinstance(obj["narrative"], str):
        raise ValueError("Expected JSON object with a 'narrative' string.")
    text = obj["narrative"].strip()
    if len(text) < 10:
        raise ValueError("Narrative seems too short (<10 chars).")
    return text

def _ensure_all_terms_in_text(narrative: str, components: List[str]) -> str:
    """
    Ensure each component term appears at least once (case-insensitive).
    If any are missing, append a compact inclusion clause to guarantee presence.
    """
    low = narrative.lower()
    missing = [c for c in components if c.lower() not in low]
    if not missing:
        return narrative
    suffix = " Includes: " + "; ".join(missing) + "."
    return (narrative + suffix).strip()

def _call_narrative(client: OpenAI, model: str, prompt: str, img_data_urls: List[str], edited_components: List[str]) -> Dict[str, Any]:
    """
    Single call: pass the authoritative component list + all images; get back { "narrative": ... }.
    """
    edited_block = json.dumps({"components": edited_components}, ensure_ascii=False)
    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": f"AUTHORITATIVE_COMPONENT_LIST_JSON:\n{edited_block}"},
    ] + [{"type": "input_image", "image_url": url} for url in img_data_urls]

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": content,
        }],
        temperature=0.2,
        max_output_tokens=600,
    )
    text = _extract_output_text(resp)
    if not text:
        raise RuntimeError("Empty response from model.")
    return _coerce_json(text)

# ---------------- Main ----------------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")

    # Gather all PNGs in the input directory
    png_paths = sorted(INPUT_DIR.glob("*.png"))
    if not png_paths:
        raise SystemExit(f"No .png files found in {INPUT_DIR}")

    img_urls = [image_to_data_url(p) for p in png_paths]
    components = _load_components(INPUT_COMPONENTS)  # authoritative (edited by user)
    client = OpenAI()

    obj = _call_narrative(client, MODEL, PROMPT_NARRATIVE_ONLY, img_urls, components)
    narrative = _validate_narrative(obj)
    narrative = _ensure_all_terms_in_text(narrative, components)

    OUT_NARR.parent.mkdir(parents=True, exist_ok=True)
    OUT_NARR.write_text(json.dumps({"narrative": narrative}, indent=2), encoding="utf-8")
    print(f"✅ Wrote {OUT_NARR.resolve()}")
    print(json.dumps({"narrative": narrative}, indent=2))

if __name__ == "__main__":
    main()
