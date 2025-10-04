#!/usr/bin/env python3
"""
run_sam_with_mask.py

Reads paired inputs from:  ./bbx_input/{k}.png and ./bbx_input/{k}.json
  JSON format:
  {
    "size": [H, W],        # optional, used for coordinate scaling if mismatched
    "boxes": [
      {"name":"seat_slab1", "bbox":[x0,y0,x1,y1]}, ...
    ]
  }

SAM weights expected at:   ./sam_vit_h_4b8939.pth
Writes outputs to:         ./bbx_output/
  - {k}_{name}_mask.png     (binary mask 0/255)
  - {k}_overlay.png         (colored contours for all parts)
  - {k}_parts.json          (per-part bbox, area, RLE)

Works well on Mac M-series (uses SamPredictor; no auto generator).
"""

from pathlib import Path
import os, json
import numpy as np
import cv2
import torch

from segment_anything import sam_model_registry, SamPredictor


ROOT = Path.cwd()
IN_DIR  = ROOT / "bbx_input"
OUT_DIR = ROOT / "bbx_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model config (you can override via env vars)
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", str(ROOT / "sam_vit_h_4b8939.pth"))

# ------------- helpers -------------
def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_boxes_json(path: Path):
    data = json.loads(path.read_text())
    # Expected: {"size":[H,W], "boxes":[{"name":..., "bbox":[x0,y0,x1,y1]}, ...]}
    Hjson, Wjson = None, None
    if "size" in data and isinstance(data["size"], (list, tuple)) and len(data["size"]) == 2:
        Hjson, Wjson = int(round(float(data["size"][0]))), int(round(float(data["size"][1])))
    boxes = []
    for item in data.get("boxes", []):
        name = item.get("name")
        x0, y0, x1, y1 = [float(v) for v in item.get("bbox", [0,0,0,0])]
        boxes.append({"name": name, "bbox": [x0, y0, x1, y1]})
    return (Hjson, Wjson), boxes

def scale_box(box, src_wh, dst_wh):
    """Scale [x0,y0,x1,y1] from (Wsrc,Hsrc) to (Wdst,Hdst) if needed."""
    x0, y0, x1, y1 = box
    Wsrc, Hsrc = src_wh
    Wdst, Hdst = dst_wh
    if Wsrc == 0 or Hsrc == 0:
        return [x0, y0, x1, y1]
    sx, sy = Wdst / Wsrc, Hdst / Hsrc
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]

def clip_box(box, w, h):
    x0, y0, x1, y1 = box
    x0 = float(np.clip(x0, 0, w-1))
    x1 = float(np.clip(x1, 0, w-1))
    y0 = float(np.clip(y0, 0, h-1))
    y1 = float(np.clip(y1, 0, h-1))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return [x0, y0, x1, y1]

def mask_bbox(seg: np.ndarray):
    ys, xs = np.where(seg)
    if ys.size == 0:
        return [0,0,0,0]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]

def rle_encode_boolmask(mask_bool: np.ndarray):
    """COCO RLE if pycocotools is present; else simple uncompressed RLE."""
    try:
        from pycocotools import mask as mutils
        rle = mutils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        return rle
    except Exception:
        arr = np.asfortranarray(mask_bool).reshape((-1,), order="F").astype(np.uint8)
        counts, prev, run = [], 0, 0
        for v in arr:
            if v != prev:
                counts.append(run)
                run = 1
                prev = v
            else:
                run += 1
        counts.append(run)
        return {"size": list(mask_bool.shape), "counts": counts}

def name_color(i: int):
    hue = (37 * (i+1)) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0])

# ------------- main -------------
def main():
    device = pick_device()
    print(f"[INFO] device={device}")
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {IN_DIR}")

    # Build SAM predictor
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)
    predictor = SamPredictor(sam)

    # Pair images with JSONs
    images = sorted([p for p in IN_DIR.glob("*.png")], key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not images:
        raise FileNotFoundError(f"No .png images in {IN_DIR}")

    for img_path in images:
        stem = img_path.stem
        json_path = IN_DIR / f"{stem}.json"
        if not json_path.exists():
            print(f"[WARN] Missing JSON for {img_path.name}; skipping.")
            continue

        # Load image
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Could not read {img_path}; skipping.")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # Load boxes (+ optional size for scaling)
        (Hjson, Wjson), boxes = load_boxes_json(json_path)
        if not boxes:
            print(f"[WARN] No boxes in {json_path.name}; skipping.")
            continue

        # Set image in predictor
        predictor.set_image(rgb)

        overlay = bgr.copy()
        parts_items = []
        for idx, entry in enumerate(boxes):
            name = entry["name"] or f"part_{idx}"
            # If size provided, scale coords to actual image size
            box = entry["bbox"]
            if Hjson is not None and Wjson is not None and (Hjson != H or Wjson != W):
                box = scale_box(box, (Wjson, Hjson), (W, H))
            # Clip to image
            x0, y0, x1, y1 = clip_box(box, W, H)
            box_np = np.array([x0, y0, x1, y1], dtype=np.float32)

            # Predict masks for this box, choose best by score
            masks, scores, _ = predictor.predict(box=box_np, multimask_output=True)
            best = int(np.argmax(scores))
            seg = masks[best].astype(bool)

            # Save per-part binary mask
            out_mask = (seg.astype(np.uint8)) * 255
            cv2.imwrite(str(OUT_DIR / f"{stem}_{name}_mask.png"), out_mask)

            # Draw contour on overlay
            color = name_color(idx)
            cnts, _ = cv2.findContours(out_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, color, 2)
            # Also draw the (possibly scaled) box lightly for reference
            cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), color, 1, lineType=cv2.LINE_AA)

            # Collect JSON info
            parts_items.append({
                "name": name,
                "score": float(scores[best]),
                "bbox": mask_bbox(seg),
                "area": int(seg.sum()),
                "rle": rle_encode_boolmask(seg),
            })

        # Save overlay + parts.json
        cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), overlay)
        with open(OUT_DIR / f"{stem}_parts.json", "w", encoding="utf-8") as f:
            json.dump({"image": f"{stem}.png", "height": H, "width": W, "parts": parts_items}, f, indent=2)

        print(f"[OK] {stem}: wrote {len(parts_items)} parts to {OUT_DIR}")

if __name__ == "__main__":
    main()
