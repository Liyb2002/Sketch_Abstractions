#!/usr/bin/env python3
"""
visualize_sam_with_mask.py

Visualize SAM part masks alongside your inputs (strokes + bounding boxes).

Reads:
  - ./bbx_input/{k}.png
  - ./bbx_input/{k}.json          # {"size":[H,W], "boxes":[{"name":..., "bbox":[x0,y0,x1,y1]}, ...]}
  - ./bbx_output/{k}_parts.json   # {"parts":[{"name":..., "bbox":[...], "area":..., "rle":...}, ...]}
  - ./bbx_output/{k}_{name}_mask.png   # 0/255 binary per part (from run_sam_with_mask.py)

Writes (creates ./bbx_output/vis/):
  - {k}_input_boxes.png
  - {k}_masks_overlay.png
  - {k}_side_by_side.png
"""

from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path.cwd()
IN_DIR   = ROOT / "bbx_input"
OUT_DIR  = ROOT / "bbx_output"
VIS_DIR  = OUT_DIR / "vis"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_boxes_json(path: Path):
    data = json.loads(path.read_text())
    Hjson = Wjson = None
    if "size" in data and isinstance(data["size"], (list, tuple)) and len(data["size"]) == 2:
        Hjson = int(round(float(data["size"][0])))
        Wjson = int(round(float(data["size"][1])))
    boxes = []
    for it in data.get("boxes", []):
        name = it.get("name") or ""
        x0, y0, x1, y1 = [float(v) for v in it.get("bbox", [0,0,0,0])]
        boxes.append({"name": name, "bbox": [x0, y0, x1, y1]})
    return (Hjson, Wjson), boxes

def scale_box(box, src_wh, dst_wh):
    x0,y0,x1,y1 = box
    Wsrc,Hsrc = src_wh
    Wdst,Hdst = dst_wh
    if Wsrc == 0 or Hsrc == 0: return [x0,y0,x1,y1]
    sx, sy = Wdst / Wsrc, Hdst / Hsrc
    return [x0*sx, y0*sy, x1*sx, y1*sy]

def id_color_from_name(name: str):
    # deterministic color per name (BGR)
    h = (abs(hash(name)) % 180)
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def overlay_mask(base_bgr: np.ndarray, mask_u8: np.ndarray, color_bgr, alpha=0.45, draw_contour=True):
    out = base_bgr.copy()
    m = mask_u8 > 0
    if m.any():
        # fill
        fill = np.zeros_like(base_bgr, dtype=np.uint8)
        fill[m] = color_bgr
        out[m] = (alpha * fill[m] + (1 - alpha) * base_bgr[m]).astype(np.uint8)
        # contour
        if draw_contour:
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color_bgr, 2, lineType=cv2.LINE_AA)
    return out

def draw_boxes(img_bgr: np.ndarray, boxes, Hjson, Wjson):
    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy()
    for it in boxes:
        name = it["name"]
        box = it["bbox"]
        if Hjson is not None and Wjson is not None and (Hjson != H or Wjson != W):
            box = scale_box(box, (Wjson, Hjson), (W, H))
        x0,y0,x1,y1 = [int(round(v)) for v in box]
        color = id_color_from_name(name)
        cv2.rectangle(vis, (x0,y0), (x1,y1), color, 2, lineType=cv2.LINE_AA)
        if name:
            cv2.putText(vis, name, (x0, max(0, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

def make_legend(part_names: list, swatch=18, pad=6, font_scale=0.5):
    # vertical legend image
    if not part_names:
        return np.zeros((1,1,3), dtype=np.uint8)
    rows = []
    for nm in part_names:
        color = id_color_from_name(nm)
        chip = np.full((swatch, swatch, 3), color, dtype=np.uint8)
        text_img = np.zeros((swatch, 240, 3), dtype=np.uint8)
        cv2.putText(text_img, nm, (0, swatch-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (230,230,230), 1, cv2.LINE_AA)
        row = np.hstack([chip, np.full((swatch, pad, 3), 0, dtype=np.uint8), text_img])
        rows.append(row)
    legend = np.vstack(rows)
    border = 12
    legend = cv2.copyMakeBorder(legend, border, border, border, border, cv2.BORDER_CONSTANT, value=(30,30,30))
    return legend

def side_by_side(left_bgr, right_bgr, legend=None):
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    def pad_to(img, h):
        if img.shape[0] == h: return img
        top = (h - img.shape[0]) // 2
        bottom = h - img.shape[0] - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    L = pad_to(left_bgr, h)
    R = pad_to(right_bgr, h)
    combo = np.hstack([L, R])
    if legend is not None and legend.size > 1:
        # scale legend to image height
        scale = h / legend.shape[0]
        legend_resized = cv2.resize(legend, (int(legend.shape[1]*scale*0.7), h), interpolation=cv2.INTER_AREA)
        combo = np.hstack([combo, legend_resized])
    return combo

# ---------- main ----------
def main():
    images = sorted([p for p in IN_DIR.glob("*.png")], key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not images:
        raise FileNotFoundError(f"No .png in {IN_DIR}")

    for img_path in images:
        stem = img_path.stem
        json_in  = IN_DIR / f"{stem}.json"
        parts_js = OUT_DIR / f"{stem}_parts.json"

        if not json_in.exists():
            print(f"[WARN] missing bbx JSON for {stem}, skipping.")
            continue
        if not parts_js.exists():
            print(f"[WARN] missing parts JSON for {stem}, skipping.")
            continue

        # load base image
        base = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if base is None:
            print(f"[WARN] cannot open {img_path}")
            continue
        H, W = base.shape[:2]

        # load boxes & draw
        (Hj, Wj), boxes = load_boxes_json(json_in)
        vis_boxes = draw_boxes(base, boxes, Hj, Wj)
        cv2.imwrite(str(VIS_DIR / f"{stem}_input_boxes.png"), vis_boxes)

        # load parts meta & masks
        parts_meta = json.loads(parts_js.read_text()).get("parts", [])
        part_names = [p.get("name", f"part{i}") for i, p in enumerate(parts_meta)]

        # overlay masks
        overlay = base.copy()
        for meta in parts_meta:
            name = meta.get("name", "")
            mask_path = OUT_DIR / f"{stem}_{name}_mask.png"
            if not mask_path.exists():
                # if name contains spaces or special chars that were sanitized;
                # try a fallback: replace spaces with underscores
                alt = OUT_DIR / f"{stem}_{name.replace(' ', '_')}_mask.png"
                mask_path = alt if alt.exists() else mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            color = id_color_from_name(name)
            overlay = overlay_mask(overlay, mask, color, alpha=0.45, draw_contour=True)

        cv2.imwrite(str(VIS_DIR / f"{stem}_masks_overlay.png"), overlay)

        # side-by-side + legend
        legend = make_legend(part_names)
        sbs = side_by_side(vis_boxes, overlay, legend)
        cv2.imwrite(str(VIS_DIR / f"{stem}_side_by_side.png"), sbs)

        print(f"[OK] {stem}: wrote input_boxes, masks_overlay, side_by_side to {VIS_DIR}")

if __name__ == "__main__":
    main()
