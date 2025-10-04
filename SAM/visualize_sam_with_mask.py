#!/usr/bin/env python3
"""
visualize_bbx_masks.py

Visualize priors vs. SAM-refined masks.

Reads:
  ./bbx_mask_input/{k}.png            # base stroke render (k is an integer)
  ./bbx_mask_input/{k}_{name}.png     # prior (binary mask, 0/255) per component
  ./bbx_mask_output/{k}_parts.json    # metadata written by run_sam_with_mask.py
  ./bbx_mask_output/{k}_{name}_mask.png  # refined mask per component

Writes (creates ./bbx_mask_output/vis):
  {k}_input_overlay.png    # priors overlaid on base
  {k}_result_overlay.png   # refined results overlaid on base
  {k}_side_by_side.png     # input vs result + legend
"""

from pathlib import Path
import re, json
import cv2
import numpy as np

ROOT      = Path.cwd()
IN_DIR    = ROOT / "bbx_mask_input"
OUT_DIR   = ROOT / "bbx_mask_output"
VIS_DIR   = OUT_DIR / "vis"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def list_views(dir_path: Path):
    return sorted([p for p in dir_path.glob("*.png") if re.fullmatch(r"\d+\.png", p.name)],
                  key=lambda p: int(p.stem))

def id_color_from_name(name: str):
    # deterministic BGR from name
    h = (abs(hash(name)) % 180)
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def overlay_mask(base_bgr: np.ndarray, mask_u8: np.ndarray, color_bgr, alpha=0.45, draw_contour=True):
    out = base_bgr.copy()
    m = mask_u8 > 0
    if m.any():
        fill = np.zeros_like(base_bgr, dtype=np.uint8)
        fill[m] = color_bgr
        out[m] = (alpha * fill[m] + (1 - alpha) * base_bgr[m]).astype(np.uint8)
        if draw_contour:
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color_bgr, 2, lineType=cv2.LINE_AA)
    return out

def make_legend(names: list, swatch=18, pad=8, font_scale=0.5):
    if not names:
        return np.zeros((1,1,3), dtype=np.uint8)
    rows = []
    # fixed width column for labels; adjust if you have long names
    text_w = max(180, max([len(n) for n in names]) * 8)
    for nm in names:
        color = id_color_from_name(nm)
        chip = np.full((swatch, swatch, 3), color, dtype=np.uint8)
        text_img = np.zeros((swatch, text_w, 3), dtype=np.uint8)
        cv2.putText(text_img, nm, (0, swatch-4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (230,230,230), 1, cv2.LINE_AA)
        row = np.hstack([chip, np.full((swatch, pad, 3), 0, dtype=np.uint8), text_img])
        rows.append(row)
    legend = np.vstack(rows)
    border = 12
    legend = cv2.copyMakeBorder(legend, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=(30,30,30))
    return legend

def side_by_side(left_bgr, right_bgr, legend=None):
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    def pad_to(img, hh):
        if img.shape[0] == hh: return img
        top = (hh - img.shape[0]) // 2
        bottom = hh - img.shape[0] - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    L = pad_to(left_bgr, h)
    R = pad_to(right_bgr, h)
    combo = np.hstack([L, R])
    if legend is not None and legend.size > 1:
        # scale legend height to match images
        scale = h / legend.shape[0]
        new_w = int(legend.shape[1] * min(1.0, scale*0.8))
        legend_rs = cv2.resize(legend, (new_w, h), interpolation=cv2.INTER_AREA)
        combo = np.hstack([combo, legend_rs])
    return combo

# ---------- main ----------
def main():
    base_imgs = list_views(IN_DIR)
    if not base_imgs:
        raise FileNotFoundError(f"No base images like '0.png' in {IN_DIR}")

    for base_path in base_imgs:
        stem = base_path.stem
        base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
        if base is None:
            print(f"[WARN] cannot read {base_path}, skipping")
            continue
        H, W = base.shape[:2]

        # --- gather names from results json if present, else infer from files
        parts_json = OUT_DIR / f"{stem}_parts.json"
        names = []
        if parts_json.exists():
            try:
                data = json.loads(parts_json.read_text())
                names = [p.get("name", f"part{i}") for i, p in enumerate(data.get("parts", []))]
            except Exception:
                names = []
        if not names:
            # fallback: look for refined masks, strip suffix
            refined = sorted(OUT_DIR.glob(f"{stem}_*_mask.png"))
            for p in refined:
                nm = p.stem[len(stem)+1:-5] if p.stem.endswith("_mask") else p.stem[len(stem)+1:]
                names.append(nm)
            names = sorted(set(names))

        # --- INPUT overlay (priors)
        input_overlay = base.copy()
        for nm in names:
            prior_path = IN_DIR / f"{stem}_{nm}.png"
            if not prior_path.exists():
                # skip missing prior
                continue
            prior = cv2.imread(str(prior_path), cv2.IMREAD_GRAYSCALE)
            if prior is None:
                continue
            if prior.shape[:2] != (H, W):
                prior = cv2.resize(prior, (W, H), interpolation=cv2.INTER_NEAREST)
            color = id_color_from_name(nm)
            input_overlay = overlay_mask(input_overlay, prior, color, alpha=0.45, draw_contour=True)
        cv2.imwrite(str(VIS_DIR / f"{stem}_input_overlay.png"), input_overlay)

        # --- RESULT overlay (refined)
        result_overlay = base.copy()
        for nm in names:
            mask_path = OUT_DIR / f"{stem}_{nm}_mask.png"
            if not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            color = id_color_from_name(nm)
            result_overlay = overlay_mask(result_overlay, mask, color, alpha=0.45, draw_contour=True)
        cv2.imwrite(str(VIS_DIR / f"{stem}_result_overlay.png"), result_overlay)

        # --- Side by side with legend
        legend = make_legend(names)
        sbs = side_by_side(input_overlay, result_overlay, legend)
        cv2.imwrite(str(VIS_DIR / f"{stem}_side_by_side.png"), sbs)

        print(f"[OK] view {stem}: wrote input_overlay, result_overlay, side_by_side -> {VIS_DIR}")

if __name__ == "__main__":
    main()
