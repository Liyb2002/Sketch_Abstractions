#!/usr/bin/env python3
"""
run_sam_with_mask.py

Refine prior (non-axis-aligned) masks with SAM.

Inputs (in ./bbx_mask_input):
  - {k}.png                 # base image for view k  (k is an integer: 0,1,2,...)
  - {k}_{name}.png          # binary prior(s) per component, 0/255 or 0/1

Outputs (in ./bbx_mask_output):
  - {k}_{name}_mask.png     # refined binary mask (0/255)
  - {k}_overlay.png         # colored overlay of all refined parts
  - {k}_parts.json          # metadata (bbox, area, score, RLE per part)

Env (optional):
  - SAM_MODEL_TYPE   (default: "vit_h")
  - SAM_CHECKPOINT   (default: "./sam_vit_h_4b8939.pth")
"""

from pathlib import Path
import os, re, json
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- paths ---
ROOT    = Path.cwd()
IN_DIR  = ROOT / "bbx_mask_input"
OUT_DIR = ROOT / "bbx_mask_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- model config ---
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", str(ROOT / "sam_vit_h_4b8939.pth"))

# --- helpers ---
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def name_color(i: int):
    hue = (37 * (i + 1)) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])

def mask_bbox(seg: np.ndarray):
    ys, xs = np.where(seg)
    if ys.size == 0:
        return [0, 0, 0, 0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]

def rle_encode_boolmask(mask_bool: np.ndarray):
    try:
        from pycocotools import mask as mutils
        rle = mutils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        return rle
    except Exception:
        arr = np.asfortranarray(mask_bool).reshape((-1,), order="F").astype(np.uint8)
        counts = []
        prev = 0
        run = 0
        for v in arr:
            if v != prev:
                counts.append(run)
                run = 1
                prev = v
            else:
                run += 1
        counts.append(run)
        return {"size": list(mask_bool.shape), "counts": counts}

def list_views(in_dir: Path):
    # base images named like 0.png, 1.png, ...
    return sorted(
        [p for p in in_dir.glob("*.png") if re.fullmatch(r"\d+\.png", p.name)],
        key=lambda p: int(p.stem),
    )

# Convert a full-res prior (HxW, 0..255) into SAM mask logits [1,256,256]
# aligned with the current image embedding in predictor.
def prior_to_mask_input(prior_u8: np.ndarray, predictor: SamPredictor) -> torch.Tensor:
    H, W = prior_u8.shape
    prob = (prior_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    prob = cv2.GaussianBlur(prob, (3, 3), 0)

    enc_size = predictor.model.image_encoder.img_size  # usually 1024
    scale = enc_size / max(H, W)
    newh, neww = int(round(H * scale)), int(round(W * scale))
    prob_resized = cv2.resize(prob, (neww, newh), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = enc_size - newh, enc_size - neww
    prob_square = cv2.copyMakeBorder(
        prob_resized, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0.0
    )

    lowres = cv2.resize(prob_square, (256, 256), interpolation=cv2.INTER_LINEAR)

    eps = 1e-4
    lowres = np.clip(lowres, eps, 1.0 - eps)
    logits = np.log(lowres / (1.0 - lowres)).astype(np.float32)

    # IMPORTANT: return [1,256,256] (no channel dim) — SAM will add it internally
    t = torch.from_numpy(logits)[None, ...]
    return t.to(next(predictor.model.parameters()).device)

# --- main ---
def main():
    device = pick_device()
    print(f"[INFO] device={device}")
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Missing input dir: {IN_DIR}")

    # build SAM
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)
    predictor = SamPredictor(sam)

    # iterate views
    images = list_views(IN_DIR)
    if not images:
        raise FileNotFoundError(f"No base images like '0.png' in {IN_DIR}")

    for ipath in images:
        stem = ipath.stem
        bgr = cv2.imread(str(ipath), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read {ipath}, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        predictor.set_image(rgb)

        # find priors: files {stem}_*.png (excluding the base itself)
        priors = sorted([p for p in IN_DIR.glob(f"{stem}_*.png")])
        if not priors:
            print(f"[WARN] no priors for view {stem} in {IN_DIR}")
            continue

        overlay = bgr.copy()
        parts_items = []

        for idx, pth in enumerate(priors):
            name = pth.stem[len(stem) + 1:] if pth.stem.startswith(f"{stem}_") else pth.stem

            prior = cv2.imread(str(pth), cv2.IMREAD_GRAYSCALE)
            if prior is None:
                print(f"[WARN] cannot read prior {pth}")
                continue
            if prior.shape[:2] != (H, W):
                # keep nearest to preserve edges
                prior = cv2.resize(prior, (W, H), interpolation=cv2.INTER_NEAREST)

            # optional loose focus box from prior bounds (helps decoder)
            ys, xs = np.where(prior > 0)
            if ys.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            box_np = np.array([x0, y0, x1, y1], dtype=np.float32)

            # build mask_input logits (shape [1,256,256])
            mask_input = prior_to_mask_input(prior, predictor)

            # predict single refined mask
            masks, scores, _ = predictor.predict(
                box=box_np,                 # optional but useful
                mask_input=mask_input,      # NOTE: [1,256,256]
                multimask_output=False
            )
            seg = masks[0].astype(bool)
            score = float(scores[0])

            # save binary mask
            out_mask = (seg.astype(np.uint8)) * 255
            cv2.imwrite(str(OUT_DIR / f"{stem}_{name}_mask.png"), out_mask)

            # overlay (fill + contour)
            color = name_color(idx)
            overlay[seg] = (0.45 * np.array(color, np.uint8) + 0.55 * overlay[seg]).astype(np.uint8)
            cnts, _ = cv2.findContours(out_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, color, 2, lineType=cv2.LINE_AA)

            parts_items.append({
                "name": name,
                "score": score,
                "bbox": mask_bbox(seg),
                "area": int(seg.sum()),
                "rle": rle_encode_boolmask(seg),
            })

        # write overlay + manifest
        cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), overlay)
        with open(OUT_DIR / f"{stem}_parts.json", "w", encoding="utf-8") as f:
            json.dump({"image": f"{stem}.png", "height": H, "width": W, "parts": parts_items}, f, indent=2)

        print(f"[OK] {stem}: refined {len(parts_items)} parts -> {OUT_DIR}")

if __name__ == "__main__":
    main()
