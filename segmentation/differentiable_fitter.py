#!/usr/bin/env python3
"""
differentiable_fitter.py  (translation-only)

Optimize ONLY per-part world translations (no size changes) to fit 3D stroke samples.
- Keeps l,w,h fixed exactly as in ../input/sketch_program_ir.json
- Fits strokes to the union of cuboid EDGES (good for wireframe strokes)
- Saves learned per-part offsets to ../input/fit_translations.json
- Does NOT modify the IR on disk (no size or anchor edits)

Inputs:
  ../input/sketch_program_ir.json
  (caller passes sample_points, feature_lines arrays)

API:
  run_differentiable_fit(input_dir: Path, sample_points, feature_lines,
                         steps=200, lr=1e-2) -> Executor

Conventions:
- feature_lines[-1] is a type id: 1 line, 2 circle, 3 cylinder face, 4 arc, 5 spline, 6 sphere

Notes:
- Attach graph is respected; we only add small learnable world offsets per prototype part.
- Translate-array copies inherit the prototype's learned offset.
- If you want me to also bake these offsets back into the IR (anchors or extra translate ops), ping me.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn

from program_executor import Executor

Tensor = torch.Tensor


# ----------------------- Basic helpers -----------------------

def flatten_points(sample_points: List[List[List[float]]]) -> Tensor:
    """
    Flatten a list of per-stroke 3D samples into a single (M,3) tensor.
    """
    pts = []
    for stroke in sample_points:
        for p in stroke:
            if isinstance(p, (list, tuple)) and len(p) == 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
    if not pts:
        return torch.empty(0, 3)
    return torch.tensor(pts, dtype=torch.float32)


def flatten_points_with_masks(
    sample_points: List[List[List[float]]],
    feature_lines: List[List[float]],
    edge_types: Tuple[int, ...] = (1,)  # stroke type ids to include in EDGE loss
) -> Tuple[Tensor, Tensor]:
    """
    Flattens sample_points (list per stroke) and builds a boolean mask indicating
    which per-point samples belong to strokes whose type is in `edge_types`.
    Assumes order-alignment between feature_lines and sample_points.
    Returns:
      pts: (M,3) float32
      mask_edge: (M,) bool
    """
    pts = []
    mask = []

    for fl, stroke_pts in zip(feature_lines, sample_points):
        t = int(fl[-1]) if isinstance(fl, list) and len(fl) >= 1 else -1
        is_edge_like = (t in edge_types)
        for p in stroke_pts:
            if isinstance(p, (list, tuple)) and len(p) == 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
                mask.append(is_edge_like)

    if not pts:
        return torch.empty(0, 3), torch.empty(0, dtype=torch.bool)

    return torch.tensor(pts, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def build_band_weight(feature_lines: List[List[float]]) -> float:
    """
    Slightly increase the surface band term if closed-ish primitives are present.
    """
    type_ids = [int(st[-1]) for st in feature_lines if isinstance(st, list) and len(st) >= 1]
    closed_like = any(t in (2, 3, 6) for t in type_ids)  # circle, cylinder-face, sphere
    return 0.35 if closed_like else 0.25


# ----------------------- Differentiable executor (translation-only) -----------------------

class DifferentiableExecutor(nn.Module):
    """
    Materializes part placements from the IR, then adds learnable world offsets per prototype part.
    Sizes (l,w,h) are constants from the IR; ONLY translations are learnable.
    """
    def __init__(self, ir: Dict, device: str = "cpu"):
        super().__init__()
        self.P = ir["program"]
        self.bb = self.P["bblock"]
        self.device = torch.device(device)

        # Fixed sizes per named cuboid (exclude bbox)
        self.part_names: List[str] = [c["var"] for c in self.P.get("cuboids", [])]
        self.size_const = {
            c["var"]: torch.tensor([float(c["l"]), float(c["w"]), float(c["h"])],
                                   dtype=torch.float32, device=self.device)
            for c in self.P.get("cuboids", [])
        }

        # Learnable per-part world translation offsets (prototype-level; copies inherit)
        self._dorigin = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device))
            for name in self.part_names
        })

    def sizes(self, name: str) -> Tensor:
        if name == "bbox":
            return torch.tensor([float(self.bb["l"]), float(self.bb["w"]), float(self.bb["h"])],
                                dtype=torch.float32, device=self.device)
        return self.size_const[name]

    def _apply_offset(self, name: str, origin: Tensor) -> Tensor:
        # Only real prototype parts receive learnable offsets
        if name in self._dorigin:
            return origin + self._dorigin[name]
        return origin

    def forward(self) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Execute attach -> translate-array (axis-aligned, translation only).
        Returns dict name -> (origin[min], size). Includes bbox.
        """
        out: Dict[str, Tuple[Tensor, Tensor]] = {}
        bb_min = torch.tensor(self.bb.get("min", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
        out["bbox"] = (bb_min, self.sizes("bbox"))

        # Attach multi-pass (grounded by bbox)
        pending = list(self.P.get("attach", []))
        last_len = None
        while pending and last_len != len(pending):
            last_len = len(pending)
            next_p = []
            for a in pending:
                A, B = a["a"], a["b"]
                x1, y1, z1, x2, y2, z2 = [torch.tensor(float(a[k]), device=self.device) for k in ("x1", "y1", "z1", "x2", "y2", "z2")]
                if A in out and B in out:
                    continue
                elif B in out:
                    oB, sB = out[B]
                    sA = self.sizes(A)
                    pB = oB + sB * torch.stack([x2, y2, z2])
                    pA = sA * torch.stack([x1, y1, z1])
                    oA = pB - pA
                    out[A] = (self._apply_offset(A, oA), sA)
                elif A in out:
                    oA, sA = out[A]
                    sB = self.sizes(B)
                    pA = oA + sA * torch.stack([x1, y1, z1])
                    pB = sB * torch.stack([x2, y2, z2])
                    oB = pA - pB
                    out[B] = (self._apply_offset(B, oB), sB)
                else:
                    next_p.append(a)
            pending = next_p
        if pending:
            raise RuntimeError(f"Unresolved attaches (grounding order violated): {pending}")

        # Translate arrays (explicit copies share prototype offset)
        copies: List[Tuple[str, Tensor, Tensor]] = []
        for t in self.P.get("translate", []):
            src = t["c"]; axis = t["axis"].upper(); n = int(t["n"]); d = float(t["d"])
            if src not in out or n <= 0:
                continue
            oS, sS = out[src]
            _, bb_s = out["bbox"]
            ax = {"X": 0, "Y": 1, "Z": 2}[axis]
            total = d * bb_s[ax]
            step = total / float(n)
            for i in range(1, n + 1):
                offset = torch.zeros(3, device=self.device); offset[ax] = step * i
                nm = f"{src}_T{i}"
                copies.append((nm, oS + offset, sS))
        for nm, oC, sC in copies:
            proto = nm.split("_T")[0]
            if proto in self._dorigin:
                out[nm] = (oC + self._dorigin[proto], sC)
            else:
                out[nm] = (oC, sC)

        return out

    def pack_boxes(self, out: Dict[str, Tuple[Tensor, Tensor]], skip_bbox: bool = True) -> Tensor:
        rows = []
        for k, (o, s) in out.items():
            if skip_bbox and k == "bbox":
                continue
            rows.append(torch.cat([o, s], dim=0))  # [x0,y0,z0,l,w,h]
        return torch.stack(rows, dim=0) if rows else torch.empty(0, 6, device=self.device)


# ----------------------- Distances: surface & edges -----------------------


def _dist_point_to_segments(points: torch.Tensor, segs: torch.Tensor) -> torch.Tensor:
    """
    points: (M,3)
    segs:   (N,12,2,3)
    returns: (M,N,12) Euclidean distances to each segment
    """
    if points.dim() != 2 or points.size(-1) != 3:
        raise ValueError(f"[points] expected (M,3), got {tuple(points.shape)}")
    if segs.dim() != 4 or segs.size(-1) != 3 or segs.size(-2) != 2 or segs.size(-3) != 12:
        raise ValueError(f"[segs] expected (N,12,2,3), got {tuple(segs.shape)}")

    M = points.size(0)
    p = points.view(M, 1, 1, 3)         # (M,1,1,3)
    a = segs[:, :, 0, :].unsqueeze(0)   # (1,N,12,3)
    b = segs[:, :, 1, :].unsqueeze(0)   # (1,N,12,3)

    ab = b - a                          # (1,N,12,3)
    ap = p - a                          # (M,N,12,3)

    denom = (ab * ab).sum(dim=-1).clamp_min(1e-12)  # (1,N,12)
    t = (ap * ab).sum(dim=-1) / denom               # (M,N,12)
    t = t.clamp(0.0, 1.0).unsqueeze(-1)             # (M,N,12,1)

    proj = a + t * ab                               # (M,N,12,3)
    d = ((p - proj) ** 2).sum(dim=-1).sqrt()        # (M,N,12)
    return d


def union_softmin_edges(points: torch.Tensor, boxes: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    """
    Soft-min distance from points (M,3) to the union of all edges
    of axis-aligned boxes (N,6: [x0,y0,z0,l,w,h]).
    Returns (M,)
    """
    if boxes.numel() == 0:
        return torch.full((points.shape[0],), 1e3, device=points.device)
    if boxes.dim() != 2 or boxes.size(-1) != 6:
        raise ValueError(f"[boxes] expected (N,6), got {tuple(boxes.shape)}")

    O = boxes[:, 0:3]   # (N,3)
    S = boxes[:, 3:6]   # (N,3)

    # Corners: (N,8,3)
    N = O.size(0)
    device = boxes.device
    mask = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ], dtype=torch.float32, device=device)  # (8,3)

    corners = O[:, None, :].expand(N, 8, 3) + S[:, None, :].expand(N, 8, 3) * mask  # (N,8,3)

    # Edges: (N,12,2,3)
    idx_pairs = torch.tensor([
        [0, 1], [2, 3], [4, 5], [6, 7],   # +X edges
        [0, 2], [1, 3], [4, 6], [5, 7],   # +Y edges
        [0, 4], [1, 5], [2, 6], [3, 7],   # +Z edges
    ], dtype=torch.long, device=device)    # (12,2)
    a = corners[:, idx_pairs[:, 0], :]     # (N,12,3)
    b = corners[:, idx_pairs[:, 1], :]     # (N,12,3)
    segs = torch.stack([a, b], dim=2).contiguous()  # (N,12,2,3)

    # Distances to all edges of all boxes -> (M, N, 12)
    d_all = _dist_point_to_segments(points, segs)   # (M,N,12)
    d_min_per_box = d_all.min(dim=-1).values        # (M,N)

    v = -d_min_per_box / tau
    return -tau * torch.logsumexp(v, dim=1)         # (M,)


# ----------------------- Public entrypoint -----------------------

def run_differentiable_fit(input_dir: Path,
                           sample_points: List[List[List[float]]],
                           feature_lines: List[List[float]],
                           steps: int = 200,
                           lr: float = 1e-2) -> Executor:
    """
    Optimize per-part translations to fit stroke samples to edges.
    DOES NOT modify sizes; IR on disk is left unchanged.
    Writes learned offsets to ../input/fit_translations.json.
    Returns a fresh program_executor.Executor for the original IR.
    """
    ir_path = input_dir / "sketch_program_ir.json"
    if not ir_path.exists():
        raise SystemExit(f"IR not found: {ir_path}")

    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dex = DifferentiableExecutor(ir, device=device).to(device)
    opt = torch.optim.Adam(dex.parameters(), lr=lr)

    # Prepare all points + boolean mask for edge-loss points
    # By default, only straight lines (type 1) contribute to the edge snapping loss.
    pts_all, mask_edge = flatten_points_with_masks(sample_points, feature_lines, edge_types=(1,))
    pts_all = pts_all.to(device)
    mask_edge = mask_edge.to(device)

    if pts_all.numel() == 0:
        print("No sample points found; skipping optimization.")
        return Executor(ir)

    pts_edge = pts_all[mask_edge]
    if pts_edge.numel() == 0:
        print("⚠️ No points matched edge_types for edge loss; edge term will be skipped.")

    # Optimize translations
    for step in range(steps):
        opt.zero_grad()

        placed = dex()
        boxes = dex.pack_boxes(placed, skip_bbox=True)  # (N,6)

        # Main objective: snap selected strokes' points to cuboid EDGES
        if pts_edge.numel() > 0:
            d_edge = union_softmin_edges(pts_edge, boxes, tau=0.02)   # (Me,)
            loss_edge = d_edge.mean()
        else:
            loss_edge = torch.tensor(0.0, device=device)

        loss = 1.0 * loss_edge
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == steps - 1:
            print(f"[fit-T] step {step:03d} | "
                  f"loss={loss.item():.6f}  edge={loss_edge.item():.6f} ")

    # ---- Save learned translations (sidecar JSON) ----
    with torch.no_grad():
        offsets = {k: dex._dorigin[k].detach().cpu().tolist() for k in dex.part_names}
    sidecar = {
        "note": "per-part world translation offsets (prototype-level). Copies inherit prototype offset.",
        "offsets_xyz": offsets,
    }
    (input_dir / "fit_translations.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"✅ Saved learned translations to {(input_dir / 'fit_translations.json')}")

    # Return a fresh classic Executor (unmodified IR)
    return Executor(ir)
