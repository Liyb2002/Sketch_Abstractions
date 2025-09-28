#!/usr/bin/env python3
"""
differentiable_fitter.py

Differentiable fitting of a ShapeAssembly-like IR to 3D stroke samples.

Inputs (fixed paths):
  ../input/sketch_program_ir.json
  ../input/perturbed_feature_lines.json  (provided by caller for plotting, not read here)

API:
  run_differentiable_fit(input_dir: Path, sample_points, feature_lines, steps=200) -> Executor

- sample_points: list[list[[x,y,z], ...], ...]  # same shape as feature_lines; each sublist = one stroke
- feature_lines: list[list[... , type_id]]      # last entry is type id (int)
  Types (your convention):
    1 line, 2 circle, 3 cylinder face, 4 arc, 5 spline, 6 sphere

What it does:
  * Builds a differentiable executor (no rotations) with per-cuboid (l,w,h) and per-cuboid translation offsets as learnable params.
  * Minimizes a soft-min **edge** distance from stroke samples to the union of cuboid edges.
  * Adds a small surface band term and L2 regularizers (sizes, translations).
  * Writes updated sizes back into ../input/sketch_program_ir.json.
  * Returns a fresh program_executor.Executor for the updated IR.

Notes:
  - Attach anchors stay fixed in [0,1] (we only add small per-part world translations here).
  - Reflect/translate copies share size with their prototype and inherit its learned offset.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from program_executor import Executor

Tensor = torch.Tensor
INPUT_REL = Path("..") / "input"  # will be resolved against current working dir by caller


# ----------------------- Small helpers -----------------------

def _t(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def flatten_points(sample_points: List[List[List[float]]]) -> Tensor:
    pts = []
    for stroke in sample_points:
        for p in stroke:
            if isinstance(p, (list, tuple)) and len(p) == 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
    if not pts:
        return torch.empty(0, 3)
    return torch.tensor(pts, dtype=torch.float32)


def build_type_weights(feature_lines: List[List[float]]) -> Tuple[float, float]:
    """
    Return (edge_weight, band_weight) depending on presence of closed primitives.
    If we see circle/cylinder-face/sphere => use a slightly larger surface band weight.
    """
    type_ids = [int(st[-1]) for st in feature_lines if isinstance(st, list) and len(st) >= 1]
    closed_like = any(t in (2, 3, 6) for t in type_ids)
    return (1.0, 0.35 if closed_like else 0.25)


# ----------------------- Differentiable Executor -----------------------

class DifferentiableExecutor(nn.Module):
    def __init__(self, ir: Dict, device: str = "cpu", learn_anchors: bool = False):
        super().__init__()
        self.P = ir["program"]
        self.bb = self.P["bblock"]
        self.device = torch.device(device)

        # Trainable sizes per named cuboid (excluding bbox)
        self.part_names: List[str] = [c["var"] for c in self.P.get("cuboids", [])]
        self.sizes_init = {
            c["var"]: torch.tensor([float(c["l"]), float(c["w"]), float(c["h"])], dtype=torch.float32)
            for c in self.P.get("cuboids", [])
        }
        self._size_raw = nn.ParameterDict({
            name: nn.Parameter(self._inv_softplus(self.sizes_init[name]).to(self.device))
            for name in self.part_names
        })

        # Trainable per-part world translation offsets (prototype-level; copies inherit)
        self._dorigin = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device))
            for name in self.part_names
        })

        self.learn_anchors = learn_anchors  # reserved (off)
        # (If needed later: register per-attach params via sigmoid to keep in [0,1])

    @staticmethod
    def _inv_softplus(x: Tensor) -> Tensor:
        return torch.log(torch.expm1(x.clamp_min(1e-6)))

    @staticmethod
    def _softplus_pos(r: Tensor) -> Tensor:
        return torch.nn.functional.softplus(r) + 1e-6

    def sizes(self, name: str) -> Tensor:
        if name == "bbox":
            return torch.tensor([float(self.bb["l"]), float(self.bb["w"]), float(self.bb["h"])],
                                dtype=torch.float32, device=self.device)
        return self._softplus_pos(self._size_raw[name])

    def _apply_offset(self, name: str, origin: Tensor) -> Tensor:
        # Only learned for real cuboids; bbox and synthetic names unchanged
        if name in self._dorigin:
            return origin + self._dorigin[name]
        return origin

    def forward(self) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Execute attach -> reflect -> translate (axis-aligned, translation only).
        Returns dict name -> (origin[3], size[3]). Includes bbox.
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

        # Translate arrays (explicit copies share size & offset with prototype)
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
            # Inherit prototype offset: use the source name before "_T"
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


# ----------------------- SDF & Edge distances -----------------------

def sdf_to_box(points: Tensor, origin: Tensor, size: Tensor) -> Tensor:
    """
    Signed distance from points (...,3) to axis-aligned box defined by min=origin, size.
    Negative inside, positive outside.
    """
    center = origin + 0.5 * size
    half = 0.5 * size
    d = torch.abs(points - center) - half
    outside = torch.clamp(d, min=0.0)
    dist_out = torch.linalg.norm(outside, dim=-1)
    dist_in = torch.clamp(torch.max(d, dim=-1).values, max=0.0)
    return dist_out + dist_in  # signed


def union_softmin(points: Tensor, boxes: Tensor, tau: float = 0.02) -> Tensor:
    """
    Soft minimum SDF to union of boxes.
    boxes: (N,6) with [x0,y0,z0,l,w,h]
    returns (M,) distances for points (M,3).
    """
    if boxes.numel() == 0:
        return torch.full((points.shape[0],), 1e3, device=points.device)
    P = points[:, None, :]     # (M,1,3)
    O = boxes[None, :, 0:3]    # (1,N,3)
    S = boxes[None, :, 3:6]    # (1,N,3)
    d = sdf_to_box(P, O, S)    # (M,N)
    v = -d / tau
    return -tau * torch.logsumexp(v, dim=1)


def _corners_from_min_size(origin: Tensor, size: Tensor) -> Tensor:
    """
    origin,size: (N,3) -> corners (N,8,3)
    Generates 8 corners of an axis-aligned box from min-corner and size.
    """
    N = origin.shape[0]
    device = origin.device
    O = origin[:, None, :].expand(N, 8, 3)
    # Binary corner mask [0/1] for (x,y,z)
    mask = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ], dtype=torch.float32, device=device)                # (8,3)
    SZ = size[:, None, :].expand(N, 8, 3) * mask         # (N,8,3)
    return O + SZ                                        # (N,8,3)


def _edges_from_corners(c: Tensor) -> Tensor:
    """
    c: (N,8,3) -> edges (N,12,2,3)
    Corner indexing:
      0:(0,0,0) 1:(1,0,0) 2:(0,1,0) 3:(1,1,0) 4:(0,0,1) 5:(1,0,1) 6:(0,1,1) 7:(1,1,1)
    """
    idx_pairs = torch.tensor([
        [0, 1], [2, 3], [4, 5], [6, 7],   # +X edges
        [0, 2], [1, 3], [4, 6], [5, 7],   # +Y edges
        [0, 4], [1, 5], [2, 6], [3, 7],   # +Z edges
    ], dtype=torch.long, device=c.device)                 # (12,2)
    a = c[:, idx_pairs[:, 0], :]  # (N,12,3)
    b = c[:, idx_pairs[:, 1], :]  # (N,12,3)
    segs = torch.stack([a, b], dim=2)  # (N,12,2,3)
    return segs


def _dist_point_to_segments(points: torch.Tensor, segs: torch.Tensor) -> torch.Tensor:
    """
    points: (M,3)
    segs:   expected (N,12,2,3)  but we reshape defensively.
    returns: (M,N,12) Euclidean distances to each segment
    """
    # ---- Defensive shape normalization ----
    if points.dim() != 2 or points.size(-1) != 3:
        raise ValueError(f"[points] expected (M,3), got {tuple(points.shape)}")

    if segs.dim() != 4:
        # Try to coerce: total elems must be divisible by (2*3)
        if segs.numel() % 6 != 0:
            raise ValueError(f"[segs] invalid numel={segs.numel()}, cannot reshape to (*,12,2,3)")
        # Heuristic: assume K=12 edges; infer N
        K = 12
        # infer N from total // (K*2*3)
        N = segs.numel() // (K * 2 * 3)
        segs = segs.view(N, K, 2, 3)
    else:
        # Ensure it is (N,12,2,3); if K!=12 we won't proceed
        N, K, two, three = segs.shape
        if K != 12 or two != 2 or three != 3:
            raise ValueError(f"[segs] expected (N,12,2,3), got {tuple(segs.shape)}")

    # Now guaranteed: points (M,3), segs (N,12,2,3)
    M = points.size(0)
    N = segs.size(0)
    K = segs.size(1)

    p = points.view(M, 1, 1, 3)         # (M,1,1,3)
    a = segs[:, :, 0, :].unsqueeze(0)   # (1,N,K,3)
    b = segs[:, :, 1, :].unsqueeze(0)   # (1,N,K,3)

    ab = b - a                          # (1,N,K,3)
    ap = p - a                          # (M,N,K,3)

    denom = (ab * ab).sum(dim=-1).clamp_min(1e-12)  # (1,N,K)
    t = (ap * ab).sum(dim=-1) / denom               # (M,N,K)
    t = t.clamp(0.0, 1.0).unsqueeze(-1)             # (M,N,K,1)

    proj = a + t * ab                               # (M,N,K,3)
    d = ((p - proj) ** 2).sum(dim=-1).sqrt()        # (M,N,K)
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
    Optimize cuboid sizes and per-part translations to fit stroke samples to edges.
    Writes updated sizes back to ../input/sketch_program_ir.json.
    Returns a fresh program_executor.Executor for the updated IR.
    """
    ir_path = input_dir / "sketch_program_ir.json"
    if not ir_path.exists():
        raise SystemExit(f"IR not found: {ir_path}")

    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dex = DifferentiableExecutor(ir, device=device).to(device)
    opt = torch.optim.Adam(dex.parameters(), lr=lr)

    # Prepare stroke points
    pts = flatten_points(sample_points).to(device)
    if pts.numel() == 0:
        print("No sample points found; skipping optimization.")
        return Executor(ir)

    # Weights
    w_edge, w_band = build_type_weights(feature_lines)

    # Cache initial sizes for regularization
    size_init = {k: v.to(device) for k, v in dex.sizes_init.items()}

    band_cap = 0.003  # surface band clamp
    for step in range(steps):
        opt.zero_grad()
        placed = dex()
        boxes = dex.pack_boxes(placed, skip_bbox=True)  # (N,6)

        # Main: pull stroke samples to *edges* of boxes
        d_edge = union_softmin_edges(pts, boxes, tau=0.02)   # (M,)
        loss_edge = d_edge.mean()

        # Light surface "band" to avoid drifting (symmetric near-surface pull)
        d_surf = union_softmin(pts, boxes, tau=0.02)         # (M,)
        loss_band = d_surf.abs().clamp_max(band_cap).mean()

        # Regularization (sizes near init; translations small)
        reg = torch.tensor(0.0, device=device)
        for name in dex.part_names:
            reg = reg + torch.mean((dex.sizes(name) - size_init[name])**2)
            reg = reg + 1e-2 * torch.mean(dex._dorigin[name]**2)
        reg = 1e-2 * reg

        loss = w_edge * loss_edge + w_band * loss_band + reg
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == steps - 1:
            print(f"[fit] step {step:03d} | "
                  f"loss={loss.item():.6f}  edge={loss_edge.item():.6f}  band={loss_band.item():.6f}  reg={reg.item():.6f}")

    # ---- Write updated sizes back into IR ----
    with torch.no_grad():
        for c in ir["program"].get("cuboids", []):
            name = c["var"]
            s = dex.sizes(name).cpu().numpy().tolist()
            c["l"], c["w"], c["h"] = float(s[0]), float(s[1]), float(s[2])

    ir_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
    print(f"✅ Optimized sizes written to {ir_path}")

    # NOTE: We don't write translations back into the IR to avoid breaking the attach graph.
    # If you want to persist them, we can (option A) bake small deltas into attach anchors,
    # or (option B) add a 'tweak' translate op per part. For now, we keep them for fitting only.
    # Quick debug print:
    with torch.no_grad():
        avg_off = {k: dex._dorigin[k].detach().cpu().numpy().tolist() for k in dex.part_names}
    print("[fit] learned per-part translation offsets (world coords):")
    for k, v in avg_off.items():
        print(f"  {k}: {v}")

    # Return a fresh classic Executor for downstream use/vis
    return Executor(ir)
