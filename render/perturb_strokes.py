import numpy as np
import copy
import random
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection


def do_perturb(stroke_node_features, perturb_factor=0.002):
    """
    Step 0 (skeleton):
    - Walk each stroke, detect its type (row[9]).
    - For now, do nothing (pass) in every case.
    - Return [] per your instruction.

    Final (future) output format per stroke:
      a list of 10 points, each point is [x, y, z],
      i.e. [[x1,y1,z1], ..., [x10,y10,z10]]
    """
    result = []  # For now, empty as all functions are pass

    for stroke in stroke_node_features:
        t = stroke[9]

        if t == 1:
            # Straight Line
            result.append(perturb_straight_line(stroke))

        elif t == 2:
            # Circle
            result.append(perturb_circle(stroke))

        elif t == 3:
            # Cylinder face
            result.extend(perturb_cylinder_face(stroke))

        elif t == 4:
            # Arc
            result.append(perturb_arc(stroke))

        elif t == 5:
            # Spline
            result.append(perturb_spline(stroke))

        elif t == 6:
            # Sphere
            pass
        else:
            # Unknown type
            pass

    return result



def perturb_straight_line(stroke, rng=None):
    """
    Adapted from your reference logic for straight lines.

    Input:
        stroke: [x1,y1,z1, x2,y2,z2, 0,0,0, 1]

    Output:
        list of 10 points, each [x,y,z]
    """
    if rng is None:
        rng = random

    # ------ vector helpers ------
    def v_add(a, b):   return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def v_sub(a, b):   return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
    def v_mul(a, s):   return [a[0]*s, a[1]*s, a[2]*s]
    def v_len(a):      return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    def v_norm(a):
        L = v_len(a)
        return [1.0, 0.0, 0.0] if L < 1e-12 else [a[0]/L, a[1]/L, a[2]/L]
    def v_lerp(a, b, t):
        return [a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t, a[2]*(1-t)+b[2]*t]
    def gauss3(scale):
        return [rng.gauss(0.0, scale), rng.gauss(0.0, scale), rng.gauss(0.0, scale)]
    def uniform3(a, b):
        return [rng.uniform(a, b), rng.uniform(a, b), rng.uniform(a, b)]

    # ------ parse endpoints ------
    p1 = [float(stroke[0]), float(stroke[1]), float(stroke[2])]
    p2 = [float(stroke[3]), float(stroke[4]), float(stroke[5])]
    L = v_len(v_sub(p2, p1))
    if L < 1e-12:
        # Degenerate: just return 10 identical points
        return [p1[:] for _ in range(10)]

    # ------ cad2sketch-style random strengths ------
    point_jitter_ratio   = rng.uniform(0.0005, 0.001)
    endpoint_shift_ratio = rng.uniform(0.008, 0.012)
    overdraw_ratio       = rng.uniform(0.04, 0.08)

    point_jitter   = point_jitter_ratio * L
    endpoint_shift = endpoint_shift_ratio * L
    overdraw       = overdraw_ratio * L

    # ------ create a small polyline (5 evenly spaced points) ------
    base_ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    pts = [v_lerp(p1, p2, t) for t in base_ts]

    # ------ perturb original geometry ------
    # endpoints: uniform cube shift; interior: Gaussian per axis
    for i in range(len(pts)):
        if i == 0 or i == len(pts)-1:
            shift = uniform3(-endpoint_shift, endpoint_shift)
        else:
            shift = gauss3(point_jitter)
        pts[i] = v_add(pts[i], shift)

    # ------ overdraw at both ends ------
    v_start = v_sub(pts[1], pts[0])
    v_end   = v_sub(pts[-2], pts[-1])
    v_start = v_norm(v_start)
    v_end   = v_norm(v_end)
    pts[0]  = v_sub(pts[0],  v_mul(v_start, overdraw))
    pts[-1] = v_sub(pts[-1], v_mul(v_end,   overdraw))

    # ------ resample 10 evenly spaced points between new endpoints ------
    start = pts[0]
    end   = pts[-1]
    ts = [i/9.0 for i in range(10)]
    resampled = [v_lerp(start, end, t) for t in ts]

    # ------ jitter interior points only ------
    for i in range(1, len(resampled)-1):
        resampled[i] = v_add(resampled[i], gauss3(point_jitter))

    return resampled



def perturb_circle(stroke, start_angle=0.0, rng=None):
    """
    Perturb/synthesize a hand-drawn-looking CLOSED circle polyline (10 points).
    Input stroke: [cx,cy,cz, nx,ny,nz, 0, radius, 0, 2]  (radius at index 7)
    Returns: list of 10 points [x,y,z], with pts[-1] == pts[0].
    """
    if rng is None:
        rng = random

    # --- parse ---
    cx, cy, cz = float(stroke[0]), float(stroke[1]), float(stroke[2])
    nx, ny, nz = float(stroke[3]), float(stroke[4]), float(stroke[5])
    R = float(stroke[7])

    # --- vec utils ---
    def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    def add(a, b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def mul(a, s): return [a[0]*s, a[1]*s, a[2]*s]
    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    def norm(a):
        L2 = dot(a, a)
        if L2 <= 0.0:
            return [0.0, 0.0, 1.0]
        L = math.sqrt(L2)
        return [a[0]/L, a[1]/L, a[2]/L]

    center = [cx, cy, cz]
    n = norm([nx, ny, nz])
    if abs(n[0]) + abs(n[1]) + abs(n[2]) < 1e-12:
        n = [0.0, 0.0, 1.0]

    # --- build in-plane orthonormal basis (right-handed) ---
    ref = [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = norm(cross(n, ref))
    if abs(u[0]) + abs(u[1]) + abs(u[2]) < 1e-12:
        ref = [0.0, 1.0, 0.0]
        u = norm(cross(n, ref))
    v = cross(n, u)

    if R <= 1e-12:
        # Degenerate: return closed “point circle”
        p = center[:]
        return [p, p, p, p, p, p, p, p, p, p]

    # --- ellipse-ish params & jitter (non-uniform angles) ---
    rx = R * rng.uniform(0.9, 1.1)
    ry = R * rng.uniform(0.9, 1.1)
    phi = rng.uniform(0.0, 2.0 * math.pi)             # in-plane rotation
    jitter_2d = rng.uniform(0.001, 0.004) * R         # wobble in plane
    angle_jitter = 0.15                                # radians; < (2π/9)/2 so order stays increasing

    cphi, sphi = math.cos(phi), math.sin(phi)

    # --- 9 slightly non-uniform angles (then close with first) ---
    pts = []
    for i in range(9):  # 0..8
        base = start_angle + 2.0 * math.pi * (i / 9.0)
        t = base + rng.uniform(-angle_jitter, angle_jitter)

        # ellipse in local 2D (then rotate by phi)
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        xr = cphi * x - sphi * y
        yr = sphi * x + cphi * y

        # in-plane jitter
        xr += rng.gauss(0.0, jitter_2d)
        yr += rng.gauss(0.0, jitter_2d)

        # back to 3D
        p = add(center, add(mul(u, xr), mul(v, yr)))
        pts.append(p)

    # optional gentle seam smoothing before closing
    seam_shift_len = rng.uniform(0.05, 0.1) * R
    seam_shift = [rng.gauss(0.0, seam_shift_len),
                  rng.gauss(0.0, seam_shift_len),
                  rng.gauss(0.0, seam_shift_len)]
    for idx, w in zip([8, 7, 6], [1.0, 0.8, 0.6]):
        pts[idx] = add(pts[idx], mul(seam_shift, w))

    # close the loop
    pts.append(pts[0][:])
    return pts



def perturb_cylinder_face(stroke, rng=None):
    """
    Build & perturb the 4 boundary lines of a cylinder face.

    Input stroke format:
      [lx,ly,lz, ux,uy,uz, 0, radius, 0, 3]
      where (lx,ly,lz) = lower circle center, (ux,uy,uz) = upper circle center

    Returns:
      [
        [ [x,y,z] * 10 ],  # line 1
        [ [x,y,z] * 10 ],  # line 2
        [ [x,y,z] * 10 ],  # line 3
        [ [x,y,z] * 10 ]   # line 4
      ]
    """
    if rng is None:
        rng = random

    # ---- tiny vec helpers ----
    def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    def add(a,b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def sub(a,b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
    def mul(a,s): return [a[0]*s, a[1]*s, a[2]*s]
    def norm(a):
        L2 = dot(a,a)
        if L2 <= 0.0: return [0.0,0.0,1.0]
        L = math.sqrt(L2);  return [a[0]/L, a[1]/L, a[2]/L]
    def cross(a,b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    # ---- parse cylinder ----
    lx, ly, lz = float(stroke[0]), float(stroke[1]), float(stroke[2])
    ux, uy, uz = float(stroke[3]), float(stroke[4]), float(stroke[5])
    R = float(stroke[7])

    lower = [lx, ly, lz]
    upper = [ux, uy, uz]
    axis_vec = sub(upper, lower)
    L = math.sqrt(dot(axis_vec, axis_vec))
    if L <= 1e-12:
        axis_vec = [0.0, 0.0, 1.0]; L = 1.0
        upper = add(lower, axis_vec)

    w = norm(axis_vec)  # cylinder axis direction

    # Orthonormal in-plane basis (u, v), perpendicular to axis
    ref = [1.0, 0.0, 0.0] if abs(w[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = norm(cross(w, ref))
    if abs(u[0]) + abs(u[1]) + abs(u[2]) < 1e-12:
        ref = [0.0, 1.0, 0.0]
        u = norm(cross(w, ref))
    v = cross(w, u)  # right-handed: u × v = w

    # Base angles at quarter turns, with a tiny random offset to avoid perfect symmetry
    base_angles = [0.0, math.pi/2, math.pi, 3*math.pi/2]
    jitter = math.pi / 36.0  # ±5 degrees
    start_angle = rng.uniform(0.0, 2.0*math.pi)  # random phase
    angles = [start_angle + a + rng.uniform(-jitter, jitter) for a in base_angles]

    # Fallback if radius too small: just return 4 perturbed axis lines
    if R <= 1e-12:
        return [
            perturb_straight_line([*lower, *upper, 0,0,0, 1], rng=rng),
            perturb_straight_line([*lower, *upper, 0,0,0, 1], rng=rng),
            perturb_straight_line([*lower, *upper, 0,0,0, 1], rng=rng),
            perturb_straight_line([*lower, *upper, 0,0,0, 1], rng=rng),
        ]

    lines = []
    for a in angles:
        # point on lower/upper circles at angle a
        offset = add(mul(u, R*math.cos(a)), mul(v, R*math.sin(a)))
        p_low = add(lower, offset)
        p_up  = add(upper,  offset)

        # build a "straight line stroke" and perturb using your approved routine
        stroke_line = [p_low[0], p_low[1], p_low[2], p_up[0], p_up[1], p_up[2], 0,0,0, 1]
        line_pts = perturb_straight_line(stroke_line, rng=rng)  # expects to return 10 [x,y,z]
        lines.append(line_pts)

    return lines



def perturb_arc(stroke, arc_fraction=None,
                noise_scale_ratio=0.0001,
                endpoint_shift_ratio=0.002,
                rng=None):
    """
    Input (arc stroke of 10 values):
      [cx,cy,cz, nx,ny,nz, radius, angle_start, sweep, 4]
    Output:
      list of 10 points [x,y,z] along a perturbed arc, preserving direction.
    """
    if rng is None:
        rng = random

    # ---- parse ----
    cx, cy, cz = float(stroke[0]), float(stroke[1]), float(stroke[2])
    nx, ny, nz = float(stroke[3]), float(stroke[4]), float(stroke[5])
    R          = float(stroke[6])
    a0         = float(stroke[7])   # start angle (radians)
    sw         = float(stroke[8])   # sweep (radians), sign = direction

    # ---- vec helpers (no numpy) ----
    def dot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    def add(a,b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def sub(a,b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
    def mul(a,s): return [a[0]*s, a[1]*s, a[2]*s]
    def norm(a):
        L2 = dot(a,a)
        if L2 <= 0.0: return [0.0,0.0,1.0]
        L = math.sqrt(L2); return [a[0]/L, a[1]/L, a[2]/L]
    def cross(a,b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    def gauss3(s): return [rng.gauss(0.0, s), rng.gauss(0.0, s), rng.gauss(0.0, s)]

    center = [cx, cy, cz]
    n = norm([nx, ny, nz])

    # in-plane orthonormal basis (right-handed: u × v = n)
    ref = [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = norm(cross(n, ref))
    if abs(u[0]) + abs(u[1]) + abs(u[2]) < 1e-12:
        ref = [0.0, 1.0, 0.0]
        u = norm(cross(n, ref))
    v = cross(n, u)

    # guard: tiny radius or zero sweep → return degenerate straight-ish sample
    N = 10
    if R <= 1e-12 or abs(sw) <= 1e-12:
        # sample a straight segment of near-zero length at center
        p = center[:]
        return [p[:] for _ in range(N)]

    # --- randomize strengths (like your interpolation reference) ---
    if arc_fraction is None:
        arc_fraction = rng.uniform(0.5, 1.3)  # 1.0 ~ original arc, <1 blends straighter
    noise_scale   = R * noise_scale_ratio
    endpoint_shift= R * endpoint_shift_ratio

    # --- non-uniform, but monotonic arc parameters in [0,1] ---
    # jitter interior parameters slightly, keep sorted to preserve direction
    ts = [0.0] + sorted(max(0.0, min(1.0, i/(N-1) + rng.uniform(-0.06, 0.06)))
                        for i in range(1, N-1)) + [1.0]

    # --- sample ideal arc points (respect sign of sweep) ---
    arc_pts = []
    for t in ts:
        ang = a0 + sw * t
        x = R * math.cos(ang)
        y = R * math.sin(ang)
        arc_pts.append(add(center, add(mul(u, x), mul(v, y))))

    # --- endpoints & chord for scaling ---
    start = arc_pts[0][:]
    end   = arc_pts[-1][:]
    chord_len = math.sqrt(max(1e-18, dot(sub(end, start), sub(end, start))))

    # shift endpoints a bit
    start = add(start, gauss3(endpoint_shift))
    end   = add(end,   gauss3(endpoint_shift))

    # --- blend arc with straight line + interior jitter ---
    blended = []
    for j, t in enumerate(ts):
        line_pt = add(mul(start, 1.0 - t), mul(end, t))
        arc_pt  = arc_pts[j]
        p = add(mul(line_pt, 1.0 - arc_fraction), mul(arc_pt, arc_fraction))
        if 0 < j < N-1:
            # interior jitter
            p = add(p, gauss3(noise_scale))
        blended.append(p)

    return blended






def perturb_spline(stroke, samples=10, curvature_boost=1.8, max_mid_offset_ratio=0.75):
    """
    Make a 3-CP spline (quadratic Bézier) visibly curvier by boosting the middle control.

    Input:
      stroke = [x0,y0,z0,  x1,y1,z1,  x2,y2,z2,  5]

    Params:
      samples               : number of points to sample (default 10)
      curvature_boost       : how much to push the middle control away from chord
      max_mid_offset_ratio  : clamp for |P1' - midpoint| relative to chord length

    Returns:
      list of `samples` points [x,y,z] along the boosted quadratic Bézier.
    """
    # --- parse controls ---
    P0 = [float(stroke[0]), float(stroke[1]), float(stroke[2])]
    P1 = [float(stroke[3]), float(stroke[4]), float(stroke[5])]
    P2 = [float(stroke[6]), float(stroke[7]), float(stroke[8])]

    # --- tiny vec utils ---
    def add(a,b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def sub(a,b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
    def mul(a,s): return [a[0]*s, a[1]*s, a[2]*s]
    def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    def cross(a,b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    def length(a): 
        from math import sqrt
        return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
    def norm(a):
        L = length(a)
        return [0.0,0.0,1.0] if L < 1e-12 else [a[0]/L, a[1]/L, a[2]/L]

    # --- chord / midpoint ---
    chord = sub(P2, P0)
    L = length(chord)
    if L < 1e-12:
        # Degenerate: return all the same point
        return [P0[:] for _ in range(max(2, samples))]

    M = mul(add(P0, P2), 0.5)
    w = sub(P1, M)  # P1 offset from chord midpoint

    # If P1 is too close to chord, synthesize a perpendicular direction
    if length(w) < 0.02 * L:
        u = norm(chord)
        # pick a ref not parallel to u
        ref = [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0]
        n = norm(cross(u, ref))
        w_dir = norm(cross(n, u))  # perpendicular to chord
        w = mul(w_dir, 0.25 * L)   # give it some curvature baseline

    # Boost curvature and clamp
    w = mul(w, curvature_boost)
    max_off = max_mid_offset_ratio * L
    if length(w) > max_off:
        w = mul(norm(w), max_off)

    P1_boost = add(M, w)

    # --- sample quadratic Bézier with boosted middle ---
    if samples < 2:
        samples = 2

    pts = []
    for i in range(samples):
        t = i / (samples - 1)
        one_t = 1.0 - t
        b0 = one_t * one_t
        b1 = 2.0 * one_t * t
        b2 = t * t
        p = add(add(mul(P0, b0), mul(P1_boost, b1)), mul(P2, b2))
        pts.append(p)

    return pts






# ---------------------------------------------------------------------------------------- #

def vis_perturbed_strokes(perturbed_feature_lines, perturbed_construction_lines, *,
                          color="black", linewidth=0.8, show=True):
    """
    Visualize perturbed strokes with equal scaling across x/y/z.

    Parameters
    ----------
    perturbed_feature_lines : list[list[[x,y,z], ...]] or nested lists
        Each item is a polyline (list of 3D points) or a list of polylines.
    perturbed_construction_lines : list[list[[x,y,z], ...]] or nested lists
        Each item is a polyline (list of 3D points) or a list of polylines.
    color : str
        Line color for all strokes.
    linewidth : float
        Base width for feature lines. Construction lines are thinner.
    show : bool
        If True, calls plt.show() at the end.

    Notes
    -----
    - Feature lines use alpha in [0.9, 1.0]
    - Construction lines use alpha in [0.2, 0.5] and are thinner
    """

    # ---------- helpers ----------
    def is_point(p):
        return isinstance(p, (list, tuple)) and len(p) == 3 and all(isinstance(v, (int, float)) for v in p)

    def is_polyline(obj):
        return isinstance(obj, (list, tuple)) and len(obj) > 0 and is_point(obj[0])

    def flatten_polylines(obj):
        out = []
        if is_polyline(obj):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                out.extend(flatten_polylines(item))
        return out

    feat_lines = flatten_polylines(perturbed_feature_lines)
    cons_lines = flatten_polylines(perturbed_construction_lines)

    if not feat_lines and not cons_lines:
        raise ValueError("You must provide at least one polyline in feature or construction lines.")

    # ---------- bounds ----------
    x_min = y_min = z_min = float("inf")
    x_max = y_max = z_max = float("-inf")

    def update_bounds(lines):
        nonlocal x_min, y_min, z_min, x_max, y_max, z_max
        for pts in lines:
            for x, y, z in pts:
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if z < z_min: z_min = z
                if x > x_max: x_max = x
                if y > y_max: y_max = y
                if z > z_max: z_max = z

    update_bounds(feat_lines)
    update_bounds(cons_lines)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_diff == 0:
        max_diff = 1.0
    half = max_diff / 2.0

    # ---------- plot ----------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.grid(False)

    # Feature lines: thicker, alpha 0.9–1.0
    for pts in feat_lines:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        ax.plot(xs, ys, zs, color=color,
                linewidth=linewidth,
                alpha=random.uniform(0.9, 1.0))

    # Construction lines: thinner, alpha 0.2–0.5
    cons_width = max(0.1, 0.6 * linewidth)  # thinner than feature lines
    for pts in cons_lines:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        ax.plot(xs, ys, zs, color=color,
                linewidth=cons_width,
                alpha=random.uniform(0.2, 0.5))

    # Equalize axes
    ax.set_xlim([x_center - half, x_center + half])
    ax.set_ylim([y_center - half, y_center + half])
    ax.set_zlim([z_center - half, z_center + half])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=100, azim=45)

    if show:
        plt.show()

    return fig, ax



