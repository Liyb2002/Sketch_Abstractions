import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

import random

# stroke types:
# 1)Straight Line: Point_1 (3 value), Point_2 (3 value), 0, 0, 0, 1
# 2)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 3)Cylinder face: Center (3 value), normal (3 value), height, radius, 0, 3
# 4)Arc: Start S (3 values), End E (3 values), Center C (3 values), 4
# 5)Spline: Control_point_1 (3 value), Control_point_2 (3 value), Control_point_3 (3 value), 5
# 6)Sphere: center_x, center_y, center_z, axis_nx,  axis_ny,  axis_nz, 0,        radius,   0,     6


def vis_stroke_node_features(stroke_node_features):
    # Initialize the 3D plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')  # Turn off axis background and borders

    # Initialize min and max limits
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')


    # Plot all strokes in blue with perturbations
    for idx, stroke in enumerate(stroke_node_features):
        start, end = stroke[:3], stroke[3:6]
        

        # Update min and max limits based on strokes (ignoring circles)
        if stroke[-1] == 1:
            # Straight line: [start(x,y,z), end(x,y,z), 0,0,0, 1]
            start = np.array(stroke[0:3], dtype=float)
            end   = np.array(stroke[3:6], dtype=float)

            x_values = [start[0], end[0]]
            y_values = [start[1], end[1]]
            z_values = [start[2], end[2]]

            # Update bounds
            x_min, x_max = min(x_min, *x_values), max(x_max, *x_values)
            y_min, y_max = min(y_min, *y_values), max(y_max, *y_values)
            z_min, z_max = min(z_min, *z_values), max(z_max, *z_values)

            ax.plot(x_values, y_values, z_values, color='black', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 2:
            # Circle: [center(3), normal(3), 0, radius, 0, 2]
            cx, cy, cz, nx, ny, nz, _, r, _ = (float(v) for v in stroke[:9])
            center = np.array([cx, cy, cz], dtype=float)
            normal = np.array([nx, ny, nz], dtype=float)

            # Normalize normal; if degenerate, skip
            nlen = np.linalg.norm(normal)
            if nlen < 1e-12:
                continue
            normal /= nlen

            # Build in-plane orthonormal basis (xdir, ydir) ⟂ normal
            up = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(normal, up)
            if np.linalg.norm(xdir) < 1e-12:
                up = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(normal, up)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(normal, xdir)

            # Sample full circle
            theta = np.linspace(0.0, 2.0 * np.pi, 200)
            pts = center[None, :] + r * (np.cos(theta)[:, None] * xdir + np.sin(theta)[:, None] * ydir)
            x_values, y_values, z_values = pts[:, 0], pts[:, 1], pts[:, 2]

            # (By your convention, we do NOT update bounds for circles)
            ax.plot(x_values, y_values, z_values, color='black', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 3:
            # Cylinder face: [lower_center(3), upper_center(3), 0.0, radius, 0.0, 3]
            # total length = 10; last is type code (3)
            lx, ly, lz, ux, uy, uz, _zero0, r, _zero1 = (float(v) for v in stroke[:9])

            L = np.array([lx, ly, lz], dtype=float)  # lower circle center
            U = np.array([ux, uy, uz], dtype=float)  # upper circle center
            axis_vec = U - L
            h = np.linalg.norm(axis_vec)             # cylinder height

            if h < 1e-12 or r <= 0.0:
                continue

            # unit axis direction (lower -> upper)
            n = axis_vec / h

            # build orthonormal basis in the rim plane (perpendicular to n)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(n, ref)
            if np.linalg.norm(xdir) < 1e-12:
                ref = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(n, ref)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(n, xdir)

            # four boundary directions at 0°, 90°, 180°, 270°
            for ang in (0.0, 0.5*np.pi, np.pi, 1.5*np.pi):
                radial = np.cos(ang)*xdir + np.sin(ang)*ydir

                # corresponding boundary points on lower/upper rims
                p_low = L + r * radial
                p_up  = U + r * radial

                ax.plot([p_low[0], p_up[0]],
                        [p_low[1], p_up[1]],
                        [p_low[2], p_up[2]],
                        color='black', alpha=1, linewidth=0.5)

                # bounds
                x_min = min(x_min, p_low[0], p_up[0]); x_max = max(x_max, p_low[0], p_up[0])
                y_min = min(y_min, p_low[1], p_up[1]); y_max = max(y_max, p_low[1], p_up[1])
                z_min = min(z_min, p_low[2], p_up[2]); z_max = max(z_max, p_low[2], p_up[2])
            continue
            
        if stroke[-1] == 4:
            # Arc encoded as: [sx,sy,sz, ex,ey,ez, cx,cy,cz, 4]
            sx, sy, sz, ex, ey, ez, cx, cy, cz = (float(v) for v in stroke[:9])
            S = np.array([sx, sy, sz])
            E = np.array([ex, ey, ez])
            C = np.array([cx, cy, cz])

            vS = S - C; vE = E - C
            rS = np.linalg.norm(vS); rE = np.linalg.norm(vE)
            r  = 0.5 * (rS + rE)
            if r < 1e-12:
                continue

            vS /= rS; vE /= rE  # unit
            # normal from start to end (right-hand)
            n = np.cross(vS, vE)
            nlen = np.linalg.norm(n)
            if nlen < 1e-12:
                # degenerate (collinear) — just draw a line
                ax.plot([sx, ex], [sy, ey], [sz, ez], color='black', alpha=1, linewidth=0.5)
                continue
            n /= nlen

            # In-plane basis: xdir along start vector, ydir = n × xdir
            xdir = vS
            ydir = np.cross(n, xdir)

            # signed sweep from S to E
            cosang = np.clip(np.dot(vS, vE), -1.0, 1.0)
            sweep  = np.arccos(cosang)
            # orientation sign
            if np.dot(n, np.cross(vS, vE)) < 0:
                sweep = -sweep

            # sample (should be ±pi/2 for quarter arcs)
            theta = np.linspace(0.0, sweep, 100)
            pts = C + r*(np.cos(theta)[:,None]*xdir + np.sin(theta)[:,None]*ydir)
            x_values, y_values, z_values = pts[:,0], pts[:,1], pts[:,2]

            # update bounds
            x_min, x_max = min(x_min, x_values.min()), max(x_max, x_values.max())
            y_min, y_max = min(y_min, y_values.min()), max(y_max, y_values.max())
            z_min, z_max = min(z_min, z_values.min()), max(z_max, z_values.max())

            ax.plot(x_values, y_values, z_values, color='black', alpha=1, linewidth=0.5)
            continue
        

        if stroke[-1] == 5:
            # Spline encoded as 3 control points + type 5
            p0 = np.array(stroke[0:3], dtype=float)
            p1 = np.array(stroke[3:6], dtype=float)
            p2 = np.array(stroke[6:9], dtype=float)

            # Quadratic Bézier sampling: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
            t = np.linspace(0.0, 1.0, 100)
            one_minus_t = 1.0 - t
            bez_x = (one_minus_t**2) * p0[0] + 2 * one_minus_t * t * p1[0] + (t**2) * p2[0]
            bez_y = (one_minus_t**2) * p0[1] + 2 * one_minus_t * t * p1[1] + (t**2) * p2[1]
            bez_z = (one_minus_t**2) * p0[2] + 2 * one_minus_t * t * p1[2] + (t**2) * p2[2]

            # Update bounds
            x_min, x_max = min(x_min, bez_x.min()), max(x_max, bez_x.max())
            y_min, y_max = min(y_min, bez_y.min()), max(y_max, bez_y.max())
            z_min, z_max = min(z_min, bez_z.min()), max(z_max, bez_z.max())

            # Plot the spline
            ax.plot(bez_x, bez_y, bez_z, color='black', alpha=1, linewidth=0.5)
            continue

        if stroke[-1] == 6:
            # Sphere: [cx,cy,cz, nx,ny,nz, 0, r, 0, 6]
            cx, cy, cz, nx, ny, nz, _, r, _ = (float(v) for v in stroke[:9])
            C = np.array([cx, cy, cz], dtype=float)
            n = np.array([nx, ny, nz], dtype=float)

            # Normalize axis
            nlen = np.linalg.norm(n)
            if nlen < 1e-12:
                continue
            n /= nlen

            # Build an orthonormal basis in the equatorial plane (xdir, ydir) ⟂ n
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            xdir = np.cross(n, ref)
            if np.linalg.norm(xdir) < 1e-12:
                ref = np.array([0.0, 1.0, 0.0])
                xdir = np.cross(n, ref)
            xdir /= np.linalg.norm(xdir)
            ydir = np.cross(n, xdir)

            # Four plane normals for four great circles
            normals = [
                n,                                   # equator
                xdir,                                # meridian 1
                ydir,                                # meridian 2
                (xdir + ydir) / np.linalg.norm(xdir + ydir)  # tilted meridian (45°)
            ]

            theta = np.linspace(0.0, 2.0*np.pi, 200)
            c, s = np.cos(theta), np.sin(theta)

            for pn in normals:
                # Build in-plane basis (u,v) for this circle plane (normal = pn)
                # Choose a ref not parallel to pn
                ref2 = np.array([0.0, 0.0, 1.0]) if abs(pn[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
                u = np.cross(pn, ref2)
                if np.linalg.norm(u) < 1e-12:
                    ref2 = np.array([0.0, 1.0, 0.0])
                    u = np.cross(pn, ref2)
                u /= np.linalg.norm(u)
                v = np.cross(pn, u)

                pts = C[None, :] + r * (c[:, None]*u[None, :] + s[:, None]*v[None, :])
                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

                # update bounds
                x_min = min(x_min, x.min()); x_max = max(x_max, x.max())
                y_min = min(y_min, y.min()); y_max = max(y_max, y.max())
                z_min = min(z_min, z.min()); z_max = max(z_max, z.max())

                ax.plot(x, y, z, color='black', alpha=1, linewidth=0.5)
            continue


    # Compute the center and rescale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    max_diff = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_xlim([x_center - max_diff / 2, x_center + max_diff / 2])
    ax.set_ylim([y_center - max_diff / 2, y_center + max_diff / 2])
    ax.set_zlim([z_center - max_diff / 2, z_center + max_diff / 2])



    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Show plot
    plt.show()



def compute_global_threshold(feature_lines):
    """
    Compute global_threshold from a list of strokes (feature_lines).

    Format (straight line only):
      [x1, y1, z1, x2, y2, z2, 0, 0, 0, 1]

    Returns:
      float: avg(straight_line_lengths) * 0.1
             Returns 0.0 if no straight lines are found.
    """
    lengths = []

    for stroke in feature_lines:
        # minimal shape & straight-line flag
        if not stroke or len(stroke) < 10:
            continue
        flag = stroke[-1]
        if isinstance(flag, (int, float)) and abs(flag - 1) < 1e-9:
            x1, y1, z1, x2, y2, z2 = stroke[0], stroke[1], stroke[2], stroke[3], stroke[4], stroke[5]
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length > 0:
                lengths.append(length)

    if not lengths:
        return 0.0  # no straight lines found

    avg_len = sum(lengths) / len(lengths)
    return avg_len * 0.08




def vis_perturbed_strokes(
    perturbed_feature_lines,
    perturbed_construction_lines,
    *,
    color="black",
    linewidth=0.8,
    show=True,
    ax=None,
    elev=100,
    azim=45,
    deterministic_alpha=False,
    return_bounds=False,
):
    """
    Visualize perturbed strokes with equal scaling across x/y/z.

    Parameters
    ----------
    perturbed_feature_lines : polyline | line_group | nested list
        - polyline: [[x,y,z], [x,y,z], ...]
        - line_group: [polyline, polyline, ...] (nesting allowed)
    perturbed_construction_lines : same as above
    color : str
        Line color for all strokes.
    linewidth : float
        Base width for feature lines. Construction lines are thinner.
    show : bool
        If True, calls plt.show() (only if we created the axes here).
    ax : mpl_toolkits.mplot3d axes or None
        Provide to draw on an existing 3D axes (e.g., for multi-view screenshots).
    elev, azim : float
        Camera angles for view_init.
    deterministic_alpha : bool
        If True, uses fixed alphas (features=1.0, constructions=0.35).
    return_bounds : bool
        If True, returns (fig, ax, (x_min, x_max, y_min, y_max, z_min, z_max)).
        Otherwise returns (fig, ax) for backward compatibility.
    """

    # ---------- helpers ----------
    def is_number(v):
        return isinstance(v, (int, float))

    def is_point(p):
        return (
            isinstance(p, (list, tuple))
            and len(p) == 3
            and all(is_number(v) for v in p)
        )

    def is_polyline(obj):
        # A non-empty sequence of points
        return (
            isinstance(obj, (list, tuple))
            and len(obj) > 0
            and all(is_point(p) for p in obj)
        )

    def flatten_polylines(obj):
        """
        Recursively collect all polylines from:
        - a single polyline
        - a line group: list/tuple of polylines or nested groups
        - arbitrarily nested structures mixing the above
        """
        out = []
        if obj is None:
            return out
        if is_polyline(obj):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                out.extend(flatten_polylines(item))
        else:
            # ignore scalars/unknowns silently here; validation happens later
            pass
        return out

    feat_lines = flatten_polylines(perturbed_feature_lines)
    cons_lines = flatten_polylines(perturbed_construction_lines)

    # Validate that inputs were structurally OK (i.e., contained at least one polyline)
    if not feat_lines and not cons_lines:
        raise ValueError(
            "No valid polylines found. Provide a polyline [[x,y,z], ...] or a line group "
            "[polyline, polyline, ...] (nesting allowed)."
        )

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
    created_here = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        created_here = True
        ax.set_axis_off()
        ax.grid(False)
    else:
        fig = ax.get_figure()

    # Feature lines: thicker, alpha 0.9–1.0 (or fixed)
    feat_alpha = 1.0 if deterministic_alpha else None
    for pts in feat_lines:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        ax.plot(
            xs, ys, zs,
            color=color,
            linewidth=linewidth,
            alpha=(feat_alpha if feat_alpha is not None else random.uniform(0.9, 1.0))
        )

    # Construction lines: thinner, alpha 0.2–0.5 (or fixed)
    cons_width = max(0.1, 0.6 * linewidth)
    cons_alpha = 0.35 if deterministic_alpha else None
    for pts in cons_lines:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        ax.plot(
            xs, ys, zs,
            color=color,
            linewidth=cons_width,
            alpha=(cons_alpha if cons_alpha is not None else random.uniform(0.2, 0.5))
        )

    # Equalize axes
    ax.set_xlim([x_center - half, x_center + half])
    ax.set_ylim([y_center - half, y_center + half])
    ax.set_zlim([z_center - half, z_center + half])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=elev, azim=azim)

    if show and created_here:
        plt.show()

    if return_bounds:
        return fig, ax, (x_min, x_max, y_min, y_max, z_min, z_max)
    return fig, ax

