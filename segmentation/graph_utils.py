import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


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
