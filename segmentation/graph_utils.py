import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

import random

# stroke types:
# 1)Straight Line: Point_1 (3 value), Point_2 (3 value), 0, 0, 0, 1
# 2)Cicles: Center (3 value), normal (3 value), 0, radius, 0, 2
# 3)Cylinder face: [lower_center(3), upper_center(3), 0.0, radius, 0.0, 3]
# 4)Arc: Start S (3 values), End E (3 values), Center C (3 values), 4
# 5)Spline: Control_point_1 (3 value), Control_point_2 (3 value), Control_point_3 (3 value), 5
# 6)Sphere: center_x, center_y, center_z, axis_nx,  axis_ny,  axis_nz, 0,        radius,   0,     6



def intersection_pairs(sample_points, feature_lines, global_thresh):
    """
    Return all (i, j) stroke index pairs (i < j) whose sampled points come within
    'global_thresh' Euclidean distance of each other.

    sample_points[i] can be:
      - [ (x,y,z), (x,y,z), ... ]                          # flat list of points
      - [ [(x,y,z),...], [(x,y,z),...], ... ]              # list of polylines, each a list of points

    feature_lines is accepted for interface consistency (unused here).
    """
    n = len(sample_points)
    thr = float(global_thresh)
    thr2 = thr * thr

    # Helper: flatten one stroke's points, regardless of nesting (1 level max)
    def _flatten(pts):
        flat = []
        if not pts:
            return flat
        first = pts[0] if isinstance(pts, (list, tuple)) and pts else None

        # Case A: flat list of points
        if isinstance(first, (list, tuple)) and len(first) == 3:
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) == 3:
                    flat.append((float(p[0]), float(p[1]), float(p[2])))
            return flat

        # Case B: list of polylines (each a list of points)
        for seg in pts:
            if not seg:
                continue
            if isinstance(seg, (list, tuple)) and isinstance(seg[0], (list, tuple)) and len(seg[0]) == 3:
                for p in seg:
                    if isinstance(p, (list, tuple)) and len(p) == 3:
                        flat.append((float(p[0]), float(p[1]), float(p[2])))
        return flat

    # Pre-flatten points per stroke and build padded AABBs for quick pruning
    flat_pts = []
    bboxes = []  # (xmin, ymin, zmin, xmax, ymax, zmax) padded by thr
    for pts in sample_points:
        fp = _flatten(pts)
        flat_pts.append(fp)
        if not fp:
            bboxes.append((0.0, 0.0, 0.0, -1.0, -1.0, -1.0))  # empty bbox
            continue
        x0 = y0 = z0 = float('inf')
        x1 = y1 = z1 = float('-inf')
        for (x, y, z) in fp:
            if x < x0: x0 = x
            if y < y0: y0 = y
            if z < z0: z0 = z
            if x > x1: x1 = x
            if y > y1: y1 = y
            if z > z1: z1 = z
        bboxes.append((x0 - thr, y0 - thr, z0 - thr, x1 + thr, y1 + thr, z1 + thr))

    def _boxes_overlap(a, b):
        # Axis-aligned overlap
        return not (
            a[3] < b[0] or b[3] < a[0] or  # x
            a[4] < b[1] or b[4] < a[1] or  # y
            a[5] < b[2] or b[5] < a[2]     # z
        )

    pairs = []
    for i in range(n):
        pi = flat_pts[i]
        if not pi:
            continue
        bi = bboxes[i]
        for j in range(i + 1, n):
            pj = flat_pts[j]
            if not pj:
                continue
            if not _boxes_overlap(bi, bboxes[j]):
                continue

            # Brute force with early exit
            found = False
            for (x1, y1, z1) in pi:
                # Slight micro-optimization: hoist
                for (x2, y2, z2) in pj:
                    dx = x1 - x2
                    dy = y1 - y2
                    dz = z1 - z2
                    if dx*dx + dy*dy + dz*dz <= thr2:
                        found = True
                        break
                if found:
                    break

            if found:
                pairs.append((i, j))

    return pairs





def perpendicular_pairs(feature_lines, global_thresh):
    """
    Detect perpendicular relationships per your simplified rules:

    1) line-line : angle in [85°, 95°]  (|dot| <= cos 85°)
    2) line-circle : min endpoint distance to circle center is ~ on the circle:
         abs(dist(P_end, C_circle) - R_circle) < global_thresh
    3) line-sphere : same as line-circle but with sphere center/radius
    4) circle-cylinder : centers coincide within global_thresh, OR
         |C_circle - (C_cyl + H*n)| <= global_thresh, OR
         |C_circle - (C_cyl - H*n)| <= global_thresh
       (n is the normalized cylinder normal)

    Returns:
      List[Tuple[int, int]] with i<j
    """

    # ----------------- tiny vector helpers -----------------
    def _dot(ax, ay, az, bx, by, bz):
        return ax*bx + ay*by + az*bz

    def _norm(ax, ay, az):
        return (ax*ax + ay*ay + az*az) ** 0.5

    def _sub(ax, ay, az, bx, by, bz):
        return (ax - bx, ay - by, az - bz)

    def _dist(ax, ay, az, bx, by, bz):
        dx, dy, dz = ax - bx, ay - by, az - bz
        return (dx*dx + dy*dy + dz*dz) ** 0.5

    def _unit(ax, ay, az):
        n = _norm(ax, ay, az)
        if n == 0.0:
            return None
        return (ax/n, ay/n, az/n)

    # ----------------- parsers by type -----------------
    def _type_code(stroke):
        return int(round(float(stroke[-1])))

    def _is_line(s):     return _type_code(s) == 1
    def _is_circle(s):   return _type_code(s) == 2
    def _is_cylinder(s): return _type_code(s) == 3
    def _is_sphere(s):   return _type_code(s) == 6

    def _line_endpoints(s):
        # [x1,y1,z1, x2,y2,z2, 0,0,0, 1]
        return (float(s[0]), float(s[1]), float(s[2])), (float(s[3]), float(s[4]), float(s[5]))

    def _line_dir(s):
        (x1,y1,z1), (x2,y2,z2) = _line_endpoints(s)
        u = _unit(x2-x1, y2-y1, z2-z1)
        return u  # None if degenerate

    def _circle_center_normal_radius(s):
        # [cx,cy,cz, nx,ny,nz, 0, radius, 0, 2]
        cx, cy, cz = float(s[0]), float(s[1]), float(s[2])
        nx, ny, nz = float(s[3]), float(s[4]), float(s[5])
        r = float(s[7])
        return (cx, cy, cz), (nx, ny, nz), r

    def _cylinder_center_normal_height_radius(s):
        # [cx,cy,cz, nx,ny,nz, height, radius, 0, 3]
        cx, cy, cz = float(s[0]), float(s[1]), float(s[2])
        nx, ny, nz = float(s[3]), float(s[4]), float(s[5])
        h = float(s[6])
        r = float(s[7])
        return (cx, cy, cz), (nx, ny, nz), h, r

    def _sphere_center_radius(s):
        # [cx,cy,cz, ax,ay,az, 0, radius, 0, 6]
        cx, cy, cz = float(s[0]), float(s[1]), float(s[2])
        r = float(s[7])
        return (cx, cy, cz), r

    # ----------------- rule checks -----------------
    PERP_DOT_MAX = 0.0873  # ~= cos(85°)

    def _line_line_perp(s1, s2):
        d1 = _line_dir(s1)
        d2 = _line_dir(s2)
        if (d1 is None) or (d2 is None):
            return False
        dot = abs(_dot(d1[0], d1[1], d1[2], d2[0], d2[1], d2[2]))
        return dot <= PERP_DOT_MAX

    def _line_circle_perp(line_s, circ_s):
        (x1,y1,z1), (x2,y2,z2) = _line_endpoints(line_s)
        (ccx, ccy, ccz), _n, r = _circle_center_normal_radius(circ_s)
        # use absolute difference to test "near circumference"
        d1 = abs(_dist(x1,y1,z1, ccx,ccy,ccz) - r)
        d2 = abs(_dist(x2,y2,z2, ccx,ccy,ccz) - r)
        return (d1 < global_thresh) or (d2 < global_thresh)

    def _line_sphere_perp(line_s, sph_s):
        (x1,y1,z1), (x2,y2,z2) = _line_endpoints(line_s)
        (scx, scy, scz), r = _sphere_center_radius(sph_s)
        d1 = abs(_dist(x1,y1,z1, scx,scy,scz) - r)
        d2 = abs(_dist(x2,y2,z2, scx,scy,scz) - r)
        return (d1 < global_thresh) or (d2 < global_thresh)

    def _circle_cylinder_perp(circ_s, cyl_s):
        (ccx, ccy, ccz), _cn, _cr = _circle_center_normal_radius(circ_s)
        (ycx, ycy, ycz), yn, h, _yr = _cylinder_center_normal_height_radius(cyl_s)
        n = _unit(yn[0], yn[1], yn[2])
        if n is None:
            return False
        # centers equal
        if _dist(ccx,ccy,ccz, ycx,ycy,ycz) <= global_thresh:
            return True
        # center = cyl_center ± h * n
        offx, offy, offz = h*n[0], h*n[1], h*n[2]
        upx, upy, upz = ycx + offx, ycy + offy, ycz + offz
        dwnx, dwny, dwnz = ycx - offx, ycy - offy, ycz - offz
        if _dist(ccx,ccy,ccz, upx,upy,upz) <= global_thresh:
            return True
        if _dist(ccx,ccy,ccz, dwnx,dwny,dwnz) <= global_thresh:
            return True
        return False

    # ----------------- main loop over relevant type pairs only -----------------
    n = len(feature_lines)
    pairs = []
    for i in range(n):
        si = feature_lines[i]
        ti = _type_code(si)
        if ti not in (1, 2, 3, 6):
            continue  # skip arc(4)/spline(5)/others by your spec
        for j in range(i+1, n):
            sj = feature_lines[j]
            tj = _type_code(sj)
            if tj not in (1, 2, 3, 6):
                continue

            ok = False
            # line-line
            if ti == 1 and tj == 1:
                ok = _line_line_perp(si, sj)

            # line-circle
            elif (ti == 1 and tj == 2):
                ok = _line_circle_perp(si, sj)
            elif (ti == 2 and tj == 1):
                ok = _line_circle_perp(sj, si)

            # line-sphere
            elif (ti == 1 and tj == 6):
                ok = _line_sphere_perp(si, sj)
            elif (ti == 6 and tj == 1):
                ok = _line_sphere_perp(sj, si)

            # circle - cylinder face
            elif (ti == 2 and tj == 3):
                ok = _circle_cylinder_perp(si, sj)
            elif (ti == 3 and tj == 2):
                ok = _circle_cylinder_perp(sj, si)

            if ok:
                pairs.append((i, j))

    return pairs


def find_planar_loops(feature_lines, global_thresh, angle_tol_deg=5.0):
    """
    Find planar loops of size 3 or 4 using only straight lines (type 1).

    A valid loop:
      - uses 3 or 4 straight-line strokes,
      - the 2*E endpoints cluster into exactly E vertices (E=3 or 4),
      - every vertex appears exactly twice (degree-2),
      - all endpoints lie on one plane within global_thresh,
      - every edge direction lies in that plane within ±angle_tol_deg.

    Returns:
      loops: List[List[int]]  # each is a list of 3 or 4 stroke ids
    """
    thr = float(global_thresh)
    thr2 = thr * thr

    # ------------- tiny vector helpers -------------
    def _dot(ax, ay, az, bx, by, bz):
        return ax*bx + ay*by + az*bz

    def _sub(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

    def _add(a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

    def _norm(ax, ay, az):
        return (ax*ax + ay*ay + az*az) ** 0.5

    def _unit(v):
        n = _norm(v[0], v[1], v[2])
        if n == 0.0:
            return None
        return (v[0]/n, v[1]/n, v[2]/n)

    def _cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    def _dist2(a, b):
        dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
        return dx*dx + dy*dy + dz*dz

    # angle tolerance: line direction must be within 90°±tol of plane normal
    # i.e., |dot(dir, n)| <= sin(tol)
    # For 5°, sin ≈ 0.08716; we’ll compute a small-angle approx for flexibility.
    PI = 3.141592653589793
    rad = angle_tol_deg * PI / 180.0
    sin_tol = rad - (rad*rad*rad)/6.0  # good for small angles (<=10°)
    if sin_tol < 0:  # just in case
        sin_tol = 0.0872

    # ------------- collect straight lines -------------
    # For each straight line stroke: endpoints, direction, length
    straight_ids = []
    endpoints_by_stroke = {}  # s -> (p1, p2)
    dir_by_stroke = {}        # s -> unit direction
    for s, stroke in enumerate(feature_lines):
        if not stroke or len(stroke) < 10:
            continue
        t = int(round(float(stroke[-1])))
        if t != 1:
            continue  # only straight lines
        x1, y1, z1 = float(stroke[0]), float(stroke[1]), float(stroke[2])
        x2, y2, z2 = float(stroke[3]), float(stroke[4]), float(stroke[5])
        p1 = (x1, y1, z1)
        p2 = (x2, y2, z2)
        d = _unit((x2-x1, y2-y1, z2-z1))
        if d is None:
            continue  # degenerate
        straight_ids.append(s)
        endpoints_by_stroke[s] = (p1, p2)
        dir_by_stroke[s] = d

    if not straight_ids:
        return []

    # ------------- make vertex registry by clustering endpoints -------------
    # Assign each endpoint to a vertex id if within 'thr' of an existing one.
    vertices = []  # list of 3D points (representatives)
    def _vid_for_point(p):
        # linear scan (ok for typical sizes); merges within thr
        for vid, q in enumerate(vertices):
            if _dist2(p, q) <= thr2:
                return vid
        vertices.append(p)
        return len(vertices) - 1

    # For each straight stroke, map to vertex ids
    edge_of_stroke = {}  # s -> (v1, v2)
    for s in straight_ids:
        p1, p2 = endpoints_by_stroke[s]
        v1 = _vid_for_point(p1)
        v2 = _vid_for_point(p2)
        if v1 == v2:
            continue  # skip zero-length after clustering
        edge_of_stroke[s] = (v1, v2)

    # ------------- build vertex graph (adjacency) and edge map -------------
    adj = {}  # vid -> set(neighbor vids)
    edge_to_stroke = {}  # (min(v1,v2), max(v1,v2)) -> stroke id (choose first)
    for s, (v1, v2) in edge_of_stroke.items():
        a, b = (v1, v2) if v1 < v2 else (v2, v1)
        if a == b:
            continue
        # keep the first stroke for this geometric edge
        if (a, b) not in edge_to_stroke:
            edge_to_stroke[(a, b)] = s
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    V = len(vertices)
    for vid in range(V):
        adj.setdefault(vid, set())

    # ------------- helpers to validate loops -------------
    def _plane_from_three_points(a, b, c):
        # returns (n_unit, d) for plane n·x + d = 0, or (None, None) if degenerate
        ab = _sub(b, a)
        ac = _sub(c, a)
        n = _cross(ab, ac)
        n = _unit(n)
        if n is None:
            return None, None
        d = -_dot(n[0], n[1], n[2], a[0], a[1], a[2])
        return n, d

    def _point_plane_dist_abs(n, d, p):
        # |n·p + d|
        return abs(_dot(n[0], n[1], n[2], p[0], p[1], p[2]) + d)

    def _loop_vertex_counts_ok(stroke_ids):
        # Gather vertex counts; each must appear exactly twice; #unique == len(strokes)
        counts = {}
        for s in stroke_ids:
            if s not in edge_of_stroke:
                return False
            v1, v2 = edge_of_stroke[s]
            counts[v1] = counts.get(v1, 0) + 1
            counts[v2] = counts.get(v2, 0) + 1
        if len(counts) != len(stroke_ids):
            return False
        for v, c in counts.items():
            if c != 2:
                return False
        return True

    def _loop_connected(stroke_ids):
        # Check the subgraph induced by these edges is a single cycle (connected).
        if not stroke_ids:
            return False
        # build local adjacency on loop vertices
        ladj = {}
        verts_in = set()
        for s in stroke_ids:
            v1, v2 = edge_of_stroke[s]
            verts_in.add(v1); verts_in.add(v2)
            ladj.setdefault(v1, set()).add(v2)
            ladj.setdefault(v2, set()).add(v1)
        # BFS from an arbitrary vertex
        start = next(iter(verts_in))
        stack = [start]
        seen = set([start])
        while stack:
            u = stack.pop()
            for w in ladj.get(u, ()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen == verts_in

    def _loop_planar_ok(stroke_ids):
        # Build a plane from three non-collinear vertices; check endpoints & directions.
        # Collect unique vertex ids in this loop
        vids = []
        for s in stroke_ids:
            v1, v2 = edge_of_stroke[s]
            if v1 not in vids: vids.append(v1)
            if v2 not in vids: vids.append(v2)
        if len(vids) < 3:
            return False
        # find non-collinear triple
        n = None; d = None
        found = False
        L = len(vids)
        for i in range(L-2):
            for j in range(i+1, L-1):
                for k in range(j+1, L):
                    n_try, d_try = _plane_from_three_points(vertices[vids[i]], vertices[vids[j]], vertices[vids[k]])
                    if n_try is not None:
                        n, d = n_try, d_try
                        found = True
                        break
                if found: break
            if found: break
        if not found:
            return False
        # endpoints close to plane
        for s in stroke_ids:
            p1, p2 = endpoints_by_stroke[s]
            if _point_plane_dist_abs(n, d, p1) > thr:
                return False
            if _point_plane_dist_abs(n, d, p2) > thr:
                return False
        # directions lie in plane (within ±angle tolerance)
        for s in stroke_ids:
            dx, dy, dz = dir_by_stroke[s]
            if abs(_dot(dx, dy, dz, n[0], n[1], n[2])) > sin_tol:
                return False
        return True

    loops = []
    seen = set()  # frozenset of stroke ids to avoid duplicates

    # --------- TRIANGLES (3-cycles) ----------
    for a in range(V):
        na = adj.get(a, set())
        if not na:
            continue
        for b in sorted(na):
            if b <= a:
                continue
            nb = adj.get(b, set())
            # common neighbors c
            common = na.intersection(nb)
            for c in sorted(common):
                if c <= b or c == a:
                    continue
                # edges: (a,b), (b,c), (c,a) must exist
                e1 = (a, b) if a < b else (b, a)
                e2 = (b, c) if b < c else (c, b)
                e3 = (c, a) if c < a else (a, c)
                if e1 not in edge_to_stroke or e2 not in edge_to_stroke or e3 not in edge_to_stroke:
                    continue
                s1 = edge_to_stroke[e1]
                s2 = edge_to_stroke[e2]
                s3 = edge_to_stroke[e3]
                group = [s1, s2, s3]
                key = frozenset(group)
                if key in seen:
                    continue
                # structural checks
                if not _loop_vertex_counts_ok(group):
                    continue
                if not _loop_connected(group):
                    continue
                # planar checks
                if not _loop_planar_ok(group):
                    continue
                loops.append(group)
                seen.add(key)

    # --------- QUADS (4-cycles) ----------
    # Enumerate 4-cycles (a,b,c,d) with edges (a,b),(b,c),(c,d),(d,a).
    for a in range(V):
        nbrs_a = sorted(adj.get(a, set()))
        # choose two distinct neighbors b,d of a
        for bi in range(len(nbrs_a)):
            b = nbrs_a[bi]
            if b <= a:
                continue
            for di in range(bi+1, len(nbrs_a)):
                d = nbrs_a[di]
                if d <= a or d == b:
                    continue
                # common neighbors of b and d give candidates for c
                common = adj.get(b, set()).intersection(adj.get(d, set()))
                for c in sorted(common):
                    if c == a or c == b or c == d:
                        continue
                    # canonicalization to reduce duplicates:
                    # require a to be the smallest vertex id in the 4-tuple,
                    # and b < d to fix orientation
                    if not (a < b and a < c and a < d and b < d):
                        continue
                    # edges present?
                    e_ab = (a, b) if a < b else (b, a)
                    e_bc = (b, c) if b < c else (c, b)
                    e_cd = (c, d) if c < d else (d, c)
                    e_da = (d, a) if d < a else (a, d)
                    if e_ab not in edge_to_stroke or e_bc not in edge_to_stroke \
                       or e_cd not in edge_to_stroke or e_da not in edge_to_stroke:
                        continue
                    group = [edge_to_stroke[e_ab],
                             edge_to_stroke[e_bc],
                             edge_to_stroke[e_cd],
                             edge_to_stroke[e_da]]
                    key = frozenset(group)
                    if key in seen:
                        continue
                    # structural checks
                    if not _loop_vertex_counts_ok(group):
                        continue
                    if not _loop_connected(group):
                        continue
                    # planar checks
                    if not _loop_planar_ok(group):
                        continue
                    loops.append(group)
                    seen.add(key)

    return loops



def find_entity_pairs(feature_lines, global_thresh):
    """
    Find (circle, cylinder-face) pairs under the NEW cylinder spec.

    New Cylinder face (type 3):
      [lower_center(3), upper_center(3), 0.0, radius, 0.0, 3]

    Circle (type 2):
      [cx, cy, cz, nx, ny, nz, 0, radius, 0, 2]

    Relation: circle_center is within global_thresh of
      - lower_center  OR
      - upper_center  OR
      - midpoint( (lower+upper)/2 )

    Returns:
      list of (i, j) with i < j, where one is a circle (2) and the other a cylinder (3).
    """
    thr = float(global_thresh)
    thr2 = thr * thr

    def _type_code(s):
        return int(round(float(s[-1])))

    def _dist2(ax, ay, az, bx, by, bz):
        dx, dy, dz = ax - bx, ay - by, az - bz
        return dx*dx + dy*dy + dz*dz

    # parsers for the two types we care about
    def _circle_center(s):
        # [cx,cy,cz, nx,ny,nz, 0, r, 0, 2]
        return (float(s[0]), float(s[1]), float(s[2]))

    def _cylinder_lower_upper_radius(s):
        # [lx,ly,lz, ux,uy,uz, 0.0, r, 0.0, 3]
        lx, ly, lz = float(s[0]), float(s[1]), float(s[2])
        ux, uy, uz = float(s[3]), float(s[4]), float(s[5])
        r = float(s[7])
        return (lx, ly, lz), (ux, uy, uz), r

    n = len(feature_lines)
    pairs = []

    for i in range(n):
        si = feature_lines[i]
        ti = _type_code(si)
        if ti not in (2, 3):
            continue
        for j in range(i + 1, n):
            sj = feature_lines[j]
            tj = _type_code(sj)
            if {ti, tj} != {2, 3}:
                continue

            # order as (circle, cylinder)
            if ti == 2 and tj == 3:
                circ, cyl = si, sj
            else:
                circ, cyl = sj, si

            cx, cy, cz = _circle_center(circ)
            (lx, ly, lz), (ux, uy, uz), _r = _cylinder_lower_upper_radius(cyl)
            mx, my, mz = (0.5*(lx+ux), 0.5*(ly+uy), 0.5*(lz+uz))

            # within tolerance of any of the three centers
            if _dist2(cx, cy, cz, lx, ly, lz) <= thr2 \
               or _dist2(cx, cy, cz, ux, uy, uz) <= thr2 \
               or _dist2(cx, cy, cz, mx, my, mz) <= thr2:
                pairs.append((i, j))

    return pairs


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




# ========================================================================================== #



def stroke_cuboid_distance_matrix(sample_points, components):
    """
    Build a (num_strokes x num_cuboids) matrix of average min distances
    from each stroke to each cuboid's edges.

    Inputs:
      - sample_points: List over strokes. Each stroke is either:
            [(x,y,z), ...]                    # flat list of points
        or  [ [(x,y,z),...], [(x,y,z),...], ... ]  # list of polylines
      - components: List[CuboidGeom], each exposing .edges() -> list of 12 (a,b) endpoints

    Returns:
      - D: np.ndarray of shape (num_strokes, num_cuboids), dtype=float
           D[i, j] = stroke_to_cuboid_distance(points_i, components[j].edges())
           Empty strokes get +inf.
    """

    def _flatten_to_numpy(pts):
        # No imports; assume np is available in your environment (as in your snippet).
        if not pts:
            return np.empty((0, 3), dtype=float)

        first = pts[0] if isinstance(pts, (list, tuple)) and pts else None

        # Case A: flat list of points
        if isinstance(first, (list, tuple)) and len(first) == 3:
            arr = np.asarray(pts, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
            # fallback if odd shapes
            buf = []
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) == 3:
                    buf.append([float(p[0]), float(p[1]), float(p[2])])
            return np.asarray(buf, dtype=float) if buf else np.empty((0, 3), dtype=float)

        # Case B: list of polylines (each a list of points)
        buf = []
        for seg in pts:
            if not seg:
                continue
            if isinstance(seg, (list, tuple)) and seg and isinstance(seg[0], (list, tuple)) and len(seg[0]) == 3:
                for p in seg:
                    if isinstance(p, (list, tuple)) and len(p) == 3:
                        buf.append([float(p[0]), float(p[1]), float(p[2])])
        return np.asarray(buf, dtype=float) if buf else np.empty((0, 3), dtype=float)

    num_strokes = len(sample_points)
    num_cuboids = len(components)

    # Pre-extract edges for all cuboids once
    edges_list = [comp.edges() for comp in components]

    D = np.full((num_strokes, num_cuboids), np.inf, dtype=float)
    for si, stroke in enumerate(sample_points):
        P = _flatten_to_numpy(stroke)
        if P.size == 0:
            continue
        for cj, edges in enumerate(edges_list):
            D[si, cj] = stroke_to_cuboid_distance(P, edges)

    return D



def stroke_to_cuboid_distance(P, edges, eps=1e-12):
    """
    Average-of-min-edge distance from a stroke to a cuboid.

    Args:
      P     : np.ndarray of shape (N, 3) — sampled points for one stroke.
      edges : iterable of ((ax,ay,az), (bx,by,bz)) for the cuboid's 12 edges.
      eps   : small number to guard degenerate edges.

    Returns:
      float distance = mean_i  min_e  dist(point_i, segment_e)
      If P is empty, returns +inf.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    if P.size == 0:
        return float('inf')

    # Keep the best (smallest) squared distance per point across all edges
    min_d2 = np.full(P.shape[0], np.inf, dtype=float)

    for a, b in edges:
        A = np.asarray(a, dtype=float)
        B = np.asarray(b, dtype=float)
        U = B - A
        denom = float(U[0]*U[0] + U[1]*U[1] + U[2]*U[2])

        if denom < eps:
            # Degenerate edge: treat as a point
            diff = P - A
            d2 = (diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2)
        else:
            # Project each point onto the segment, clamp t to [0,1]
            AP = P - A
            t = (AP[:, 0]*U[0] + AP[:, 1]*U[1] + AP[:, 2]*U[2]) / denom
            t = np.clip(t, 0.0, 1.0)
            C = A + t[:, None] * U  # closest point on segment for each P
            diff = P - C
            d2 = (diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2)

        # Best edge so far for each point
        min_d2 = np.minimum(min_d2, d2)

    # Average of per-point minimal distances (not squared)
    return float(np.sqrt(min_d2).mean())



def distances_to_confidence(D, global_thresh, eps=1e-12):
    """
    Convert (num_strokes x num_cuboids) distances to per-stroke confidences.
    Uses RBF weights: w = exp(-(d/tau)^2), tau = global_thresh.
    Rows are normalized to sum to 1 when there is any finite entry.
    """
    D = np.asarray(D, dtype=float)
    tau = max(float(global_thresh * 5), eps)

    # RBF weights; invalid distances -> 0
    W = np.exp(-np.square(D / tau))
    W[~np.isfinite(D)] = 0.0

    # Row-wise normalize (avoid shape mismatch)
    row_sum = W.sum(axis=1)                 # shape: (num_strokes,)
    mask = row_sum > 0                      # shape: (num_strokes,)
    W[mask] = W[mask] / row_sum[mask][:, None]  # broadcast divide per selected row
    # rows with no finite entries stay all-zeros
    return W



def print_confidence_preview(C, D, components, global_thresh, top_k=3, max_rows=5, indices=None):
    """
    Pretty-print top-k cuboids per stroke with (confidence, distance).
      C: (num_strokes x num_cuboids) confidence matrix
      D: (num_strokes x num_cuboids) distance matrix
      components: list of executed cuboids (each has .name)
      global_thresh: tau used in distances_to_confidence
      top_k: how many cuboids to show per stroke
      max_rows: how many strokes from the top to preview (ignored if 'indices' provided)
      indices: optional explicit list of stroke indices to preview
    """
    import numpy as np

    print(f"[conf] tau (global_thresh) = {float(global_thresh):.6g}")
    if C is None or D is None or C.size == 0 or D.size == 0:
        print("[conf] Empty matrices.")
        return

    cuboid_names = [getattr(c, "name", f"cuboid_{j}") for j, c in enumerate(components)]
    num_strokes = C.shape[0]
    num_cuboids = C.shape[1]

    if indices is None:
        indices = list(range(min(max_rows, num_strokes)))

    def _fmt(x):  # distance formatter
        return "inf" if not np.isfinite(x) else f"{float(x):.4g}"

    for i in indices:
        if i < 0 or i >= num_strokes:
            continue
        # If the row has no finite distances, just say so
        if not np.isfinite(D[i]).any():
            print(f"stroke {i:3d} → (no valid samples)")
            continue

        order = np.argsort(-C[i])  # descending by confidence
        k = min(top_k, num_cuboids)
        parts = []
        for j in order[:k]:
            name = cuboid_names[j] if j < len(cuboid_names) else f"cuboid_{j}"
            parts.append(f"{name}(p={C[i,j]:.3f}, d={_fmt(D[i,j])})")
        print(f"stroke {i:3d} → " + ", ".join(parts))



# ========================================================================================== #


def best_stroke_for_each_component(C_init, D=None):
    """
    Select one UNIQUE anchor stroke per component (column).
    Preference: highest confidence in C_init; tie-break by smaller distance D.

    Args:
      C_init : (N x K) initial confidences
      D      : optional (N x K) distances; used only for tie-breaks and
               as a fallback when a component has all equal confidences.

    Returns:
      anchor_idx_per_comp : list[int] length K; -1 if no feasible stroke
      anchor_mask         : (N,) bool array; True where the stroke is an anchor
    """
    import numpy as np

    C = np.asarray(C_init, dtype=float)
    N, K = C.shape
    D = np.asarray(D, dtype=float) if D is not None else None

    # Build candidate list of (score, -distance, stroke_i, comp_k) pairs
    # Use -distance so that smaller distance sorts earlier.
    candidates = []
    for k in range(K):
        for i in range(N):
            score = C[i, k]
            if not np.isfinite(score):
                continue
            if D is not None and np.isfinite(D[i, k]):
                dist_key = -float(D[i, k])
            else:
                dist_key = -1e9  # neutral tie-break if no D
            candidates.append((float(score), dist_key, i, k))

    # If there are no candidates at all, return empty anchors
    if not candidates:
        return [-1] * K, np.zeros((N,), dtype=bool)

    # Sort by score desc, then by distance asc (since we negated it)
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)

    anchor_idx_per_comp = [-1] * K
    stroke_used = np.zeros((N,), dtype=bool)
    comps_assigned = 0

    for score, _negdist, i, k in candidates:
        if anchor_idx_per_comp[k] != -1:
            continue
        if stroke_used[i]:
            continue
        anchor_idx_per_comp[k] = i
        stroke_used[i] = True
        comps_assigned += 1
        if comps_assigned == K:
            break

    # Build mask
    anchor_mask = np.zeros((N,), dtype=bool)
    for idx in anchor_idx_per_comp:
        if idx >= 0:
            anchor_mask[idx] = True

    return anchor_idx_per_comp, anchor_mask



def propagate_confidences_safe(
    C_init,
    intersect_pairs,
    perp_pairs,
    loops,
    circle_cyl_pairs,
    w_self=1.0,
    w_inter=0.1,
    w_perp=0.1,
    w_loop=0.3,
    w_circle_cyl=0.3,
    iters=10,
    alpha=0.75,
    use_trust=True,
    anchor_mask=None,   # rows to keep fixed (already one-hot)
):
    import numpy as np

    C = np.asarray(C_init, dtype=float).copy()
    N, K = C.shape

    def _build_neighbors(pairs):
        neigh = [set() for _ in range(N)]
        for a, b in pairs:
            if 0 <= a < N and 0 <= b < N and a != b:
                neigh[a].add(b); neigh[b].add(a)
        return [list(s) for s in neigh]

    interN = _build_neighbors(intersect_pairs)
    perpN  = _build_neighbors(perp_pairs)

    loopN = [set() for _ in range(N)]
    for grp in loops:
        g = [i for i in grp if 0 <= i < N]
        for u in g:
            for v in g:
                if u != v:
                    loopN[u].add(v)
    loopN = [list(s) for s in loopN]

    circleCylN = _build_neighbors(circle_cyl_pairs)

    def _avg_neighbors(prev, nbrs, trust=None):
        if not nbrs:
            return np.zeros(K, dtype=float)
        if trust is None:
            return np.mean(prev[nbrs, :], axis=0)
        t = np.clip(trust[nbrs], 0.0, 1.0)
        s = t.sum()
        if s <= 0:
            return np.zeros(K, dtype=float)
        return (prev[nbrs, :] * t[:, None]).sum(axis=0) / s

    if anchor_mask is None:
        anchor_mask = np.zeros((N,), dtype=bool)
    else:
        anchor_mask = np.asarray(anchor_mask, dtype=bool)

    for _ in range(max(1, int(iters))):
        prev = C.copy()

        # neighbor trust (margin top1 - top2); optional
        if use_trust and K >= 2:
            top2_idx = np.argpartition(-prev, kth=1, axis=1)[:, :2]
            row = np.arange(N)[:, None]
            top_vals = prev[row, top2_idx]
            trust = np.abs(top_vals[:, 0] - top_vals[:, 1])
        else:
            trust = None

        mix = np.zeros_like(prev)

        for i in range(N):
            if anchor_mask[i]:
                # DO NOT change anchors; keep their current (one-hot) row.
                mix[i] = prev[i]
                continue

            acc = np.zeros(K, dtype=float)
            acc += w_self * prev[i]
            if interN[i]:
                acc += w_inter * _avg_neighbors(prev, interN[i], trust)
            if perpN[i]:
                acc += w_perp * _avg_neighbors(prev, perpN[i], trust)
            if loopN[i]:
                acc += w_loop * _avg_neighbors(prev, loopN[i], trust)
            if circleCylN[i]:
                acc += w_circle_cyl * _avg_neighbors(prev, circleCylN[i], trust)

            mix[i] = acc

        # Row-normalize mix (non-anchors only)
        rs = mix.sum(axis=1, keepdims=True)
        non_anchor = ~anchor_mask
        mask_rows = (rs[:, 0] > 0) & non_anchor
        mix[mask_rows] /= rs[mask_rows]

        # Restart toward C_init (non-anchors only)
        C[non_anchor] = (1.0 - alpha) * C_init[non_anchor] + alpha * mix[non_anchor]

        # Final row-normalize (non-anchors only)
        rs = C.sum(axis=1, keepdims=True)
        mask_rows = (rs[:, 0] > 0) & non_anchor
        C[mask_rows] /= rs[mask_rows]
        # Anchors remain untouched

    return C



def make_anchor_onehots(C_init, anchor_idx_per_comp):
    """
    Force anchor strokes to be one-hot on their component (once).
    Returns:
      C0        : C_init copy with anchors set to one-hot
      mask      : (N,) bool True at anchor rows
      onehots   : (N x K) one-hot rows (zero elsewhere) for convenience
    """
    import numpy as np
    C_init = np.asarray(C_init, dtype=float)
    N, K = C_init.shape
    C0 = C_init.copy()
    mask = np.zeros((N,), dtype=bool)
    onehots = np.zeros_like(C0)

    for k, i in enumerate(anchor_idx_per_comp):
        if i is None or i < 0 or i >= N:
            continue
        mask[i] = True
        onehots[i, k] = 1.0
        C0[i, :] = 0.0
        C0[i, k] = 1.0

    return C0, mask, onehots




# ========================================================================================== #




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




def visualize_strokes_by_confidence(sample_points, C, components=None, title="Assignments"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    from matplotlib.lines import Line2D

    sample_points = sample_points or []
    C = np.asarray(C, dtype=float)
    N, K = C.shape if C.ndim == 2 else (len(sample_points), 0)

    names = [
        getattr(components[j], "name", f"cuboid_{j}") if (components and j < len(components)) else f"cuboid_{j}"
        for j in range(K)
    ]

    row_sum = C.sum(axis=1) if K > 0 else np.zeros((N,), dtype=float)
    labels = np.full(N, -1, dtype=int)
    has_mass = row_sum > 0
    if K > 0:
        labels[has_mass] = np.argmax(C[has_mass], axis=1)

    if K <= 20:
        base = get_cmap('tab20')(np.linspace(0, 1, 20))
        colors = base[:K] if K > 0 else np.empty((0, 4))
    else:
        colors = get_cmap('hsv')(np.linspace(0, 1, K, endpoint=False))
    color_unassigned = (0.6, 0.6, 0.6, 1.0)

    def _iter_polylines(pts):
        if not pts:
            return
        first = pts[0] if isinstance(pts, (list, tuple)) and pts else None
        if isinstance(first, (list, tuple)) and len(first) == 3:
            arr = np.asarray(pts, dtype=float).reshape(-1, 3)
            if arr.size:
                yield arr
        else:
            for seg in pts:
                if not seg:
                    continue
                arr = np.asarray(seg, dtype=float).reshape(-1, 3)
                if arr.size:
                    yield arr

    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    for i, stroke in enumerate(sample_points):
        col = color_unassigned if labels[i] < 0 else colors[labels[i]]
        for arr in _iter_polylines(stroke):                
            if arr.shape[0] >= 2:
                ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], linewidth=1.6, alpha=0.95, color=col)
            else:
                ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=8, alpha=0.95, color=col)
            mins = np.minimum(mins, arr.min(axis=0))
            maxs = np.maximum(maxs, arr.max(axis=0))

    # Equal aspect
    if np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs)):
        ranges = maxs - mins
        max_range = ranges.max() if ranges.max() > 0 else 1.0
        centers = (maxs + mins) / 2.0
        ext = max_range / 2.0
        ax.set_xlim(centers[0] - ext, centers[0] + ext)
        ax.set_ylim(centers[1] - ext, centers[1] + ext)
        ax.set_zlim(centers[2] - ext, centers[2] + ext)

    # ---- Hide axes, grids, and numbers (keep legend) ----
    # Try the simple way first:
    try:
        ax.set_axis_off()
    except Exception:
        pass
    # Belt & suspenders for 3D:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_ticks([])
            axis.set_ticklabels([])
            # Make panes invisible
            pane = getattr(axis, "pane", None)
            if pane is not None:
                pane.set_edgecolor((1, 1, 1, 0))
                pane.set_alpha(0.0)
        except Exception:
            pass
    # Remove axis lines if present (older mpl):
    for attr in ("w_xaxis", "w_yaxis", "w_zaxis"):
        if hasattr(ax, attr):
            try:
                getattr(ax, attr).line.set_lw(0.0)
            except Exception:
                pass

    # Legend: explanation for the lines (keep this visible)
    used = sorted(set([l for l in labels if l >= 0]))
    legend_elems = [Line2D([0], [0], color=color_unassigned, lw=2, label="unassigned")] if (-1 in labels) else []
    for j in used:
        nm = names[j] if j < len(names) else f"cuboid_{j}"
        legend_elems.append(Line2D([0], [0], color=colors[j], lw=2, label=nm))
    if legend_elems:
        ax.legend(handles=legend_elems, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.show()



def visualize_anchors(sample_points, anchor_mask, title="Anchors (red)"):
    """
    Show anchored strokes in red, others in black. Works with:
      - stroke = [(x,y,z), ...]   OR
      - stroke = [ [(x,y,z),...], [(x,y,z),...], ... ]
    """
    import numpy as np
    import matplotlib.pyplot as plt

    sample_points = sample_points or []
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    N = len(sample_points)
    if anchor_mask.shape != (N,):
        raise ValueError("anchor_mask must have shape (num_strokes,)")

    def _iter_polylines(pts):
        if not pts: return
        first = pts[0] if isinstance(pts, (list, tuple)) and pts else None
        if isinstance(first, (list, tuple)) and len(first) == 3:
            arr = np.asarray(pts, dtype=float).reshape(-1, 3)
            if arr.size: yield arr
        else:
            for seg in pts:
                if not seg: continue
                arr = np.asarray(seg, dtype=float).reshape(-1, 3)
                if arr.size: yield arr

    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if title: ax.set_title(title)

    col_anchor = (1.0, 0.0, 0.0, 1.0)   # red
    col_other  = (0.0, 0.0, 0.0, 1.0)   # black

    # Draw: anchors on top for visibility (draw others first)
    order = list(range(N))
    others = [i for i in order if not anchor_mask[i]]
    anchors = [i for i in order if anchor_mask[i]]

    for group, color, lw in ((others, col_other, 1.0), (anchors, col_anchor, 1.6)):
        for i in group:
            for arr in _iter_polylines(sample_points[i]):
                if arr.shape[0] >= 2:
                    ax.plot(arr[:,0], arr[:,1], arr[:,2], color=color, linewidth=lw, alpha=0.95)
                else:
                    ax.scatter(arr[:,0], arr[:,1], arr[:,2], s=10 if anchor_mask[i] else 6, color=color, alpha=0.95)
                mins = np.minimum(mins, arr.min(axis=0))
                maxs = np.maximum(maxs, arr.max(axis=0))

    # Equal aspect box
    if np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs)):
        span = (maxs - mins)
        r = span.max() if span.max() > 0 else 1.0
        c = (maxs + mins) / 2.0
        ax.set_xlim(c[0]-r/2, c[0]+r/2)
        ax.set_ylim(c[1]-r/2, c[1]+r/2)
        ax.set_zlim(c[2]-r/2, c[2]+r/2)

    # Hide grid/axes/ticks
    try: ax.set_axis_off()
    except Exception: pass
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_ticks([]); axis.set_ticklabels([])
            pane = getattr(axis, "pane", None)
            if pane is not None:
                pane.set_edgecolor((1,1,1,0)); pane.set_alpha(0.0)
        except Exception: pass

    plt.tight_layout()
    plt.show()



def plot_strokes_and_program(sample_points, components, title=None):
    """
    Minimal 3D viz: strokes (black) + cuboids (thin blue), no labels/legend/axes.
    Strokes are drawn first so cuboids appear on top.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- helpers ----
    def _iter_polylines(pts):
        if not pts:
            return
        first = pts[0] if isinstance(pts, (list, tuple)) and pts else None
        # flat list of points
        if isinstance(first, (list, tuple)) and len(first) == 3:
            arr = np.asarray(pts, dtype=float).reshape(-1, 3)
            if arr.size:
                yield arr
        # list of polylines
        else:
            for seg in pts:
                if not seg:
                    continue
                arr = np.asarray(seg, dtype=float).reshape(-1, 3)
                if arr.size:
                    yield arr

    def _bounds_from_edges(edges):
        if not edges:
            return np.empty((0,3), dtype=float)
        pts = []
        for a,b in edges:
            pts.append(a); pts.append(b)
        return np.asarray(pts, dtype=float).reshape(-1,3)

    # ---- plot ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if title:
        ax.set_title(title)

    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = -mins

    # 1) strokes in black (background)
    stroke_color = (0, 0, 0, 1)
    for stroke in (sample_points or []):
        for arr in _iter_polylines(stroke):
            if arr.shape[0] >= 2:
                ax.plot(arr[:,0], arr[:,1], arr[:,2], color=stroke_color, linewidth=1.0, alpha=0.9)
            else:
                ax.scatter(arr[:,0], arr[:,1], arr[:,2], color=stroke_color, s=6, alpha=0.9)
            mins = np.minimum(mins, arr.min(axis=0))
            maxs = np.maximum(maxs, arr.max(axis=0))

    # 2) cuboids as thin blue wireframes (foreground)
    cube_color = (0.2, 0.4, 1.0, 1.0)
    for comp in (components or []):
        try:
            edges = comp.edges()
        except Exception:
            edges = []
        for a, b in edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=cube_color, linewidth=0.8, alpha=0.95)
        E = _bounds_from_edges(edges)
        if E.size:
            mins = np.minimum(mins, E.min(axis=0))
            maxs = np.maximum(maxs, E.max(axis=0))

    # equal aspect
    if np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs)):
        spans = maxs - mins
        r = spans.max() if spans.max() > 0 else 1.0
        c = (maxs + mins) / 2.0
        ax.set_xlim(c[0]-r/2, c[0]+r/2)
        ax.set_ylim(c[1]-r/2, c[1]+r/2)
        ax.set_zlim(c[2]-r/2, c[2]+r/2)

    # hide grid, ticks, numbers, axes
    try: ax.set_axis_off()
    except Exception: pass
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_ticks([]); axis.set_ticklabels([])
            pane = getattr(axis, "pane", None)
            if pane is not None:
                pane.set_edgecolor((1,1,1,0)); pane.set_alpha(0.0)
        except Exception: pass

    plt.tight_layout()
    plt.show()
