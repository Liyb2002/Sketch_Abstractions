def projection_lines(feature_lines, tol=1e-4, angle_tol_rel=0.15, min_gap=1e-4):
    """
    Projection lines:
      (A) Orthogonal projections between near-parallel planar quads (as before).
      (B) In-plane tangents around circular faces: for each circle (top/bottom of a cylinder, or standalone),
          draw four tangent segments in the circle's plane (±u, ±v offsets), clipped to local extents.

    Returns only NEW straight lines in your 10-value format: [x1,y1,z1, x2,y2,z2, 0,0,0, 1].
    """

    TYPE_LINE, TYPE_CIRCLE, TYPE_CYL_FACE = 1, 2, 3

    # -----------------------
    # basic vector ops (no imports)
    # -----------------------
    def sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    def add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    def scl(a,s): return (a[0]*s, a[1]*s, a[2]*s)
    def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    def nrm2(a): return dot(a,a)
    def nrm(a):
        d2 = nrm2(a)
        if d2 <= 0.0: return (0.0,0.0,0.0)
        d = d2 ** 0.5
        return (a[0]/d, a[1]/d, a[2]/d)
    def dist2(a,b): return nrm2(sub(a,b))
    def eq_pt(a,b): return dist2(a,b) <= tol*tol

    def eq_seg(a1, a2, b1, b2):
        return (eq_pt(a1,b1) and eq_pt(a2,b2)) or (eq_pt(a1,b2) and eq_pt(a2,b1))

    def point_on_segment(p, a, b):
        ab = sub(b,a); ap = sub(p,a)
        L2 = nrm2(ab)
        if L2 <= tol*tol:  # degenerate
            return eq_pt(p,a)
        t = dot(ap, ab) / L2
        if t < -tol or t > 1+tol:
            return False
        perp = sub(ap, scl(ab, t))
        return nrm2(perp) <= tol*tol

    def seg_contained_in(a1, a2, b1, b2):
        return point_on_segment(a1, b1, b2) and point_on_segment(a2, b1, b2)

    def make_line(p1, p2):
        return [p1[0],p1[1],p1[2], p2[0],p2[1],p2[2], 0.0,0.0,0.0, TYPE_LINE]

    # -----------------------
    # Separate primitives & cache existing straight segments
    # -----------------------
    existing_segments, straight_segs, circles, cylinders = [], [], [], []
    for v in feature_lines:
        t = int(v[9])
        if t == TYPE_LINE:
            p1 = (v[0], v[1], v[2]); p2 = (v[3], v[4], v[5])
            if dist2(p1,p2) > tol*tol:
                existing_segments.append((p1,p2))
                straight_segs.append((p1,p2))
        elif t == TYPE_CIRCLE:
            circles.append(v)
        elif t == TYPE_CYL_FACE:
            cylinders.append(v)

    def is_duplicate(p1,p2, new_segments):
        if dist2(p1,p2) <= max(tol*tol, min_gap*min_gap):
            return True
        for (q1,q2) in existing_segments:
            if eq_seg(p1,p2,q1,q2) or seg_contained_in(p1,p2,q1,q2):
                return True
        for (r1,r2) in new_segments:
            if eq_seg(p1,p2,r1,r2) or seg_contained_in(p1,p2,r1,r2):
                return True
        return False

    new_lines, new_segments = [], []

    # ============================================================
    # (A) (optional) planar quads → face-to-face orthogonal projections
    # ------------------------------------------------------------
    # If you already have this elsewhere, you can comment this block out.
    # ============================================================
    # cluster endpoints -> vertices
    vertices = []
    def find_or_add(pt):
        for i,q in enumerate(vertices):
            if eq_pt(pt,q): return i
        vertices.append(pt); return len(vertices)-1

    edges = []
    for (a,b) in straight_segs:
        ia, ib = find_or_add(a), find_or_add(b)
        if ia != ib:
            e = (min(ia,ib), max(ia,ib))
            if e not in edges: edges.append(e)

    adj = {i:set() for i in range(len(vertices))}
    for (i,j) in edges:
        adj[i].add(j); adj[j].add(i)

    # enumerate simple 4-cycles
    quads = set()
    for a in range(len(vertices)):
        for b in adj[a]:
            if b == a: continue
            for c in adj[b]:
                if c in (a,b): continue
                for d in adj[c]:
                    if d in (a,b,c): continue
                    if a in adj[d]:
                        cyc = [a,b,c,d]
                        rots = [
                            tuple(cyc),(b,c,d,a),(c,d,a,b),(d,a,b,c),
                            tuple(reversed(cyc)), tuple(reversed((b,c,d,a))),
                            tuple(reversed((c,d,a,b))), tuple(reversed((d,a,b,c)))
                        ]
                        quads.add(min(rots))

    def coplanar(pA,pB,pC,pD):
        n = cross(sub(pB,pA), sub(pC,pA))
        h = dot(n, sub(pD,pA))
        return abs(h) <= tol * (1.0 + (nrm2(n)**0.5))

    def rel_cross(u,v):
        nu, nv = (nrm2(u)**0.5), (nrm2(v)**0.5)
        if nu <= tol or nv <= tol: return 1.0
        cr = cross(u,v)
        return (nrm2(cr)**0.5) / (nu*nv)

    def is_parallelogram(pA,pB,pC,pD):
        AB, BC, CD, DA = sub(pB,pA), sub(pC,pB), sub(pD,pC), sub(pA,pD)
        return (rel_cross(AB,CD) <= angle_tol_rel) and (rel_cross(BC,DA) <= angle_tol_rel)

    faces = []
    for (a,b,c,d) in quads:
        A,B,C,D = vertices[a], vertices[b], vertices[c], vertices[d]
        if coplanar(A,B,C,D) and is_parallelogram(A,B,C,D):
            AB, AD = sub(B,A), sub(D,A)
            n = nrm(cross(AB, AD))
            if nrm2(n) <= tol*tol: continue
            faces.append({"A":A,"B":B,"C":C,"D":D,"AB":AB,"AD":AD,"n":n})

    # pair near-parallel faces
    def normals_parallel(n1, n2):
        cr = cross(n1, n2)
        return (nrm2(cr)**0.5) <= angle_tol_rel or (nrm2(cross(n1, scl(n2,-1.0)))**0.5) <= angle_tol_rel

    face_pairs = []
    for i in range(len(faces)):
        for j in range(i+1, len(faces)):
            if normals_parallel(faces[i]["n"], faces[j]["n"]):
                face_pairs.append((i, j))

    def in_face(q, face):
        A, AB, AD = face["A"], face["AB"], face["AD"]
        AQ = sub(q, A)
        a11 = dot(AB, AB); a22 = dot(AD, AD); a12 = dot(AB, AD)
        b1  = dot(AQ, AB); b2  = dot(AQ, AD)
        det = a11*a22 - a12*a12
        if abs(det) <= tol: return False
        u = ( b1*a22 - b2*a12) / det
        v = (-b1*a12 + b2*a11) / det
        eps = 1e-6
        return (-eps <= u <= 1+eps) and (-eps <= v <= 1+eps)

    def project_to_plane(p, dirv, Q0, nQ):
        denom = dot(dirv, nQ)
        if abs(denom) <= tol:  # parallel to plane
            return None
        t = dot(sub(Q0, p), nQ) / denom
        return add(p, scl(dirv, t))

    # face-to-face projections
    for (ia, ib) in face_pairs:
        FA, FB = faces[ia], faces[ib]
        for src, dst in ((FA, FB), (FB, FA)):
            for p in [src["A"], src["B"], src["C"], src["D"]]:
                q = project_to_plane(p, src["n"], dst["A"], dst["n"])
                if q is not None and in_face(q, dst) and not is_duplicate(p, q, new_segments):
                    new_segments.append((p, q))
                    new_lines.append(make_line(p, q))

    # ============================================================
    # (B) In-plane tangent lines for circular faces
    # ------------------------------------------------------------
    # Helper: orthonormal basis (u,v) in plane with normal n
    # ============================================================
    def plane_basis(n):
        n = nrm(n)
        h = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = nrm(cross(n, h))
        if nrm2(u) <= tol*tol:
            h = (0.0, 0.0, 1.0)
            u = nrm(cross(n, h))
        v = nrm(cross(n, u))
        return u, v  # both ⟂ n, and ⟂ each other

    # Project a point to plane (P0,n) and return (u,v) coords
    def to_uv(p, P0, n, u, v):
        # p_proj = p - n * dot(n, p-P0), then coordinates along u and v
        w = sub(p, P0)
        w_perp = sub(w, scl(n, dot(w, n)))
        return (dot(w_perp, u), dot(w_perp, v))

    # Given a plane (P0,n) and radius r, estimate extents along u & v
    def estimate_extents(P0, n, r, k=1.5):
        u, v = plane_basis(n)
        # initialize with a reasonable fallback around the circle
        u_min, u_max = -k*r, k*r
        v_min, v_max = -k*r, k*r
        # enlarge using projections of all straight endpoints
        for (a,b) in straight_segs:
            for p in (a, b):
                uu, vv = to_uv(p, P0, n, u, v)
                if uu < u_min: u_min = uu
                if uu > u_max: u_max = uu
                if vv < v_min: v_min = vv
                if vv > v_max: v_max = vv
        return u, v, (u_min, u_max, v_min, v_max)

    # Tangents in plane: two parallel to u at v=±r, two parallel to v at u=±r
    def add_circle_tangents(P0, n, r):
        if r <= tol:  # nothing to do
            return
        u, v, (u_min, u_max, v_min, v_max) = estimate_extents(P0, n, r, k=1.5)

        # lines parallel to u (vary u), at v = ±r
        for sign in (+1.0, -1.0):
            off = scl(v, sign * r)
            p1 = add(P0, add(scl(u, u_min), off))
            p2 = add(P0, add(scl(u, u_max), off))
            if not is_duplicate(p1, p2, new_segments):
                new_segments.append((p1, p2))
                new_lines.append(make_line(p1, p2))

        # lines parallel to v (vary v), at u = ±r
        for sign in (+1.0, -1.0):
            off = scl(u, sign * r)
            p1 = add(P0, add(scl(v, v_min), off))
            p2 = add(P0, add(scl(v, v_max), off))
            if not is_duplicate(p1, p2, new_segments):
                new_segments.append((p1, p2))
                new_lines.append(make_line(p1, p2))

    # Circles from cylinders: two faces at lower & upper centers, normal = axis
    for v in cylinders:
        L = (v[0], v[1], v[2])
        U = (v[3], v[4], v[5])
        axis = sub(U, L)
        if nrm2(axis) <= tol*tol:  # invalid cylinder
            continue
        n = nrm(axis)
        r = abs(v[7])
        # bottom face @ L
        add_circle_tangents(L, n, r)
        # top face @ U
        add_circle_tangents(U, n, r)

    # Standalone circles too
    for c in circles:
        C = (c[0], c[1], c[2])
        n = nrm((c[3], c[4], c[5]))
        r = abs(c[7])
        add_circle_tangents(C, n, r)

    return new_lines




def bounding_box_lines(geometry, tol=1e-6, samples_spline=101):
    """
    Returns the 12 edges of the axis-aligned bounding box (AABB) for mixed 3D geometry.
    Accepts either:
      - polylines (possibly nested): [[ [x,y,z], ...], ...]
      - 10-value primitives with type tag at v[9], using YOUR formats:

        1) Straight Line:  P1(x,y,z), P2(x,y,z), 0,        0,      0,      1
        2) Circle:         C(x,y,z),  n(nx,ny,nz), 0,      r,      0,      2
        3) Cylinder face:  C(x,y,z),  n(nx,ny,nz), height, r,      0,      3
        4) Arc:            S(x,y,z),  E(x,y,z),    C(x,y,z),       4
        5) Spline (quad):  P0(x,y,z), P1(x,y,z),   P2(x,y,z),      5
        6) Sphere:         Cx, Cy, Cz, nx, ny, nz, 0,      r,      0,      6
    """
    import math

    TYPE_LINE, TYPE_CIRCLE, TYPE_CYL, TYPE_ARC, TYPE_SPLINE, TYPE_SPHERE = 1,2,3,4,5,6

    # -------- basic ops --------
    def sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    def add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    def scl(a,s): return (a[0]*s, a[1]*s, a[2]*s)
    def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    def nrm2(a): return dot(a,a)
    def nrm(a):
        d2 = nrm2(a)
        if d2 <= 0.0: return (0.0,0.0,0.0)
        d = d2**0.5
        return (a[0]/d, a[1]/d, a[2]/d)

    # -------- helpers to detect input shape --------
    def is_point(p):
        return isinstance(p, (list, tuple)) and len(p) == 3 and all(isinstance(x, (int, float)) for x in p)

    def is_polyline(obj):
        return isinstance(obj, (list, tuple)) and len(obj) > 0 and is_point(obj[0])

    def flatten_polylines(obj):
        out = []
        if is_polyline(obj):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                out.extend(flatten_polylines(it))
        return out

    # -------- track extrema --------
    INF = 10**30
    xmin = ymin = zmin =  INF
    xmax = ymax = zmax = -INF

    def upd(p):
        nonlocal xmin, ymin, zmin, xmax, ymax, zmax
        x,y,z = p
        if x < xmin: xmin = x
        if y < ymin: ymin = y
        if z < zmin: zmin = z
        if x > xmax: xmax = x
        if y > ymax: ymax = y
        if z > zmax: zmax = z

    def plane_basis(n):
        n = nrm(n)
        h = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = nrm(cross(n, h))
        if nrm2(u) <= tol*tol:
            h = (0.0, 0.0, 1.0)
            u = nrm(cross(n, h))
        v = nrm(cross(n, u))
        return u, v

    # exact AABB for full 3D circle
    def extend_with_circle(center, normal, r):
        cx, cy, cz = center
        nx, ny, nz = nrm(normal)
        rx = r * (1.0 - nx*nx)**0.5
        ry = r * (1.0 - ny*ny)**0.5
        rz = r * (1.0 - nz*nz)**0.5
        upd((cx - rx, cy - ry, cz - rz))
        upd((cx + rx, cy + ry, cz + rz))

    # arc AABB by checking endpoints + axis-wise stationary points within the sweep
    def clamp_arc_extrema(center, normal, r, theta0, sweep):
        u, v = plane_basis(normal)
        C = center
        theta1 = theta0 + sweep

        # candidate angles: endpoints + where d/dθ of each coord = 0
        cand = [theta0, theta1]

        A = u  # coefficient for cos
        B = v  # coefficient for sin

        def add_stationary(ax):
            Au, Bv = (A[ax]*r, B[ax]*r)
            if abs(Au) <= tol and abs(Bv) <= tol:
                return
            th = math.atan2(Bv, Au)
            # also opposite angle (π apart)
            for base in (th, th + math.pi):
                # normalize to [theta0, theta0+2π) then test inclusion
                rel = (base - theta0) % (2*math.pi)
                abs_th = theta0 + rel
                if sweep >= 0:
                    inside = (abs_th >= min(theta0, theta1) - 1e-12) and (abs_th <= max(theta0, theta1) + 1e-12)
                else:
                    inside = (abs_th <= max(theta0, theta1) + 1e-12) or (abs_th >= min(theta0, theta1) - 1e-12)
                if inside:
                    cand.append(abs_th)

        add_stationary(0); add_stationary(1); add_stationary(2)

        def eval_arc(th):
            return add(C, add(scl(u, r*math.cos(th)), scl(v, r*math.sin(th))))

        for th in cand:
            upd(eval_arc(th))

    # quadratic Bézier sampler (for spline bbox)
    def bezier2(p0, p1, p2, t):
        s = 1.0 - t
        return (s*s*p0[0] + 2*s*t*p1[0] + t*t*p2[0],
                s*s*p0[1] + 2*s*t*p1[1] + t*t*p2[1],
                s*s*p0[2] + 2*s*t*p1[2] + t*t*p2[2])

    # -------- process input --------
    polylines = flatten_polylines(geometry)
    if polylines:
        for pts in polylines:
            for p in pts:
                upd((float(p[0]), float(p[1]), float(p[2])))
    else:
        # Assume 10-value primitives per your schema
        for v in geometry:
            t = int(v[9])

            if t == TYPE_LINE:
                # P1, P2
                upd((v[0], v[1], v[2]))
                upd((v[3], v[4], v[5]))

            elif t == TYPE_CIRCLE:
                # C, n, 0, r, 0
                C = (v[0], v[1], v[2])
                n = (v[3], v[4], v[5])
                r = abs(v[7])  # radius at index 7
                extend_with_circle(C, n, r)

            elif t == TYPE_CYL:
                # Cylinder face: C, n, height, r, 0
                C = (v[0], v[1], v[2])
                n = nrm((v[3], v[4], v[5]))
                h = float(v[6])
                r = abs(v[7])
                # centers of the two circular caps
                half = 0.5 * h
                Lc = add(C, scl(n, -half))
                Uc = add(C, scl(n,  half))
                extend_with_circle(Lc, n, r)
                extend_with_circle(Uc, n, r)

            elif t == TYPE_ARC:
                # Arc: S, E, C
                S = (v[0], v[1], v[2])
                E = (v[3], v[4], v[5])
                C = (v[6], v[7], v[8])

                # update endpoints regardless
                upd(S); upd(E); upd(C)

                # derive plane & circle params
                SC = sub(S, C)
                EC = sub(E, C)
                rS = math.sqrt(max(nrm2(SC), 0.0))
                rE = math.sqrt(max(nrm2(EC), 0.0))
                r = 0.5*(rS + rE)
                if r <= tol:
                    continue  # degenerate arc

                n = cross(SC, EC)
                if nrm2(n) <= tol*tol:
                    # S, E, C collinear → treat as chord endpoints
                    continue

                # basis: u along S from C, v = (n × u)
                u = nrm(SC)
                w = nrm(n)
                v_ = nrm(cross(w, u))

                # angle from S to E in this (u,v_) basis (CCW)
                x = dot(EC, u)
                y = dot(EC, v_)
                theta0 = 0.0
                sweep = math.atan2(y, x)
                if sweep < 0:
                    sweep += 2*math.pi  # choose minor-arc CCW S→E

                clamp_arc_extrema(C, w, r, theta0, sweep)

            elif t == TYPE_SPLINE:
                # quadratic Bézier: P0, P1, P2
                p0 = (v[0], v[1], v[2])
                p1 = (v[3], v[4], v[5])
                p2 = (v[6], v[7], v[8])
                N = max(3, int(samples_spline))
                for i in range(N):
                    t_ = i/(N-1)
                    upd(bezier2(p0, p1, p2, t_))

            elif t == TYPE_SPHERE:
                # Sphere: center..., radius at index 7
                C = (v[0], v[1], v[2])
                r = abs(v[7])
                upd((C[0]-r, C[1]-r, C[2]-r))
                upd((C[0]+r, C[1]+r, C[2]+r))

            else:
                # fallback: try first two triplets if present
                upd((v[0], v[1], v[2]))
                upd((v[3], v[4], v[5]))

    # -------- build 12 bbox edges --------
    if not (xmin <= xmax and ymin <= ymax and zmin <= zmax):
        return []

    X = [xmin, xmax]; Y = [ymin, ymax]; Z = [zmin, zmax]
    corners = [(X[i], Y[j], Z[k]) for i in (0,1) for j in (0,1) for k in (0,1)]
    idx = {(i,j,k): (i<<2) | (j<<1) | k for i in (0,1) for j in (0,1) for k in (0,1)}
    edges_idx = [
        (idx[(0,0,0)], idx[(1,0,0)]), (idx[(0,0,0)], idx[(0,1,0)]),
        (idx[(1,1,0)], idx[(0,1,0)]), (idx[(1,1,0)], idx[(1,0,0)]),
        (idx[(0,0,1)], idx[(1,0,1)]), (idx[(0,0,1)], idx[(0,1,1)]),
        (idx[(1,1,1)], idx[(0,1,1)]), (idx[(1,1,1)], idx[(1,0,1)]),
        (idx[(0,0,0)], idx[(0,0,1)]), (idx[(1,0,0)], idx[(1,0,1)]),
        (idx[(0,1,0)], idx[(0,1,1)]), (idx[(1,1,0)], idx[(1,1,1)]),
    ]
    def make_line(p1, p2):
        return [p1[0],p1[1],p1[2], p2[0],p2[1],p2[2], 0.0,0.0,0.0, TYPE_LINE]

    out = []
    for i,j in edges_idx:
        p1, p2 = corners[i], corners[j]
        out.append(make_line(p1, p2))
    return out
