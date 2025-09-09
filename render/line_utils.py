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

def bounding_box_lines(feature_lines, tol=1e-6, samples_spline=101):
    """
    Return the 12 straight-line edges of the EXACT axis-aligned bounding box of the geometry.

    Primitive format (10 values), type code at v[9]:
      1 Line:      [p1(3), p2(3), 0, 0, 0, 1]
      2 Circle:    [center(3), normal(3), 0, radius, 0, 2]
      3 Cylinder:  [lowerC(3), upperC(3), 0, radius, 0, 3]
      4 Arc:       [center(3), normal(3), radius, angle_start, sweep, 4]
      5 Spline:    [cp1(3), cp2(3), cp3(3), 5]  # treated as quadratic Bézier for bbox (sampled)
      6 Sphere:    [center(3), axis(3), 0, radius, 0, 6]

    Returns: list of 12 primitives (each [x1,y1,z1, x2,y2,z2, 0,0,0, 1]).
    """

    TYPE_LINE, TYPE_CIRCLE, TYPE_CYL, TYPE_ARC, TYPE_SPLINE, TYPE_SPHERE = 1,2,3,4,5,6

    # ---- basic ops (no imports) ----
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

    # Track extrema
    INF = 10**30
    xmin, ymin, zmin =  INF,  INF,  INF
    xmax, ymax, zmax = -INF, -INF, -INF
    def upd(p):
        nonlocal xmin, ymin, zmin, xmax, ymax, zmax
        x,y,z = p
        if x < xmin: xmin = x
        if y < ymin: ymin = y
        if z < zmin: zmin = z
        if x > xmax: xmax = x
        if y > ymax: ymax = y
        if z > zmax: zmax = z

    # Orthonormal basis in plane with normal n (unit)
    def plane_basis(n):
        n = nrm(n)
        # pick a helper not parallel to n
        h = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = nrm(cross(n, h))
        if nrm2(u) <= tol*tol:
            h = (0.0, 0.0, 1.0)
            u = nrm(cross(n, h))
        v = nrm(cross(n, u))
        return u, v

    # --- exact AABB of a full circle in 3D ---
    # For axis e=(1,0,0),(0,1,0),(0,0,1), the radius projected onto e is r*sqrt(1 - (n·e)^2)
    def extend_with_circle(center, normal, r):
        cx, cy, cz = center
        nx, ny, nz = nrm(normal)
        rx = r * (1.0 - nx*nx)**0.5
        ry = r * (1.0 - ny*ny)**0.5
        rz = r * (1.0 - nz*nz)**0.5
        upd((cx - rx, cy - ry, cz - rz))
        upd((cx + rx, cy + ry, cz + rz))

    # --- exact AABB of an arc (subset of circle) ---
    # Param: P(θ) = C + r*(u cosθ + v sinθ), θ ∈ [θ0, θ1] with θ1=θ0+sweep
    def norm_angle(a):
        # map to [-pi, pi) without imports
        # Using iterative wrapping to avoid math library
        while a >= 3.141592653589793: a -= 6.283185307179586
        while a < -3.141592653589793: a += 6.283185307179586
        return a

    def angle_in_interval(theta, t0, t1):
        # works for both positive/negative sweep; interval is directed
        # normalize so t0 is start, t1 is end along sweep direction
        if t1 >= t0:
            return (theta >= t0 - 1e-12) and (theta <= t1 + 1e-12)
        else:
            # wrap across -pi/pi
            return (theta >= t0 - 1e-12) or (theta <= t1 + 1e-12)

    def clamp_arc_extrema(center, normal, r, theta0, sweep):
        # Build plane basis
        u, v = plane_basis(normal)
        # Candidate angles: endpoints plus where x,y,z reach stationary points
        # For a coordinate axis ex, f(θ) = dot(ex, r*(u cosθ + v sinθ))
        # df/dθ = 0 => -r*(dot(ex,u) sinθ - dot(ex,v) cosθ)=0 => tanθ = dot(ex,v)/dot(ex,u)
        C = center
        theta1 = theta0 + sweep

        # components of u,v along x,y,z
        A = (u[0], u[1], u[2])  # dot(ex,u) etc.
        B = (v[0], v[1], v[2])  # dot(ex,v) etc.

        cand = [theta0, theta1]

        def add_stationary(ax):  # ax = 0 for x, 1 for y, 2 for z
            Au, Bv = (A[ax], B[ax])
            if abs(Au) <= tol and abs(Bv) <= tol:
                return
            # θ* = atan2(Bv, Au)
            # We avoid math.atan2; approximate with piecewise to keep zero-import:
            # Use a simple rational approximation of atan2 for robustness:
            def atan2_approx(y, x):
                if abs(x) > abs(y):
                    t = y / (abs(x) + 1e-30)
                    # approx atan(t) ~ t*(1 - 0.28*t^2)
                    a = t*(1.0 - 0.28*t*t)
                    return a if x > 0 else (a + (0.0 if t>=0 else -3.141592653589793) + (0.0 if t<0 else 3.141592653589793))
                else:
                    t = x / (abs(y) + 1e-30)
                    a = 1.5707963267948966 - t*(1.0 - 0.28*t*t)
                    return a if y > 0 else -a
            th = atan2_approx(Bv, Au)
            th = norm_angle(th)
            th2 = norm_angle(th + 3.141592653589793)  # +π gives the opposite extremum
            for cand_th in (th, th2):
                # shift into reference frame of theta0..theta1
                # normalize relative to theta0
                rel = norm_angle(cand_th - theta0)
                # reconstitute absolute angle nearest to interval
                # Map back to global angle space around [theta0, theta1]
                abs_th = norm_angle(theta0 + rel)
                if angle_in_interval(abs_th, theta0, theta1):
                    cand.append(abs_th)

        add_stationary(0)  # x
        add_stationary(1)  # y
        add_stationary(2)  # z

        # Evaluate positions and update bbox
        def eval_arc(th):
            c, s = cos_sin(th)
            offs = add(scl(u, c*r), scl(v, s*r))
            return add(C, offs)

        # Minimal cos/sin without imports (Cordic-ish tiny approx acceptable for bbox):
        def cos_sin(t):
            # Reduce to [-pi, pi]
            t = norm_angle(t)
            # Use 5th-order minimax-ish polynomial (good enough here):
            tt = t*t
            cos_t = 1 - tt/2 + tt*tt/24
            sin_t = t - t*tt/6 + t*tt*tt/120
            return cos_t, sin_t

        for th in cand:
            p = eval_arc(th)
            upd(p)

    # --- quadratic Bézier sampling (for your 3-CP spline) ---
    def bezier2(p0, p1, p2, t):
        s = 1.0 - t
        return (s*s*p0[0] + 2*s*t*p1[0] + t*t*p2[0],
                s*s*p0[1] + 2*s*t*p1[1] + t*t*p2[1],
                s*s*p0[2] + 2*s*t*p1[2] + t*t*p2[2])

    # ---- accumulate exact/tight bbox ----
    for v in feature_lines:
        t = int(v[9])

        if t == TYPE_LINE:
            upd((v[0], v[1], v[2]))
            upd((v[3], v[4], v[5]))

        elif t == TYPE_CIRCLE:
            C = (v[0], v[1], v[2])
            n = (v[3], v[4], v[5])
            r = abs(v[7])
            extend_with_circle(C, n, r)

        elif t == TYPE_CYL:
            L = (v[0], v[1], v[2])
            U = (v[3], v[4], v[5])
            axis = sub(U, L)
            n = nrm(axis)
            r = abs(v[7])
            # two circular caps at L and U in plane normal n
            extend_with_circle(L, n, r)
            extend_with_circle(U, n, r)

        elif t == TYPE_ARC:
            C = (v[0], v[1], v[2])
            n = (v[3], v[4], v[5])
            r = abs(v[6])
            theta0 = v[7]
            sweep  = v[8]
            clamp_arc_extrema(C, n, r, theta0, sweep)

        elif t == TYPE_SPLINE:
            p0 = (v[0], v[1], v[2])
            p1 = (v[3], v[4], v[5])
            p2 = (v[6], v[7], v[8])
            # sample densely (exact Bézier extrema solving would need more algebra)
            N = max(3, int(samples_spline))
            for i in range(N):
                t_ = i/(N-1)
                upd(bezier2(p0, p1, p2, t_))

        elif t == TYPE_SPHERE:
            C = (v[0], v[1], v[2])
            r = abs(v[7])
            upd((C[0]-r, C[1]-r, C[2]-r))
            upd((C[0]+r, C[1]+r, C[2]+r))

        else:
            # fallback to first two points if present
            upd((v[0], v[1], v[2]))
            upd((v[3], v[4], v[5]))

    # Degenerate?
    if not (xmin <= xmax and ymin <= ymax and zmin <= zmax):
        return []

    # Build 8 corners
    X = [xmin, xmax]; Y = [ymin, ymax]; Z = [zmin, zmax]
    corners = [
        (X[i], Y[j], Z[k]) for i in (0,1) for j in (0,1) for k in (0,1)
    ]
    idx = {(i,j,k): (i<<2) | (j<<1) | k for i in (0,1) for j in (0,1) for k in (0,1)}
    # 12 edges
    edges_idx = [
        # bottom (z=zmin)
        (idx[(0,0,0)], idx[(1,0,0)]),
        (idx[(0,0,0)], idx[(0,1,0)]),
        (idx[(1,1,0)], idx[(0,1,0)]),
        (idx[(1,1,0)], idx[(1,0,0)]),
        # top (z=zmax)
        (idx[(0,0,1)], idx[(1,0,1)]),
        (idx[(0,0,1)], idx[(0,1,1)]),
        (idx[(1,1,1)], idx[(0,1,1)]),
        (idx[(1,1,1)], idx[(1,0,1)]),
        # verticals
        (idx[(0,0,0)], idx[(0,0,1)]),
        (idx[(1,0,0)], idx[(1,0,1)]),
        (idx[(0,1,0)], idx[(0,1,1)]),
        (idx[(1,1,0)], idx[(1,1,1)]),
    ]

    def make_line(p1, p2):
        return [p1[0],p1[1],p1[2], p2[0],p2[1],p2[2], 0.0,0.0,0.0, TYPE_LINE]

    # Emit all 12 edges (exact bbox), without filtering out overlaps on purpose
    out = []
    for i,j in edges_idx:
        p1, p2 = corners[i], corners[j]
        out.append(make_line(p1, p2))
    return out
