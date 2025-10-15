# compute_sketch_points.py
from typing import List, Tuple

Vec3 = Tuple[float, float, float]

def _dir_to_basis(extrude_dir: str) -> Tuple[str, int, Vec3, Vec3, Vec3, Tuple[str, str]]:
    """
    Map an extrusion direction like 'z', '-x', 'y' to:
      axis   : 'x' | 'y' | 'z'
      sign   : +1 or -1
      n      : normal vector of the sketch plane (aligned with extrude_dir)
      u, v   : orthonormal in-plane basis for the sketch
      len_map: which parameter names correspond to u and v ('x'|'y'|'z')
    """
    sign = -1 if extrude_dir.startswith('-') else 1
    axis = extrude_dir[1:] if sign == -1 else extrude_dir

    normals = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}
    n_base = normals.get(axis, (0.0, 0.0, 1.0))
    n = (sign * n_base[0], sign * n_base[1], sign * n_base[2])

    # Pick in-plane axes (u, v) and which lengths feed them.
    if axis == 'x':        # sketch plane is YZ
        u, v = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        len_map = ('y', 'z')
    elif axis == 'y':      # sketch plane is XZ
        u, v = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        len_map = ('x', 'z')
    else:                  # 'z' -> sketch plane is XY
        u, v = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        len_map = ('x', 'y')

    return axis, sign, n, u, v, len_map


def _vadd(a: Vec3, b: Vec3, s: float) -> List[float]:
    return [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]


def compute_rectangle(
    x_length: float, y_length: float, z_length: float,
    center: Vec3, extrude_dir: str
) -> List[List[float]]:
    """
    Returns a 4-point loop: [p1, p2, p3, p4]
    Order matches your original: lower-left, lower-right, upper-right, upper-left
    with 'v' treated as 'up' in the sketch plane.
    """
    _, _, _, u, v, len_map = _dir_to_basis(extrude_dir)

    len_lookup = {'x': x_length, 'y': y_length, 'z': z_length}
    len_u = len_lookup[len_map[0]]
    len_v = len_lookup[len_map[1]]
    hu, hv = len_u / 2.0, len_v / 2.0
    c = (center[0], center[1], center[2])

    p1 = _vadd(_vadd(c, u, -hu), v, -hv)
    p2 = _vadd(_vadd(c, u,  hu), v, -hv)
    p3 = _vadd(_vadd(c, u,  hu), v,  hv)
    p4 = _vadd(_vadd(c, u, -hu), v,  hv)
    return [p1, p2, p3, p4]


def compute_triangle(
    x_length: float, y_length: float, z_length: float,
    center: Vec3, extrude_dir: str
) -> List[List[float]]:
    """
    Isosceles triangle centered at 'center'.
    Base along ±u at v = -hv, apex at v = +hv (matches your XY version).
    Returns 3 points: [left_base, right_base, apex]
    """
    _, _, _, u, v, len_map = _dir_to_basis(extrude_dir)

    len_lookup = {'x': x_length, 'y': y_length, 'z': z_length}
    len_u = len_lookup[len_map[0]]
    len_v = len_lookup[len_map[1]]
    hu, hv = len_u / 2.0, len_v / 2.0
    c = (center[0], center[1], center[2])

    left_base  = _vadd(_vadd(c, u, -hu), v, -hv)
    right_base = _vadd(_vadd(c, u,  hu), v, -hv)
    apex       = _vadd(c, v, hv)
    return [left_base, right_base, apex]


def compute_circle(
    x_length: float, y_length: float, z_length: float,
    center: Vec3, extrude_dir: str
) -> Tuple[float, List[float], List[float]]:
    """
    Returns (radius, center, normal) to match build_circle(radius, center, normal).
    Radius mirrors your previous behavior: uses the 'u' in-plane length / 2.
    """
    _, _, n, _, _, len_map = _dir_to_basis(extrude_dir)

    len_lookup = {'x': x_length, 'y': y_length, 'z': z_length}
    len_u = len_lookup[len_map[0]]  # keep behavior consistent with prior x_len/2 on Z-up
    radius = len_u / 2.0

    return radius, [center[0], center[1], center[2]], [n[0], n[1], n[2]]
