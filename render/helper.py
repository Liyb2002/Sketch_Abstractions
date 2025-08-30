from math import pi, atan2, fmod


def _norm_angle(a):
    # normalize to [0, 2π)
    twopi = 2.0 * pi
    a = fmod(a, twopi)
    if a < 0.0:
        a += twopi
    return a

def _angle_diff(a2, a1):
    # signed shortest diff a2 - a1 in (-π, π]
    twopi = 2.0 * pi
    d = _norm_angle(a2) - _norm_angle(a1)
    if d > pi:
        d -= twopi
    if d <= -pi:
        d += twopi
    return d
