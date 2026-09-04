"""
Re-solve pitch trajectories at the midpoint of home plate (y = 8.5 in),
since the ABS zone is judged there, not at the front (y = 17 in) where
Statcast's plate_x/plate_z are reported.

Ball-radius correction (confirmed 2026-09-04 against real ABS challenge
data): ABS applies the standard "any part of the ball over the zone"
rule. MLB's edge_distance measures from the ball's EDGE; a trajectory
solve gives the ball's CENTER. The two differ by exactly one ball radius
(fit against 11 real challenges: mean offset 0.1205 ft vs a real ball
radius of 0.1208 ft -- match to 0.006 in). Center distance and
radius-corrected distance are kept as separate named outputs; do not
collapse them into one number, since the raw center distance is still
what you get from geometry alone and the correction is a separate,
falsifiable physical claim.
"""
import numpy as np

Y_MIDPLATE = 8.5 / 12.0
HALF_WIDTH = (17.0 / 12.0) / 2.0  # 0.708 ft
BALL_RADIUS_FT = 0.1208  # 2.9" diameter ball, radius = 1.45/12 ft


def solve_t_at_y(y0, vy0, ay, y_target):
    """Time at which the ball crosses y_target. Ball travels in -y direction."""
    a, b, c = 0.5 * ay, vy0, (y0 - y_target)
    disc = b**2 - 4 * a * c
    disc = np.where(disc < 0, np.nan, disc)
    t1 = (-b - np.sqrt(disc)) / (2 * a)
    t2 = (-b + np.sqrt(disc)) / (2 * a)
    # ball crosses y_target twice (math has two roots); we want the smaller
    # positive one, since the larger root is after the ball has already
    # passed y_target once (a is tiny relative to b, so the "wrong" root
    # is a large, physically meaningless time)
    return np.where((t1 > 0) & (t1 < t2), t1, t2)


def location_at_midplate(y0, vy0, ay, x0, vx0, ax, z0, vz0, az):
    t = solve_t_at_y(y0, vy0, ay, Y_MIDPLATE)
    x = x0 + vx0 * t + 0.5 * ax * t**2
    z = z0 + vz0 * t + 0.5 * az * t**2
    return x, z


# MEASURED FACT (2026-09-05), contra the project walkthrough: plate_x/plate_z
# are ALREADY reported at the middle of the plate (y = 8.5/12), not the front.
# Solving the trajectory to y=8.5/12 reproduces plate_x/plate_z to 0.0000 in
# over 940 pitches, while y=17/12 (front) is off by 0.25 in horizontally and
# 0.98 in vertically. True in 2024 as well, so it is a long-standing convention
# and not an ABS-era change. location_at_midplate() above is therefore correct
# but redundant -- plate_x/plate_z can be used directly, which matters because
# the bulk CSV export ships no x0/y0/z0 anchor to re-solve from.


def center_distance_to_zone(x, z, batter_height_ft, top_pct=0.535, bot_pct=0.270):
    """
    Signed Euclidean distance from the ball's CENTER (x, z) to the ABS zone
    rectangle boundary. Positive = inside (distance to the nearest edge).
    Negative = outside: axis-aligned to the nearest edge when only one axis
    is violated, Euclidean to the nearest corner when both are (inflating a
    rectangle isn't axis-independent at the corners, so neither is this).
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    top = top_pct * batter_height_ft
    bot = bot_pct * batter_height_ft

    dx_out = np.maximum(np.abs(x) - HALF_WIDTH, 0.0)
    dz_out = np.maximum(np.maximum(bot - z, 0.0), np.maximum(z - top, 0.0))
    outside = (dx_out > 0) | (dz_out > 0)
    corner_dist = np.hypot(dx_out, dz_out)

    inside_margin = np.minimum(np.minimum(HALF_WIDTH - np.abs(x), z - bot), top - z)

    return np.where(outside, -corner_dist, inside_margin)


def ball_edge_distance(center_dist, radius=BALL_RADIUS_FT):
    """
    Signed distance under the "any part of the ball over the zone" rule:
    positive means the ball's edge reaches the zone (an effective strike),
    negative means it doesn't (an effective ball), even though the value
    is derived from a center-based measurement. This is the number that
    should line up with MLB's edge_distance once you take abs().
    """
    return center_dist + radius


def in_abs_zone(x, z, batter_height_ft, top_pct=0.535, bot_pct=0.270, radius=BALL_RADIUS_FT):
    center = center_distance_to_zone(x, z, batter_height_ft, top_pct, bot_pct)
    return ball_edge_distance(center, radius) > 0


# Listed heights are published rounded to the nearest whole inch (confirmed
# 2026-09-05: every listed height in the dataset is an exact integer number
# of inches). A batter's true height is therefore ~Uniform(listed - 0.5in,
# listed + 0.5in) unless we've independently backed out their actual
# measured height from real challenges (see scripts/verify_ball_radius.py).
HEIGHT_ROUNDING_HALFWIDTH_FT = 0.5 / 12.0


def p_inside_zone(x, z, height_ft, height_uncertainty_ft=0.0, n_grid=21,
                   location_sigma_ft=0.0,
                   top_pct=0.535, bot_pct=0.270, radius=BALL_RADIUS_FT):
    """
    Probability the ball is inside the ABS zone, integrating over uncertainty
    in batter height and (optionally) in the tracked location itself.

    height_uncertainty_ft: 0 for a batter with a trusted measured height;
        HEIGHT_ROUNDING_HALFWIDTH_FT for a listed-height-only batter, whose
        true height is ~Uniform(listed - 0.5in, listed + 0.5in).

    location_sigma_ft: perpendicular location uncertainty. With this at 0 and a
        measured height, the result is a hard 0/1 call -- which makes any
        challenge policy degenerate ("challenge exactly when the call is
        wrong"), since it assumes the decision-maker knows the true location.
        A non-zero sigma represents what is actually knowable at decision time.
        Handled as P(signed distance + noise > 0) = Phi(d / sigma), exact for
        the one-dimensional perpendicular case that dominates near an edge.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    height_ft = np.atleast_1d(np.asarray(height_ft, dtype=float))

    if height_uncertainty_ft == 0:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-height_uncertainty_ft, height_uncertainty_ft, n_grid)

    probs = np.zeros(np.broadcast_shapes(x.shape, z.shape, height_ft.shape), dtype=float)
    for dh in offsets:
        d = ball_edge_distance(
            center_distance_to_zone(x, z, height_ft + dh, top_pct, bot_pct), radius)
        if location_sigma_ft > 0:
            probs = probs + _norm_cdf(d / location_sigma_ft)
        else:
            probs = probs + (d > 0)
    return probs / len(offsets)


def _norm_cdf(t):
    from math import sqrt
    from scipy.special import erf
    return 0.5 * (1.0 + erf(np.asarray(t, dtype=float) / sqrt(2.0)))
