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
                   top_pct=0.535, bot_pct=0.270, radius=BALL_RADIUS_FT):
    """
    Probability the ball is inside the ABS zone, integrating over uncertainty
    in batter height. Pass height_uncertainty_ft=0 (default) for a batter
    with a trusted measured height -> reduces to a deterministic 0/1 call.
    For a listed-height-only batter, pass HEIGHT_ROUNDING_HALFWIDTH_FT to
    integrate over the rounding-to-nearest-inch uncertainty.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    height_ft = np.atleast_1d(np.asarray(height_ft, dtype=float))
    if height_uncertainty_ft == 0:
        return in_abs_zone(x, z, height_ft, top_pct, bot_pct, radius).astype(float)
    grid = np.linspace(-height_uncertainty_ft, height_uncertainty_ft, n_grid)
    probs = np.zeros(np.broadcast_shapes(x.shape, z.shape, height_ft.shape), dtype=float)
    for dh in grid:
        probs = probs + in_abs_zone(x, z, height_ft + dh, top_pct, bot_pct, radius)
    return probs / n_grid
