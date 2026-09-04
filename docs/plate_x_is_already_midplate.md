# `plate_x` / `plate_z` are already at the middle of the plate

**Finding:** Statcast's `plate_x` and `plate_z` are the pitch location at
**y = 8.5/12 ft — the middle of home plate**, not the front. The trajectory
re-solve normally prescribed for ABS work is unnecessary.

This contradicts the project walkthrough this repo was built from, which frames
the re-solve as the technically demanding step of the whole project:

> The ABS zone is defined as: 17 inches wide, top at 53.5% of the batter's
> measured height, bottom at 27%. Critically, **location is captured over the
> middle of the plate, not the front.**
>
> Statcast's `plate_x` / `plate_z` are reported at the *front* of the plate. So
> you can't use them directly — you have to re-solve the trajectory.

The first half is right: ABS does judge at the middle of the plate. The second
half is wrong: so does `plate_x`.

## Measurement

Take every pitch carrying both the trajectory parameters (`x0,y0,z0`,
`vx0,vy0,vz0`, `ax,ay,az`) and the reported `plate_x`/`plate_z`, solve the
constant-acceleration trajectory to a candidate y-plane, and compare.

n = 1,220 pitches (2026 games 822974, 822730, 824510, 823702):

| candidate plane | mean abs residual, x | mean abs residual, z |
|---|---|---|
| front of plate, y = 17/12 | 0.2529 in | 0.9758 in |
| **middle of plate, y = 8.5/12** | **0.00000 in** | **0.00000 in** |
| back of plate, y = 0 | 0.2533 in | 0.9877 in |

The middle-of-plate residual is zero to machine precision — max as well as mean.

## This is an identity, not an approximation

The y-plane was **not** fitted before comparing; 8.5/12 was tested as a fixed
hypothesis. Fitting it afterwards as a check, by solving per pitch for the time
at which the trajectory reaches the reported `plate_z` and evaluating y there:

```
implied y: mean = 0.708333 ft   sd = 0.000000   95% CI = [0.708333, 0.708333]
8.5/12                        = 0.708333 ft
difference                    = 0.00000 inches
```

Zero variance across 1,220 pitches. `plate_x`/`plate_z` are not merely close to
the midplate value — they are computed as the trajectory evaluated at y = 8.5/12.

Both feeds publish the value at full float64 precision (14–19 significant
decimals), so there is no rounding to hide behind, and the bulk CSV export
(`pybaseball.statcast()`) is byte-identical to the Savant `gf` feed on the same
pitches (mean and max difference both 0.000000 in over 342 pitches). CSV
`at_bat_number` matches gf `ab_number` with no offset.

## Not an ABS-era change

The same test on a 2024 game (746865, 262 pitches, pre-ABS) gives the same
answer: 0.0000 in at y = 8.5/12, 0.3475 in / 0.9225 in at the front. This is a
long-standing convention, not something introduced for the challenge system.

## Why it still mattered to build the re-solve

`src/geometry.py::location_at_midplate` is redundant given the above, but
building it was not wasted:

1. It is what proved the equivalence. Without an independent midplate
   calculation there is nothing to compare `plate_x` against.
2. It validated against MLB's own `edge_distance` at R² = 1.0000 (slope 0.9999,
   intercept 0.1208 ft = the ball radius, mean residual 0.000 in over 1,136
   side-bound challenges). That is what establishes that **ABS judges at
   midplate** — a separate claim from what `plate_x` means, and one that
   `plate_x` alone cannot settle.

## The practical consequence

The bulk Statcast CSV exports `vx0/vy0/vz0` and `ax/ay/az` but **not** the
`x0/y0/z0` position anchor those are defined against. A trajectory re-solve is
therefore impossible from the CSV alone. Anyone following the re-solve
prescription against bulk data will find themselves stuck, reach for
`release_pos_x/y/z` as a substitute anchor, and silently introduce error —
release position is at y ≈ 54 ft while the velocities and accelerations are
defined at y = 50 ft, so the two are not mutually consistent.

Use `plate_x` / `plate_z` directly. They are already the number you want.

## Reproducing

```bash
python -c "
import sys; sys.path.insert(0,'src')
from geometry import solve_t_at_y, Y_MIDPLATE
# see git history for the full comparison script
"
```
