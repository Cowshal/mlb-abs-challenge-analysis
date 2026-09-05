"""
Verify the ball-radius correction to the ABS geometry model on a large
sample, then use the confirmed radius to back out each batter's actual
(measured) zone top/bottom from their challenged pitches.

Two-phase design, deliberately height-independent where possible:

Phase A (radius fit): restrict to challenges where the SIDE boundary is
binding (|x_mid| close to the fixed half-width, comfortably clear of top/
bottom regardless of height). Regress MLB's edge_distance (sign-corrected
by the true call) on our center distance. Slope should be ~1.0, intercept
should be ~ball radius. This fit needs no batter height at all.

Phase B (height back-out): for challenges where TOP or BOTTOM is binding,
invert the now-confirmed radius-corrected equation to solve for the true
zone edge MLB actually used, and therefore the batter's measured height.
Median per batter, compared against listed height.

Run: python scripts/verify_ball_radius.py
Output: prints regression + residual report, writes data/measured_heights.parquet
"""
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries
from geometry import location_at_midplate, center_distance_to_zone, HALF_WIDTH, BALL_RADIUS_FT

START_DATE = "2026-04-01"
END_DATE = "2026-05-15"
N_WORKERS = 6

PITCH_FIELDS = [
    "game_pk", "play_id", "batter", "batter_name", "call_name",
    "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az",
]


def final_game_pks():
    """Every completed game_pk in the window, deduplicated.

    Same schedule-API duplicate-listing issue fixed in
    collect_abs_challenges.py: a game_pk can appear under more than one
    date entry (doubleheader/makeup games), which without deduping here
    would fetch and append that game's challenges twice.
    """
    print(f"fetching schedule {START_DATE}..{END_DATE} ...")
    r = get_with_retries(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "startDate": START_DATE, "endDate": END_DATE},
    )
    pks = []
    seen = set()
    for day in r.json().get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            pk = g["gamePk"]
            if pk in seen:
                continue
            seen.add(pk)
            pks.append(pk)
    print(f"got {len(pks)} final games")
    return pks


def fetch_challenges(game_pk):
    url = f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
    try:
        blob = get_with_retries(url).json()
    except Exception as e:
        return game_pk, [], str(e)

    rows, seen = [], set()
    for side in ("team_home", "team_away"):
        for p in blob.get(side) or []:
            if not p.get("is_abs_challenge"):
                continue
            pid = p.get("play_id")
            if pid in seen:
                continue
            seen.add(pid)
            ac = p.get("abs_challenge") or {}
            rec = {f: p.get(f) for f in PITCH_FIELDS}
            rec["is_overturned"] = ac.get("is_overturned")
            rec["edge_distance"] = ac.get("edge_distance")
            rows.append(rec)
    return game_pk, rows, None


def collect():
    pks = final_game_pks()
    all_rows, n_errors, done = [], 0, 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(fetch_challenges, pk): pk for pk in pks}
        for fut in as_completed(futures):
            game_pk, rows, err = fut.result()
            done += 1
            if err:
                n_errors += 1
                print(f"  [{done}/{len(pks)}] game {game_pk} FAILED: {err}")
            else:
                all_rows.extend(rows)
            if done % 25 == 0 or done == len(pks):
                print(f"  progress: {done}/{len(pks)} games, "
                      f"{len(all_rows)} challenges so far, {n_errors} errors")
    print(f"\ntotal challenges collected: {len(all_rows)} (errors: {n_errors})")
    return pd.DataFrame(all_rows)


def fetch_heights(batter_ids):
    print(f"fetching listed heights for {len(batter_ids)} batters ...")
    heights = {}
    ids = list(batter_ids)
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = get_with_retries(
            "https://statsapi.mlb.com/api/v1/people",
            params={"personIds": ",".join(str(b) for b in chunk)},
        )
        for person in r.json().get("people", []):
            h = person.get("height", "")
            if h:
                feet, inches = h.replace('"', "").split("'")
                heights[person["id"]] = int(feet) + int(inches.strip()) / 12.0
    print(f"  got heights for {len(heights)}/{len(ids)} batters")
    return heights


def main():
    df = collect()
    df = df.dropna(subset=["edge_distance", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az"])
    df = df[df["call_name"].isin(["Strike", "Ball"])].reset_index(drop=True)
    print(f"usable rows after dropping incomplete/non-called pitches: {len(df)}")

    x_mid, z_mid = location_at_midplate(
        df.y0.values, df.vy0.values, df.ay.values,
        df.x0.values, df.vx0.values, df.ax.values,
        df.z0.values, df.vz0.values, df.az.values,
    )
    df["x_mid"] = x_mid
    df["z_mid"] = z_mid

    heights = fetch_heights(df.batter.unique())
    df["listed_height_ft"] = df.batter.map(heights)
    df = df.dropna(subset=["listed_height_ft"]).reset_index(drop=True)

    # single source of truth for the distance value: geometry.py's corner-aware
    # rectangle distance, NOT a naive per-axis pick (an earlier version of this
    # script hand-rolled an axis-argmin heuristic that silently mislabeled any
    # pitch that was outside on one axis but had a smaller raw margin on
    # another -- classification below is now derived the same way the geometry
    # module itself decides inside/outside, so it can't diverge again)
    df["center_margin"] = center_distance_to_zone(
        df.x_mid.values, df.z_mid.values, df.listed_height_ft.values
    )

    top0 = 0.535 * df.listed_height_ft
    bot0 = 0.270 * df.listed_height_ft
    dx_out = np.maximum(df.x_mid.abs() - HALF_WIDTH, 0.0)
    dz_out = np.maximum(np.maximum(bot0 - df.z_mid, 0.0), np.maximum(df.z_mid - top0, 0.0))
    outside = (dx_out > 0) | (dz_out > 0)

    side_margin = HALF_WIDTH - df.x_mid.abs()
    bot_margin = df.z_mid - bot0
    top_margin = top0 - df.z_mid
    inside_label = pd.concat([side_margin, bot_margin, top_margin], axis=1,
                              keys=["side", "bot", "top"]).idxmin(axis=1)

    axis = np.select(
        [outside & (dx_out > 0) & (dz_out > 0), outside & (dx_out > 0),
         outside & (dz_out > 0) & (df.z_mid > top0), outside & (dz_out > 0)],
        ["corner", "side", "top", "bot"],
        default=None,
    )
    df["axis"] = np.where(outside, axis, inside_label)

    # ground truth sign: final call_name reflects the post-review ruling
    sign_true = np.where(df.call_name == "Strike", 1, -1)
    df["signed_edge_distance"] = sign_true * df.edge_distance

    # ---- Phase A: height-independent radius fit on side-bound challenges ----
    side = df[df.axis == "side"].copy()
    print(f"\n=== PHASE A: radius fit on {len(side)} side-bound challenges (height-independent) ===")
    if len(side) >= 5:
        slope, intercept = np.polyfit(side.center_margin, side.signed_edge_distance, 1)
        pred = slope * side.center_margin + intercept
        resid_ft = side.signed_edge_distance - pred
        resid_in = resid_ft.abs() * 12
        r2 = 1 - (resid_ft**2).sum() / ((side.signed_edge_distance - side.signed_edge_distance.mean())**2).sum()
        print(f"slope = {slope:.4f} (expect ~1.0)")
        print(f"intercept (fitted ball radius) = {intercept:.4f} ft = {intercept*12:.3f} in "
              f"(expect ~{BALL_RADIUS_FT:.4f} ft = {BALL_RADIUS_FT*12:.3f} in)")
        print(f"R^2 = {r2:.4f}")
        print(f"residual |error| in inches: mean={resid_in.mean():.3f}  "
              f"median={resid_in.median():.3f}  p95={resid_in.quantile(0.95):.3f}")
    else:
        print("not enough side-bound challenges in this window to fit reliably")

    # ---- Phase A cross-check: full-sample fit using fixed BALL_RADIUS_FT ----
    pred_fixed = df.center_margin + BALL_RADIUS_FT
    resid_fixed_in = (df.signed_edge_distance - pred_fixed).abs() * 12
    print(f"\n=== Cross-check: ALL {len(df)} challenges, fixed radius={BALL_RADIUS_FT} "
          f"(listed height, so this also carries height error) ===")
    print(f"residual |error| in inches: mean={resid_fixed_in.mean():.3f}  "
          f"median={resid_fixed_in.median():.3f}  p95={resid_fixed_in.quantile(0.95):.3f}")

    # ---- Phase B: height back-out on top/bottom-bound challenges ----
    vertical = df[df.axis.isin(["top", "bot"])].copy()
    print(f"\n=== PHASE B: height back-out on {len(vertical)} top/bottom-bound challenges ===")

    # m = signed_edge_distance - radius = true center margin from the true edge
    m = vertical.signed_edge_distance - BALL_RADIUS_FT
    true_top = np.where(vertical.axis == "top", vertical.z_mid + m, np.nan)
    true_bot = np.where(vertical.axis == "bot", vertical.z_mid - m, np.nan)
    implied_height = np.where(
        vertical.axis == "top", true_top / 0.535,
        np.where(vertical.axis == "bot", true_bot / 0.270, np.nan)
    )
    vertical["implied_height_ft"] = implied_height

    per_batter = vertical.groupby(["batter", "batter_name"]).agg(
        n_pitches=("implied_height_ft", "size"),
        measured_height_ft=("implied_height_ft", "median"),
        listed_height_ft=("listed_height_ft", "first"),
    ).reset_index()
    per_batter["diff_in"] = (per_batter.measured_height_ft - per_batter.listed_height_ft) * 12

    print(f"batters with >=1 usable vertical challenge: {len(per_batter)}")
    robust = per_batter[per_batter.n_pitches >= 3].copy()
    print(f"batters with >=3 usable vertical challenges (robust median): {len(robust)}")

    print(f"\n=== ROBUST SET (n>=3), listed-vs-measured height diff, inches ===")
    d = robust.diff_in
    print(f"  mean signed        = {d.mean():.3f}")
    print(f"  mean ABSOLUTE      = {d.abs().mean():.3f}")
    print(f"  median ABSOLUTE    = {d.abs().median():.3f}")
    print(f"  std (signed)       = {d.std():.3f}")
    print(f"  p95 (absolute)     = {d.abs().quantile(0.95):.3f}")

    counts, edges = np.histogram(d, bins=12)
    print(f"\n  histogram of (measured - listed), inches:")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * c
        print(f"    [{lo:6.2f}, {hi:6.2f}) {bar} {c}")

    print(f"\n  10 largest |diff| in the robust set:")
    top10 = robust.reindex(robust.diff_in.abs().sort_values(ascending=False).index).head(10)
    print(top10[["batter_name", "n_pitches", "listed_height_ft", "measured_height_ft", "diff_in"]]
          .to_string(index=False))

    # ---- circularity check: top-derived vs bottom-derived height, independently ----
    per_batter_axis = vertical.groupby(["batter", "batter_name", "axis"]).agg(
        n=("implied_height_ft", "size"), h=("implied_height_ft", "median")
    ).reset_index()
    wide = per_batter_axis.pivot_table(index=["batter", "batter_name"], columns="axis",
                                        values=["n", "h"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    both = wide.dropna(subset=["h_top", "h_bot"]).copy()
    both["diff_top_minus_bot_in"] = (both.h_top - both.h_bot) * 12
    print(f"\n=== CIRCULARITY CHECK: batters with >=1 top-bound AND >=1 bottom-bound "
          f"challenge, heights derived independently ===")
    print(f"  n batters with both: {len(both)}")
    if len(both):
        dd = both.diff_top_minus_bot_in
        print(f"  mean |top-derived - bottom-derived| (in) = {dd.abs().mean():.3f}")
        print(f"  median |...|                              = {dd.abs().median():.3f}")
        print(f"  p95 |...|                                 = {dd.abs().quantile(0.95):.3f}")
        both_robust = both[(both.n_top >= 2) & (both.n_bot >= 2)]
        print(f"  n batters with >=2 pitches on EACH side: {len(both_robust)}")
        if len(both_robust):
            dd2 = both_robust.diff_top_minus_bot_in
            print(f"  restricted to that set: mean|diff|={dd2.abs().mean():.3f}  "
                  f"median|diff|={dd2.abs().median():.3f}  p95={dd2.abs().quantile(0.95):.3f}")

    Path("data").mkdir(exist_ok=True)
    per_batter.to_parquet("data/measured_heights.parquet", index=False)
    vertical.to_parquet("data/measured_heights_pitch_level.parquet", index=False)
    print("\nsaved data/measured_heights.parquet and data/measured_heights_pitch_level.parquet")


if __name__ == "__main__":
    main()
