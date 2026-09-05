"""
Reproduce the season-wide ball-radius classification-accuracy figures cited
in CLAUDE.md and README.md (naive center-only classification disagrees with
the final call on ~34% of all season challenges; ~67% within a
one-ball-radius borderline band; the radius-corrected model matches the
final call on ~99.4%). This existed previously only as a hardcoded finding
in CLAUDE.md with no script behind it -- and it was computed on the
season's challenge table BEFORE the schedule-doubleheader dedup fix
(scripts/collect_abs_challenges.py), so it needs independent reproduction
on the corrected data, not just a re-quote.

plate_x/plate_z in data/abs_challenges.parquet are already the MIDPLATE
location for 2026 data (confirmed in CLAUDE.md), so no trajectory re-solve
is needed here. Batter height uses the measured value from
scripts/verify_ball_radius.py where available (more accurate), falling
back to listed height (fetched from statsapi) otherwise.

CIRCULARITY GUARD: 1,198 of the ~9,071 season challenges are the exact
top/bottom-bound pitches scripts/verify_ball_radius.py used to back out
that batter's measured height in the first place (confirmed by play_id
overlap between data/measured_heights_pitch_level.parquet and
data/abs_challenges.parquet). Classifying one of those pitches with a
height fit that includes that same pitch's own edge_distance is testing
the height back-out formula's ability to invert itself, not the ball-radius
correction's real-world accuracy -- it manufactures near-100% "accuracy" on
that subset. For exactly those 1,198 pitches, height is recomputed
leave-one-out (median of the batter's OTHER vertical challenges only,
falling back to listed height if none remain); every other pitch (the
other ~87% of the season, including this same batter's own side/corner-axis
and out-of-window pitches) uses the normal in-sample measured/listed height,
since there's no such circularity for those.

Run: python scripts/validate_ball_radius_classification.py
Reads: data/abs_challenges.parquet, data/measured_heights.parquet,
       data/measured_heights_pitch_level.parquet
Output: prints the disagreement/accuracy report; writes
        data/ball_radius_classification_check.parquet (one-row summary)
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries
from geometry import center_distance_to_zone, ball_edge_distance, BALL_RADIUS_FT

MODEL_VERSION = "ball_radius_classification_check_v1"


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


def leave_one_out_heights(vertical):
    """For each vertical-bound pitch used to fit its batter's measured height,
    recompute that height from the batter's OTHER vertical pitches only (both
    axes pooled, matching how verify_ball_radius.py's per-batter median is
    built) -- so testing classification on this exact pitch isn't testing the
    height-fit's ability to reproduce its own input. Returns {play_id: height_ft}
    for pitches with >=1 other vertical pitch to fall back on; pitches whose
    batter has no other vertical challenge are left out (caller falls back to
    listed height for those)."""
    loo = {}
    for batter, g in vertical.groupby("batter"):
        heights = g.implied_height_ft.values
        play_ids = g.play_id.values
        if len(heights) < 2:
            continue
        for i, pid in enumerate(play_ids):
            others = np.delete(heights, i)
            loo[pid] = np.median(others)
    return loo


def main():
    df = pd.read_parquet("data/abs_challenges.parquet")
    df = df.dropna(subset=["plate_x", "plate_z", "call_name"]).reset_index(drop=True)
    df = df[df.call_name.isin(["Strike", "Ball"])].reset_index(drop=True)
    print(f"season challenges with usable location + final call: {len(df)}")

    measured = pd.read_parquet("data/measured_heights.parquet")
    measured_map = dict(zip(measured.batter, measured.measured_height_ft))

    vertical = pd.read_parquet("data/measured_heights_pitch_level.parquet")
    loo_map = leave_one_out_heights(vertical)
    # batters whose single vertical pitch has no leave-one-out fallback need a
    # listed height fetched too, even though they otherwise have a measured one
    single_pitch_batters = [b for b, g in vertical.groupby("batter") if len(g) < 2]

    need_listed = sorted(set(
        [b for b in df.batter.unique() if b not in measured_map] + single_pitch_batters))
    listed_map = fetch_heights(need_listed)

    height_ft = df.batter.map(measured_map)
    height_source = np.where(df.batter.isin(measured_map), "measured", "listed")
    fallback = df.batter.map(listed_map)
    height_ft = height_ft.where(height_ft.notna(), fallback)
    df["height_ft"] = height_ft
    df["height_source"] = height_source
    is_circular = df.play_id.isin(set(vertical.play_id))
    has_loo = df.play_id.isin(loo_map)
    loo_height = df.play_id.map(loo_map)
    listed_for_circular = df.batter.map(listed_map)
    resolved_height = loo_height.where(has_loo, listed_for_circular)
    df["height_source"] = np.where(
        is_circular,
        np.where(has_loo, "measured_loo", "listed_loo_fallback"),
        df.height_source)
    df["height_ft"] = np.where(is_circular, resolved_height, df["height_ft"])
    print(f"circularity guard applied to {is_circular.sum()} pitches "
          f"(this batter's own height-fit input, now leave-one-out or listed)")

    before = len(df)
    df = df.dropna(subset=["height_ft"]).reset_index(drop=True)
    print(f"dropped {before - len(df)} challenges with no height (measured or listed); "
          f"{len(df)} remain ({(df.height_source == 'measured').sum()} measured, "
          f"{(df.height_source == 'measured_loo').sum()} measured-leave-one-out, "
          f"{(df.height_source == 'listed').sum()} listed, "
          f"{(df.height_source == 'listed_loo_fallback').sum()} listed-fallback-for-circular)")

    center = center_distance_to_zone(df.plate_x.values, df.plate_z.values, df.height_ft.values)
    edge = ball_edge_distance(center)

    naive_call = np.where(center > 0, "Strike", "Ball")
    corrected_call = np.where(edge > 0, "Strike", "Ball")
    truth = df.call_name.values

    naive_disagree = naive_call != truth
    corrected_match = corrected_call == truth
    borderline = np.abs(center) < BALL_RADIUS_FT

    n = len(df)
    n_border = borderline.sum()
    naive_rate = naive_disagree.mean()
    naive_border_rate = naive_disagree[borderline].mean()
    corrected_rate = corrected_match.mean()
    corrected_border_rate = corrected_match[borderline].mean()

    print(f"\n=== SEASON-WIDE BALL-RADIUS CLASSIFICATION CHECK (n={n}) ===")
    print(f"naive (center-only) disagreement with final call: {naive_rate*100:.1f}% "
          f"({naive_disagree.sum()} of {n})")
    print(f"borderline band (|center| < 1 ball radius): n={n_border} "
          f"({n_border/n*100:.1f}% of season)")
    print(f"  naive disagreement within borderline band: {naive_border_rate*100:.1f}%")
    print(f"radius-corrected model matches final call: {corrected_rate*100:.2f}% overall, "
          f"{corrected_border_rate*100:.2f}% within borderline band")

    out = pd.DataFrame([{
        "n_challenges": n,
        "naive_disagreement_rate": naive_rate,
        "n_borderline": int(n_border),
        "naive_disagreement_rate_borderline": naive_border_rate,
        "corrected_match_rate": corrected_rate,
        "corrected_match_rate_borderline": corrected_border_rate,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }])
    Path("data").mkdir(exist_ok=True)
    out.to_parquet("data/ball_radius_classification_check.parquet", index=False)
    print("\nsaved data/ball_radius_classification_check.parquet")


if __name__ == "__main__":
    main()
