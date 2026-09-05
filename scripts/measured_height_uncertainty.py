"""
Reproduce the height-rounding-uncertainty figures cited in the README
limitations section (listed vs. measured height): for the vertical-bound
challenged pitches belonging to the robust-set batters (>=3 usable vertical
challenges, so their measured height is a reliable median), how often does
using LISTED height instead of MEASURED height flip the deterministic
in/out call, and how often does the +/-0.5in rounding uncertainty alone
make the call genuinely ambiguous?

This existed previously only as a hardcoded figure in CLAUDE.md with no
script behind it. Written so the number can be regenerated, not just quoted.

Run: python scripts/measured_height_uncertainty.py
Reads: data/measured_heights.parquet, data/measured_heights_pitch_level.parquet
       (both written by scripts/verify_ball_radius.py)
Output: prints the flip-rate / ambiguous-rate report; writes
        data/height_uncertainty_check.parquet (one-row summary)
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geometry import in_abs_zone, p_inside_zone, HEIGHT_ROUNDING_HALFWIDTH_FT

MODEL_VERSION = "height_uncertainty_check_v1"
ROBUST_MIN_PITCHES = 3


def main():
    per_batter = pd.read_parquet("data/measured_heights.parquet")
    vertical = pd.read_parquet("data/measured_heights_pitch_level.parquet")

    robust = per_batter[per_batter.n_pitches >= ROBUST_MIN_PITCHES]
    print(f"robust-set batters (n>={ROBUST_MIN_PITCHES} usable vertical challenges): {len(robust)}")

    v = vertical[vertical.batter.isin(robust.batter)].merge(
        robust[["batter", "measured_height_ft"]], on="batter", how="left")
    print(f"vertical-bound challenged pitches for those batters: {len(v)}")

    call_listed = in_abs_zone(v.x_mid.values, v.z_mid.values, v.listed_height_ft.values)
    call_measured = in_abs_zone(v.x_mid.values, v.z_mid.values, v.measured_height_ft.values)
    flips = call_listed != call_measured
    flip_rate = flips.mean()

    p = p_inside_zone(v.x_mid.values, v.z_mid.values, v.listed_height_ft.values,
                       height_uncertainty_ft=HEIGHT_ROUNDING_HALFWIDTH_FT)
    ambiguous = (p > 0.05) & (p < 0.95)
    ambiguous_rate = ambiguous.mean()

    print(f"\nusing listed instead of measured height flips the deterministic "
          f"in/out call on {flip_rate*100:.2f}% of pitches ({flips.sum()} of {len(v)})")
    print(f"rounding uncertainty alone (+/-0.5in) makes the call genuinely "
          f"ambiguous (0.05<P<0.95) on {ambiguous_rate*100:.2f}% of pitches "
          f"({ambiguous.sum()} of {len(v)})")

    out = pd.DataFrame([{
        "n_robust_batters": len(robust),
        "n_vertical_pitches": len(v),
        "flip_rate": flip_rate,
        "ambiguous_rate": ambiguous_rate,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }])
    Path("data").mkdir(exist_ok=True)
    out.to_parquet("data/height_uncertainty_check.parquet", index=False)
    print("\nsaved data/height_uncertainty_check.parquet")


if __name__ == "__main__":
    main()
