"""
Precompute everything the in-game decision tool needs so the deployed app
never touches DuckDB, refits sigma, or re-solves the DP -- it only looks
things up and does arithmetic, same discipline as build_app_data.py.

Two artifacts, neither of which exists as a saved file anywhere else in the
pipeline (both are recomputed inline, on the fly, by other scripts):

1. data/re_2026.parquet -- the SEASON-2026-ONLY run-expectancy lookup used to
   compute dre for a given (balls, strikes, outs, bases). This is NOT the
   same as data/run_expectancy.parquet, which pools 2024-2026 and is what the
   README/CLAUDE.md warn against using for the decision model, since the run
   environment drifts by season. build_challenge_opportunities.py computes a
   2026-only lookup inline via load_re_lookup(season=2026) and never saves
   it; this script saves that exact same lookup so the app can use it too.

2. data/posterior_lookup.parquet -- P(challenge succeeds | observation o), by
   role, tabulated over a grid of o. This is perception.build_posterior_lookup's
   output, which src/abs_policy.py normally computes on the fly during a DP
   solve. Saving it directly means the app can convert "where you think the
   pitch was" into "P(the call was wrong)" with one np.interp call, using the
   exact same math and the exact same empirical prior (the full 2026
   opportunity set) as the canonical model -- not a re-derivation.

Run: python scripts/build_decision_tool_data.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from run_expectancy import load_re_lookup
from geometry import center_distance_to_zone, ball_edge_distance
from perception import build_posterior_lookup

MODEL_VERSION = "decision_tool_data_v1"
SEASON = 2026


def main():
    generated_at = datetime.now(timezone.utc).isoformat()
    con = duckdb.connect("data/baseball.duckdb")

    print(f"building the {SEASON}-only RE lookup (same method as "
          f"build_challenge_opportunities.py, saved this time)...")
    re_dict = load_re_lookup(con, season=SEASON)
    re_rows = [{"balls": b, "strikes": s, "outs": o, "r1": r1, "r2": r2, "r3": r3,
               "run_exp": v} for (b, s, o, r1, r2, r3), v in re_dict.items()]
    re_df = pd.DataFrame(re_rows)
    re_df["model_version"] = MODEL_VERSION
    re_df["generated_at"] = generated_at
    re_df.to_parquet("data/re_2026.parquet", index=False)
    print(f"  saved {len(re_df)} states -> data/re_2026.parquet")

    print("\nbuilding the posterior lookup (P(win) | observation), by role...")
    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    opp["d"] = ball_edge_distance(center_distance_to_zone(
        opp.x_mid.values, opp.z_mid.values, opp.height_ft.values))
    sigma_role = pd.read_parquet("data/perception_sigma.parquet")
    sig = dict(zip(sigma_role.side, sigma_role.sigma_ft))

    rows = []
    for side in ("batting", "fielding"):
        d_base = opp[opp.challenger == side].d.values
        o_grid, p_grid = build_posterior_lookup(d_base, sig[side], side)
        for o, p in zip(o_grid, p_grid):
            rows.append({"role": side, "o_ft": o, "p_win": p})
        print(f"  {side}: sigma={sig[side]*12:.3f} in, "
              f"{len(o_grid)} grid points, o range [{o_grid.min():.2f}, {o_grid.max():.2f}] ft")

    post = pd.DataFrame(rows)
    post["model_version"] = MODEL_VERSION
    post["generated_at"] = generated_at
    post.to_parquet("data/posterior_lookup.parquet", index=False)
    print(f"  saved {len(post)} rows -> data/posterior_lookup.parquet")


if __name__ == "__main__":
    main()
