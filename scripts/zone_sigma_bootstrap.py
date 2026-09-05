"""
Bootstrap the zone-region sigma sensitivity check (scripts/zone_sigma_refit.py).

Why this exists: the dedup fix removed 1.8% of challenge rows (165 of 9,197)
and the headline sensitivity number moved 37x (+0.02 -> +0.74 runs/season).
That's a suspiciously large move for a 1.8% row change, and the 3x3-region x
2-role split means some (role, region) cells are fit on only ~200-1,100
challenged pitches -- thin enough that the "which pitches happened to get
challenged" sampling noise could plausibly explain a swing this size on its
own, with no need to invoke the duplicate games specifically.

Design: hold the challenge-opportunity POPULATION fixed (the "base" distribution
per region used as the geometric prior in build_posterior_lookup) and resample
WITH REPLACEMENT only the CHALLENGED subset actually used to fit sigma in
each (role, region) cell -- the same n as observed, drawn from the same cell.
Refit sigma from the resampled sample, propagate to a fresh p_post for every
opportunity, and rerun the full DP (solve/option_values/simulate/summarize)
to get a new zone-region decision gap. Compare against the FIXED canonical
"optimal @ player sigma" baseline (read from disk, never resampled) to get one
bootstrap draw of the sensitivity move. Repeat N times.

This isolates exactly one source of noise: which pitches were challenged.
It does NOT resample games, so it will not by itself distinguish "duplicate
games specifically" from "any resampling of the challenge sample" -- but it
directly answers the operative question, is +0.74 (or +0.02) a number this
data can actually support, or does it swing wildly under ordinary resampling
noise.

Run: python scripts/zone_sigma_bootstrap.py [N_BOOT]
Output: prints the bootstrap distribution; writes
        data/zone_sigma_sensitivity_bootstrap.parquet (one row per replicate)
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from abs_policy import reach_probabilities, solve, option_values, simulate, summarize
from geometry import center_distance_to_zone, ball_edge_distance
from perception import fit_sigma, build_posterior_lookup, posterior_from_observation
from zone_analysis import add_zone_regions

MIN_CHALLENGED_N = 30
BASE_SEED = 17
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 150


def main():
    t_start = time.time()
    con = duckdb.connect("data/baseball.duckdb", read_only=True)

    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    opp["d"] = ball_edge_distance(center_distance_to_zone(
        opp.x_mid.values, opp.z_mid.values, opp.height_ft.values))
    opp["t"] = (opp.inning - 1) * 2 + np.where(opp.inning_topbot == "Bot", 2, 1)

    stand = con.execute("""
        SELECT DISTINCT game_pk, at_bat_number, pitch_number, stand
        FROM statcast WHERE game_year = 2026 AND stand IS NOT NULL
    """).df()
    for c in ("game_pk", "at_bat_number", "pitch_number"):
        stand[c] = stand[c].astype(np.int64)
        opp[c] = opp[c].astype(np.int64)
    opp = opp.merge(stand, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    opp = add_zone_regions(opp).reset_index(drop=True)

    all_pitches = con.execute(
        "SELECT game_pk, inning, inning_topbot FROM statcast WHERE game_year = 2026").df()
    q = reach_probabilities(all_pitches)

    sigma_role_df = pd.read_parquet("data/perception_sigma.parquet")
    role_sigma = dict(zip(sigma_role_df.side, sigma_role_df.sigma_ft))

    canonical = pd.read_parquet("data/policy_decomposition.parquet")
    ply_row = canonical[canonical.label == "optimal @ player sigma"].iloc[0]
    fixed_gap_per_game = ply_row.runs_per_team_game

    # ---- fixed pieces, computed once ----
    cells = []
    for side in ("batting", "fielding"):
        for region in sorted(opp.zone_region.unique()):
            base = opp[(opp.challenger == side) & (opp.zone_region == region)]
            chal_d = base[base.was_challenged].d.values
            cells.append({
                "side": side, "region": region,
                "base_d": base.d.values, "chal_d": chal_d, "n": len(chal_d),
            })
    print(f"cells: {len(cells)}, n_challenged range "
          f"{min(c['n'] for c in cells)}-{max(c['n'] for c in cells)}")

    dre_full = opp.dre.values
    challenger_full = opp.challenger.values
    d_full = opp.d.values

    # precompute group membership once per side: list of positional-index arrays,
    # one per (game_pk, inning, inning_topbot) half-inning -- these never change
    # across bootstrap replicates, only p_post does.
    group_idx = {}
    for side in ("batting", "fielding"):
        sub = opp[opp.challenger == side]
        pos = np.flatnonzero((challenger_full == side))
        # use pandas groupby only ONCE to get positional indices, reused every replicate
        idx_map = sub.groupby(["game_pk", "inning", "inning_topbot"], sort=False).indices
        group_idx[side] = [pos[np.asarray(v)] for v in idx_map.values()]
    print(f"precompute done in {time.time()-t_start:.1f}s, starting {N_BOOT} bootstrap replicates")

    rows = []
    for b in range(N_BOOT):
        t0 = time.time()
        rng = np.random.default_rng(BASE_SEED * 1000 + b)
        p_post = np.empty(len(opp))
        for c in cells:
            if c["n"] < MIN_CHALLENGED_N:
                sigma = role_sigma[c["side"]]
            else:
                boot_chal = rng.choice(c["chal_d"], size=c["n"], replace=True)
                sigma, _, _ = fit_sigma(c["base_d"], boot_chal, c["side"])
            m = (challenger_full == c["side"]) & \
                (opp.zone_region.values == c["region"])
            if not m.any():
                continue
            d_base_side = d_full[challenger_full == c["side"]]
            o_grid, p_grid = build_posterior_lookup(d_base_side, sigma, c["side"])
            obs = d_full[m] + rng.normal(0.0, sigma, size=m.sum())
            p_post[m] = posterior_from_observation(obs, o_grid, p_grid)

        pools = {
            side: [np.column_stack([p_post[idx], dre_full[idx]]) for idx in group_idx[side]]
            for side in ("batting", "fielding")
        }
        W = solve(pools, q, team_bats_on_even_t=True, seed=int(rng.integers(0, 2**31 - 1)))
        ov = option_values(W)
        opp_iter = opp.copy()
        opp_iter["p_post"] = p_post
        opp_iter["won_if_challenged"] = np.where(
            opp_iter.challenger == "batting", opp_iter.d < 0, opp_iter.d > 0)
        sim, _ = simulate(opp_iter, ov, seed=int(rng.integers(0, 2**31 - 1)))
        result = summarize(sim, opp_iter.game_pk.nunique())
        move = (result["runs_per_team_game"] - fixed_gap_per_game) * 162
        rows.append({"replicate": b, "runs_per_team_game": result["runs_per_team_game"],
                      "move_runs_per_season": move})
        if (b + 1) % 10 == 0 or b == 0:
            print(f"  [{b+1}/{N_BOOT}] move={move:+.3f}  ({time.time()-t0:.1f}s/iter, "
                  f"{time.time()-t_start:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df["model_version"] = "zone_sigma_bootstrap_v1"
    df["generated_at"] = datetime.now(timezone.utc).isoformat()
    df.to_parquet("data/zone_sigma_sensitivity_bootstrap.parquet", index=False)

    moves = df.move_runs_per_season.values
    print(f"\n=== BOOTSTRAP DISTRIBUTION OF THE SENSITIVITY MOVE (N={len(moves)}) ===")
    print(f"mean   = {moves.mean():+.3f}")
    print(f"std    = {moves.std():.3f}")
    print(f"min/max = {moves.min():+.3f} / {moves.max():+.3f}")
    for p in (2.5, 25, 50, 75, 97.5):
        print(f"p{p:>5} = {np.percentile(moves, p):+.3f}")
    print(f"\npoint estimates for reference: pre-dedup +0.02, post-dedup (current, full data) +0.74")
    print(f"fraction of replicates <= 0.10 (near the old '+0.02, negligible' framing): "
          f"{(moves <= 0.10).mean()*100:.0f}%")
    print(f"fraction of replicates >= 0.50 (near the new '+0.74' framing): "
          f"{(moves >= 0.50).mean()*100:.0f}%")
    print(f"\ntotal time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
