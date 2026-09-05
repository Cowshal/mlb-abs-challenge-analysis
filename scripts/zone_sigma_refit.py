"""
Sensitivity check triggered by scripts/zone_analysis.py: the role gap in
challenge success swings 37.7 percentage points across zone regions
(LR test p<0.0001, n=9,034) -- large enough that "one sigma per role,
everywhere in the zone" is worth checking, not just flagging as a footnote.

This is a SENSITIVITY ANALYSIS, not a replacement for the canonical model.
data/option_values.parquet, MODEL_VERSION="role_sigma_v1", and every headline
number in the README/app/writeup are untouched by this script. It answers one
question: if sigma is allowed to vary by zone region instead of being pooled
across the whole zone, how much does the headline decision gap move?

Method: reuses abs_policy.solve() / option_values() / simulate() / summarize()
unchanged -- those functions only consume (p_success, dre) per opportunity and
don't care how p_success was derived. Only the posterior-building step is
replaced: instead of one sigma per role applied to every pitch, each pitch
gets the sigma fitted for its own (role, zone region).

Run: python scripts/zone_sigma_refit.py
Output: data/zone_sigma.parquet             (fitted sigma per role x region)
        data/zone_sigma_sensitivity.parquet (headline numbers, zone-sigma vs
                                              role-sigma, one row each)
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from abs_policy import reach_probabilities, solve, option_values, simulate, summarize
from geometry import center_distance_to_zone, ball_edge_distance
from perception import fit_sigma, build_posterior_lookup, posterior_from_observation
from zone_analysis import add_zone_regions  # reuse the exact same region definition

MODEL_VERSION = "zone_sigma_refit_v1"
MIN_CHALLENGED_N = 30
SEED = 17


def main():
    generated_at = datetime.now(timezone.utc).isoformat()
    con = duckdb.connect("data/baseball.duckdb")

    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    opp["d"] = ball_edge_distance(center_distance_to_zone(
        opp.x_mid.values, opp.z_mid.values, opp.height_ft.values))
    opp["t"] = (opp.inning - 1) * 2 + np.where(opp.inning_topbot == "Bot", 2, 1)

    print("pulling batter stand for the full opportunity set (needed for the "
          "same batter-relative in/away axis as zone_analysis.py)...")
    stand = con.execute("""
        SELECT DISTINCT game_pk, at_bat_number, pitch_number, stand
        FROM statcast WHERE game_year = 2026 AND stand IS NOT NULL
    """).df()
    for c in ("game_pk", "at_bat_number", "pitch_number"):
        stand[c] = stand[c].astype(np.int64)
        opp[c] = opp[c].astype(np.int64)
    opp = opp.merge(stand, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    print(f"{len(opp):,} of the full opportunity set matched to a batter stand")

    opp = add_zone_regions(opp)

    act = opp[opp.was_challenged]
    sigma_role_df = pd.read_parquet("data/perception_sigma.parquet")
    role_sigma = dict(zip(sigma_role_df.side, sigma_role_df.sigma_ft))

    print("\n=== FITTING SIGMA PER (ROLE, ZONE REGION) ===")
    rows = []
    region_sigma = {}
    for side in ("batting", "fielding"):
        for region in sorted(opp.zone_region.unique()):
            base = opp[(opp.challenger == side) & (opp.zone_region == region)]
            chal = base[base.was_challenged]
            n = len(chal)
            if n < MIN_CHALLENGED_N:
                sigma = role_sigma[side]
                note = f"fallback to pooled role sigma (only {n} challenges)"
            else:
                sigma, cutoff, ll = fit_sigma(base.d.values, chal.d.values, side)
                note = "fitted"
            region_sigma[(side, region)] = sigma
            rows.append({"challenger": side, "zone_region": region, "n_challenged": n,
                        "sigma_in": sigma * 12, "note": note})
            print(f"  {side:9s} {region:12s} n={n:4d}  sigma={sigma*12:.3f} in  ({note})")

    zone_sigma_df = pd.DataFrame(rows)
    zone_sigma_df["model_version"] = MODEL_VERSION
    zone_sigma_df["generated_at"] = generated_at
    zone_sigma_df.to_parquet("data/zone_sigma.parquet", index=False)

    print("\n=== RE-RUNNING THE DP WITH PER-REGION SIGMA ===")
    all_pitches = con.execute(
        "SELECT game_pk, inning, inning_topbot FROM statcast WHERE game_year = 2026").df()
    q = reach_probabilities(all_pitches)

    rng = np.random.default_rng(SEED)
    opp = opp.copy()
    p_post = np.empty(len(opp))
    for (side, region), sigma in region_sigma.items():
        m = ((opp.challenger == side) & (opp.zone_region == region)).values
        if not m.any():
            continue
        d_base_side = opp.d.values[opp.challenger.values == side]
        o_grid, p_grid = build_posterior_lookup(d_base_side, sigma, side)
        obs = opp.d.values[m] + rng.normal(0.0, sigma, size=m.sum())
        p_post[m] = posterior_from_observation(obs, o_grid, p_grid)
    opp["p_post"] = p_post
    opp["won_if_challenged"] = np.where(opp.challenger == "batting", opp.d < 0, opp.d > 0)

    pools = {side: [g[["p_post", "dre"]].to_numpy()
                    for _, g in opp[opp.challenger == side].groupby(
                        ["game_pk", "inning", "inning_topbot"])]
             for side in ("batting", "fielding")}
    W = solve(pools, q, team_bats_on_even_t=True, seed=SEED)
    ov_zone = option_values(W)
    sim_zone, _ = simulate(opp, ov_zone, seed=SEED)
    zone_result = summarize(sim_zone, opp.game_pk.nunique())
    zone_result["label"] = "optimal @ zone-region sigma"

    print("\n=== COMPARISON: pooled role sigma vs per-region sigma ===")
    canonical = pd.read_parquet("data/policy_decomposition.parquet")
    ply_row = canonical[canonical.label == "optimal @ player sigma"].iloc[0]
    obs_row = canonical[canonical.label == "observed 2026"].iloc[0]

    comparison = pd.DataFrame([
        {"label": "observed 2026", "runs_per_team_game": obs_row.runs_per_team_game},
        {"label": "optimal @ player sigma (role-only, canonical)",
         "runs_per_team_game": ply_row.runs_per_team_game},
        {"label": "optimal @ zone-region sigma (sensitivity)",
         "runs_per_team_game": zone_result["runs_per_team_game"]},
    ])
    comparison["decision_gap_vs_observed_per_game"] = (
        comparison.runs_per_team_game - obs_row.runs_per_team_game)
    comparison["decision_gap_vs_observed_per_season"] = (
        comparison.decision_gap_vs_observed_per_game * 162)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    move = (zone_result["runs_per_team_game"] - ply_row.runs_per_team_game) * 162
    print(f"\nheadline decision gap moves by {move:+.2f} runs/team-season "
          f"when sigma is allowed to vary by zone region instead of being "
          f"pooled per role.")

    comparison["model_version"] = MODEL_VERSION
    comparison["generated_at"] = generated_at
    comparison.to_parquet("data/zone_sigma_sensitivity.parquet", index=False)
    print("\nsaved data/zone_sigma.parquet, data/zone_sigma_sensitivity.parquet")


if __name__ == "__main__":
    main()
