"""
Estimate players' perceptual noise (sigma) from WHERE they chose to challenge,
independently of how often they challenged or how often they were right.

The identifying idea: under perfect information nobody challenges a pitch that
sits well inside the zone. The observed distribution of true pitch locations
among actual challenges therefore reveals how noisy the player's read is. Volume
and success rate are never referenced, so the estimate cannot absorb the
decision gap we are trying to measure.

Run: python scripts/estimate_perception_sigma.py
Output: data/perception_sigma.parquet
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geometry import center_distance_to_zone, ball_edge_distance
from perception import fit_sigma, likelihood_surface, challenge_probability

# Separate from abs_policy.MODEL_VERSION -- this versions the sigma
# ESTIMATION method, not the downstream policy model that consumes it.
SIGMA_MODEL_VERSION = "perception_sigma_v1"


def main():
    opp = pd.read_parquet("data/challenge_opportunities.parquet")

    # d = signed ball-edge distance, positive = pitch is in the zone
    opp["d"] = ball_edge_distance(center_distance_to_zone(
        opp.x_mid.values, opp.z_mid.values, opp.height_ft.values))

    chal = pd.read_parquet("data/abs_challenges.parquet")
    chal["game_pk"] = chal.game_pk.astype(np.int64)
    chal["ab_number"] = chal.ab_number.astype(np.int64)
    chal["pitch_number"] = chal.pitch_number.astype(np.int64)
    chal = chal.drop_duplicates(subset=["game_pk", "ab_number", "pitch_number"])
    print(f"{len(chal):,} challenges collected; {len(opp):,} opportunities")

    key = ["game_pk", "at_bat_number", "pitch_number"]
    opp["game_pk"] = opp.game_pk.astype(np.int64)
    opp["at_bat_number"] = opp.at_bat_number.astype(np.int64)
    opp["pitch_number"] = opp.pitch_number.astype(np.int64)
    marks = chal.rename(columns={"ab_number": "at_bat_number"})[
        key + ["is_overturned", "challenging_player_type", "edge_distance"]]
    opp = opp.merge(marks, on=key, how="left", indicator=True)
    opp["challenged"] = opp._merge == "both"
    print(f"{opp.challenged.sum():,} challenges matched into the opportunity set "
          f"({100*opp.challenged.sum()/len(chal):.1f}% of collected)")

    # sanity: our signed d should agree with MLB's edge_distance magnitude
    m = opp[opp.challenged & opp.edge_distance.notna()]
    agree = (np.abs(np.abs(m.d) - m.edge_distance) * 12)
    print(f"|our d| vs MLB edge_distance on matched challenges: "
          f"mean={agree.mean():.4f} in  median={agree.median():.4f} in")

    rows = []
    for side in ("batting", "fielding"):
        base = opp[opp.challenger == side]
        d_base = base.d.values
        d_chal = base[base.challenged].d.values
        if len(d_chal) < 50:
            print(f"{side}: only {len(d_chal)} challenges, skipping")
            continue

        sigma, cutoff, ll = fit_sigma(d_base, d_chal, side)
        wins = (d_chal < 0) if side == "batting" else (d_chal > 0)
        print(f"\n=== {side} ===")
        print(f"  base n={len(d_base):,}  challenged n={len(d_chal):,}")
        print(f"  observed success rate in this sample: {wins.mean():.3f} "
              f"(NOT used in the fit)")
        print(f"  fitted sigma  = {sigma*12:.3f} inches")
        print(f"  fitted cutoff = {cutoff*12:+.3f} inches "
              f"(observed location at which they pull the trigger 50% of the time)")
        print(f"  log-likelihood = {ll:,.1f}")

        # how deep into the zone do challenges actually go? the direct evidence
        hopeless = (d_chal > 0) if side == "batting" else (d_chal < 0)
        print(f"  challenges that were hopeless (wrong side of the boundary): "
              f"{hopeless.mean():.3f}")
        q = np.percentile(np.abs(d_chal[hopeless]) * 12, [50, 90, 99]) if hopeless.any() else []
        if len(q):
            print(f"  depth on the wrong side, inches: p50={q[0]:.2f} p90={q[1]:.2f} p99={q[2]:.2f}")

        rows.append({"side": side, "sigma_ft": sigma, "sigma_in": sigma * 12,
                     "cutoff_ft": cutoff, "n_challenged": len(d_chal), "loglik": ll})

        # identifiability: is the optimum a peak or a ridge?
        sig_grid = np.linspace(max(sigma * 0.3, 0.005), sigma * 2.5, 25)
        cut_grid = np.linspace(cutoff - 0.25, cutoff + 0.25, 25)
        S = likelihood_surface(d_base, d_chal, side, sig_grid, cut_grid)
        S = S - S.max()
        i, j = np.unravel_index(np.argmax(S), S.shape)
        # profile curvature along each axis through the optimum
        prof_sigma = S[:, j]
        prof_cut = S[i, :]
        drop_sigma = prof_sigma.max() - np.median(prof_sigma)
        drop_cut = prof_cut.max() - np.median(prof_cut)
        print(f"  identifiability: optimum at sigma={sig_grid[i]*12:.3f} in, "
              f"cutoff={cut_grid[j]*12:+.3f} in")
        print(f"    log-lik drop from peak to median along sigma axis : {drop_sigma:8.1f}")
        print(f"    log-lik drop from peak to median along cutoff axis: {drop_cut:8.1f}")
        # 2-unit log-likelihood interval on sigma (approx 95%)
        ok = sig_grid[prof_sigma > prof_sigma.max() - 2]
        print(f"    sigma within 2 log-lik units: "
              f"[{ok.min()*12:.3f}, {ok.max()*12:.3f}] inches")
        if drop_sigma < 5 or drop_cut < 5:
            print("    WARNING: shallow in at least one direction -- possible ridge")

    out = pd.DataFrame(rows)
    out["model_version"] = SIGMA_MODEL_VERSION
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    out.to_parquet("data/perception_sigma.parquet", index=False)
    print("\nsaved data/perception_sigma.parquet")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
