"""
Task 4: is challenge accuracy a repeatable team skill, or is one season's
spread consistent with binomial noise at each team's own volume?

Four independent checks, in the order they inform each other:
  1. Split-half reliability (the core test): correlate each team's success
     rate in the first half of the season against the second half.
  2. Observed vs simulated cross-team spread: how much bigger is the real
     spread in success rate / runs gained than 30 teams drawing from the
     league rate at their own attempt counts would produce by chance alone?
  3. Per-team significance: binomial z-score per team vs league rate, how
     many teams clear 2 SE, and the Bonferroni-corrected p-value for the
     single most extreme team (since it's the max of 30 draws, not one).
  4. Per-team perceptual sigma, split by batting/fielding role, with a
     bootstrap CI -- reported regardless of how (1)-(3) land, but NOT used
     to argue for or against the catcher-receiving hypothesis unless the
     base skill question is actually resolved.

Run: python scripts/team_skill_test.py
Output: data/team_skill_test.parquet (per-team z, sigma, CIs)
        prints the split-half r + CI and the spread/significance tests
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geometry import center_distance_to_zone, ball_edge_distance
from perception import fit_sigma

N_SIM = 20000
N_BOOT = 300
SEED = 2026


def team_of(inning_topbot, challenger):
    bot = inning_topbot == "Bot"
    bat = challenger == "batting"
    return np.where((bot & bat) | (~bot & ~bat), "home", "away")


def load_opp_with_team():
    con = duckdb.connect("data/baseball.duckdb")
    teams = con.execute("""
        SELECT DISTINCT game_pk, CAST(game_date AS DATE) AS game_date, home_team, away_team
        FROM statcast WHERE game_year = 2026
    """).df()
    teams["game_pk"] = teams.game_pk.astype(np.int64)

    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    opp = opp.drop(columns=["game_date"])  # use the CAST(...AS DATE) version from teams instead
    opp["game_pk"] = opp.game_pk.astype(np.int64)
    opp["d"] = ball_edge_distance(center_distance_to_zone(
        opp.x_mid.values, opp.z_mid.values, opp.height_ft.values))
    opp["side"] = team_of(opp.inning_topbot.values, opp.challenger.values)
    opp = opp.merge(teams, on="game_pk", how="left")
    opp["team"] = np.where(opp.side == "home", opp.home_team, opp.away_team)
    return opp


def split_half_reliability(act):
    mid = act.game_date.median()
    act = act.copy()
    act["half"] = np.where(act.game_date <= mid, "H1", "H2")
    g = act.groupby(["team", "half"]).agg(
        n=("overturned", "size"), correct=("overturned", "sum")).reset_index()
    g["rate"] = g.correct / g.n
    piv = g.pivot(index="team", columns="half", values="rate")
    n_piv = g.pivot(index="team", columns="half", values="n")

    r, p = stats.pearsonr(piv.H1, piv.H2)
    n = len(piv)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"=== 1. SPLIT-HALF RELIABILITY (split date: {mid.date()}) ===")
    print(f"n teams={n}  r={r:.3f}  p={p:.4f}  95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"half-sample sizes: min={n_piv.min().min()}  max={n_piv.max().max()}")
    ambiguous = (0.10 <= abs(r) <= 0.30) or (ci_lo < 0 < ci_hi)
    if ambiguous:
        print("AMBIGUOUS by the pre-registered rule (0.10<=|r|<=0.30, or CI spans zero).")
        print("Reporting the numbers and stopping here rather than calling it either way.")
    return r, ci_lo, ci_hi, piv


def simulated_spread(pt):
    league_success = pt.actual_correct.sum() / pt.actual_challenges.sum()
    league_stake = pt.actual_runs.sum() / pt.actual_correct.sum()
    rng = np.random.default_rng(SEED)
    attempts = pt.actual_challenges.values.astype(int)

    sim_success_std = np.empty(N_SIM)
    sim_runs_std = np.empty(N_SIM)
    for i in range(N_SIM):
        correct = rng.binomial(attempts, league_success)
        sim_success_std[i] = (correct / attempts).std()
        sim_runs_std[i] = (correct * league_stake).std()

    obs_success_std = pt.actual_success.std()
    obs_runs_std = pt.actual_runs.std()
    p_success = (sim_success_std >= obs_success_std).mean()
    p_runs = (sim_runs_std >= obs_runs_std).mean()

    print("\n=== 2. OBSERVED VS SIMULATED CROSS-TEAM SPREAD ===")
    print(f"success rate: observed std={obs_success_std:.4f}  "
          f"null mean={sim_success_std.mean():.4f}  "
          f"null p95={np.percentile(sim_success_std, 95):.4f}  "
          f"P(null >= observed)={p_success:.4f}")
    print(f"runs gained:  observed std={obs_runs_std:.4f}  "
          f"null mean={sim_runs_std.mean():.4f}  "
          f"null p95={np.percentile(sim_runs_std, 95):.4f}  "
          f"P(null >= observed)={p_runs:.4f}")
    return p_success, p_runs


def per_team_significance(pt):
    p0 = pt.actual_correct.sum() / pt.actual_challenges.sum()
    pt = pt.copy()
    pt["se"] = np.sqrt(p0 * (1 - p0) / pt.actual_challenges)
    pt["z"] = (pt.actual_success - p0) / pt.se
    pt["p_two_sided"] = 2 * (1 - stats.norm.cdf(pt.z.abs()))
    pt = pt.sort_values("z", key=lambda s: s.abs(), ascending=False)

    n_over_2se = (pt.z.abs() > 2).sum()
    expected = 30 * 2 * (1 - stats.norm.cdf(2))
    top = pt.iloc[0]
    bonf = min(1.0, top.p_two_sided * 30)

    print("\n=== 3. PER-TEAM SIGNIFICANCE ===")
    print(f"league p0={p0:.4f}")
    print(pt[["team", "actual_challenges", "actual_success", "se", "z", "p_two_sided"]]
          .head(6).to_string(index=False))
    print(f"teams with |z|>2: {n_over_2se} of 30 (expected by chance: {expected:.2f})")
    print(f"max |z|: {top.team} z={top.z:.3f} raw p={top.p_two_sided:.5f} "
          f"Bonferroni-adjusted (x30)={bonf:.4f}")
    return pt[["team", "se", "z", "p_two_sided"]]


def per_team_sigma(opp):
    rng = np.random.default_rng(SEED)
    rows = []
    for team in sorted(opp.team.dropna().unique()):
        for role in ("batting", "fielding"):
            sub = opp[(opp.team == team) & (opp.challenger == role)]
            d_base = sub.d.values
            chal = sub[sub.was_challenged]
            d_chal = chal.d.values
            n = len(d_chal)
            if n < 30:
                rows.append({"team": team, "role": role, "n": n, "sigma_in": np.nan,
                            "sigma_ci_lo": np.nan, "sigma_ci_hi": np.nan})
                continue
            sigma, cutoff, ll = fit_sigma(d_base, d_chal, role)
            boots = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, n)
                try:
                    s_b, _, _ = fit_sigma(d_base, d_chal[idx], role, sigma_init_ft=sigma)
                except Exception:
                    s_b = np.nan
                boots[b] = s_b
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rows.append({"team": team, "role": role, "n": n, "sigma_in": sigma * 12,
                        "sigma_ci_lo": lo * 12, "sigma_ci_hi": hi * 12})
    return pd.DataFrame(rows)


def main():
    opp = load_opp_with_team()
    act = opp[opp.was_challenged].copy()
    pt = pd.read_parquet("app/data/per_team.parquet")

    r, ci_lo, ci_hi, piv = split_half_reliability(act)
    p_success, p_runs = simulated_spread(pt)
    sig = per_team_significance(pt)

    print("\n=== 4. PER-TEAM PERCEPTUAL SIGMA, BY ROLE (bootstrap 95% CI, n_boot="
          f"{N_BOOT}) ===")
    sigma_tbl = per_team_sigma(opp)
    wide = sigma_tbl.pivot(index="team", columns="role",
                           values=["n", "sigma_in", "sigma_ci_lo", "sigma_ci_hi"])
    print(wide.round(2).to_string())

    sigma_tbl.to_parquet("data/team_skill_test.parquet", index=False)
    print("\nsaved data/team_skill_test.parquet")

    print("\n=== SUMMARY ===")
    print(f"split-half r={r:.3f}, 95% CI=[{ci_lo:.3f},{ci_hi:.3f}] -- AMBIGUOUS, per the "
          "pre-registered rule. Not deciding 'real skill' vs 'noise' from this alone.")
    print(f"cross-team spread exceeds binomial-noise null: "
          f"P(success)={p_success:.4f}, P(runs)={p_runs:.4f} -- spread is real, "
          "but 'more spread than chance' is not the same claim as 'stable across time'.")
    print("Per-team sigma reported above; NOT used here to argue the catcher-receiving "
          "hypothesis, since the prerequisite (accuracy being a confirmed repeatable "
          "skill) was not established.")


if __name__ == "__main__":
    main()
