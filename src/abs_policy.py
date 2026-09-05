"""
Optimal ABS challenge policy by backward induction.

State: V(t, j, k)
    t = half-inning index (1..18 regulation, then extras)
    j = index of the challengeable called pitch within half-inning t
    k = incorrect challenges already spent (0, 1, 2)

V is the expected additional runs the team captures, from that point to the end
of the game, by using its remaining challenges optimally. It is a pure option
value: the worth of still holding the right, not the value of any single call.

At each opportunity with success probability p and flip value dre:
    challenge:  p*(dre + V(next, k)) + (1-p)*V(next, k+1)
    decline:    V(next, k)
so challenge iff  p*dre > (1-p)*C(k),  where C(k) = V(k) - V(k+1) is the run
value of one incorrect-challenge token. The cost carries (1-p), not 1 -- a
correct challenge is free. That asymmetry is the whole model.

Boundaries:
    V(t, j, 2) = 0   -- rights exhausted, no future option value (absorption)
    V(T, ., k) = 0   -- game over
Extra innings restore one challenge: crossing into one maps k -> max(k-1, 0).

The opportunity sequence is bootstrapped from real half-innings so that the
number of chances and the joint (p, dre) distribution are both empirical --
sampled jointly, never independently, since borderline pitches and high-leverage
situations are correlated.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

N_REGULATION_HALF_INNINGS = 18
MAX_HALF_INNINGS = 30  # allow extras out to the 15th
N_SAMPLES = 4000
SEED = 17

# Bump this whenever the policy model itself changes (not for data refreshes
# with the same model). "role_sigma_v1" = role-specific perceptual sigma
# (batting vs fielding), fit from where players chose to challenge -- the
# model behind every headline number in the README and writeup. A prior
# version used one pooled sigma across roles, which the README documents as
# reversing the catcher-vs-batter finding; that version must never be the one
# a saved artifact silently reflects.
MODEL_VERSION = "role_sigma_v1"


def _stamp(df, generated_at):
    """Attach model provenance to a DataFrame before it's saved. Every
    parquet under data/ and app/data/ produced by this module carries these
    two columns so a mixed-vintage set can be detected by inspection instead
    of by a reader noticing the numbers don't add up."""
    df = df.copy()
    df["model_version"] = MODEL_VERSION
    df["generated_at"] = generated_at
    return df


def load_pools(path="data/challenge_opportunities.parquet"):
    """Group opportunities into half-innings, split by which side may challenge."""
    df = pd.read_parquet(path)
    pools = {}
    for side in ("batting", "fielding"):
        sub = df[df.challenger == side]
        groups = sub.groupby(["game_pk", "inning", "inning_topbot"])
        pools[side] = [g[["p_success", "dre"]].to_numpy() for _, g in groups]
    return pools


def reach_probabilities(df_all_pitches):
    """q[t] = P(half-inning t is played | half-inning t-1 was played)."""
    hi = df_all_pitches[["game_pk", "inning", "inning_topbot"]].drop_duplicates()
    hi["t"] = (hi.inning - 1) * 2 + np.where(hi.inning_topbot == "Bot", 2, 1)
    counts = hi.groupby("t").game_pk.nunique().sort_index()
    q = {}
    for t in range(1, MAX_HALF_INNINGS + 1):
        cur, prev = counts.get(t, 0), counts.get(t - 1, None)
        q[t] = 1.0 if t == 1 else (0.0 if prev in (None, 0) else min(1.0, cur / prev))
    return q


def solve_half_inning(opps, W_next, C_absorb=0.0):
    """
    Backward chain through the opportunities inside one half-inning.
    W_next is [V(t+1, 0)] indexed by k. Returns value at the start, by k.
    """
    U = np.array([W_next[0], W_next[1], C_absorb], dtype=float)
    for p, dre in opps[::-1]:
        nxt = U.copy()
        for k in (0, 1):
            decline = nxt[k]
            challenge = p * (dre + nxt[k]) + (1.0 - p) * nxt[k + 1]
            U[k] = max(decline, challenge)
        U[2] = 0.0
    return U


def solve(pools, q, team_bats_on_even_t=True, n_samples=N_SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    W = {MAX_HALF_INNINGS + 1: np.zeros(3)}

    for t in range(MAX_HALF_INNINGS, 0, -1):
        nxt = W[t + 1]
        # entering an extra inning restores one challenge
        is_extra_inning_start = t > N_REGULATION_HALF_INNINGS and (t % 2 == 1)
        if is_extra_inning_start:
            nxt = np.array([nxt[max(k - 1, 0)] for k in range(3)])

        bats = (t % 2 == 0) if team_bats_on_even_t else (t % 2 == 1)
        pool = pools["batting"] if bats else pools["fielding"]

        idx = rng.integers(0, len(pool), size=n_samples)
        acc = np.zeros(3)
        for i in idx:
            acc += solve_half_inning(pool[i], nxt)
        val = acc / n_samples

        val = q.get(t, 0.0) * val  # half-inning may not be played at all
        val[2] = 0.0
        W[t] = val
    return W


def option_values(W):
    """C(k) = V(k) - V(k+1): run value of one incorrect-challenge token."""
    rows = []
    for t in sorted(k for k in W if k <= MAX_HALF_INNINGS):
        rows.append({
            "t": t, "inning": (t + 1) // 2,
            "half": "Top" if t % 2 == 1 else "Bot",
            "V_k0": W[t][0], "V_k1": W[t][1],
            "C_k0": W[t][0] - W[t][1], "C_k1": W[t][1] - W[t][2],
        })
    return pd.DataFrame(rows)


def threshold(C, dre):
    """Minimum success probability that justifies challenging: C / (dre + C)."""
    return C / (dre + C)


def simulate(opp, ov, seed=SEED):
    """
    Play the optimal policy through real 2026 games with the actual resource
    dynamics: two challenges, one spent only on an INCORRECT call, rights gone
    after two incorrect, one restored each extra inning.

    Decisions are made on the player's NOISY POSTERIOR (column p_post) and
    outcomes are resolved against the TRUE location (column won_if_challenged).
    Those must be separate: recomputing p with a wider sigma and then also
    resolving the coin flip against that same p would corrupt the result in both
    directions -- it would let the model both decide and be graded on beliefs it
    invented, and no information gap could ever appear.

    A team's opportunities are its called strikes while batting plus its called
    balls while fielding: the home team bats in Bot halves and fields in Top.
    """
    rng = np.random.default_rng(seed)
    cmap = ov.set_index("t")[["C_k0", "C_k1"]].to_dict("index")
    last = cmap[N_REGULATION_HALF_INNINGS]

    opp = opp.sort_values(["game_pk", "inning", "inning_topbot",
                           "at_bat_number", "pitch_number"])
    is_bot = opp.inning_topbot == "Bot"
    is_bat = opp.challenger == "batting"
    opp = opp.assign(team=np.where((is_bot & is_bat) | (~is_bot & ~is_bat), "home", "away"))

    rows, fires = [], []
    for (game_pk, team), g in opp.groupby(["game_pk", "team"], sort=False):
        k, used, correct, runs = 0, 0, 0, 0.0
        by_role = {"batting": [0, 0], "fielding": [0, 0]}  # [used, correct]
        prev_inning = None
        for r in g.itertuples():
            if prev_inning is not None and r.inning > prev_inning and r.inning > 9:
                k = max(k - 1, 0)  # extra inning restores one challenge
            prev_inning = r.inning
            if k >= 2:
                continue
            C = cmap.get(r.t, last)["C_k0" if k == 0 else "C_k1"]
            if r.p_post * r.dre > (1 - r.p_post) * C:
                used += 1
                by_role[r.challenger][0] += 1
                won = bool(r.won_if_challenged)   # resolved against TRUTH
                if won:
                    correct += 1
                    runs += r.dre
                    by_role[r.challenger][1] += 1
                else:
                    k += 1
                fires.append({"game_pk": game_pk, "team": team, "role": r.challenger,
                              "dre": r.dre, "won": won, "batter": r.batter})
        rows.append({"game_pk": game_pk, "team": team,
                     "used": used, "correct": correct, "runs": runs,
                     "bat_used": by_role["batting"][0], "bat_correct": by_role["batting"][1],
                     "fld_used": by_role["fielding"][0], "fld_correct": by_role["fielding"][1]})
    return pd.DataFrame(rows), pd.DataFrame(fires)


def summarize(sim, n_games):
    def rate(u, c):
        return c.sum() / u.sum() if u.sum() else float("nan")
    return {
        "challenges_per_team_game": sim.used.sum() / len(sim),
        "success_rate": rate(sim.used, sim.correct),
        "runs_per_team_game": sim.runs.sum() / len(sim),
        "bat_per_team_game": sim.bat_used.sum() / len(sim),
        "bat_success": rate(sim.bat_used, sim.bat_correct),
        "fld_per_team_game": sim.fld_used.sum() / len(sim),
        "fld_success": rate(sim.fld_used, sim.fld_correct),
    }


def run_scenario(opp, sigma_ft, q, seed=SEED, label=""):
    """
    Full pipeline at one perceptual sigma: draw what the player sees, form the
    posterior, re-solve the option values against that belief distribution, then
    play it out resolving on truth.

    Re-solving the DP matters -- the worth of a challenge token depends on how
    well you can pick, so a noisier player has different option values, not just
    a different threshold.
    """
    from perception import build_posterior_lookup, posterior_from_observation

    # sigma may be a single value or per-role; catchers see the pitch from
    # behind and batters from the side, so their noise is genuinely different
    # and pooling the two throws away a real distinction.
    sig = sigma_ft if isinstance(sigma_ft, dict) else {
        "batting": sigma_ft, "fielding": sigma_ft}

    rng = np.random.default_rng(seed)
    opp = opp.copy()
    p_post = np.empty(len(opp))
    for side in ("batting", "fielding"):
        m = (opp.challenger == side).values
        d = opp.d.values[m]
        o_grid, p_grid = build_posterior_lookup(d, sig[side], side)
        obs = d + rng.normal(0.0, sig[side], size=len(d))
        p_post[m] = posterior_from_observation(obs, o_grid, p_grid)
    opp["p_post"] = p_post
    opp["won_if_challenged"] = np.where(opp.challenger == "batting", opp.d < 0, opp.d > 0)

    pools = {side: [g[["p_post", "dre"]].to_numpy()
                    for _, g in opp[opp.challenger == side].groupby(
                        ["game_pk", "inning", "inning_topbot"])]
             for side in ("batting", "fielding")}
    W = solve(pools, q, team_bats_on_even_t=True, seed=seed)
    ov = option_values(W)
    sim, fires = simulate(opp, ov, seed=seed)
    out = summarize(sim, opp.game_pk.nunique())
    out["label"] = label
    out["sigma_in"] = np.mean(list(sig.values())) * 12
    out["sigma_bat_in"] = sig["batting"] * 12
    out["sigma_fld_in"] = sig["fielding"] * 12
    out["C_k0"] = ov[ov.t == 1].iloc[0].C_k0
    out["C_k1"] = ov[ov.t == 1].iloc[0].C_k1
    return out, ov, sim, fires


def main():
    import duckdb
    con = duckdb.connect("data/baseball.duckdb")
    all_pitches = con.execute("""
        SELECT game_pk, inning, inning_topbot FROM statcast WHERE game_year = 2026
    """).df()

    pools = load_pools()
    q = reach_probabilities(all_pitches)
    print(f"pools: {len(pools['batting'])} batting half-innings, "
          f"{len(pools['fielding'])} fielding half-innings")
    print(f"reach probability: t=18 -> {q.get(18):.3f}, t=19 -> {q.get(19):.3f}, "
          f"t=21 -> {q.get(21):.3f}")

    results = {}
    for label, even in (("home (bats bottom halves)", True), ("away (bats top halves)", False)):
        W = solve(pools, q, team_bats_on_even_t=even)
        ov = option_values(W)
        results[label] = ov
        start = ov[ov.t == 1].iloc[0]
        print(f"\n=== {label} ===")
        print(f"start of game: V(k=0)={start.V_k0:.4f}  V(k=1)={start.V_k1:.4f} runs")
        print(f"               C(0)={start.C_k0:.4f}  C(1)={start.C_k1:.4f} runs")
        assert start.C_k0 <= start.C_k1 + 1e-9, "C(0) <= C(1) violated -- bug"
        # Only regulation is shown. Extra-inning rows carry their own reach
        # probability inside the chain, so they are not on the same conditioning
        # basis as regulation rows and shouldn't be read side by side.
        print(ov[ov.t <= N_REGULATION_HALF_INNINGS].to_string(index=False))

    ov = results["home (bats bottom halves)"]
    print("\n=== hard check: C(0) <= C(1) at every t ===")
    bad = ov[ov.C_k0 > ov.C_k1 + 1e-9]
    print("PASS" if bad.empty else f"FAIL at {len(bad)} half-innings:\n{bad}")

    print("\n=== optimal challenge threshold p* by flip value, start of game ===")
    print("(diagnostic only, pooled single-sigma model -- NOT saved to disk and NOT")
    print(" the model behind the app or the headline numbers. See decomposition()")
    print(" below for the role-specific-sigma model, which owns data/option_values.parquet.)")
    s = ov[ov.t == 1].iloc[0]
    rows = []
    for dre in (0.05, 0.10, 0.15, 0.25, 0.40, 0.62, 1.00):
        rows.append({"dre_runs": dre,
                     "p*_k0": threshold(s.C_k0, dre), "p*_k1": threshold(s.C_k1, dre)})
    print(pd.DataFrame(rows).to_string(index=False))

    decomposition(q)


HAWKEYE_SIGMA_FT = 0.5 / 12.0


def observed_row(opp):
    """Actual 2026 behaviour, straight from the challenge flags on the pitches."""
    j = opp[opp.was_challenged]
    n_games = opp.game_pk.nunique()
    out = {
        "label": "observed 2026",
        "sigma_in": np.nan,
        "challenges_per_team_game": len(j) / n_games / 2,
        "success_rate": j.overturned.mean(),
        "runs_per_team_game": j.dre[j.overturned].sum() / n_games / 2,
    }
    for role, pfx in (("batting", "bat"), ("fielding", "fld")):
        s = j[j.challenger == role]
        out[f"{pfx}_per_team_game"] = len(s) / n_games / 2
        out[f"{pfx}_success"] = s.overturned.mean()
    return out


def decomposition(q):
    generated_at = datetime.now(timezone.utc).isoformat()

    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    from geometry import center_distance_to_zone, ball_edge_distance
    opp["d"] = ball_edge_distance(center_distance_to_zone(
        opp.x_mid.values, opp.z_mid.values, opp.height_ft.values))
    opp["t"] = (opp.inning - 1) * 2 + np.where(opp.inning_topbot == "Bot", 2, 1)

    sigmas = pd.read_parquet("data/perception_sigma.parquet")
    player_sigma = dict(zip(sigmas.side, sigmas.sigma_ft))
    print(f"\nfitted player sigma by role (inches): "
          f"{ {k: round(v*12, 3) for k, v in player_sigma.items()} }")

    rows = [observed_row(opp)]
    r_ply, ov_ply, _, fires_ply = run_scenario(opp, player_sigma, q,
                                               label="optimal @ player sigma")
    rows.append(r_ply)

    # THE canonical option-value table: role-specific sigma, the same model
    # behind every other number in this function. This is the only place in
    # the codebase that writes data/option_values.parquet -- previously
    # main()'s pooled-sigma diagnostic also wrote here, silently overwriting
    # this with a superseded model every run. See MODEL_VERSION above.
    _stamp(ov_ply, generated_at).to_parquet("data/option_values.parquet", index=False)
    print("saved data/option_values.parquet (role_sigma_v1, canonical)")
    r_ceil, _, _, fires_ceil = run_scenario(opp, HAWKEYE_SIGMA_FT, q,
                                            label=f"ceiling @ sigma={HAWKEYE_SIGMA_FT*12:.1f}in")
    rows.append(r_ceil)
    res = pd.DataFrame(rows)

    print("\n=== THREE-WAY DECOMPOSITION (per team per game) ===")
    cols = ["label", "sigma_bat_in", "sigma_fld_in", "challenges_per_team_game",
            "success_rate", "runs_per_team_game"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    obs, ply, hawk = (res.runs_per_team_game.iloc[i] for i in (0, 1, 2))
    print(f"\n  decision gap    (player-optimal - observed) = {ply-obs:+.3f} runs/team-game"
          f"  = {(ply-obs)*162:+.1f} runs/team-season   <-- actionable, sigma-independent")
    print(f"  information gap (ceiling - player-optimal)  = {hawk-ply:+.3f} runs/team-game"
          f"  = {(hawk-ply)*162:+.1f} runs/team-season   <-- depends on an ASSUMED ceiling")

    print("\n=== BY CHALLENGER ROLE ===")
    rcols = ["label", "bat_per_team_game", "bat_success", "fld_per_team_game", "fld_success"]
    print(res[rcols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- is the optimal policy "challenge more" or "challenge different"? ---
    print("\n=== LEVERAGE OF CHALLENGED PITCHES (dre distribution) ===")
    act = opp[opp.was_challenged]
    lev = []
    for lbl, dre, won, role in (
            ("observed 2026", act.dre, act.overturned, act.challenger),
            ("optimal @ player sigma", fires_ply.dre, fires_ply.won, fires_ply.role)):
        for r in ("batting", "fielding", "all"):
            m = slice(None) if r == "all" else (role == r).values
            d, w = np.asarray(dre)[m], np.asarray(won)[m]
            lev.append({"policy": lbl, "role": r, "n": len(d),
                        "mean_dre": d.mean(), "median_dre": np.median(d),
                        "p90_dre": np.percentile(d, 90),
                        "runs_per_overturn": d[w].sum() / max(w.sum(), 1),
                        "success": w.mean()})
    lev = pd.DataFrame(lev)
    print(lev.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- ceiling is an assumption, so show it as a curve, not a point ---
    print("\n=== SENSITIVITY: information gap vs assumed ceiling sigma ===")
    print("(the decision gap does not appear here -- it uses only the fitted "
          "player sigma and is unaffected by this assumption)")
    sens = []
    for s_in in (0.10, 0.25, 0.50, 0.75, 1.00):
        r, _, _, _ = run_scenario(opp, s_in / 12.0, q, label=f"ceiling {s_in}in")
        sens.append({"ceiling_sigma_in": s_in,
                     "runs_per_team_game": r["runs_per_team_game"],
                     "info_gap_runs_per_team_game": r["runs_per_team_game"] - ply,
                     "info_gap_runs_per_team_season": (r["runs_per_team_game"] - ply) * 162})
    sens = pd.DataFrame(sens)
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    _stamp(res, generated_at).to_parquet("data/policy_decomposition.parquet", index=False)
    _stamp(lev, generated_at).to_parquet("data/leverage_comparison.parquet", index=False)
    _stamp(sens, generated_at).to_parquet("data/ceiling_sensitivity.parquet", index=False)
    _stamp(fires_ply, generated_at).to_parquet("data/optimal_fires.parquet", index=False)
    print("\nsaved decomposition, leverage, sensitivity, and fires "
          f"(model_version={MODEL_VERSION}, generated_at={generated_at})")


if __name__ == "__main__":
    main()
