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
import numpy as np
import pandas as pd

N_REGULATION_HALF_INNINGS = 18
MAX_HALF_INNINGS = 30  # allow extras out to the 15th
N_SAMPLES = 4000
SEED = 17


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

    rows = []
    for (game_pk, team), g in opp.groupby(["game_pk", "team"], sort=False):
        k, used, correct, runs = 0, 0, 0, 0.0
        prev_inning = None
        for r in g.itertuples():
            if prev_inning is not None and r.inning > prev_inning and r.inning > 9:
                k = max(k - 1, 0)  # extra inning restores one challenge
            prev_inning = r.inning
            if k >= 2:
                continue
            C = cmap.get(r.t, last)["C_k0" if k == 0 else "C_k1"]
            if r.p_success * r.dre > (1 - r.p_success) * C:
                used += 1
                if rng.random() < r.p_success:
                    correct += 1
                    runs += r.dre
                else:
                    k += 1
        rows.append({"team": team, "used": used, "correct": correct, "runs": runs})

    sim = pd.DataFrame(rows)
    return sim.groupby("team").agg(
        challenges_per_game=("used", "mean"),
        correct_per_game=("correct", "mean"),
        success_rate=("correct", lambda s: s.sum() / max(sim.loc[s.index, "used"].sum(), 1)),
        runs_gained_per_game=("runs", "mean"),
    ).reset_index()


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
    s = ov[ov.t == 1].iloc[0]
    rows = []
    for dre in (0.05, 0.10, 0.15, 0.25, 0.40, 0.62, 1.00):
        rows.append({"dre_runs": dre,
                     "p*_k0": threshold(s.C_k0, dre), "p*_k1": threshold(s.C_k1, dre)})
    print(pd.DataFrame(rows).to_string(index=False))

    ov.to_parquet("data/option_values.parquet", index=False)
    print("\nsaved data/option_values.parquet")

    # Behavioural sanity check: how many challenges does the optimal policy
    # actually pull the trigger on, per team per game? Observed 2026 behaviour
    # is ~2 per team per game. Wildly more or fewer means the option value is
    # mis-scaled even if C(0) <= C(1) holds.
    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    opp["t"] = (opp.inning - 1) * 2 + np.where(opp.inning_topbot == "Bot", 2, 1)
    cmap = ov.set_index("t")[["C_k0", "C_k1"]].to_dict("index")
    c0 = opp.t.map(lambda t: cmap.get(t, cmap[N_REGULATION_HALF_INNINGS])["C_k0"])
    c1 = opp.t.map(lambda t: cmap.get(t, cmap[N_REGULATION_HALF_INNINGS])["C_k1"])
    gain = opp.p_success * opp.dre
    opp["fire_k0"] = gain > (1 - opp.p_success) * c0
    opp["fire_k1"] = gain > (1 - opp.p_success) * c1

    n_games = opp.game_pk.nunique()
    print("\n=== upper bound (IGNORES token exhaustion -- not a real count) ===")
    for side in ("batting", "fielding"):
        s = opp[opp.challenger == side]
        print(f"  {side:9s} k=0 threshold fires on {s.fire_k0.sum() / n_games:5.2f}/game "
              f"of {len(s)/n_games:5.1f} opportunities/game")

    print("\n=== simulation WITH token dynamics (2 challenges, lost only when wrong) ===")
    sim = simulate(opp, ov, seed=SEED)
    print(sim.to_string(index=False))
    print(f"\n  observed 2026 behaviour: ~4.1 challenges/game across both teams "
          f"(~2.1 per team per game)")


if __name__ == "__main__":
    main()
