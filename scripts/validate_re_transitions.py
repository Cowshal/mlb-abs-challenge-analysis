"""
Transition-level validation of our run expectancy table against Statcast's
delta_run_exp.

Aggregate sign-and-magnitude agreement (ball positive, strike negative,
growing with leverage) would still look correct if we had an off-by-one in
the count transition -- e.g. a ball on 2-1 mapped to the 2-2 cell instead of
3-1. That preserves the overall pattern but corrupts individual cells, and
individual cells are exactly what the decision model consumes. So compare
per (count, base state, outs) cell.

For each called strike / ball:
    implied_delta = RE(next state) - RE(current state) + runs scored on the play
and compare against the mean delta_run_exp Statcast assigns to that same
transition.

Run: python scripts/validate_re_transitions.py
"""
import duckdb
import numpy as np
import pandas as pd

FLAG_THRESHOLD = 0.02  # runs

# walk: (r1,r2,r3) -> (r1',r2',r3'), runs forced in
WALK_ADVANCE = {
    (False, False, False): ((True, False, False), 0),
    (True,  False, False): ((True, True,  False), 0),
    (False, True,  False): ((True, True,  False), 0),
    (False, False, True):  ((True, False, True),  0),
    (True,  True,  False): ((True, True,  True),  0),
    (True,  False, True):  ((True, True,  True),  0),
    (False, True,  True):  ((True, True,  True),  0),
    (True,  True,  True):  ((True, True,  True),  1),
}


def main():
    con = duckdb.connect("data/baseball.duckdb")

    re_df = con.execute("""
        SELECT balls, strikes, outs_when_up AS outs, r1, r2, r3,
               AVG(runs_rest_of_inning) AS run_exp, COUNT(*) AS n
        FROM pitch_state_with_target
        GROUP BY 1,2,3,4,5,6
    """).df()
    RE = {(int(r.balls), int(r.strikes), int(r.outs), bool(r.r1), bool(r.r2), bool(r.r3)): r.run_exp
          for r in re_df.itertuples()}
    print(f"RE table cells: {len(RE)}")

    obs = con.execute("""
        SELECT p.balls, p.strikes, p.outs_when_up AS outs, p.r1, p.r2, p.r3, s.description,
               AVG(s.delta_run_exp) AS statcast_delta, COUNT(*) AS n
        FROM pitch_state_with_target p
        JOIN statcast s
          ON p.game_pk = s.game_pk AND p.at_bat_number = s.at_bat_number
         AND p.pitch_number = s.pitch_number
        WHERE s.description IN ('called_strike', 'ball')
        GROUP BY 1,2,3,4,5,6,7
    """).df()
    print(f"observed (count, base state, outs, call) cells: {len(obs)}")

    rows = []
    for r in obs.itertuples():
        b, s, o = int(r.balls), int(r.strikes), int(r.outs)
        bases = (bool(r.r1), bool(r.r2), bool(r.r3))
        cur = RE.get((b, s, o, *bases))
        if cur is None:
            continue

        if r.description == "ball":
            if b < 3:
                nxt, runs = RE.get((b + 1, s, o, *bases)), 0
            else:
                new_bases, runs = WALK_ADVANCE[bases]
                nxt = RE.get((0, 0, o, *new_bases))
        else:  # called_strike
            if s < 2:
                nxt, runs = RE.get((b, s + 1, o, *bases)), 0
            else:
                runs = 0
                nxt = 0.0 if o + 1 >= 3 else RE.get((0, 0, o + 1, *bases))

        if nxt is None:
            continue
        rows.append({
            "balls": b, "strikes": s, "outs": o, "r1": bases[0], "r2": bases[1], "r3": bases[2],
            "description": r.description, "n": r.n,
            "implied_delta": nxt - cur + runs,
            "statcast_delta": r.statcast_delta,
        })

    df = pd.DataFrame(rows)
    df["diff"] = df.implied_delta - df.statcast_delta
    print(f"comparable cells: {len(df)}  (total pitches covered: {df.n.sum():,})")

    corr = df.implied_delta.corr(df.statcast_delta)
    w = df.n / df.n.sum()
    wcorr = np.cov(df.implied_delta, df.statcast_delta, aweights=w)[0, 1] / (
        np.sqrt(np.cov(df.implied_delta, aweights=w)) * np.sqrt(np.cov(df.statcast_delta, aweights=w)))
    print(f"\ncorrelation (unweighted) = {corr:.4f}")
    print(f"correlation (pitch-weighted) = {float(wcorr):.4f}")

    d = df["diff"]
    print(f"\ndifference (implied - statcast), runs:")
    print(f"  mean={d.mean():+.4f}  median={d.median():+.4f}  std={d.std():.4f}")
    print(f"  mean abs={d.abs().mean():.4f}  p95 abs={d.abs().quantile(0.95):.4f}  max abs={d.abs().max():.4f}")

    counts, edges = np.histogram(d, bins=10)
    print("\n  histogram of differences (runs):")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"    [{lo:+.3f}, {hi:+.3f}) {'#' * int(c * 60 / max(counts))} {c}")

    flagged = df[d.abs() > FLAG_THRESHOLD].copy()
    print(f"\n=== cells differing by more than {FLAG_THRESHOLD} runs: "
          f"{len(flagged)} / {len(df)} ({100*len(flagged)/len(df):.1f}%), "
          f"covering {flagged.n.sum():,} pitches ({100*flagged.n.sum()/df.n.sum():.1f}%) ===")
    if len(flagged):
        show = flagged.reindex(flagged["diff"].abs().sort_values(ascending=False).index)
        print(show.head(25).to_string(index=False))

    df.to_parquet("data/re_transition_check.parquet", index=False)
    print("\nsaved data/re_transition_check.parquet")


if __name__ == "__main__":
    main()
