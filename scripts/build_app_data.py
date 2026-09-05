"""
Precompute everything the Streamlit app needs into app/data/.

The app must not run a Statcast pull, solve the DP, or touch DuckDB. It loads
these files and does nothing but filter and plot.

Run: python scripts/build_app_data.py
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries

OUT = Path("app/data")
SEASON = 2026

# Every file abs_policy.py's decomposition() saves in one run is stamped with
# the SAME model_version + generated_at (see src/abs_policy.py::_stamp). If
# any of them disagree, some subset was regenerated independently and the set
# is mixed-vintage -- exactly the bug that shipped a superseded pooled-sigma
# threshold model to the app while every other number reflected the
# role-specific model. Fail loudly instead of silently propagating that.
POLICY_ARTIFACTS = ["policy_decomposition", "ceiling_sensitivity",
                    "leverage_comparison", "optimal_fires", "option_values"]


def validate_provenance():
    stamps = {}
    for name in POLICY_ARTIFACTS:
        path = Path(f"data/{name}.parquet")
        if not path.exists():
            raise RuntimeError(f"data/{name}.parquet is missing -- run "
                               f"`python src/abs_policy.py` first.")
        df = pd.read_parquet(path)
        if "model_version" not in df.columns or "generated_at" not in df.columns:
            raise RuntimeError(
                f"data/{name}.parquet has no provenance stamp (model_version/"
                f"generated_at). It predates the versioning fix -- regenerate "
                f"everything with `python src/abs_policy.py`.")
        stamps[name] = (df.model_version.iloc[0], df.generated_at.iloc[0])

    versions = {v for v, _ in stamps.values()}
    timestamps = {t for _, t in stamps.values()}
    if len(versions) > 1 or len(timestamps) > 1:
        detail = "\n".join(f"  {n}: model_version={v}  generated_at={t}"
                           for n, (v, t) in stamps.items())
        raise RuntimeError(
            "MIXED-VINTAGE policy artifacts -- these five files must all come "
            f"from the same abs_policy.py run:\n{detail}\n"
            "Re-run `python src/abs_policy.py` (it regenerates all five "
            "together) rather than any partial/manual regeneration.")

    model_version, generated_at = next(iter(stamps.values()))

    sigma = pd.read_parquet("data/perception_sigma.parquet")
    if "model_version" not in sigma.columns:
        raise RuntimeError("data/perception_sigma.parquet has no provenance stamp -- "
                           "regenerate with `python scripts/estimate_perception_sigma.py`.")
    sigma_version, sigma_at = sigma.model_version.iloc[0], sigma.generated_at.iloc[0]
    if sigma_at > generated_at:
        print(f"  WARNING: perception_sigma.parquet ({sigma_at}) is NEWER than the "
              f"policy artifacts it feeds ({generated_at}) -- the policy model may "
              f"not reflect the latest sigma fit. Consider re-running abs_policy.py.")

    print(f"provenance OK: policy model_version={model_version}, generated_at={generated_at}")
    print(f"               sigma model_version={sigma_version}, generated_at={sigma_at}")
    return model_version, generated_at


def fetch_names(ids):
    names = {}
    ids = [int(i) for i in ids]
    for i in range(0, len(ids), 50):
        r = get_with_retries("https://statsapi.mlb.com/api/v1/people",
                             params={"personIds": ",".join(str(b) for b in ids[i:i + 50])})
        for p in r.json().get("people", []):
            names[p["id"]] = p.get("fullName")
    return names


def team_of(inning_topbot, challenger):
    """Which side (home/away) an opportunity belongs to."""
    bot = inning_topbot == "Bot"
    bat = challenger == "batting"
    return np.where((bot & bat) | (~bot & ~bat), "home", "away")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_version, generated_at = validate_provenance()
    con = duckdb.connect("data/baseball.duckdb")

    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    fires = pd.read_parquet("data/optimal_fires.parquet")

    # ---- 1. headline decomposition + sensitivity + leverage ----
    for name in ("policy_decomposition", "ceiling_sensitivity", "leverage_comparison"):
        pd.read_parquet(f"data/{name}.parquet").to_parquet(OUT / f"{name}.parquet", index=False)

    # ---- 2. optimal threshold surface: p* = C / (dre + C) ----
    ov = pd.read_parquet("data/option_values.parquet")
    ov = ov[ov.t <= 18]
    dre_grid = np.round(np.arange(0.02, 1.21, 0.02), 3)
    rows = []
    for r in ov.itertuples():
        for dre in dre_grid:
            for k, C in ((0, r.C_k0), (1, r.C_k1)):
                rows.append({"inning": r.inning, "half": r.half, "t": r.t,
                             "challenges_remaining": 2 - k, "dre": dre,
                             "p_star": C / (dre + C)})
    threshold_surface = pd.DataFrame(rows)
    threshold_surface["model_version"] = model_version
    threshold_surface["generated_at"] = generated_at
    threshold_surface.to_parquet(OUT / "threshold_surface.parquet", index=False)

    # ---- 3. per-team runs left on the table ----
    teams = con.execute(f"""
        SELECT DISTINCT game_pk, home_team, away_team
        FROM statcast WHERE game_year = {SEASON}
    """).df()
    teams["game_pk"] = teams.game_pk.astype(np.int64)

    act = opp[opp.was_challenged].copy()
    act["side"] = team_of(act.inning_topbot.values, act.challenger.values)
    act = act.merge(teams, on="game_pk", how="left")
    act["team"] = np.where(act.side == "home", act.home_team, act.away_team)

    fires = fires.merge(teams, left_on="game_pk", right_on="game_pk", how="left")
    fires["team_abbr"] = np.where(fires.team == "home", fires.home_team, fires.away_team)

    actual = act.groupby("team").apply(
        lambda g: pd.Series({"actual_challenges": len(g),
                             "actual_correct": int(g.overturned.sum()),
                             "actual_runs": g.dre[g.overturned].sum()}),
        include_groups=False).reset_index()
    optimal = fires.groupby("team_abbr").apply(
        lambda g: pd.Series({"optimal_challenges": len(g),
                             "optimal_correct": int(g.won.sum()),
                             "optimal_runs": g.dre[g.won].sum()}),
        include_groups=False).reset_index().rename(columns={"team_abbr": "team"})

    games = pd.concat([
        teams[["game_pk", "home_team"]].rename(columns={"home_team": "team"}),
        teams[["game_pk", "away_team"]].rename(columns={"away_team": "team"}),
    ]).groupby("team").game_pk.nunique().rename("games").reset_index()

    per_team = actual.merge(optimal, on="team", how="outer").merge(games, on="team", how="left")
    per_team["runs_left_on_table"] = per_team.optimal_runs - per_team.actual_runs
    per_team["runs_left_per_game"] = per_team.runs_left_on_table / per_team.games
    per_team["actual_success"] = per_team.actual_correct / per_team.actual_challenges
    per_team["optimal_success"] = per_team.optimal_correct / per_team.optimal_challenges
    per_team["full_season_pace"] = per_team.runs_left_per_game * 162
    per_team = per_team.sort_values("runs_left_on_table", ascending=False)
    per_team["model_version"] = model_version
    per_team["generated_at"] = generated_at
    per_team.to_parquet(OUT / "per_team.parquet", index=False)

    # ---- 4. per-batter (batting-role challenges only) ----
    act_b = act[act.challenger == "batting"]
    fir_b = fires[fires.role == "batting"]
    ab = act_b.groupby("batter").apply(
        lambda g: pd.Series({"actual_challenges": len(g),
                             "actual_runs": g.dre[g.overturned].sum(),
                             "actual_correct": int(g.overturned.sum())}),
        include_groups=False).reset_index()
    ob = fir_b.groupby("batter").apply(
        lambda g: pd.Series({"optimal_challenges": len(g),
                             "optimal_correct": int(g.won.sum()),
                             "optimal_runs": g.dre[g.won].sum()}),
        include_groups=False).reset_index()
    per_batter = ab.merge(ob, on="batter", how="outer").fillna(0)
    per_batter["runs_left_on_table"] = per_batter.optimal_runs - per_batter.actual_runs
    per_batter = per_batter[per_batter.optimal_challenges >= 5]
    names = fetch_names(per_batter.batter.unique())
    per_batter["player"] = per_batter.batter.map(names)
    per_batter["actual_success"] = np.where(
        per_batter.actual_challenges > 0,
        per_batter.actual_correct / per_batter.actual_challenges.replace(0, np.nan), np.nan)
    per_batter["optimal_success"] = per_batter.optimal_correct / per_batter.optimal_challenges
    per_batter = per_batter.sort_values("runs_left_on_table", ascending=False)
    per_batter["model_version"] = model_version
    per_batter["generated_at"] = generated_at
    per_batter.to_parquet(OUT / "per_batter.parquet", index=False)

    # ---- 5. fitted perception sigma ----
    pd.read_parquet("data/perception_sigma.parquet").to_parquet(
        OUT / "perception_sigma.parquet", index=False)

    total = sum(f.stat().st_size for f in OUT.glob("*.parquet"))
    print(f"wrote {len(list(OUT.glob('*.parquet')))} files to {OUT}, "
          f"{total/1e6:.2f} MB total (budget 20 MB)")
    for f in sorted(OUT.glob("*.parquet")):
        print(f"  {f.name:32s} {f.stat().st_size/1e3:8.1f} KB")


if __name__ == "__main__":
    main()
