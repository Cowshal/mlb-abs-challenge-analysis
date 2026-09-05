"""
Follow-up to team_skill_test.py: does the cross-team spread in challenge
accuracy (real, p=0.003 vs binomial chance, but team-level split-half
r=0.24 with a CI crossing zero) resolve as a personnel-composition effect
rather than a team skill?

Two checks:
  1. Split-half reliability computed on individual challengers instead of
     teams, at several minimum-challenges-per-half thresholds. If it reads
     meaningfully higher than the team-level 0.24 and clears zero, the trait
     travels with players, not team affiliation -- rosters churn mid-season
     and dilute the team-level signal.
  2. Whether each top-ranked team's primary catcher (by innings caught)
     individually ranks among the league's most accurate catcher-challengers,
     using data/team_decomposition.parquet's quality vs volume split to see
     whether an exceptional catcher tracks with a quality-driven edge
     specifically (Cincinnati, Colorado) and not a volume-driven one
     (Minnesota, Chicago).

Run: python scripts/player_skill_test.py
Output: data/player_skill_test.parquet   (per-threshold reliability table)
        data/catcher_check.parquet        (primary catcher per team + rank)
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries

MODEL_VERSION = "player_skill_test_v1"
THRESHOLDS = (5, 8, 10, 15)
TOP_N_TEAMS = 5


def player_split_half(ch):
    mid = ch.game_date.median()
    ch = ch.copy()
    ch["half"] = np.where(ch.game_date <= mid, "H1", "H2")

    rows = []
    for min_n in THRESHOLDS:
        g = ch.groupby(["challenging_player_id", "half"]).agg(
            n=("overturned", "size"), correct=("overturned", "sum")).reset_index()
        g["rate"] = g.correct / g.n
        piv = g.pivot(index="challenging_player_id", columns="half", values="rate")
        n_piv = g.pivot(index="challenging_player_id", columns="half", values="n")
        keep = (n_piv.H1 >= min_n) & (n_piv.H2 >= min_n)
        piv = piv[keep]
        n = len(piv)
        r, p = stats.pearsonr(piv.H1, piv.H2)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        rows.append({"min_challenges_per_half": min_n, "n_players": n,
                     "r": r, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi})
    return pd.DataFrame(rows)


def catcher_check(ch):
    con = duckdb.connect("data/baseball.duckdb")

    cat = ch[ch.challenging_player_type == "catcher"]
    cat_stats = cat.groupby("challenging_player_id").agg(
        n=("overturned", "size"), correct=("overturned", "sum")).reset_index()
    cat_stats["rate"] = cat_stats.correct / cat_stats.n
    cat_stats = cat_stats[cat_stats.n >= 5]
    cat_stats["pct_rank"] = cat_stats.rate.rank(pct=True)

    pitches = con.execute("""
        SELECT home_team, away_team, inning_topbot, fielder_2 AS catcher, COUNT(*) AS pitches
        FROM statcast WHERE game_year = 2026 AND fielder_2 IS NOT NULL
        GROUP BY 1, 2, 3, 4
    """).df()
    pitches["fielding_team"] = np.where(
        pitches.inning_topbot == "Top", pitches.home_team, pitches.away_team)
    by_team_catcher = pitches.groupby(["fielding_team", "catcher"]).pitches.sum().reset_index()
    primary = by_team_catcher.loc[by_team_catcher.groupby("fielding_team").pitches.idxmax()]
    primary = primary.rename(columns={"fielding_team": "team", "catcher": "primary_catcher_id"})
    primary["primary_catcher_id"] = primary.primary_catcher_id.astype(np.int64)

    ids = primary.primary_catcher_id.unique().tolist()
    names = {}
    for i in range(0, len(ids), 50):
        r = get_with_retries("https://statsapi.mlb.com/api/v1/people",
                             params={"personIds": ",".join(str(int(x)) for x in ids[i:i + 50])})
        for p in r.json().get("people", []):
            names[p["id"]] = p.get("fullName")
    primary["name"] = primary.primary_catcher_id.map(names)

    merged = primary.merge(
        cat_stats.rename(columns={"challenging_player_id": "primary_catcher_id"}),
        on="primary_catcher_id", how="left")

    decomp = pd.read_parquet("data/team_decomposition.parquet")
    merged = merged.merge(
        decomp[["team", "total_ratio", "quality_ratio", "attempts_ratio"]],
        on="team", how="left")
    merged["top_team"] = merged.team.isin(
        decomp.sort_values("total_ratio", ascending=False).head(TOP_N_TEAMS).team)
    return merged.sort_values("total_ratio", ascending=False)


def main():
    generated_at = datetime.now(timezone.utc).isoformat()

    ch = pd.read_parquet("data/abs_challenges.parquet")
    con = duckdb.connect("data/baseball.duckdb")
    dates = con.execute(
        "SELECT DISTINCT game_pk, CAST(game_date AS DATE) AS game_date "
        "FROM statcast WHERE game_year = 2026").df()
    dates["game_pk"] = dates.game_pk.astype(np.int64)
    ch["game_pk"] = ch.game_pk.astype(np.int64)
    ch = ch.merge(dates, on="game_pk", how="left")
    ch["overturned"] = ch.is_overturned.fillna(False).astype(bool)

    print("=== PLAYER-LEVEL SPLIT-HALF RELIABILITY (vs team-level r=0.24) ===")
    sh = player_split_half(ch)
    print(sh.round(4).to_string(index=False))
    sh["model_version"] = MODEL_VERSION
    sh["generated_at"] = generated_at
    sh.to_parquet("data/player_skill_test.parquet", index=False)

    print("\n=== PRIMARY CATCHER PER TEAM: INDIVIDUAL CHALLENGE ACCURACY RANK ===")
    cc = catcher_check(ch)
    print(cc[["team", "name", "n", "rate", "pct_rank", "quality_ratio", "top_team"]]
          .round(3).to_string(index=False))
    cc["model_version"] = MODEL_VERSION
    cc["generated_at"] = generated_at
    cc.to_parquet("data/catcher_check.parquet", index=False)

    print("\nsaved data/player_skill_test.parquet, data/catcher_check.parquet")


if __name__ == "__main__":
    main()
