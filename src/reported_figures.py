"""
Single source of truth for the hardcoded statistics that appear in prose --
README.md, docs/writeup.md, and the app's copy in app/streamlit_app.py.

Why this exists
---------------
build_app_data.py's provenance guards stop a *mixed-vintage data set* from
shipping. They do nothing about prose going stale while the data underneath
it moves every night. A scheduled refresh that updates the charts but not the
sentence next to them ships a site that contradicts itself.

  collect()  -- reads the pipeline's parquet outputs and returns the current
                value of every tracked figure, plus the data-through date.
                scripts/build_reported_figures.py writes this to
                data/reported_figures.json (and app/data/, for the footer and
                for the couple of app strings that now read their counts
                straight from it instead of hardcoding).

  check()    -- re-derives those values and, for each figure, greps the prose
                file(s) it appears in for the number as written. Returns a
                list of Problem records for anything that has drifted past
                its per-figure tolerance, OR whose anchoring text can no
                longer be found. The pipeline turns a non-empty list into a
                failed build.

Tolerance philosophy (revised 2026-09-09 after five days of new games moved
28 figures and tripped the gate on rounding-boundary noise)
-----------------------------------------------------------------------------
A figure trips the gate only when a reader would notice an inconsistency
between the prose and a chart or table on the SAME page -- not on every
rounding boundary. Concretely:

  * kind="point"  -- tol is set to roughly the resolution a reader could
                     resolve against the on-page chart. Bar charts ~0.01-0.02
                     on a runs axis; text-only numbers get looser tol because
                     nobody eyeballs "53.7%" against "55%".
  * kind="range"  -- the prose states a band ("approximately 8-10 runs"); the
                     Occ pattern captures (lo, hi) and the check passes while
                     the computed value sits within [lo-tol, hi+tol]. Used for
                     figures that have visibly wandered across refreshes (the
                     decision gap: 10 -> 9 -> 8) so a point claim was dishonest.
  * kind="floor"  -- the prose states a lower bound ("9,000+ challenges"); the
                     check passes while computed >= stated-tol. Used for counts
                     that only grow during the season -- a fixed absolute
                     tolerance makes no sense for them, and "9,000+" stays true
                     from here to game 2430. The app reads the live counts from
                     reported_figures.json rather than hardcoding them at all.

Adding a figure: append a Figure(...) to FIGURES. See NOT_TRACKED at the
bottom for what is intentionally excluded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP_DATA = ROOT / "app" / "data"
DATA = ROOT / "data"


@lru_cache(maxsize=None)
def _app(name: str) -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / f"{name}.parquet")


@lru_cache(maxsize=None)
def _pipe(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


# ---------------------------------------------------------------- helpers

_OBS = "observed 2026"
_OPT = "optimal @ player sigma"
_CEIL = "ceiling @ sigma=0.5in"
_SEASON_GAMES = 162


def _dec_row(label: str) -> pd.Series:
    d = _app("policy_decomposition")
    return d.loc[d.label == label].iloc[0]


def _lev(policy: str, role: str) -> pd.Series:
    lv = _app("leverage_comparison")
    return lv[(lv.policy == policy) & (lv.role == role)].iloc[0]


def _role_per_game(policy: str, role: str) -> float:
    label = _OBS if policy == _OBS else _OPT
    total = _dec_row(label).challenges_per_team_game
    return float(total * _lev(policy, role).n / _lev(policy, "all").n)


def parse_number(token: str) -> float:
    t = token.strip().replace(",", "").replace("−", "-").replace("%", "")
    t = t.lstrip("+").strip()
    return float(t)


def data_through() -> str:
    opp = DATA / "challenge_opportunities.parquet"
    if opp.exists():
        gd = pd.read_parquet(opp, columns=["game_date"]).game_date
        return str(pd.to_datetime(gd).max().date())
    gd = pd.read_parquet(DATA / "statcast_2026.parquet", columns=["game_date"]).game_date
    return str(pd.to_datetime(gd).max().date())


# ---------------------------------------------------------------- figure model

@dataclass(frozen=True)
class Occ:
    path: str        # relative to repo root
    # regex. kind point/floor: exactly one () group (the number).
    # kind range: exactly two () groups (low, high).
    pattern: str


@dataclass(frozen=True)
class Figure:
    key: str
    compute: Callable[[], float]
    occ: tuple[Occ, ...]
    tol: float
    render: Callable[[float], str] = staticmethod(lambda v: f"{v:.4g}")
    unit: str = ""
    kind: str = "point"          # "point" | "range" | "floor"
    why: str = ""                # one-line tolerance rationale (into the JSON)


@dataclass(frozen=True)
class Problem:
    key: str
    path: str
    line: Optional[int]
    detail: str

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.key}] {loc} -- {self.detail}"


def _r(dp):     return lambda v: f"{v:.{dp}f}"
def _rint(v):   return f"{int(round(v))}"
def _rcomma(v): return f"{int(round(v)):,}"


# ---------------------------------------------------------------- THE FIGURES

FIGURES: list[Figure] = [

    # ---- coverage counts (grow all season -> floor claims) -----------------
    Figure(
        "n_opportunities",
        lambda: float(_lev(_OBS, "all").n),
        (Occ("README.md", r"2026 MLB season,\s*([\d,]+)\+ challenges across"),
         Occ("docs/writeup.md", r"Across more than\s*([\d,]+)\s*\n?challenges in the 2026 season")),
        tol=300, render=_rcomma, unit="challenges", kind="floor",
        why="monotonically growing count; '9,000+' stays true all season; "
            "fires only if the challenge table regresses below the stated floor",
    ),
    Figure(
        "n_games_2026",
        lambda: float(pd.read_parquet(DATA / "challenge_opportunities.parquet",
                                      columns=["game_pk"]).game_pk.nunique()),
        (Occ("README.md", r"challenges across\s*([\d,]+)\+ games"),),
        tol=60, render=_rcomma, unit="games", kind="floor",
        why="growing count; floor claim; fires only on a data regression",
    ),

    # ---- headline decomposition --------------------------------------------
    Figure(
        "obs_challenges_per_game",
        lambda: float(_dec_row(_OBS).challenges_per_team_game),
        (Occ("README.md", r"\*\*Observed 2026\*\*\s*\|\s*([\d.]+)\s*\|"),),
        tol=0.15, render=_r(2), unit="/team-game",
        why="decomposition bar chart; ~0.1 bar resolution",
    ),
    Figure(
        "obs_challenges_per_game_1dp",
        lambda: float(_dec_row(_OBS).challenges_per_team_game),
        (Occ("README.md", r"league behaviour \(([\d.]+) challenges per"),
         Occ("docs/writeup.md", r"currently challenge about ([\d.]+) times a game")),
        tol=0.2, render=_r(1),
        why="rounded prose ('about 2.1 a game'); wide enough to survive the "
            "2.1/2.2 rounding boundary",
    ),
    Figure(
        "obs_success_rate",
        lambda: float(_dec_row(_OBS).success_rate) * 100,
        (Occ("README.md", r"\*\*Observed 2026\*\*\s*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("README.md", r"\|\s*observed\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("docs/writeup.md", r"league-wide success rate is ([\d.]+)%"),
         Occ("docs/writeup.md", r"to match the observed ([\d.]+)%\s*\n?\s*success rate")),
        tol=1.5, render=_r(1), unit="pp",
        why="text-only percentage; a reader does not eyeball 53.7% vs 55%",
    ),
    Figure(
        "obs_success_rate_54",
        lambda: float(_dec_row(_OBS).success_rate) * 100,
        (Occ("README.md", r"at a ([\d.]+)% success rate\) does not"),
         Occ("app/streamlit_app.py", r"and win \*\*([\d.]+)%\*\* of the")),
        tol=1.5, render=_rint,
        why="rounded prose ('54%'); text only",
    ),
    Figure(
        "obs_runs_per_game",
        lambda: float(_dec_row(_OBS).runs_per_team_game),
        (Occ("README.md", r"\*\*Observed 2026\*\*\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),
         Occ("docs/writeup.md", r"observed 2026 behavior \(([\d.]+)\)")),
        tol=0.02, render=_r(3), unit="runs/team-game",
        why="decomposition bar chart on a runs axis; ~0.02 is the smallest "
            "gap a reader could resolve between bars",
    ),
    Figure(
        "opt_challenges_per_game",
        lambda: float(_dec_row(_OPT).challenges_per_team_game),
        (Occ("README.md", r"\*\*Optimal, same information\*\*\s*\|\s*([\d.]+)\s*\|"),),
        tol=0.2, render=_r(2),
        why="chart bar; the optimal policy re-solves each refresh so it moves "
            "more than the observed row",
    ),
    Figure(
        "opt_success_rate",
        lambda: float(_dec_row(_OPT).success_rate) * 100,
        (Occ("README.md", r"\*\*Optimal, same information\*\*\s*\|[^|]*\|\s*([\d.]+)%"),
         Occ("README.md", r"\|\s*optimal\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("docs/writeup.md", r"wins a \*smaller\* share:\s*([\d.]+)%")),
        tol=2.0, render=_r(1), unit="pp",
        why="text; a solved-policy quantity that is noisier than the observed rate",
    ),
    Figure(
        "opt_runs_per_game",
        lambda: float(_dec_row(_OPT).runs_per_team_game),
        (Occ("README.md", r"\*\*Optimal, same information\*\*\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),
         Occ("docs/writeup.md", r"the optimal policy given the same information \(([\d.]+)\)")),
        tol=0.02, render=_r(3),
        why="chart bar resolution",
    ),
    Figure(
        "ceil_challenges_per_game",
        lambda: float(_dec_row(_CEIL).challenges_per_team_game),
        (Occ("README.md", r"High-precision benchmark \(σ = 0\.5 in\)\s*\|\s*([\d.]+)\s*\|"),),
        tol=0.3, render=_r(2),
        why="chart bar; the high-precision benchmark is the noisiest of the "
            "three policies",
    ),
    Figure(
        "ceil_success_rate",
        lambda: float(_dec_row(_CEIL).success_rate) * 100,
        (Occ("README.md", r"High-precision benchmark \(σ = 0\.5 in\)\s*\|[^|]*\|\s*([\d.]+)%"),),
        tol=2.0, render=_r(1), unit="pp",
        why="text",
    ),
    Figure(
        "ceil_runs_per_game",
        lambda: float(_dec_row(_CEIL).runs_per_team_game),
        (Occ("README.md", r"High-precision benchmark \(σ = 0\.5 in\)\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),
         Occ("docs/writeup.md", r"high-precision benchmark at sigma = 0\.5 in \(([\d.]+)\)")),
        tol=0.02, render=_r(3),
        why="chart bar resolution",
    ),
    Figure(
        "decision_gap_runs_per_season",
        lambda: (float(_dec_row(_OPT).runs_per_team_game)
                 - float(_dec_row(_OBS).runs_per_team_game)) * _SEASON_GAMES,
        (Occ("README.md", r"teams leave roughly \*\*(\d+)[–-](\d+) runs per team-season\*\*"),
         Occ("README.md", r"\*\*Decision gap: ~(\d+)[–-](\d+) runs per team-season\.\*\*"),
         Occ("docs/writeup.md", r"a gap of roughly (\d+)[–-](\d+) runs"),
         Occ("docs/writeup.md", r"coachable gap I'm reporting \((\d+)[–-](\d+) runs\)")),
        tol=0.75, render=_rint, unit="runs/season", kind="range",
        why="has visibly wandered across refreshes (10 -> 9.6 -> 9.1 -> 8.2); "
            "prose now states an 8-10 band and the check fires only if the "
            "computed gap leaves it by more than ~1 run",
    ),

    # ---- leverage table ---------------------------------------------------
    Figure("obs_mean_dre", lambda: float(_lev(_OBS, "all").mean_dre),
           (Occ("README.md", r"\|\s*observed\s*\|\s*([\d.]+)\s*\|"),),
           tol=0.02, render=_r(3), why="leverage table; digit-level comparison"),
    Figure("obs_median_dre", lambda: float(_lev(_OBS, "all").median_dre),
           (Occ("README.md", r"\|\s*observed\s*\|[^|]*\|\s*([\d.]+)\s*\|"),),
           tol=0.02, render=_r(3), why="leverage table"),
    Figure("obs_runs_per_overturn", lambda: float(_lev(_OBS, "all").runs_per_overturn),
           (Occ("README.md", r"\|\s*observed\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),),
           tol=0.02, render=_r(3), why="leverage table"),
    Figure("opt_mean_dre", lambda: float(_lev(_OPT, "all").mean_dre),
           (Occ("README.md", r"\|\s*optimal\s*\|\s*([\d.]+)\s*\|"),),
           tol=0.02, render=_r(3), why="leverage table"),
    Figure("opt_median_dre", lambda: float(_lev(_OPT, "all").median_dre),
           (Occ("README.md", r"\|\s*optimal\s*\|[^|]*\|\s*([\d.]+)\s*\|"),),
           tol=0.02, render=_r(3), why="leverage table"),
    Figure("opt_runs_per_overturn", lambda: float(_lev(_OPT, "all").runs_per_overturn),
           (Occ("README.md", r"\|\s*optimal\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),),
           tol=0.02, render=_r(3), why="leverage table"),

    # ---- ceiling sensitivity curve (illustrative, no chart) --------------
    *[
        Figure(
            f"info_gap_season_ceiling_{int(s*100):03d}",
            (lambda s=s: float(_app("ceiling_sensitivity")
                               .set_index("ceiling_sigma_in")
                               .loc[s, "info_gap_runs_per_team_season"])),
            (Occ("README.md", rf"\|\s*{s:.2f} in\s*\|\s*([\d.]+)\s*\|"),),
            tol=3.0, render=_rint, unit="runs/season",
            why="an illustrative curve with no on-page chart; shown as whole "
                "runs; only a regime change (several runs) matters",
        )
        for s in (0.10, 0.25, 0.50, 0.75, 1.00)
    ],

    # ---- perceptual sigma ------------------------------------------------
    Figure(
        "sigma_bat_in",
        lambda: float(_app("perception_sigma").set_index("side").loc["batting", "sigma_in"]),
        (Occ("README.md", r"\|\s*batters\s*\|\s*([\d.]+) in\s*\|"),
         Occ("docs/writeup.md", r"location with about ([\d.]+) inches of noise"),
         Occ("docs/writeup.md", r"batters at ([\d.]+) inches of noise versus")),
        tol=0.12, render=_r(2), unit="in",
        why="sigma-by-role bar chart; ~0.1 in bar resolution",
    ),
    Figure(
        "sigma_fld_in",
        lambda: float(_app("perception_sigma").set_index("side").loc["fielding", "sigma_in"]),
        (Occ("README.md", r"\|\s*catchers & pitchers\s*\|\s*\*\*([\d.]+) in\*\*"),
         Occ("docs/writeup.md", r"catchers and pitchers at ([\d.]+) inches")),
        tol=0.12, render=_r(2), unit="in", why="sigma bar chart resolution",
    ),
    Figure(
        "sigma_precision_gap_pct",
        lambda: (1 - float(_app("perception_sigma").set_index("side").loc["fielding", "sigma_in"])
                 / float(_app("perception_sigma").set_index("side").loc["batting", "sigma_in"])) * 100,
        (Occ("README.md", r"read the pitch about ([\d.]+)% more precisely"),
         Occ("README.md", r"already differs by ([\d.]+)% across roles"),
         Occ("docs/writeup.md", r"a measured ([\d.]+)% difference in perceptual noise"),
         Occ("docs/writeup.md", r"a ([\d.]+)% gap between roles"),
         Occ("app/streamlit_app.py", r"read the pitch ~([\d.]+)% more precisely")),
        tol=4, render=_rint, unit="pp",
        why="a ratio of two fitted sigmas, stated as '~28%'; noisier than "
            "either input",
    ),
    Figure(
        "bat_success_modeled",
        lambda: float(_lev(_OBS, "batting").success) * 100,
        (Occ("README.md", r"\|\s*batters\s*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("README.md", r"\(([\d.]+)% / [\d.]+%\) without ever seeing it"),
         Occ("docs/writeup.md", r"predicting ([\d.]+)% for\s*\n?batters")),
        tol=1.5, render=_r(1), unit="pp",
        why="an out-of-fit check value shown as text",
    ),
    Figure(
        "fld_success_modeled",
        lambda: float(_lev(_OBS, "fielding").success) * 100,
        (Occ("README.md", r"\|\s*catchers & pitchers\s*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("README.md", r"\([\d.]+% / ([\d.]+)%\) without ever seeing it"),
         Occ("docs/writeup.md", r"([\d.]+)% for catchers and pitchers")),
        tol=1.5, render=_r(1), unit="pp", why="out-of-fit check value, text",
    ),

    # ---- reliability (fragile; prose now says so) -----------------------
    Figure(
        "team_split_half_r",
        lambda: float(np.corrcoef(_app("split_half").h1_rate, _app("split_half").h2_rate)[0, 1]),
        (Occ("README.md", r"\*\*r ≈ ([\d.]+)\*\* across all 30 teams"),
         Occ("README.md", r"[Tt]eam-level split-half reliability is currently\s+\*\*r ≈ ([\d.]+)\*\*"),
         Occ("docs/writeup.md", r"split-half correlation across all 30 teams\s+is \*\*r ≈ ([\d.]+)\*\*")),
        tol=0.12, render=_r(2),
        why="split-half r at n=30 is poorly estimated (moved 0.22->0.28 on 5 "
            "days of data); prose frames it as fragile, so the gate should "
            "fire only on a sign flip or a >0.1 move, not normal wobble",
    ),
    Figure(
        "player_split_half_r_min8",
        lambda: float(_app("player_skill_test").set_index("min_challenges_per_half").loc[8, "r"]),
        (Occ("README.md", r"the individual correlation is \*\*r ≈ ([\d.]+)\*\*"),
         Occ("docs/writeup.md", r"the individual correlation is \*\*r ≈ ([\d.]+)\*\*")),
        tol=0.12, render=_r(2),
        why="same instability as the team figure (moved 0.32->0.27 on 5 days)",
    ),
    Figure(
        "player_split_half_n_min8",
        lambda: float(_app("player_skill_test").set_index("min_challenges_per_half").loc[8, "n_players"]),
        (Occ("README.md", r"the individual correlation is \*\*r ≈ [\d.]+\*\* \(n ≈ (\d+),"),
         Occ("docs/writeup.md", r"the individual correlation is \*\*r ≈ [\d.]+\*\* \(n ≈ (\d+),")),
        tol=15, render=_rint,
        why="a sample size that grows through the season; not load-bearing",
    ),
    Figure(
        "player_split_half_r_min10",
        lambda: float(_app("player_skill_test").set_index("min_challenges_per_half").loc[10, "r"]),
        (Occ("README.md", r"at \*\*≥10\*\* it is \*\*r ≈ ([\d.]+)\*\*"),
         Occ("docs/writeup.md", r"at ≥10 it is \*\*r ≈ ([\d.]+)\*\*")),
        tol=0.12, render=_r(2),
        why="the sturdier of the player cuts, but still one partial season",
    ),
    Figure(
        "player_split_half_n_min10",
        lambda: float(_app("player_skill_test").set_index("min_challenges_per_half").loc[10, "n_players"]),
        (Occ("README.md", r"at \*\*≥10\*\* it is \*\*r ≈ [\d.]+\*\* \(n ≈ (\d+),"),
         Occ("docs/writeup.md", r"at ≥10 it is \*\*r ≈ [\d.]+\*\* \(n ≈ (\d+),")),
        tol=15, render=_rint, why="growing sample size; not load-bearing",
    ),
    Figure(
        "catcher_quality_corr_r",
        lambda: float(_app("catcher_summary").quality_corr_r.iloc[0]),
        (Occ("README.md", r"catcher-quality correlation \(\*\*r = ([\d.]+),"),
         Occ("app/streamlit_app.py", r"\*\*r = ([\d.]+), p = [\d.]+\*\*, across all 30 teams")),
        tol=0.10, render=_r(2),
        why="a 30-point correlation shown against a scatter; stable across "
            "refreshes so far (0.44 -> 0.48) but keep tol generous",
    ),
    Figure(
        "catcher_quality_corr_p",
        lambda: float(_app("catcher_summary").quality_corr_p.iloc[0]),
        (Occ("README.md", r"catcher-quality correlation \(\*\*r = [\d.]+, p = ([\d.]+)\*\*"),
         Occ("app/streamlit_app.py", r"\*\*r = [\d.]+, p = ([\d.]+)\*\*, across all 30 teams")),
        tol=0.03, render=_r(3),
        why="a p-value shown as text; nobody distinguishes 0.007 from 0.02",
    ),
    Figure(
        "cin_z_above_league",
        lambda: float(_app("team_significance").set_index("team").loc["CIN", "z"]),
        # The app's copy of this figure was made fully data-driven (it renders
        # whichever team has the largest z and its exact value), so there is no
        # static prose there to drift. README / writeup still name Cincinnati
        # and 3.4 sd explicitly and are checked here.
        (Occ("README.md", r"success rate sits ([\d.]+) standard deviations"),
         Occ("docs/writeup.md", r"Cincinnati, sits ([\d.]+) standard deviations")),
        tol=0.4, render=_r(1), unit="sd", why="text ('3.4 sd')",
    ),
    Figure(
        "cin_bonferroni_p",
        lambda: float(_app("team_significance").set_index("team").loc["CIN", "p_bonferroni"]),
        (Occ("README.md", r"Bonferroni-adjusted p ≈ ([\d.]+)\)"),
         Occ("docs/writeup.md", r"\(p ≈ ([\d.]+)\)\. There is more real")),
        tol=0.03, render=_r(2), why="text ('p ~ 0.02')",
    ),

    # ---- ball-radius classification check --------------------------------
    Figure(
        "naive_disagreement_pct",
        lambda: float(_pipe("ball_radius_classification_check").naive_disagreement_rate.iloc[0]) * 100,
        (Occ("README.md", r"centre-only error rate is \*\*~([\d.]+)%\*\*"),),
        tol=2.0, render=_r(1), unit="pp", why="text",
    ),
    Figure(
        "naive_disagreement_borderline_pct",
        lambda: float(_pipe("ball_radius_classification_check")
                      .naive_disagreement_rate_borderline.iloc[0]) * 100,
        (Occ("README.md", r"actual ruling on\s*\*\*~([\d.]+)%\*\* of them"),),
        tol=3.0, render=_r(1), unit="pp", why="text",
    ),
    Figure(
        "corrected_match_pct",
        lambda: float(_pipe("ball_radius_classification_check").corrected_match_rate.iloc[0]) * 100,
        (Occ("README.md", r"corrected model matches MLB's ruling\s+\*\*~?([\d.]+)%"),),
        tol=0.3, render=_r(1), unit="pp",
        why="precise-looking text; a reader will not spot 99.8 vs 99.9",
    ),

    # ---- listed-vs-measured height (fixed early-season window -> stable) -
    Figure(
        "n_robust_batters",
        lambda: float(_pipe("height_uncertainty_check").n_robust_batters.iloc[0]),
        (Occ("README.md", r"\*\*Listed vs\. measured height\.\*\*\s*(\d+) batters have"),),
        tol=15, render=_rint,
        why="derived from a FIXED early-season window, so nearly constant; "
            "loose tol only guards a method change",
    ),
    Figure(
        "height_flip_rate_pct",
        lambda: float(_pipe("height_uncertainty_check").flip_rate.iloc[0]) * 100,
        (Occ("README.md", r"which flips the in/out call on ([\d.]+)% of near-boundary"),),
        tol=1.0, render=_r(1), unit="pp", why="text",
    ),
    Figure(
        "height_ambiguous_rate_pct",
        lambda: float(_pipe("height_uncertainty_check").ambiguous_rate.iloc[0]) * 100,
        (Occ("README.md", r"leaves ([\d.]+)% genuinely ambiguous"),),
        tol=2.0, render=_r(1), unit="pp", why="text",
    ),

    # ---- zone-region sigma sensitivity (FROZEN -> stable, keep tight) ----
    Figure(
        "zone_sigma_gap_move_runs",
        lambda: float(
            _app("zone_sigma_sensitivity").set_index("label")
            .loc["optimal @ zone-region sigma (sensitivity)", "decision_gap_vs_observed_per_season"]
            - _app("zone_sigma_sensitivity").set_index("label")
            .loc["optimal @ player sigma (role-only, canonical)", "decision_gap_vs_observed_per_season"]),
        (Occ("README.md", r"moves the headline decision gap by \*\*\+([\d.]+)\s*\n?runs/team-season\*\*"),
         Occ("README.md", r"decision model moves the headline decision gap by \+([\d.]+) runs/season"),
         Occ("README.md", r"Treat \+([\d.]+) as one plausible point"),
         Occ("README.md", r"point estimate moved from \+0\.02 to \+([\d.]+)\s*\n?runs/season")),
        tol=0.1, render=_r(2), unit="runs/season",
        why="FROZEN artifact (zone_sigma_refit.py is not re-run daily); value "
            "does not move between refreshes, so tol stays tight",
    ),
    Figure(
        "zone_sigma_boot_ci_lo",
        lambda: float(np.percentile(_app("zone_sigma_sensitivity_bootstrap").move_runs_per_season, 2.5)),
        (Occ("README.md", r"a 95% interval of \*\*(-[\d.]+) to \+[\d.]+\s*\n?runs/season\*\*"),),
        tol=0.4, render=_r(1), unit="runs/season", why="FROZEN artifact; stable",
    ),
    Figure(
        "zone_sigma_boot_ci_hi",
        lambda: float(np.percentile(_app("zone_sigma_sensitivity_bootstrap").move_runs_per_season, 97.5)),
        (Occ("README.md", r"a 95% interval of \*\*-[\d.]+ to \+([\d.]+)\s*\n?runs/season\*\*"),),
        tol=0.4, render=_r(1), unit="runs/season", why="FROZEN artifact; stable",
    ),

    # ---- zone role-gap location dependence ------------------------------
    Figure(
        "zone_lr_swing_pp",
        lambda: float(_app("zone_interaction").swing_pp.iloc[0]),
        (Occ("README.md", r"a (\d+)-point\s*\n?swing across well-populated regions"),
         Occ("README.md", r"a (\d+)-point swing across well-populated regions"),),
        tol=6, render=_rint, unit="pp", why="text ('a 38-point swing')",
    ),

    # ---- catcher spotlight (Tyler Stephenson) ---------------------------
    Figure(
        "stephenson_pct_rank",
        lambda: float(_app("catcher_check").set_index("team").loc["CIN", "pct_rank"]) * 100,
        (Occ("README.md", r"the (\d+)th\s+percentile of all catchers"),
         Occ("docs/writeup.md", r"the (\d+)th\s+percentile of all catchers")),
        tol=8, render=_rint, why="text ('85th percentile'); rank moves as more catchers qualify",
    ),
    Figure(
        "stephenson_success_pct",
        lambda: float(_app("catcher_check").set_index("team").loc["CIN", "rate"]) * 100,
        (Occ("docs/writeup.md", r"own challenge success \(a ([\d.]+)% success rate\)"),),
        tol=5, render=_rint, unit="pp", why="text",
    ),
]


# ---------------------------------------------------------------- collect / check

def collect() -> dict:
    figs = {}
    for f in FIGURES:
        v = float(f.compute())
        figs[f.key] = {
            "value": v,
            "rendered": f.render(v),
            "unit": f.unit,
            "tol": f.tol,
            "kind": f.kind,
            "why_tol": f.why,
            "appears_in": sorted({o.path for o in f.occ}),
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_through": data_through(),
        "n_figures": len(FIGURES),
        "figures": figs,
    }


def _check_one(f: Figure, want: float, text: str, o: Occ) -> Optional[Problem]:
    m = re.search(o.pattern, text)
    if m is None:
        return Problem(f.key, o.path, None,
                       f"anchor not found -- prose rewritten or pattern stale: "
                       f"{o.pattern!r}. Data value is {f.render(want)}"
                       f"{(' ' + f.unit) if f.unit else ''}.")
    line = text[:m.start(1)].count("\n") + 1

    if f.kind == "range":
        try:
            lo, hi = parse_number(m.group(1)), parse_number(m.group(2))
        except (ValueError, IndexError):
            return Problem(f.key, o.path, line,
                           f"range pattern must capture two numeric groups: {o.pattern!r}")
        if lo - f.tol <= want <= hi + f.tol:
            return None
        return Problem(f.key, o.path, line,
                       f"prose states the band {m.group(1)}-{m.group(2)}; data is "
                       f"{f.render(want)}{(' ' + f.unit) if f.unit else ''}, outside "
                       f"[{lo - f.tol:g}, {hi + f.tol:g}]")

    try:
        got = parse_number(m.group(1))
    except ValueError:
        return Problem(f.key, o.path, line,
                       f"captured {m.group(1)!r}, not a number -- pattern group misplaced: {o.pattern!r}")

    if f.kind == "floor":
        if want >= got - f.tol:
            return None
        return Problem(f.key, o.path, line,
                       f"prose claims a floor of {m.group(1).strip()}; data has fallen to "
                       f"{f.render(want)}{(' ' + f.unit) if f.unit else ''} "
                       f"(below floor - tolerance {f.tol:g}) -- likely a data regression")

    if abs(got - want) > f.tol:
        return Problem(f.key, o.path, line,
                       f"prose says {m.group(1).strip()!r}; data is now "
                       f"{f.render(want)}{(' ' + f.unit) if f.unit else ''} "
                       f"(|diff| {abs(got - want):.4g} > tolerance {f.tol:g})")
    return None


def check(values: Optional[dict] = None) -> list[Problem]:
    if values is None:
        values = {f.key: float(f.compute()) for f in FIGURES}
    problems: list[Problem] = []
    text_cache: dict[str, str] = {}
    for f in FIGURES:
        want = values[f.key]
        for o in f.occ:
            text = text_cache.setdefault(o.path, (ROOT / o.path).read_text())
            p = _check_one(f, want, text, o)
            if p is not None:
                problems.append(p)
    return problems


# ---------------------------------------------------------------- NOT tracked
#
# Intentionally left out of FIGURES:
#
# * Physical / rule constants -- 17 in plate width, 53.5% / 27% zone edges,
#   BALL_RADIUS_FT = 0.1208, "two challenges per team", the 2.9 in ball.
#   A change here is a spec change, not drift.
# * Rounded restatements of a tracked figure in places already covered by the
#   precise figure ("about twice a game", "two out of every three", "north of
#   68%").
# * Per-row snapshot tables -- the individual team rows (SD/WSH/TB/...) and
#   player rows (Curtis Mead/...) in README, framed there as "a 2026 snapshot".
# * n_challenges_raw / n_borderline -- the method paragraph no longer states
#   exact challenge counts (it says "roughly half of all season challenges"),
#   so there is nothing to check. n_opportunities (floor) still covers the
#   header claim.
# * n_endorsed_misses / the per-game rate -- the app now prints these straight
#   from endorsed_miss_summary at load time (f"{int(miss_summary.n):,}"), so
#   they cannot drift out of sync with the page they are on.
# * stephenson's raw challenge-attempt count -- a growing number; the prose was
#   reworded to drop it and keep the success rate + percentile.
# * The cross-team spread-test p-values (0.004 / <0.0001) and the 69% / 49%
#   high-middle zone cells -- printed by team_skill_test.py / zone_analysis.py
#   but not persisted to a parquet. TODO: emit a one-row summary parquet, then
#   add figures here.
