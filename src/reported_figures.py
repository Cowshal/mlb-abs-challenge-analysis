"""
Single source of truth for the hardcoded statistics that appear in prose --
README.md, docs/writeup.md, and the app's copy in app/streamlit_app.py.

Why this exists
---------------
build_app_data.py's provenance guards stop a *mixed-vintage data set* from
shipping. They do nothing about prose going stale while the data underneath
it moves every night. A scheduled refresh that updates the charts but not the
sentence next to them ships a site that contradicts itself -- exactly the
failure mode the "a number that can't be regenerated doesn't belong" rule in
CLAUDE.md was written against, one level up.

This module:

  collect()  -- reads the pipeline's parquet outputs and returns the current
                value of every tracked figure, plus the data-through date.
                scripts/build_reported_figures.py writes this to
                data/reported_figures.json (and app/data/, for the footer).

  check()    -- re-reads those values and, for each figure, greps the prose
                file(s) it appears in for the number as written. Returns a
                list of Problem records for anything that has drifted past
                its per-figure tolerance, OR whose anchoring text can no
                longer be found (prose rewritten, pattern stale). The
                pipeline turns a non-empty list into a failed build.

Adding a figure
---------------
Append a Figure(...) to FIGURES with:
  - compute:  a zero-arg fn returning the value IN THE UNITS IT IS WRITTEN
              (percentages as 53.7, not 0.537).
  - render:   value -> the canonical string, for messages.
  - tol:      absolute tolerance, in the written units. Set it to the point
              where a reader would actually be misled, not to the noise
              floor -- this runs every night during the season.
  - occ:      one Occ(path, pattern) per place the number is written. The
              pattern is a regex with EXACTLY ONE capture group around the
              number token; the surrounding text anchors it (and is what
              trips the "prose rewritten" alarm if it changes).

Deliberately NOT tracked (rounded restatements of a figure that IS tracked,
physical constants, or per-row snapshot tables): see NOT_TRACKED at the
bottom.
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
    """A consolidated artifact from app/data/ (what the deployed site loads)."""
    return pd.read_parquet(APP_DATA / f"{name}.parquet")


@lru_cache(maxsize=None)
def _pipe(name: str) -> pd.DataFrame:
    """A pipeline artifact from data/ that isn't copied into app/data/."""
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
    """Observed/optimal challenges per team-game for one role, derived from
    the total per-team-game rate and that role's share of the challenge
    count. Done this way because policy_decomposition's *observed* row stores
    the per-game and success values under swapped-looking column names;
    leverage_comparison's n / success columns are unambiguous."""
    label = _OBS if policy == _OBS else _OPT
    total = _dec_row(label).challenges_per_team_game
    n_role = _lev(policy, role).n
    n_all = _lev(policy, "all").n
    return float(total * n_role / n_all)


def parse_number(token: str) -> float:
    """A captured prose token -> float. Handles thousands commas, a trailing
    or leading %, a leading +, and the unicode minus sign."""
    t = token.strip().replace(",", "").replace("−", "-").replace("%", "")
    t = t.lstrip("+").strip()
    return float(t)


# ---------------------------------------------------------------- data-through

def data_through() -> str:
    """Latest game date present in the pipeline's challenge-opportunity table
    (falls back to the 2026 Statcast cache). This is the date the whole site
    is current to."""
    opp = DATA / "challenge_opportunities.parquet"
    if opp.exists():
        gd = pd.read_parquet(opp, columns=["game_date"]).game_date
        return str(pd.to_datetime(gd).max().date())
    sc = DATA / "statcast_2026.parquet"
    gd = pd.read_parquet(sc, columns=["game_date"]).game_date
    return str(pd.to_datetime(gd).max().date())


# ---------------------------------------------------------------- figure model

@dataclass(frozen=True)
class Occ:
    path: str        # relative to repo root
    pattern: str     # regex; exactly one () group around the number token


@dataclass(frozen=True)
class Figure:
    key: str
    compute: Callable[[], float]
    occ: tuple[Occ, ...]
    tol: float
    render: Callable[[float], str] = staticmethod(lambda v: f"{v:.4g}")
    unit: str = ""
    note: str = ""


@dataclass(frozen=True)
class Problem:
    key: str
    path: str
    line: Optional[int]
    expected: float
    expected_str: str
    detail: str

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.key}] {loc} -- {self.detail}"


# shorthand renderers
def _r(dp):        return lambda v: f"{v:.{dp}f}"
def _rint(v):      return f"{int(round(v))}"
def _rcomma(v):    return f"{int(round(v)):,}"


# ---------------------------------------------------------------- THE FIGURES

FIGURES: list[Figure] = [

    # ---- coverage / counts --------------------------------------------------
    Figure(
        "n_opportunities",
        lambda: float(_lev(_OBS, "all").n),
        (Occ("README.md", r"season,\s*([\d,]+)\s*challenges across"),
         Occ("README.md", r"distinct from the\s*([\d,]+)\s*\n?\s*\*opportunities\*"),
         Occ("docs/writeup.md", r"Across\s*([\d,]+)\s*\n?challenges in the 2026 season"),
         Occ("app/streamlit_app.py", r"— ([\d,]+) challenges across")),
        tol=250, render=_rcomma, unit="challenges",
    ),
    Figure(
        "n_games_2026",
        lambda: float(pd.read_parquet(DATA / "challenge_opportunities.parquet",
                                      columns=["game_pk"]).game_pk.nunique()),
        (Occ("README.md", r"challenges across\s*([\d,]+)\s*games"),
         Occ("app/streamlit_app.py", r"challenges across ([\d,]+) games")),
        tol=60, render=_rcomma, unit="games",
    ),
    Figure(
        "n_challenges_raw",
        lambda: float(_pipe("ball_radius_classification_check").n_challenges.iloc[0]),
        (Occ("README.md", r"half of all\s*([\d,]+)\s*season"),
         Occ("README.md", r"Across all\s*([\d,]+)\s*challenges\s*\n?\s*unconditionally")),
        tol=250, render=_rcomma, unit="challenges",
    ),
    Figure(
        "n_borderline",
        lambda: float(_pipe("ball_radius_classification_check").n_borderline.iloc[0]),
        (Occ("README.md", r"centre-based boundary,\s*n=([\d,]+)"),),
        tol=200, render=_rcomma,
    ),

    # ---- headline decomposition ------------------------------------------------
    Figure(
        "obs_challenges_per_game",
        lambda: float(_dec_row(_OBS).challenges_per_team_game),
        (Occ("README.md", r"\*\*Observed 2026\*\*\s*\|\s*([\d.]+)\s*\|"),),
        tol=0.06, render=_r(2), unit="/team-game",
    ),
    Figure(
        "obs_challenges_per_game_1dp",
        lambda: float(_dec_row(_OBS).challenges_per_team_game),
        (Occ("README.md", r"league behaviour \(([\d.]+) challenges per"),
         Occ("app/streamlit_app.py", r"attempt about \*\*([\d.]+) challenges per game\*\*"),
         Occ("docs/writeup.md", r"currently challenge about ([\d.]+) times a game")),
        tol=0.09, render=_r(1),
    ),
    Figure(
        "obs_success_rate",
        lambda: float(_dec_row(_OBS).success_rate) * 100,
        (Occ("README.md", r"\*\*Observed 2026\*\*\s*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("README.md", r"\|\s*observed\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("docs/writeup.md", r"league-wide success rate is ([\d.]+)%"),
         Occ("docs/writeup.md", r"to match the observed ([\d.]+)%\s*\n?\s*success rate")),
        tol=0.6, render=_r(1), unit="pp",
    ),
    Figure(
        "obs_success_rate_54",
        lambda: float(_dec_row(_OBS).success_rate) * 100,
        (Occ("README.md", r"at a ([\d.]+)% success rate\) does not"),
         Occ("app/streamlit_app.py", r"and win \*\*([\d.]+)%\*\* of the")),
        tol=0.9, render=_rint,
    ),
    Figure(
        "obs_runs_per_game",
        lambda: float(_dec_row(_OBS).runs_per_team_game),
        (Occ("README.md", r"\*\*Observed 2026\*\*\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),
         Occ("docs/writeup.md", r"observed 2026 behavior \(([\d.]+)\)")),
        tol=0.006, render=_r(3), unit="runs/team-game",
    ),
    Figure(
        "opt_challenges_per_game",
        lambda: float(_dec_row(_OPT).challenges_per_team_game),
        (Occ("README.md", r"\*\*Optimal, same information\*\*\s*\|\s*([\d.]+)\s*\|"),),
        tol=0.06, render=_r(2),
    ),
    Figure(
        "opt_success_rate",
        lambda: float(_dec_row(_OPT).success_rate) * 100,
        (Occ("README.md", r"\*\*Optimal, same information\*\*\s*\|[^|]*\|\s*([\d.]+)%"),
         Occ("README.md", r"\|\s*optimal\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("docs/writeup.md", r"wins a \*smaller\* share:\s*([\d.]+)%")),
        tol=0.6, render=_r(1), unit="pp",
    ),
    Figure(
        "opt_runs_per_game",
        lambda: float(_dec_row(_OPT).runs_per_team_game),
        (Occ("README.md", r"\*\*Optimal, same information\*\*\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),
         Occ("docs/writeup.md", r"the optimal policy given the same information \(([\d.]+)\)")),
        tol=0.006, render=_r(3),
    ),
    Figure(
        "ceil_challenges_per_game",
        lambda: float(_dec_row(_CEIL).challenges_per_team_game),
        (Occ("README.md", r"Ceiling \(perfect information\)\s*\|\s*([\d.]+)\s*\|"),),
        tol=0.08, render=_r(2),
    ),
    Figure(
        "ceil_success_rate",
        lambda: float(_dec_row(_CEIL).success_rate) * 100,
        (Occ("README.md", r"Ceiling \(perfect information\)\s*\|[^|]*\|\s*([\d.]+)%"),),
        tol=0.7, render=_r(1), unit="pp",
    ),
    Figure(
        "ceil_runs_per_game",
        lambda: float(_dec_row(_CEIL).runs_per_team_game),
        (Occ("README.md", r"Ceiling \(perfect information\)\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),
         Occ("docs/writeup.md", r"perfect-information ceiling \(([\d.]+)\)")),
        tol=0.008, render=_r(3),
    ),
    Figure(
        "decision_gap_runs_per_season",
        lambda: (float(_dec_row(_OPT).runs_per_team_game)
                 - float(_dec_row(_OBS).runs_per_team_game)) * _SEASON_GAMES,
        (Occ("README.md", r"teams leave roughly \*\*(\d+) runs per team-season\*\*"),
         Occ("README.md", r"\*\*Decision gap:\s*~(\d+) runs per team-season\.\*\*"),
         Occ("docs/writeup.md", r"a gap of about (\d+) runs\."),
         Occ("docs/writeup.md", r"coachable gap I'm reporting \((\d+) runs\)")),
        tol=0.7, render=_rint, unit="runs/season",
    ),

    # ---- leverage table -----------------------------------------------------
    Figure("obs_mean_dre", lambda: float(_lev(_OBS, "all").mean_dre),
           (Occ("README.md", r"\|\s*observed\s*\|\s*([\d.]+)\s*\|"),),
           tol=0.006, render=_r(3)),
    Figure("obs_median_dre", lambda: float(_lev(_OBS, "all").median_dre),
           (Occ("README.md", r"\|\s*observed\s*\|[^|]*\|\s*([\d.]+)\s*\|"),),
           tol=0.006, render=_r(3)),
    Figure("obs_runs_per_overturn", lambda: float(_lev(_OBS, "all").runs_per_overturn),
           (Occ("README.md", r"\|\s*observed\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),),
           tol=0.006, render=_r(3)),
    Figure("opt_mean_dre", lambda: float(_lev(_OPT, "all").mean_dre),
           (Occ("README.md", r"\|\s*optimal\s*\|\s*([\d.]+)\s*\|"),),
           tol=0.006, render=_r(3)),
    Figure("opt_median_dre", lambda: float(_lev(_OPT, "all").median_dre),
           (Occ("README.md", r"\|\s*optimal\s*\|[^|]*\|\s*([\d.]+)\s*\|"),),
           tol=0.006, render=_r(3)),
    Figure("opt_runs_per_overturn", lambda: float(_lev(_OPT, "all").runs_per_overturn),
           (Occ("README.md", r"\|\s*optimal\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"),),
           tol=0.006, render=_r(3)),

    # ---- ceiling sensitivity curve ---------------------------------------------
    *[
        Figure(
            f"info_gap_season_ceiling_{int(s*100):03d}",
            (lambda s=s: float(_app("ceiling_sensitivity")
                               .set_index("ceiling_sigma_in")
                               .loc[s, "info_gap_runs_per_team_season"])),
            (Occ("README.md", rf"\|\s*{s:.2f} in\s*\|\s*([\d.]+)\s*\|"),),
            tol=0.6, render=_r(1), unit="runs/season",
        )
        for s in (0.10, 0.25, 0.50, 0.75, 1.00)
    ],

    # ---- perceptual sigma --------------------------------------------------
    Figure(
        "sigma_bat_in",
        lambda: float(_app("perception_sigma").set_index("side").loc["batting", "sigma_in"]),
        (Occ("README.md", r"\|\s*batters\s*\|\s*([\d.]+) in\s*\|"),
         Occ("docs/writeup.md", r"location with about ([\d.]+) inches of noise"),
         Occ("docs/writeup.md", r"batters at ([\d.]+) inches of noise versus")),
        tol=0.05, render=_r(2), unit="in",
    ),
    Figure(
        "sigma_fld_in",
        lambda: float(_app("perception_sigma").set_index("side").loc["fielding", "sigma_in"]),
        (Occ("README.md", r"\|\s*catchers & pitchers\s*\|\s*\*\*([\d.]+) in\*\*"),
         Occ("docs/writeup.md", r"catchers and pitchers at ([\d.]+) inches")),
        tol=0.05, render=_r(2), unit="in",
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
        tol=1.5, render=_rint, unit="pp",
    ),
    Figure(
        "bat_success_modeled",
        lambda: float(_lev(_OBS, "batting").success) * 100,
        (Occ("README.md", r"\|\s*batters\s*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("README.md", r"\(([\d.]+)% / [\d.]+%\) without ever seeing it"),
         Occ("docs/writeup.md", r"predicting ([\d.]+)% for\s*\n?batters")),
        tol=0.6, render=_r(1), unit="pp",
    ),
    Figure(
        "fld_success_modeled",
        lambda: float(_lev(_OBS, "fielding").success) * 100,
        (Occ("README.md", r"\|\s*catchers & pitchers\s*\|[^|]*\|\s*([\d.]+)%\s*\|"),
         Occ("README.md", r"\([\d.]+% / ([\d.]+)%\) without ever seeing it"),
         Occ("docs/writeup.md", r"([\d.]+)% for catchers and pitchers")),
        tol=0.6, render=_r(1), unit="pp",
    ),

    # ---- reliability ------------------------------------------------------------
    Figure(
        "team_split_half_r",
        lambda: float(np.corrcoef(_app("split_half").h1_rate, _app("split_half").h2_rate)[0, 1]),
        (Occ("README.md", r"against second-half\s*\n?gives r = ([\d.]+) across all 30 teams"),
         Occ("README.md", r"split-half reliability is r = ([\d.]+) \(95% CI \[-0\.16"),
         Occ("docs/writeup.md", r"the correlation came back at\s*\n?r = ([\d.]+), 95% confidence")),
        tol=0.04, render=_r(2),
    ),
    Figure(
        "player_split_half_r_min8",
        lambda: float(_app("player_skill_test").set_index("min_challenges_per_half").loc[8, "r"]),
        (Occ("README.md", r"gives r = ([\d.]+) \(p < 0\.001, n = 104"),
         Occ("docs/writeup.md", r"the correlation is r = ([\d.]+)\s*\n?\(p < 0\.001, 95% CI 0\.14")),
        tol=0.04, render=_r(2),
    ),
    Figure(
        "player_split_half_n_min8",
        lambda: float(_app("player_skill_test").set_index("min_challenges_per_half").loc[8, "n_players"]),
        (Occ("README.md", r"p < 0\.001, n = (\d+) players with"),
         Occ("docs/writeup.md", r"8 challenges per half \((\d+) players\)")),
        tol=4, render=_rint,
    ),
    Figure(
        "player_split_half_r_min10",
        lambda: float(_app("player_skill_test").set_index("min_challenges_per_half").loc[10, "r"]),
        (Occ("README.md", r"to\s*\n?r = ([\d.]+) \(p < 0\.001, n = 82"),
         Occ("docs/writeup.md", r"at a minimum of 10 \(82 players\), r = ([\d.]+)")),
        tol=0.04, render=_r(2),
    ),
    Figure(
        "player_split_half_n_min10",
        lambda: float(_app("player_skill_test").set_index("min_challenges_per_half").loc[10, "n_players"]),
        (Occ("README.md", r"n = (\d+) with ≥10/half"),
         Occ("docs/writeup.md", r"at a minimum of 10 \((\d+) players\)")),
        tol=4, render=_rint,
    ),
    Figure(
        "catcher_quality_corr_r",
        lambda: float(_app("catcher_summary").quality_corr_r.iloc[0]),
        (Occ("README.md", r"the r = ([\d.]+) catcher-quality correlation"),
         Occ("app/streamlit_app.py", r"\*\*r = ([\d.]+), p = 0\.01\*\*")),
        tol=0.04, render=_r(2),
    ),
    Figure(
        "catcher_quality_corr_p",
        lambda: float(_app("catcher_summary").quality_corr_p.iloc[0]),
        (Occ("app/streamlit_app.py", r"\*\*r = 0\.44, p = ([\d.]+)\*\*"),),
        tol=0.006, render=_r(2),
    ),
    Figure(
        "cin_z_above_league",
        lambda: float(_app("team_significance").set_index("team").loc["CIN", "z"]),
        (Occ("README.md", r"success rate sits ([\d.]+) standard deviations"),
         Occ("docs/writeup.md", r"Cincinnati, sits ([\d.]+) standard deviations"),
         Occ("app/streamlit_app.py", r"Cincinnati, is ([\d.]+) standard")),
        tol=0.15, render=_r(1), unit="sd",
    ),
    Figure(
        "cin_bonferroni_p",
        lambda: float(_app("team_significance").set_index("team").loc["CIN", "p_bonferroni"]),
        (Occ("README.md", r"\(Bonferroni-adjusted p ≈ ([\d.]+)\)"),
         Occ("docs/writeup.md", r"\(p ≈ ([\d.]+)\)\. There is more real")),
        tol=0.012, render=_r(2),
    ),

    # ---- ball-radius classification check --------------------------------------
    Figure(
        "naive_disagreement_pct",
        lambda: float(_pipe("ball_radius_classification_check").naive_disagreement_rate.iloc[0]) * 100,
        (Occ("README.md", r"centre-only error rate is ([\d.]+)%"),
         Occ("README.md", r"README \(both σ values, the ([\d.]+)%")),
        tol=0.6, render=_r(1), unit="pp",
    ),
    Figure(
        "naive_disagreement_borderline_pct",
        lambda: float(_pipe("ball_radius_classification_check")
                      .naive_disagreement_rate_borderline.iloc[0]) * 100,
        (Occ("README.md", r"actual ruling on\s*\n?\s*\*\*([\d.]+)%\*\* of them"),
         Occ("README.md", r"error rate \(66\.9% →\s*\n?\s*([\d.]+)%")),
        tol=1.0, render=_r(1), unit="pp",
    ),
    Figure(
        "corrected_match_pct",
        lambda: float(_pipe("ball_radius_classification_check").corrected_match_rate.iloc[0]) * 100,
        (Occ("README.md", r"the corrected model matches MLB's ruling\s*\n?\s*\*\*([\d.]+)%\*\*"),
         Occ("README.md", r"match rate did not \(([\d.]+)% vs\."),
         Occ("README.md", r"came back\s*\n?at ([\d.]+)% against a previously-quoted")),
        tol=0.1, render=_r(2), unit="pp",
    ),
    Figure(
        "corrected_match_borderline_pct",
        lambda: float(_pipe("ball_radius_classification_check")
                      .corrected_match_rate_borderline.iloc[0]) * 100,
        (Occ("README.md", r"\*\*[\d.]+%\*\* of the time overall, ([\d.]+)% within the borderline"),),
        tol=0.1, render=_r(2), unit="pp",
    ),

    # ---- listed-vs-measured height ------------------------------------------
    Figure(
        "n_robust_batters",
        lambda: float(_pipe("height_uncertainty_check").n_robust_batters.iloc[0]),
        (Occ("README.md", r"\*\*Listed vs\. measured height\.\*\*\s*(\d+) batters have"),),
        tol=3, render=_rint,
    ),
    Figure(
        "height_flip_rate_pct",
        lambda: float(_pipe("height_uncertainty_check").flip_rate.iloc[0]) * 100,
        (Occ("README.md", r"which flips the in/out call on ([\d.]+)% of near-boundary"),
         Occ("README.md", r"from 2\.04%/7\.97% to ([\d.]+)%/7\.8%")),
        tol=0.3, render=_r(1), unit="pp",
    ),
    Figure(
        "height_ambiguous_rate_pct",
        lambda: float(_pipe("height_uncertainty_check").ambiguous_rate.iloc[0]) * 100,
        (Occ("README.md", r"leaves ([\d.]+)% genuinely ambiguous"),
         Occ("README.md", r"from 2\.04%/7\.97% to 2\.1%/([\d.]+)%")),
        tol=0.4, render=_r(1), unit="pp",
    ),

    # ---- zone-region sigma sensitivity (FROZEN artifacts, see refresh_all.sh) --
    Figure(
        "zone_sigma_gap_move_runs",
        lambda: float(
            _app("zone_sigma_sensitivity")
            .set_index("label")
            .loc["optimal @ zone-region sigma (sensitivity)", "decision_gap_vs_observed_per_season"]
            - _app("zone_sigma_sensitivity")
            .set_index("label")
            .loc["optimal @ player sigma (role-only, canonical)", "decision_gap_vs_observed_per_season"]),
        (Occ("README.md", r"moves the headline decision gap by \*\*\+([\d.]+)\s*\n?runs/team-season\*\*"),
         Occ("README.md", r"decision model moves the headline decision gap by \+([\d.]+) runs/season"),
         Occ("README.md", r"Treat \+([\d.]+) as one plausible point"),
         Occ("README.md", r"point estimate moved from \+0\.02 to \+([\d.]+)\s*\n?runs/season")),
        tol=0.06, render=_r(2), unit="runs/season",
        note="frozen: zone_sigma_refit.py is not re-run by the daily pipeline",
    ),
    Figure(
        "zone_sigma_boot_ci_lo",
        lambda: float(np.percentile(_app("zone_sigma_sensitivity_bootstrap").move_runs_per_season, 2.5)),
        (Occ("README.md", r"a 95% interval of \*\*(-[\d.]+) to \+[\d.]+\s*\n?runs/season\*\*"),),
        tol=0.35, render=_r(1), unit="runs/season",
        note="frozen: zone_sigma_bootstrap.py is not re-run by the daily pipeline",
    ),
    Figure(
        "zone_sigma_boot_ci_hi",
        lambda: float(np.percentile(_app("zone_sigma_sensitivity_bootstrap").move_runs_per_season, 97.5)),
        (Occ("README.md", r"a 95% interval of \*\*-[\d.]+ to \+([\d.]+)\s*\n?runs/season\*\*"),),
        tol=0.35, render=_r(1), unit="runs/season",
        note="frozen: zone_sigma_bootstrap.py is not re-run by the daily pipeline",
    ),

    # ---- zone role-gap location dependence ------------------------------------
    Figure(
        "zone_lr_swing_pp",
        lambda: float(_app("zone_interaction").swing_pp.iloc[0]),
        (Occ("README.md", r"a (\d+)-point\s*\n?swing across well-populated regions"),
         Occ("README.md", r"a (\d+)-point swing across well-populated regions"),),
        tol=2.5, render=_rint, unit="pp",
    ),

    # ---- endorsed missed opportunities --------------------------------------
    Figure(
        "n_endorsed_misses",
        lambda: float(_app("endorsed_miss_summary").n.iloc[0]),
        (Occ("app/streamlit_app.py", r"How big are these ([\d,]+), really\?"),
         Occ("app/streamlit_app.py", r"The same ([\d,]+) pitches under three cuts")),
        tol=250, render=_rcomma,
    ),

    # ---- catcher spotlight (Tyler Stephenson) -----------------------------------
    Figure(
        "stephenson_pct_rank",
        lambda: float(_app("catcher_check").set_index("team").loc["CIN", "pct_rank"]) * 100,
        (Occ("README.md", r"ranks in the (\d+)th percentile of all catchers"),
         Occ("docs/writeup.md", r"ranks in the (\d+)th percentile of all catchers")),
        tol=4, render=_rint,
    ),
    Figure(
        "stephenson_success_pct",
        lambda: float(_app("catcher_check").set_index("team").loc["CIN", "rate"]) * 100,
        (Occ("docs/writeup.md", r"own challenge success \((\d+)% on 119 attempts\)"),),
        tol=2.5, render=_rint, unit="pp",
    ),
    Figure(
        "stephenson_attempts",
        lambda: float(_app("catcher_check").set_index("team").loc["CIN", "n"]),
        (Occ("docs/writeup.md", r"own challenge success \(\d+% on (\d+) attempts\)"),),
        tol=4, render=_rint,
    ),
]


# ---------------------------------------------------------------- collect / check

def collect() -> dict:
    """Current value of every tracked figure + the data-through date. Written
    to data/reported_figures.json by scripts/build_reported_figures.py."""
    figs = {}
    for f in FIGURES:
        v = float(f.compute())
        appears = sorted({o.path for o in f.occ})
        figs[f.key] = {
            "value": v,
            "rendered": f.render(v),
            "unit": f.unit,
            "tol": f.tol,
            "appears_in": appears,
            **({"note": f.note} if f.note else {}),
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_through": data_through(),
        "n_figures": len(FIGURES),
        "figures": figs,
    }


def check(values: Optional[dict] = None) -> list[Problem]:
    """Compare each tracked figure's freshly-computed value against the number
    as written in prose. Returns every drift and every missing anchor -- an
    empty list means the prose is consistent with the data."""
    if values is None:
        values = {f.key: float(f.compute()) for f in FIGURES}
    problems: list[Problem] = []
    text_cache: dict[str, str] = {}

    for f in FIGURES:
        want = values[f.key]
        for o in f.occ:
            text = text_cache.setdefault(o.path, (ROOT / o.path).read_text())
            m = re.search(o.pattern, text)
            if m is None:
                problems.append(Problem(
                    f.key, o.path, None, want, f.render(want),
                    f"anchor not found -- prose rewritten, or the pattern is "
                    f"stale: {o.pattern!r}. Data value is {f.render(want)}"
                    f"{(' ' + f.unit) if f.unit else ''}."))
                continue
            token = m.group(1)
            try:
                got = parse_number(token)
            except ValueError:
                problems.append(Problem(
                    f.key, o.path, text[:m.start(1)].count("\n") + 1, want, f.render(want),
                    f"captured {token!r}, which is not a number -- pattern group "
                    f"is misplaced: {o.pattern!r}"))
                continue
            if abs(got - want) > f.tol:
                line = text[:m.start(1)].count("\n") + 1
                problems.append(Problem(
                    f.key, o.path, line, want, f.render(want),
                    f"prose says {token.strip()!r}; data is now "
                    f"{f.render(want)}{(' ' + f.unit) if f.unit else ''} "
                    f"(|diff| {abs(got - want):.4g} > tolerance {f.tol:g})"))
    return problems


# ---------------------------------------------------------------- NOT tracked
#
# Intentionally left out of FIGURES, with why:
#
# * Physical / rule constants -- 17 in plate width, 53.5% / 27% zone edges,
#   BALL_RADIUS_FT = 0.1208, "two challenges per team", the 2.9 in ball.
#   These do not move with data; a change here is a spec change, not drift.
# * Rounded restatements of a figure that IS tracked -- "about twice a game",
#   "two out of every three", "43% vs 54%", "north of 68%", "about 2.0 inches"
#   in places already covered by the precise figure. Adding every paraphrase
#   multiplies false "anchor moved" alarms for no extra protection.
# * Per-row snapshot tables -- the individual team rows (SD/WSH/TB/...) and
#   player rows (Curtis Mead/Yandy Diaz/...) in README. README already frames
#   these explicitly as "a 2026 snapshot, not a proven ranking"; they churn
#   by design and are not load-bearing claims.
# * The cross-team spread-test p-values (p = 0.004 / p < 0.0001) and the
#   69% / 49% high-middle zone cells -- their exact values are printed by
#   scripts/team_skill_test.py and scripts/zone_analysis.py but not persisted
#   to a parquet. TODO: have those scripts write a one-row summary parquet,
#   then add figures here.
