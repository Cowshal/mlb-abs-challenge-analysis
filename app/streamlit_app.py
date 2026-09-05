"""
ABS Challenge Optimizer -- 2026 MLB season.

Loads precomputed parquet from app/data/ and does nothing but filter and plot.
All modelling happens upstream in src/abs_policy.py.
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="ABS Challenge Optimizer", layout="wide")


@st.cache_data
def load(name):
    return pd.read_parquet(DATA / f"{name}.parquet")


dec = load("policy_decomposition")
sens = load("ceiling_sensitivity")
lev = load("leverage_comparison")
surf = load("threshold_surface")
per_team = load("per_team")
per_batter = load("per_batter")
sigma = load("perception_sigma")

st.title("Who's Leaving Runs on the Table?")
st.caption("Optimal ABS challenge policy vs. observed behaviour, 2026 MLB season "
           "— 9,037 challenges across 2,136 games")

obs, ply, ceil = (dec.runs_per_team_game.iloc[i] for i in (0, 1, 2))
decision_gap = (ply - obs) * 162

c1, c2, c3 = st.columns(3)
c1.metric("Observed", f"{obs*162:.0f} runs/team-season",
          help="Runs actually gained through successful challenges in 2026.")
c2.metric("Optimal, same information", f"{ply*162:.0f} runs/team-season",
          help="Optimal policy played with the perceptual noise players actually have.")
c3.metric("Decision gap", f"+{decision_gap:.0f} runs/team-season",
          help="The actionable number: better decisions, identical information.")

tab1, tab2, tab3 = st.tabs(
    ["Decomposition", "Optimal threshold", "Runs left on the table"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Where the gap actually comes from")
    st.markdown(
        "Teams challenge about **2.1 times per game and succeed 54%** of the time. "
        "An optimal policy given the *same* information challenges more often and "
        "succeeds *less* often — trading hit rate for leverage — and still nets more runs."
    )
    show = dec[["label", "challenges_per_team_game", "success_rate",
                "runs_per_team_game"]].copy()
    show.columns = ["Policy", "Challenges / team-game", "Success rate", "Runs / team-game"]
    st.dataframe(show.style.format({
        "Challenges / team-game": "{:.2f}", "Success rate": "{:.1%}",
        "Runs / team-game": "{:.3f}"}), hide_index=True, width='stretch')

    st.markdown(
        f"**Decision gap: +{decision_gap:.0f} runs/team-season.** This is the part a team "
        "can capture by changing policy alone. It does not depend on any assumption "
        "about tracking precision."
    )
    st.warning(
        "The remaining gap to a perfect-information ceiling is **not** coachable, and "
        "its size depends entirely on an assumed tracking precision that this dataset "
        "cannot measure — we only ever observe Hawk-Eye's own output, never independent "
        "ground truth. Shown below as a sensitivity curve rather than a single number."
    )
    st.altair_chart(
        alt.Chart(sens).mark_line(point=True).encode(
            x=alt.X("ceiling_sigma_in:Q", title="Assumed ceiling tracking σ (inches)"),
            y=alt.Y("info_gap_runs_per_team_season:Q",
                    title="Information gap (runs / team-season)"),
            tooltip=["ceiling_sigma_in", "info_gap_runs_per_team_season"],
        ).properties(height=280), width='stretch')

    st.subheader("Fitted perceptual noise, by role")
    st.markdown(
        "Estimated from *where* players chose to challenge — never from how often "
        "they challenged or how often they were right, so the fit cannot absorb the "
        "decision gap it is meant to measure."
    )
    s = sigma[["side", "sigma_in", "n_challenged"]].copy()
    s.columns = ["Role", "Fitted σ (inches)", "Challenges"]
    st.dataframe(s.style.format({"Fitted σ (inches)": "{:.2f}"}),
                 hide_index=True, width='stretch')
    st.caption("Catchers and pitchers read the pitch ~28% more precisely than batters, "
               "consistent with seeing it from behind rather than from the side.")

    st.subheader("Challenge more, or challenge differently?")
    l = lev[lev.role == "all"][["policy", "mean_dre", "median_dre",
                                "runs_per_overturn", "success"]].copy()
    l.columns = ["Policy", "Mean stake (runs)", "Median stake (runs)",
                 "Runs per overturn", "Success rate"]
    st.dataframe(l.style.format({
        "Mean stake (runs)": "{:.3f}", "Median stake (runs)": "{:.3f}",
        "Runs per overturn": "{:.3f}", "Success rate": "{:.1%}"}),
        hide_index=True, width='stretch')
    st.caption("The optimal policy wins a *smaller* share of its challenges but each "
               "one is worth more. It is not 'challenge more' — it is 'challenge different'.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("When is a challenge worth it?")
    st.markdown(
        "Challenge when your confidence exceeds **p\\* = C / (ΔRE + C)**, where C is the "
        "run value of one incorrect-challenge token. A correct challenge is free, so the "
        "cost carries a factor of (1 − p), not 1. That asymmetry pushes the threshold "
        "far below the coin flip most players seem to use."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        inning = st.slider("Inning", 1, 9, 1)
        half = st.radio("Half", ["Top", "Bot"], horizontal=True)
        remaining = st.radio("Challenges remaining", [2, 1], horizontal=True)
    sub = surf[(surf.inning == inning) & (surf.half == half) &
               (surf.challenges_remaining == remaining)]
    with col2:
        st.altair_chart(
            alt.Chart(sub).mark_line(size=3).encode(
                x=alt.X("dre:Q", title="Runs at stake in the call (ΔRE)"),
                y=alt.Y("p_star:Q", title="Minimum confidence to challenge",
                        scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                tooltip=["dre", "p_star"],
            ).properties(height=380), width='stretch')
    st.caption(
        "A full-count flip with the bases loaded is worth roughly 0.6–1.0 runs; a 0-0 "
        "take with nobody on is worth under 0.05. The threshold moves enormously across "
        "that range, which is why a single fixed habit cannot be right."
    )

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Runs left on the table")
    st.caption("Optimal policy minus actual, using the perceptual noise players "
               "actually have. Positive means the team could have gained more.")
    view = st.radio("View", ["Team", "Batter"], horizontal=True)

    if view == "Team":
        t = per_team[["team", "games", "actual_challenges", "actual_success",
                      "actual_runs", "optimal_challenges", "optimal_runs",
                      "runs_left_on_table", "full_season_pace"]].copy()
        t.columns = ["Team", "Games", "Challenges", "Success", "Runs gained",
                     "Optimal challenges", "Optimal runs", "Runs left",
                     "162-game pace"]
        st.dataframe(t.style.format({
            "Games": "{:.0f}", "Challenges": "{:.0f}", "Optimal challenges": "{:.0f}",
            "Success": "{:.1%}", "Runs gained": "{:.2f}", "Optimal runs": "{:.2f}",
            "Runs left": "{:.2f}", "162-game pace": "{:.1f}"}),
            hide_index=True, width='stretch', height=560)
        st.altair_chart(
            alt.Chart(per_team.head(15)).mark_bar().encode(
                x=alt.X("runs_left_on_table:Q", title="Runs left on the table"),
                y=alt.Y("team:N", sort="-x", title=None),
                tooltip=["team", "runs_left_on_table"],
            ).properties(height=380), width='stretch')
    else:
        min_ch = st.slider("Minimum optimal challenges", 5, 40, 10)
        b = per_batter[per_batter.optimal_challenges >= min_ch]
        b = b[["player", "actual_challenges", "actual_success", "actual_runs",
               "optimal_challenges", "optimal_runs", "runs_left_on_table"]].copy()
        b.columns = ["Player", "Challenges", "Success", "Runs gained",
                     "Optimal challenges", "Optimal runs", "Runs left"]
        st.dataframe(b.style.format({
            "Challenges": "{:.0f}", "Optimal challenges": "{:.0f}",
            "Success": "{:.1%}", "Runs gained": "{:.2f}", "Optimal runs": "{:.2f}",
            "Runs left": "{:.2f}"}), hide_index=True, width='stretch', height=560)
        st.caption("Batting-role challenges only — a batter can only challenge called "
                   "strikes against himself.")

st.divider()
st.caption(
    "Method: run expectancy built from 2024–2026 Statcast (not `delta_run_exp`, which is "
    "count-based and strips base-out context). ABS zone geometry validated against MLB's "
    "own `edge_distance` at R² = 1.0000. Policy solved by backward induction over "
    "(half-inning × pitch × challenges spent). Caveat: the fitted σ absorbs any real "
    "variation in players' thresholds across counts, which inflates the information gap "
    "and deflates the decision gap — so the decision gap is a floor."
)
