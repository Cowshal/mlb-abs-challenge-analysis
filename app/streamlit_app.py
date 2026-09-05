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

# ---------------------------------------------------------------- intro panel
st.info(
    "**What this is.** In 2026, MLB let players challenge close ball-and-strike "
    "calls, using the same tracking system that draws the strike zone on TV. "
    "Each team starts a game with two challenges — and here's the part that "
    "matters: a challenge only costs you if you're **wrong**. Get it right and "
    "you keep it to use again. That means a player doesn't need to be *sure* "
    "before challenging, just more often right than the cost of being wrong. "
    "This page compares how players actually use their challenges to how they "
    "should, given only what a player can actually see in the moment — and "
    "finds teams are leaving **about ten runs a season** on the table, not by "
    "challenging too rarely, but by challenging the wrong pitches."
)

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
    st.markdown(
        "##### In short\n"
        "Teams challenge about **twice a game** and win about half the time. "
        "The model says the smarter move is to challenge **about three times "
        "a game** — and win a *smaller* share of them — because the calls worth "
        "challenging aren't always the ones you're most sure about. Winning "
        "fewer challenges is fine if the ones you win are worth more: the same "
        "way a hitter who trades some singles for more walks and extra-base "
        "hits can end up more valuable with a lower average."
    )
    st.divider()

    st.subheader("Where the gap actually comes from")
    st.markdown(
        "Teams challenge about **2.1 times per game and succeed 54%** of the time. "
        "An optimal policy given the *same* information challenges more often and "
        "succeeds *less* often — trading hit rate for leverage (**leverage** = how "
        "much runs are riding on this particular call) — and still nets more runs."
    )
    show = dec[["label", "challenges_per_team_game", "success_rate",
                "runs_per_team_game"]].copy()
    show.columns = ["Policy", "Challenges / team-game", "Success rate", "Runs / team-game"]
    st.dataframe(show.style.format({
        "Challenges / team-game": "{:.2f}", "Success rate": "{:.1%}",
        "Runs / team-game": "{:.3f}"}), hide_index=True, width='stretch')
    st.markdown(
        "*What this means: teams challenge about twice a game. The model says they "
        "should challenge about three times — and pick different pitches when they do.*"
    )

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
    st.markdown(
        "*What this means: this line shows how big the \"nothing a team can do about "
        "it\" part of the gap might be, depending on how good MLB's cameras actually "
        "are — a number nobody outside MLB can measure, so it's shown as a range "
        "instead of a single guess.*"
    )

    st.subheader("Fitted perceptual noise, by role")
    st.markdown(
        "**σ (sigma)** here means how precisely a player can judge where a pitch "
        "crossed the plate — a smaller number means a more precise read. It's "
        "estimated from *where* players chose to challenge — never from how often "
        "they challenged or how often they were right, so the fit cannot absorb the "
        "decision gap it is meant to measure."
    )
    s = sigma[["side", "sigma_in", "n_challenged"]].copy()
    s.columns = ["Role", "Fitted σ (inches)", "Challenges"]
    st.dataframe(s.style.format({"Fitted σ (inches)": "{:.2f}"}),
                 hide_index=True, width='stretch')
    st.markdown(
        "*What this means: catchers and pitchers read the pitch more precisely than "
        "batters do — probably because they're looking at it from directly behind the "
        "plate instead of from the side.*"
    )
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
    st.markdown(
        "*What this means: the optimal player wins fewer of the calls he challenges, "
        "but each win is worth more runs — the challenge equivalent of trading batting "
        "average for slugging.*"
    )
    st.caption("The optimal policy wins a *smaller* share of its challenges but each "
               "one is worth more. It is not 'challenge more' — it is 'challenge different'.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.markdown(
        "##### In short\n"
        "**The more a call is worth, the less sure you need to be before "
        "challenging it.** A full-count pitch with runners on can be worth over "
        "half a run if you get it right — challenge that one even if you're only "
        "**15% sure**. A first-pitch take with the bases empty is worth almost "
        "nothing — don't challenge unless you're almost certain, **about 70%**."
    )
    st.divider()

    st.subheader("When is a challenge worth it?")
    st.markdown(
        "Challenge when your confidence exceeds **p\\* = C / (ΔRE + C)**, where "
        "**ΔRE** ('runs at stake') is the change in **run expectancy** — the "
        "average number of runs a team can expect to score from this point in "
        "the inning onward — caused by the call going one way instead of the "
        "other, and **C** is the run value of one incorrect-challenge token. A "
        "correct challenge is free, so the cost carries a factor of (1 − p), not "
        "1. That asymmetry pushes the threshold far below the coin flip most "
        "players seem to use."
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
    st.markdown(
        "*What this means: the more a call is worth, the less sure you need to be "
        "before challenging it — the line falls fast because a high-stakes call is "
        "worth challenging on a hunch, while a low-stakes call is only worth it when "
        "you're nearly certain.*"
    )
    st.caption(
        "A full-count flip with the bases loaded is worth roughly 0.6–1.0 runs; a 0-0 "
        "take with nobody on is worth under 0.05. The threshold moves enormously across "
        "that range, which is why a single fixed habit cannot be right."
    )

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown(
        "##### In short\n"
        "These are the teams and players who would gain the most by challenging "
        "*smarter* — not necessarily by challenging *more*. A team or player who "
        "shows up here with very few real-world challenges isn't being penalized "
        "for challenging badly; the model is simply saying they're sitting on a "
        "right they almost never use."
    )
    st.divider()

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
        st.markdown(
            "*What this means: a team near the top of this list isn't necessarily "
            "challenging poorly today — it's the team with the most runs available "
            "if it started picking better spots to challenge.*"
        )
        st.altair_chart(
            alt.Chart(per_team.head(15)).mark_bar().encode(
                x=alt.X("runs_left_on_table:Q", title="Runs left on the table"),
                y=alt.Y("team:N", sort="-x", title=None),
                tooltip=["team", "runs_left_on_table"],
            ).properties(height=380), width='stretch')
        st.markdown(
            "*What this means: the longer the bar, the more runs that team is "
            "leaving on the table by not challenging optimally.*"
        )
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
        st.markdown(
            "*What this means: some players near the top rarely challenge in real "
            "life — the model isn't grading their judgment, it's pointing out a "
            "right they're leaving unused.*"
        )
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
