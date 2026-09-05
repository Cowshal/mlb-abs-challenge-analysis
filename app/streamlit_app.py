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

# One learned color encoding, used everywhere: gray = what actually happened,
# blue = what the model recommends, amber = an assumption-dependent ceiling.
COLOR_OBSERVED = "#64748B"
COLOR_OPTIMAL = "#2563EB"
COLOR_CEILING = "#F59E0B"

POLICY_LABELS = {
    "observed 2026": "Observed (2026)",
    "optimal @ player sigma": "Optimal (same information)",
    "ceiling @ sigma=0.5in": "Ceiling (perfect information)",
}
POLICY_COLORS = {
    "Observed (2026)": COLOR_OBSERVED,
    "Optimal (same information)": COLOR_OPTIMAL,
    "Ceiling (perfect information)": COLOR_CEILING,
}
ROLE_LABELS = {"batting": "Batters", "fielding": "Catchers & pitchers"}


def policy_color_encoding(field="Policy", legend=True):
    return alt.Color(
        f"{field}:N",
        scale=alt.Scale(domain=list(POLICY_COLORS.keys()), range=list(POLICY_COLORS.values())),
        legend=alt.Legend(title=None) if legend else None,
    )


@st.cache_data
def load(name):
    return pd.read_parquet(DATA / f"{name}.parquet")


dec = load("policy_decomposition")
dec["Policy"] = dec.label.map(POLICY_LABELS)
sens = load("ceiling_sensitivity")
lev = load("leverage_comparison")
lev["Policy"] = lev.policy.map(POLICY_LABELS)
surf = load("threshold_surface")
per_team = load("per_team")
per_batter = load("per_batter")
sigma = load("perception_sigma")
sigma["Role"] = sigma.side.map(ROLE_LABELS)

st.title("Who's leaving runs on the table?")
st.caption("Optimal ABS challenge policy vs. observed behaviour, 2026 MLB season "
           "— 9,037 challenges across 2,136 games")

# ---------------------------------------------------------------- intro panel
obs, ply, ceil = (dec.runs_per_team_game.iloc[i] for i in (0, 1, 2))
decision_gap = (ply - obs) * 162

st.info(
    "**The rule that makes this interesting.** Teams start a game with two "
    "challenges — but a **correct** challenge is given back immediately. Only "
    "a **wrong** one costs you. That means a team that keeps winning never "
    "runs out, which changes the math completely: you don't need to be "
    "*sure* before challenging, just more often right than the cost of "
    "occasionally being wrong.\n\n"
    "**What this page shows.** In 2026, MLB let players challenge close "
    "ball-and-strike calls, using the same tracking system that draws the "
    "strike zone on TV. This page compares how players actually use their "
    "challenges to how they should, given only what a player can see in the "
    "moment — and finds teams are leaving **about ten runs a season** on the "
    "table, not by challenging too rarely, but by challenging the wrong pitches."
)

c1, c2, c3 = st.columns(3)
c1.metric("Observed (per team-season)", f"{obs*162:.0f} runs",
          help="Runs actually gained through successful challenges in 2026.")
c2.metric("Optimal (per team-season)", f"{ply*162:.0f} runs",
          help="Optimal policy played with the perceptual noise players actually have.")
c3.metric("Decision gap (per team-season)", f"+{decision_gap:.0f} runs",
          help="The actionable number: better decisions, identical information.")

tab1, tab2, tab3 = st.tabs(
    ["Decomposition", "Optimal threshold", "Runs left on the table"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.markdown(
        f"""
        <div style="font-size:2.6rem; font-weight:800; color:{COLOR_OPTIMAL}; line-height:1.15;">
            +{decision_gap:.0f} runs per team, every season
        </div>
        <div style="font-size:1.05rem; color:#475569; margin-top:0.15rem; margin-bottom:1rem;">
            The edge available from challenging smarter — picking higher-value
            moments, not simply challenging more often.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "##### In short\n"
        "Teams attempt about **twice a game** and win about half the time. "
        "Because a **correct** challenge is returned — only a wrong one costs "
        "you — a team that keeps winning never runs out. The model says the "
        "smarter number is **about three attempts a game**, winning a "
        "*smaller* share of them, because the calls worth challenging aren't "
        "always the ones you're most sure about. Winning fewer of them is "
        "fine if the ones you win are worth more: the same way a hitter who "
        "trades some singles for more walks and extra-base hits can end up "
        "more valuable with a lower average."
    )
    st.divider()

    st.subheader("Where the gap actually comes from")
    st.markdown(
        "Teams attempt about **2.1 challenges per game** — already more than "
        "the two-challenge allotment, since a **correct** challenge is "
        "returned and only a wrong one costs you — and win **54%** of the "
        "time. An optimal policy, given the *same* information, attempts "
        "more (**about three a game**) and wins a *smaller* share — trading "
        "hit rate for leverage (**leverage** = how many runs are riding on "
        "this particular call) — and still nets more runs."
    )

    chart_dec = alt.Chart(dec).mark_bar(size=32).encode(
        x=alt.X("runs_per_team_game:Q", title="Runs per team, per game"),
        y=alt.Y("Policy:N", sort=list(POLICY_COLORS.keys()), title=None,
                axis=alt.Axis(labelLimit=220, labelFontSize=13)),
        color=policy_color_encoding(legend=False),
        tooltip=["Policy", alt.Tooltip("runs_per_team_game:Q", title="Runs/game", format=".3f"),
                 alt.Tooltip("challenges_per_team_game:Q", title="Attempts/game", format=".2f"),
                 alt.Tooltip("success_rate:Q", title="Success rate", format=".1%")],
    ).properties(height=190)
    st.altair_chart(chart_dec, width='stretch')

    show = dec[["Policy", "challenges_per_team_game", "success_rate", "runs_per_team_game"]].copy()
    show["success_pct"] = show.success_rate * 100
    st.dataframe(
        show, hide_index=True, width='stretch',
        column_order=["Policy", "challenges_per_team_game", "success_pct", "runs_per_team_game"],
        column_config={
            "Policy": st.column_config.TextColumn("Policy"),
            "challenges_per_team_game": st.column_config.NumberColumn(
                "Attempts per game", format="%.2f"),
            "success_pct": st.column_config.NumberColumn("Success rate", format="%.1f%%"),
            "runs_per_team_game": st.column_config.NumberColumn("Runs per game", format="%.3f"),
        },
    )
    st.markdown(
        "*What this means: teams attempt about two challenges a game today. "
        "The model says the smarter number is about three attempts a game — "
        "made possible because winning one gives it right back — and it "
        "picks different pitches when it does.*"
    )

    st.markdown(
        f"**Decision gap: +{decision_gap:.0f} runs per team-season.** This is the part "
        "a team can capture by changing policy alone. It does not depend on any "
        "assumption about tracking precision."
    )
    st.warning(
        "The remaining gap to a perfect-information ceiling is **not** coachable, and "
        "its size depends entirely on an assumed tracking precision that this dataset "
        "cannot measure — we only ever observe Hawk-Eye's own output, never independent "
        "ground truth. Shown below as a sensitivity curve rather than a single number."
    )
    st.altair_chart(
        alt.Chart(sens).mark_line(point=True, color=COLOR_CEILING, size=3).encode(
            x=alt.X("ceiling_sigma_in:Q", title="Assumed ceiling tracking precision, σ (inches)"),
            y=alt.Y("info_gap_runs_per_team_season:Q",
                    title="Information gap (runs per team-season)"),
            tooltip=[alt.Tooltip("ceiling_sigma_in:Q", title="Assumed σ (in)"),
                     alt.Tooltip("info_gap_runs_per_team_season:Q", title="Runs/season", format=".1f")],
        ).properties(height=280), width='stretch')
    st.markdown(
        "*What this means: this line shows how big the \"nothing a team can do about "
        "it\" part of the gap might be, depending on how good MLB's cameras actually "
        "are — a number nobody outside MLB can measure, so it's shown as a range "
        "instead of a single guess.*"
    )

    st.subheader("Fitted perceptual noise, by role")
    st.markdown(
        "**Perceptual σ (sigma)** here means how precisely a player can judge "
        "where a pitch crossed the plate — a smaller number means a more "
        "precise read. It's estimated from *where* players chose to "
        "challenge — never from how often they challenged or how often they "
        "were right, so the fit cannot absorb the decision gap it is meant "
        "to measure."
    )
    st.dataframe(
        sigma, hide_index=True, width='stretch',
        column_order=["Role", "sigma_in", "n_challenged"],
        column_config={
            "Role": st.column_config.TextColumn("Role"),
            "sigma_in": st.column_config.NumberColumn("Read precision, σ (inches)", format="%.2f"),
            "n_challenged": st.column_config.NumberColumn("Challenges in sample", format="%d"),
        },
    )
    st.markdown(
        "*What this means: catchers and pitchers read the pitch more precisely "
        "than batters do — probably because they're looking at it from directly "
        "behind the plate instead of from the side.*"
    )
    st.caption("Catchers and pitchers read the pitch ~28% more precisely than batters, "
               "consistent with seeing it from behind rather than from the side.")

    st.subheader("Challenge more, or challenge differently?")
    l = lev[lev.role == "all"][["Policy", "mean_dre", "median_dre", "runs_per_overturn", "success"]].copy()
    l["success_pct"] = l.success * 100
    st.dataframe(
        l, hide_index=True, width='stretch',
        column_order=["Policy", "mean_dre", "median_dre", "runs_per_overturn", "success_pct"],
        column_config={
            "Policy": st.column_config.TextColumn("Policy"),
            "mean_dre": st.column_config.NumberColumn("Mean stake (runs)", format="%.3f"),
            "median_dre": st.column_config.NumberColumn("Median stake (runs)", format="%.3f"),
            "runs_per_overturn": st.column_config.NumberColumn("Runs per overturn", format="%.3f"),
            "success_pct": st.column_config.NumberColumn("Success rate", format="%.1f%%"),
        },
    )
    st.markdown(
        "*What this means: the optimal player wins fewer of the calls he "
        "challenges, but each win is worth more runs — the challenge "
        "equivalent of trading batting average for slugging.*"
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
        half_display = st.radio("Half of the inning", ["Top", "Bottom"], horizontal=True)
        half = "Bot" if half_display == "Bottom" else "Top"
        remaining = st.radio("Challenges remaining", [2, 1], horizontal=True)
    sub = surf[(surf.inning == inning) & (surf.half == half) &
               (surf.challenges_remaining == remaining)]
    with col2:
        st.altair_chart(
            alt.Chart(sub).mark_line(size=3, color=COLOR_OPTIMAL).encode(
                x=alt.X("dre:Q", title="Value of the call, in runs (ΔRE)"),
                y=alt.Y("p_star:Q", title="Minimum confidence to challenge",
                        scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("dre:Q", title="Runs at stake"),
                         alt.Tooltip("p_star:Q", title="Min. confidence", format=".0%")],
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
        "These are the teams and players who would gain the most by "
        "challenging *smarter* — not necessarily by challenging *more*. "
        "Because a correct challenge is returned, the counts below can run "
        "well above the two-per-game allotment; they're attempts across a "
        "season, not the resource itself. A team or player who shows up here "
        "with very few real-world attempts isn't being penalized for "
        "challenging badly — the model is saying they're sitting on a right "
        "they almost never use."
    )
    st.divider()

    st.subheader("Runs left on the table")
    st.caption("Optimal policy minus actual, using the perceptual noise players "
               "actually have. Positive means the team could have gained more.")
    view = st.radio("View", ["Team", "Batter"], horizontal=True)

    if view == "Team":
        t = per_team.copy()
        t["actual_success_pct"] = t.actual_success * 100
        st.dataframe(
            t, hide_index=True, width='stretch', height=560,
            column_order=["team", "games", "actual_challenges", "actual_success_pct",
                          "actual_runs", "optimal_challenges", "optimal_runs",
                          "runs_left_on_table", "full_season_pace"],
            column_config={
                "team": st.column_config.TextColumn("Team"),
                "games": st.column_config.NumberColumn("Games", format="%d"),
                "actual_challenges": st.column_config.NumberColumn("Attempts (season)", format="%d"),
                "actual_success_pct": st.column_config.NumberColumn("Success rate", format="%.1f%%"),
                "actual_runs": st.column_config.NumberColumn("Runs gained", format="%.2f"),
                "optimal_challenges": st.column_config.NumberColumn("Optimal attempts (season)", format="%d"),
                "optimal_runs": st.column_config.NumberColumn("Optimal runs", format="%.2f"),
                "runs_left_on_table": st.column_config.NumberColumn("Runs left", format="%.2f"),
                "full_season_pace": st.column_config.NumberColumn("Pace per 162 games", format="%.1f"),
            },
        )
        st.markdown(
            "*What this means: a team near the top of this list isn't necessarily "
            "challenging poorly today — it's the team with the most runs available "
            "if it started picking better spots to challenge.*"
        )
        top15 = per_team.head(15)
        st.altair_chart(
            alt.Chart(top15).mark_bar(color=COLOR_OPTIMAL).encode(
                x=alt.X("runs_left_on_table:Q", title="Runs left on the table (season)"),
                y=alt.Y("team:N", sort="-x", title=None),
                tooltip=["team", alt.Tooltip("runs_left_on_table:Q", format=".2f")],
            ).properties(height=380), width='stretch')
        st.markdown(
            "*What this means: the longer the bar, the more runs that team is "
            "leaving on the table by not challenging optimally.*"
        )
    else:
        min_ch = st.slider("Minimum optimal attempts to include", 5, 40, 10)
        b = per_batter[per_batter.optimal_challenges >= min_ch].copy()
        # NumberColumn renders a missing rate (0 real attempts -> no rate to
        # report) as the literal text "None" rather than a blank cell -- use a
        # formatted text fallback instead.
        b["success_display"] = b.actual_success.apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        st.dataframe(
            b, hide_index=True, width='stretch', height=560,
            column_order=["player", "actual_challenges", "success_display", "actual_runs",
                          "optimal_challenges", "optimal_runs", "runs_left_on_table"],
            column_config={
                "player": st.column_config.TextColumn("Player"),
                "actual_challenges": st.column_config.NumberColumn("Attempts (season)", format="%d"),
                "success_display": st.column_config.TextColumn("Success rate"),
                "actual_runs": st.column_config.NumberColumn("Runs gained", format="%.2f"),
                "optimal_challenges": st.column_config.NumberColumn("Optimal attempts (season)", format="%d"),
                "optimal_runs": st.column_config.NumberColumn("Optimal runs", format="%.2f"),
                "runs_left_on_table": st.column_config.NumberColumn("Runs left", format="%.2f"),
            },
        )
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
