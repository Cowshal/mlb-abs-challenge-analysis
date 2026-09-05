"""
ABS Challenge Optimizer -- 2026 MLB season.

Loads precomputed parquet from app/data/ and does nothing but filter and plot.
All modelling happens upstream in src/abs_policy.py.
"""
from pathlib import Path

import altair as alt
import numpy as np
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
split_half = load("split_half")
team_sigma = load("team_sigma")
team_sig_test = load("team_significance")
player_skill_test = load("player_skill_test")
catcher_check = load("catcher_check")
catcher_population = load("catcher_population")
catcher_summary = load("catcher_summary")
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
    st.markdown(
        "**What \"confidence\" means here.** Confidence is the probability the call "
        "was actually wrong — a number, not a feeling. The model knows exactly where "
        "the pitch crossed the plate and exactly where the zone's edges were; the "
        "player doesn't. He saw it once, from the side or from behind the plate, in "
        "well under half a second. We didn't assume how precisely players judge "
        "that — we measured it from *where they actually chose to challenge*: about "
        "**2.75 inches** of error for batters and **1.99 inches** for catchers and "
        "pitchers (how far off a player's read of the pitch's location tends to be). "
        "So \"challenge at 15% confidence\" means: even if a pitch like this one is "
        "actually the wrong call only 15% of the time, challenging it is still worth "
        "it — because a correct challenge costs nothing, and this particular call is "
        "worth about 0.6 runs if you're right."
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

    # ------------------------------------------------ is it a repeatable skill?
    st.divider()
    st.subheader("Is this a repeatable team skill?")
    st.markdown(
        "##### In short\n"
        "**The clearest team-level result replicates the league-wide finding "
        "above: 28 of 30 teams individually read fielding challenges more "
        "precisely than batting challenges.** Whether the overall spread in "
        "team success is a stable *skill* is murkier — it's more than chance "
        "would produce, but fails a team-level reliability test. That's "
        "because it's personnel, not a team trait: rosters turn over "
        "mid-season, and the same reliability test reads meaningfully higher "
        "once run on individual players instead of teams. Read the team table "
        "above as a snapshot of who was on the roster in 2026, not a "
        "permanent ranking of front offices."
    )

    st.markdown("**Perceptual precision, by team and role**")
    st.markdown(
        "Fitted the same way as the league-wide estimate — never from success "
        "rate. Every one of 28 of 30 teams reads fielding challenges (catcher/"
        "pitcher) more precisely than batting challenges, matching the "
        "league-wide pattern exactly. This one doesn't depend on the "
        "reliability question below — it's a direct, team-by-team replication "
        "of the role effect."
    )
    ts = team_sigma.pivot(index="team", columns="role", values="sigma_in").reset_index()
    st.altair_chart(
        alt.Chart(ts).mark_circle(size=70, color=COLOR_OPTIMAL, opacity=0.75).encode(
            x=alt.X("batting:Q", title="Batting σ (inches)"),
            y=alt.Y("fielding:Q", title="Fielding σ (inches)"),
            tooltip=["team", alt.Tooltip("batting:Q", format=".2f"),
                     alt.Tooltip("fielding:Q", format=".2f")],
        ).properties(height=300) +
        alt.Chart(pd.DataFrame({"x": [1, 4], "y": [1, 4]})).mark_line(
            strokeDash=[4, 4], color="#94A3B8").encode(x="x", y="y"),
        width='stretch')
    st.caption(
        "Points below the dashed line read fielding more precisely than batting "
        "(28 of 30 teams). Whether teams with sharper fielding reads also win "
        "more of their challenges is a weak, not-quite-significant relationship "
        "(r = −0.35, p = 0.06) — suggestive, not confirmation. Full per-team "
        "figures, bootstrap confidence intervals, and the underlying tests are "
        "in scripts/team_skill_test.py and data/team_skill_test.parquet."
    )

    st.divider()
    st.markdown("##### Is the cross-team spread in success itself a repeatable skill?")

    sh = split_half.merge(team_sig_test[["team", "z", "p_bonferroni"]], on="team")
    r_val = np.corrcoef(sh.h1_rate, sh.h2_rate)[0, 1]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Split-half reliability — the core test**")
        base = alt.Chart(sh)
        line = alt.Chart(pd.DataFrame({"x": [0.3, 0.75], "y": [0.3, 0.75]})).mark_line(
            strokeDash=[4, 4], color="#94A3B8").encode(x="x", y="y")
        pts = base.mark_circle(size=90, color=COLOR_OPTIMAL, opacity=0.75).encode(
            x=alt.X("h1_rate:Q", title="Success rate, first half of season",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.3, 0.75])),
            y=alt.Y("h2_rate:Q", title="Success rate, second half of season",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.3, 0.75])),
            tooltip=["team", alt.Tooltip("h1_rate:Q", format=".1%"),
                     alt.Tooltip("h2_rate:Q", format=".1%")],
        )
        st.altair_chart((line + pts).properties(height=320), width='stretch')
        st.markdown(
            f"*What this means: if a team's first-half success predicted its "
            f"second-half success, the dots would hug the dashed diagonal. They "
            f"don't, consistently enough to be sure — correlation r = {r_val:.2f}, "
            f"95% CI crosses zero.*"
        )
    with col2:
        st.markdown("**But the spread itself is real**")
        st.markdown(
            "Simulating 30 league-average teams at each team's real attempt "
            "count, the *spread* in success rate we actually see is bigger than "
            "chance alone produces (p = 0.003), and the same is true for runs "
            "gained (p < 0.0001). One team, Cincinnati, is 3.4 standard "
            "deviations above the league rate — still borderline-significant "
            "(p ≈ 0.02) even after accounting for having checked all 30 teams."
        )
        st.markdown(
            "*So: real variation exists in 2026. Whether it's a stable trait "
            "that predicts next season, or a one-year cluster of borderline "
            "calls going one team's way, isn't answered by this season alone.*"
        )

    st.markdown("**Does the same test read differently at the player level?**")
    pst = player_skill_test.rename(columns={
        "min_challenges_per_half": "Min. challenges / half", "n_players": "Players",
        "r": "r", "p": "p", "ci_lo": "95% CI low", "ci_hi": "95% CI high"})
    st.dataframe(
        pst[["Min. challenges / half", "Players", "r", "p", "95% CI low", "95% CI high"]],
        hide_index=True, width='stretch',
        column_config={
            "r": st.column_config.NumberColumn(format="%.3f"),
            "p": st.column_config.NumberColumn(format="%.4f"),
            "95% CI low": st.column_config.NumberColumn(format="%.3f"),
            "95% CI high": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.markdown(
        "Same split-half design, run on individual challengers instead of "
        "teams, at a few different minimum-sample thresholds. At 8 or 10 "
        "challenges per half, reliability (r ≈ 0.28–0.37) is clearly above the "
        "team-level 0.24 and its 95% CI clears zero — unlike the team-level "
        "test. The extremes are noisier (5: too lax a bar, mostly noise; 15: "
        "only 51 players left, underpowered) rather than a genuine reversal. "
        "*This doesn't prove an organizational skill — one season still can't "
        "separate a genuinely better process from happening to roster the "
        "right people — but it does explain why the team-level test came back "
        "ambiguous despite the spread being real.*"
    )

    st.markdown("**Do the top teams' own catchers show up individually?**")
    n_pop = int(catcher_summary.population_n.iloc[0])
    min_n_pop = int(catcher_summary.min_challenges.iloc[0])
    sel_r = catcher_summary.selection_r.iloc[0]
    sel_p = catcher_summary.selection_p.iloc[0]
    qual_r = catcher_summary.quality_corr_r.iloc[0]
    qual_p = catcher_summary.quality_corr_p.iloc[0]
    qual_n = int(catcher_summary.quality_corr_n.iloc[0])
    st.markdown(
        f"**Population and ranking, stated plainly:** every catcher who "
        f"logged at least {min_n_pop} challenges of his own in 2026 — "
        f"{n_pop} of them leaguewide — ranked by *raw* individual success "
        f"rate, nothing else. That raw ranking has an obvious failure mode: "
        f"a catcher who only challenges pitches that missed by a mile would "
        f"rank high without reading close calls any better than average. We "
        f"checked for it directly — correlating each catcher's mean "
        f"challenge distance from the zone boundary against his success "
        f"rate — and found a weak, **not statistically significant** "
        f"relationship (r = {sel_r:.2f}, p = {sel_p:.2f}). So there's no "
        f"strong evidence the ranking is just 'who picks easier misses,' "
        f"but with p this close to conventional significance it's a real "
        f"caveat, not a cleared one."
    )

    pop_domain = [max(0.0, catcher_population.rate.min() - 0.03),
                  min(1.0, catcher_population.rate.max() + 0.03)]
    hist = alt.Chart(catcher_population).mark_bar(color="#CBD5E1").encode(
        x=alt.X("rate:Q", bin=alt.Bin(maxbins=22),
                title="Individual challenge success rate",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=pop_domain)),
        y=alt.Y("count():Q", title=f"Catchers (of {n_pop})"),
    )
    top5_catchers = catcher_check[catcher_check.top_team].dropna(subset=["rate"])
    marks = alt.Chart(top5_catchers).mark_rule(size=3, opacity=0.9).encode(
        x=alt.X("rate:Q", scale=alt.Scale(domain=pop_domain)),
        color=alt.Color("name:N", title="Primary catcher, top-5 team",
                        scale=alt.Scale(scheme="tableau10")),
        tooltip=["name", "team", alt.Tooltip("rate:Q", format=".1%"),
                 alt.Tooltip("n:Q", title="Challenges")],
    )
    st.altair_chart((hist + marks).properties(height=280), width='stretch')
    st.caption(
        "Where the top 5 teams' primary catchers fall in the full leaguewide "
        "distribution of individual catcher accuracy. Hover a line for the "
        "name, team, exact rate, and sample size."
    )

    top_catchers = top5_catchers[
        ["team", "name", "n", "rate", "ci_lo", "ci_hi", "pct_rank", "quality_ratio"]
    ].copy()
    for c in ("rate", "ci_lo", "ci_hi", "pct_rank"):
        top_catchers[c] *= 100
    top_catchers = top_catchers.rename(columns={
        "team": "Team", "name": "Primary catcher", "n": "Challenges",
        "rate": "Own success rate", "ci_lo": "95% CI low", "ci_hi": "95% CI high",
        "pct_rank": "League percentile (catchers)", "quality_ratio": "Team quality ratio",
    })
    st.dataframe(
        top_catchers, hide_index=True, width='stretch',
        column_config={
            "Own success rate": st.column_config.NumberColumn(format="%.1f%%"),
            "95% CI low": st.column_config.NumberColumn(format="%.1f%%"),
            "95% CI high": st.column_config.NumberColumn(format="%.1f%%"),
            "League percentile (catchers)": st.column_config.NumberColumn(format="%.0f%%"),
            "Team quality ratio": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    def _pct_equiv(rate):
        return (catcher_population.rate < rate).mean() * 100

    _stephenson = top5_catchers[top5_catchers.name == "Tyler Stephenson"].iloc[0]
    _goodman = top5_catchers[top5_catchers.name == "Hunter Goodman"].iloc[0]
    _caratini = top5_catchers[top5_catchers.name == "Victor Caratini"].iloc[0]
    _langeliers = top5_catchers[top5_catchers.name == "Shea Langeliers"].iloc[0]
    st.markdown(
        f"**How much to trust one catcher's rank:** a primary catcher gets "
        f"roughly 40–160 challenges of his own in a season — enough to see a "
        f"real signal, not enough to pin down a precise number. Converting "
        f"each catcher's 95% CI back into where it would land in the league "
        f"distribution: Shea Langeliers's CI alone spans the "
        f"{_pct_equiv(_langeliers.ci_lo):.0f}th to "
        f"{_pct_equiv(_langeliers.ci_hi):.0f}th percentile, and Victor "
        f"Caratini's the {_pct_equiv(_caratini.ci_lo):.0f}th to "
        f"{_pct_equiv(_caratini.ci_hi):.0f}th — both wide enough to include "
        f"a below-average catcher. Tyler Stephenson "
        f"({_pct_equiv(_stephenson.ci_lo):.0f}th–{_pct_equiv(_stephenson.ci_hi):.0f}th) "
        f"and Hunter Goodman "
        f"({_pct_equiv(_goodman.ci_lo):.0f}th–{_pct_equiv(_goodman.ci_hi):.0f}th) "
        f"sit on firmer ground, but even Goodman's low end isn't clearly "
        f"above average. This matches the player-level split-half "
        f"reliability above (r ≈ 0.28–0.37): individual accuracy is real, "
        f"repeatable signal, but a noisy one — this season's exact ranking "
        f"of any one catcher would likely move some by next season."
    )

    st.markdown("**Does a team's quality edge actually line up with its own catcher's accuracy?**")
    cc_all = catcher_check.dropna(subset=["rate", "quality_ratio"])
    base_scatter = alt.Chart(cc_all).mark_circle(size=80, opacity=0.55, color="#94A3B8").encode(
        x=alt.X("quality_ratio:Q", title="Team quality ratio (success × leverage vs. league, "
                "attempts held fixed)"),
        y=alt.Y("rate:Q", title="Primary catcher's own success rate", axis=alt.Axis(format="%")),
        tooltip=["team", "name", alt.Tooltip("quality_ratio:Q", format=".2f"),
                 alt.Tooltip("rate:Q", format=".1%")],
    )
    trend = base_scatter.transform_regression("quality_ratio", "rate").mark_line(
        color="#334155", strokeDash=[5, 3])
    top5_pts = alt.Chart(cc_all[cc_all.top_team]).mark_circle(
        size=160, color=COLOR_OPTIMAL).encode(
        x="quality_ratio:Q", y="rate:Q",
        tooltip=["team", "name", alt.Tooltip("quality_ratio:Q", format=".2f"),
                 alt.Tooltip("rate:Q", format=".1%")],
    )
    top5_labels = alt.Chart(cc_all[cc_all.top_team]).mark_text(
        dx=10, dy=-6, align="left", color=COLOR_OPTIMAL, fontWeight="bold").encode(
        x="quality_ratio:Q", y="rate:Q", text="team")
    st.altair_chart(
        (base_scatter + trend + top5_pts + top5_labels).properties(height=340),
        width='stretch')
    st.markdown(
        f"Every dot is one of the 30 primary catchers, not just the top 5: x is "
        f"how much of his team's edge is *quality* (success and leverage, with "
        f"attempts held fixed) rather than volume; y is that same catcher's own "
        f"individual challenge success rate. They're correlated leaguewide "
        f"(r = {qual_r:.2f}, p = {qual_p:.3f}, n = {qual_n} teams) — not just a "
        f"pattern eyeballed in 5 points. **Cincinnati (CIN) and Colorado (COL)** "
        f"got to a similar-looking runs-gained rank as **Minnesota (MIN) and "
        f"Chicago (CWS)**, but by different routes: Tyler Stephenson and Hunter "
        f"Goodman are individually sharp challengers driving a real quality "
        f"edge, while Victor Caratini and Drew Romo are average-to-below and "
        f"their teams' rank instead reflects challenging *more often* — which "
        f"matters, because a volume edge is a lineup-construction choice a new "
        f"front office could keep next season, while a quality edge tied to one "
        f"catcher walks out the door with him in a trade."
    )

    st.markdown(
        "##### What this section means\n"
        "Challenge accuracy looks like a skill that belongs to **players**, "
        "not front offices — and the players who most clearly have it are "
        "**catchers**. That reframes the question a team should be asking. "
        "It's not just *when* to challenge (the optimal-policy question this "
        "whole page is about) but *who* calls for one: a team with a sharp "
        "catcher back there has real signal worth trusting on close calls; a "
        "team with an average one is better off leaning on the same policy "
        "logic as everyone else."
    )

st.divider()
st.caption(
    "Method: run expectancy built from 2024–2026 Statcast (not `delta_run_exp`, which is "
    "count-based and strips base-out context). ABS zone geometry validated against MLB's "
    "own `edge_distance` at R² = 1.0000. Policy solved by backward induction over "
    "(half-inning × pitch × challenges spent). Caveat: the fitted σ absorbs any real "
    "variation in players' thresholds across counts, which inflates the information gap "
    "and deflates the decision gap — so the decision gap is a floor."
)

_policy_v = dec.model_version.iloc[0] if "model_version" in dec.columns else "unstamped"
_policy_t = dec.generated_at.iloc[0] if "generated_at" in dec.columns else "unknown"
_sigma_v = sigma.model_version.iloc[0] if "model_version" in sigma.columns else "unstamped"
_sigma_t = sigma.generated_at.iloc[0] if "generated_at" in sigma.columns else "unknown"
st.caption(
    f"Policy model: `{_policy_v}` (generated {_policy_t}) · "
    f"Perceptual-σ fit: `{_sigma_v}` (generated {_sigma_t})"
)
