"""
ABS Challenge Optimizer -- 2026 MLB season.

Loads precomputed parquet from app/data/ and does nothing but filter and plot.
All modelling happens upstream in src/abs_policy.py.
"""
import json
import logging
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger("abs_app")


@contextmanager
def guarded_section(name):
    """Render one independent page section (a tab). If its body raises, log
    the FULL traceback server-side -- it stays visible in `streamlit run`
    terminal output and in Streamlit Community Cloud logs -- and show a
    scoped message in place of just this section, so the other tabs and the
    footer still render. Streamlit executes every st.tabs() body on each
    rerun, so without this an exception in one tab blanks the whole page.

    This is NOT a silent catch: nothing is swallowed quietly, the traceback
    is always logged, and it is only applied at the top level of each major
    section -- not around small blocks.
    """
    try:
        yield
    except Exception:
        logger.exception("Section %r failed to render", name)
        traceback.print_exc()
        st.error(
            f"**The “{name}” section didn't load.** The rest of the "
            "page is unaffected. If this keeps happening, the app may be "
            "temporarily unavailable."
        )
        with st.expander("Technical detail"):
            st.code(traceback.format_exc())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geometry import center_distance_to_zone, ball_edge_distance, HALF_WIDTH, BALL_RADIUS_FT
from run_expectancy import flip_value
from challenge_rules import (break_even_probability, expected_challenge_value,
                             recommend)

DATA = Path(__file__).parent / "data"

st.set_page_config(
    page_title="ABS Challenge Optimizer",
    layout="wide",
    menu_items={
        "About": (
            "**MLB ABS Challenge Policy Analysis**\n\n"
            "By Kaushal Namuduri\n\n"
            "https://github.com/Cowshal/mlb-abs-challenge-analysis"
        )
    },
)

# One learned color encoding, used everywhere: gray = what actually happened,
# blue = what the model recommends, amber = an assumption-dependent benchmark.
COLOR_OBSERVED = "#64748B"
COLOR_OPTIMAL = "#2563EB"
COLOR_CEILING = "#F59E0B"

# "High-precision benchmark" is deliberately NOT "perfect information": the
# scenario still carries 0.5 in of perceptual/tracking noise (Hawk-Eye's own
# resolution floor), it just removes the human read error. A true
# perfect-information run would use the exact pitch location with zero noise.
BENCHMARK_LABEL = "High-precision benchmark (σ = 0.5 in)"
POLICY_LABELS = {
    "observed 2026": "Observed (2026)",
    "optimal @ player sigma": "Optimal (same information)",
    "ceiling @ sigma=0.5in": BENCHMARK_LABEL,
}
POLICY_COLORS = {
    "Observed (2026)": COLOR_OBSERVED,
    "Optimal (same information)": COLOR_OPTIMAL,
    BENCHMARK_LABEL: COLOR_CEILING,
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


@st.cache_data
def load_reported_figures():
    """The pipeline's data-provenance snapshot (src/reported_figures.py ->
    scripts/build_reported_figures.py). Optional: an older app/data/ set may
    predate it, so every read is defensive."""
    path = DATA / "reported_figures.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


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
zone_heatmap = load("zone_heatmap")
zone_interaction = load("zone_interaction")
zone_sigma_sensitivity = load("zone_sigma_sensitivity")
zone_sigma_bootstrap = load("zone_sigma_sensitivity_bootstrap")
option_values = load("option_values")
re_2026 = load("re_2026")
posterior_lookup = load("posterior_lookup")
decision_scatter = load("decision_scatter")
decision_boundary = load("decision_boundary")
cases = load("case_studies")
miss_by_count = load("endorsed_miss_by_count")
miss_summary = load("endorsed_miss_summary").iloc[0]
miss_hist = load("endorsed_miss_evnet_hist")
coinflip_team = load("coinflip_by_team")
coinflip_role = load("coinflip_by_role")
coinflip_summary = load("coinflip_summary").iloc[0]
sigma["Role"] = sigma.side.map(ROLE_LABELS)


@st.cache_data
def re_lookup_dict():
    return {(int(r.balls), int(r.strikes), int(r.outs), bool(r.r1), bool(r.r2), bool(r.r3)):
            r.run_exp for r in re_2026.itertuples()}


@st.cache_data
def option_value_lookup():
    return option_values.set_index("t")[["C_k0", "C_k1"]].to_dict("index")


@st.cache_data
def posterior_grids():
    """(o_grid, p_grid) per role, sorted by o -- np.interp requires ascending x."""
    out = {}
    for role in ("batting", "fielding"):
        sub = posterior_lookup[posterior_lookup.role == role].sort_values("o_ft")
        out[role] = (sub.o_ft.values, sub.p_win.values)
    return out


# Median height across every 2026 challenge opportunity's height_ft (measured
# where available, else listed) -- see data/challenge_opportunities.parquet.
# The decision tool takes a game situation, not a specific batter, so it uses
# a league-typical zone; an actual batter's zone top/bottom shifts by a few
# inches around this.
REPRESENTATIVE_HEIGHT_FT = 6.00

# Dimensionally-accurate zone plot geometry, all in inches. The rule zone
# itself is close to square (17 in wide, ~19 in tall for a 6'0" batter) --
# genuinely not "noticeably taller than wide" on its own; what makes the
# panel read as a strike zone and not an abstract square is the ball-radius
# buffer (a headline finding: ABS's real boundary is the rulebook rectangle
# inflated by one ball radius, not the rectangle itself) plus a true-to-scale
# home plate diagram anchored underneath it for orientation. The aspect
# ratio below falls out of those real dimensions -- it is not chosen to hit
# a target ratio.
ZONE_WIDTH_IN = HALF_WIDTH * 2 * 12                              # 17.0 in, rulebook width
ZONE_TOP_IN = 0.535 * REPRESENTATIVE_HEIGHT_FT * 12               # ABS top, league-average batter
ZONE_BOT_IN = 0.270 * REPRESENTATIVE_HEIGHT_FT * 12               # ABS bottom
BALL_RADIUS_IN = BALL_RADIUS_FT * 12                              # ~1.45 in
CLICK_MARGIN_IN = 4.0     # room to click a clearly-out pitch, same on every side
PLATE_GAP_IN = 1.5        # visual gap between the ball-radius boundary and the plate
PLATE_DEPTH_IN = 17.0     # true plate depth, back edge to point (rule 2.03)

PLOT_X_LO = -(ZONE_WIDTH_IN / 2 + BALL_RADIUS_IN + CLICK_MARGIN_IN)
PLOT_X_HI = +(ZONE_WIDTH_IN / 2 + BALL_RADIUS_IN + CLICK_MARGIN_IN)
PLOT_Z_HI = ZONE_TOP_IN + BALL_RADIUS_IN + CLICK_MARGIN_IN
PLATE_TOP_Z = ZONE_BOT_IN - BALL_RADIUS_IN - PLATE_GAP_IN
PLATE_BOT_Z = PLATE_TOP_Z - PLATE_DEPTH_IN
PLOT_Z_LO = PLATE_BOT_Z - 1.0

PLOT_WIDTH_IN = PLOT_X_HI - PLOT_X_LO
PLOT_HEIGHT_IN = PLOT_Z_HI - PLOT_Z_LO
ZONE_PLOT_WIDTH_PX = 380
ZONE_PLOT_HEIGHT_PX = round(ZONE_PLOT_WIDTH_PX * PLOT_HEIGHT_IN / PLOT_WIDTH_IN)


def home_plate_outline(top_z):
    """True-scale home plate footprint (17 in back edge, 8.5 in perpendicular
    sides, meeting at a point 17 in deep), back edge at height `top_z`,
    purely as a to-scale orientation reference -- not a claim about where a
    pitch's z-coordinate physically is relative to the ground."""
    w = ZONE_WIDTH_IN / 2
    pts = [(-w, top_z), (w, top_z), (w, top_z - 8.5),
           (0, top_z - PLATE_DEPTH_IN), (-w, top_z - 8.5), (-w, top_z)]
    df = pd.DataFrame(pts, columns=["x", "z"])
    df["order"] = range(len(df))
    return df


def compute_dre(balls, strikes, outs, r1, r2, r3):
    return flip_value(re_lookup_dict(), balls, strikes, outs, (r1, r2, r3))


def half_inning_index(inning, half):
    return (inning - 1) * 2 + (2 if half == "Bot" else 1)


def option_value_at(t):
    cmap = option_value_lookup()
    last = cmap[max(cmap.keys())]
    row = cmap.get(t, last)
    return row["C_k0"], row["C_k1"]


@st.cache_data
def zone_click_grid(role, height_ft=REPRESENTATIVE_HEIGHT_FT, cell_in=1.3):
    """A grid of cells covering the full dimensionally-accurate plot area
    (zone + ball-radius buffer + click margin + the space reserved for the
    plate diagram), each carrying the model's P(call was wrong) for a click
    at that location -- shaded as a heatmap and also the set of clickable
    targets for the selection."""
    nx = max(1, round(PLOT_WIDTH_IN / cell_in))
    nz = max(1, round(PLOT_HEIGHT_IN / cell_in))
    x_edges = np.linspace(PLOT_X_LO, PLOT_X_HI, nx + 1)
    z_edges = np.linspace(PLOT_Z_LO, PLOT_Z_HI, nz + 1)
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    zc = (z_edges[:-1] + z_edges[1:]) / 2

    # Outer loop over x-bins, inner loop over z-bins -- every array below uses
    # this same repeat/tile pattern so row i is one consistent (x, z) cell.
    x0, x1 = np.repeat(x_edges[:-1], nz), np.repeat(x_edges[1:], nz)
    z0, z1 = np.tile(z_edges[:-1], nx), np.tile(z_edges[1:], nx)
    xc_full, zc_full = np.repeat(xc, nz), np.tile(zc, nx)

    d = ball_edge_distance(center_distance_to_zone(
        xc_full / 12, zc_full / 12, np.full(xc_full.size, height_ft)))
    o_grid, p_grid = posterior_grids()[role]
    p_wrong = np.interp(d, o_grid, p_grid)
    return pd.DataFrame({
        "x0": x0, "x1": x1, "z0": z0, "z1": z1,
        "xc": xc_full, "zc": zc_full, "p_wrong": p_wrong,
    })


def p_wrong_given_click(x_ft, z_ft, role, height_ft):
    """Model's own estimate of P(the original call was wrong), given where the
    player believes the pitch crossed the plate. This is exactly the quantity
    the DP conditions its challenge decision on -- not a separate calculation."""
    d = ball_edge_distance(center_distance_to_zone(
        np.array([x_ft]), np.array([z_ft]), np.array([height_ft])))[0]
    o_grid, p_grid = posterior_grids()[role]
    return float(np.interp(d, o_grid, p_grid))


def fmt_pct(p):
    """Percent for display. Shows '<1%' rather than a misleading '0%' when a
    break-even rounds to zero (it does in very high-leverage late-game spots)."""
    if p != p:            # nan
        return "n/a"
    if 0 < p < 0.005:
        return "<1%"
    return f"{p:.0%}"


# ---------------------------------------------------------------- case studies
def _ord(n):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(int(n), f"{int(n)}th")


def case_zone_chart(row):
    """The same zone diagram as the 'Should I challenge?' tool -- rulebook box,
    dashed ball-radius ABS boundary, to-scale home plate -- drawn for this
    batter's height, with a marker where the pitch actually crossed the middle
    of the plate."""
    zt, zb = float(row.zone_top_in), float(row.zone_bot_in)
    margin = 2.5
    x_lo = -(ZONE_WIDTH_IN / 2 + BALL_RADIUS_IN + margin)
    x_hi = -x_lo
    z_hi = zt + BALL_RADIUS_IN + margin
    plate_top_z = zb - BALL_RADIUS_IN - PLATE_GAP_IN
    z_lo = plate_top_z - PLATE_DEPTH_IN - 1.0
    xs = alt.Scale(domain=[x_lo, x_hi])
    zs = alt.Scale(domain=[z_lo, z_hi])
    rule_zone = alt.Chart(pd.DataFrame([{
        "x0": -ZONE_WIDTH_IN / 2, "x1": ZONE_WIDTH_IN / 2, "z0": zb, "z1": zt}])
    ).mark_rect(fill="#BFDBFE", fillOpacity=0.35, stroke="#0F172A", strokeWidth=2
    ).encode(x=alt.X("x0:Q", scale=xs,
                     title="Inches from the plate's center"), x2="x1:Q",
             y=alt.Y("z1:Q", scale=zs, title="Inches off the ground"), y2="z0:Q")
    abs_boundary = alt.Chart(pd.DataFrame([{
        "x0": -ZONE_WIDTH_IN / 2 - BALL_RADIUS_IN,
        "x1": ZONE_WIDTH_IN / 2 + BALL_RADIUS_IN,
        "z0": zb - BALL_RADIUS_IN, "z1": zt + BALL_RADIUS_IN}])
    ).mark_rect(fill=None, stroke="#0F172A", strokeWidth=1.4, strokeDash=[5, 3]
    ).encode(x=alt.X("x0:Q", scale=xs), x2="x1:Q",
             y=alt.Y("z1:Q", scale=zs), y2="z0:Q")
    plate = alt.Chart(home_plate_outline(plate_top_z)).mark_line(
        color="#0F172A", strokeWidth=1.5, fill="#E2E8F0", fillOpacity=0.9
    ).encode(x=alt.X("x:Q", scale=xs), y=alt.Y("z:Q", scale=zs), order="order:O")
    pitch = alt.Chart(pd.DataFrame([{
        "x": float(row.pitch_x_in), "z": float(row.pitch_z_in)}])
    ).mark_point(size=340, shape="diamond", filled=True, color=COLOR_CEILING,
                 stroke="#0F172A", strokeWidth=1.5).encode(
        x=alt.X("x:Q", scale=xs), y=alt.Y("z:Q", scale=zs))
    w = 270
    h = min(int(w * (z_hi - z_lo) / (x_hi - x_lo)), 380)
    return (rule_zone + abs_boundary + plate + pitch).properties(width=w, height=h)

try:
    st.title("Who's leaving runs on the table?")
    _rf_head = load_reported_figures().get("figures", {})
    _n_ch = _rf_head.get("n_opportunities", {}).get("value")
    _n_gm = _rf_head.get("n_games_2026", {}).get("value")
    st.caption(
        "Optimal ABS challenge policy vs. observed behaviour, 2026 MLB season"
        + (f" — {_n_ch:,.0f} challenges across {_n_gm:,.0f} games"
           if _n_ch and _n_gm else "")
    )
    st.caption(
        "By **Kaushal Namuduri** · "
        "[github.com/Cowshal/mlb-abs-challenge-analysis]"
        "(https://github.com/Cowshal/mlb-abs-challenge-analysis)"
    )

    # ---------------------------------------------------------------- intro panel
    obs, ply, ceil = (dec.runs_per_team_game.iloc[i] for i in (0, 1, 2))
    decision_gap = (ply - obs) * 162
    obs_rate = dec.success_rate.iloc[0]
    opt_rate = dec.success_rate.iloc[1]
    obs_cpg = dec.challenges_per_team_game.iloc[0]
    opt_cpg = dec.challenges_per_team_game.iloc[1]

    st.markdown(
        f"""
        <div style="font-size:2.5rem; font-weight:800; line-height:1.12; margin-top:0.2rem;">
            MLB teams are challenging the wrong pitches.
        </div>
        <div style="font-size:1.15rem; color:#475569; margin-top:0.5rem; margin-bottom:0.4rem;">
            Teams currently win about <b>{obs_rate:.0%}</b> of their ABS challenges.
            A value-maximizing policy wins only about <b>{opt_rate:.0%}</b> —
            and still leaves teams better off by roughly
            <b>{decision_gap:.0f} runs per team-season</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Observed challenges / game", f"{obs_cpg:.1f}",
              help="Attempts per team per game in 2026. Already above the "
                   "two-challenge allotment, because a correct challenge is "
                   "returned.")
    c2.metric("Optimal challenges / game", f"{opt_cpg:.1f}",
              delta=f"{opt_cpg - obs_cpg:+.1f} vs. observed",
              help="What a value-maximizing policy attempts, using the same "
                   "in-the-moment information players actually have.")
    c3.metric("Value available", f"+{decision_gap:.0f} runs / team-season",
              help="Runs a team can capture by changing which pitches it "
                   "challenges — not by challenging more. Output of the fitted "
                   "decision model; see the Decomposition tab.")

    st.warning(
        f"**A lower success rate can produce more value.** A **{opt_rate:.0%}** "
        f"challenge strategy outperforms today's **{obs_rate:.0%}** one because "
        f"the calls it picks are worth more runs. A **correct** challenge is "
        f"returned, so only a *wrong* one costs anything — which puts the "
        f"break-even confidence far below a coin flip and makes *leverage*, not "
        f"*certainty*, the thing to optimize. **Challenge accuracy is not "
        f"challenge value.**"
    )

    st.caption(
        f"**Jump to:** the **Real games from 2026** tab for named at-bats · "
        f"the **Should I challenge?** tab for the live calculator · "
        f"the **Decomposition** tab for how the {decision_gap:.0f}-run figure "
        f"is built."
    )

    with st.expander("Which parts of this rest on statistics, and which don't"):
        st.markdown(
            f"- **The decision rule is arithmetic.** Because a correct challenge "
            f"is returned, the break-even confidence to challenge is "
            f"p\\* = C / (ΔRE + C), which follows directly from the challenge "
            f"rules and run expectancy. It is not a hypothesis test and does not "
            f"depend on any correlation clearing a significance threshold.\n"
            f"- **The ~{decision_gap:.0f}-run figure is the output of a fitted "
            f"decision model.** Its magnitude depends on modeling inputs — the "
            f"fitted perceptual noise, the posterior challenge-success "
            f"probabilities, the 2026 run-expectancy table, the empirical "
            f"opportunity distribution, and the half-inning-level continuation "
            f"value. The point estimate has moved by a run or two across data "
            f"refreshes; treat it as an **8–10 run** range, not a precise "
            f"number.\n"
            f"- **The team-vs-player skill question is separate.** It lives in "
            f"the **Runs left on the table** tab, and every statistic there is "
            f"labeled with how much weight it can bear. A confidence interval "
            f"that crosses zero on one of those supporting checks does not touch "
            f"the decision gap above."
        )

    tab1, tab_games, tab2, tab3 = st.tabs(
        ["Decomposition", "Real games from 2026", "Should I challenge?",
         "Runs left on the table"])

    # ---------------------------------------------------------------- tab 1
    with tab1, guarded_section("Decomposition"):
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

        # -------------------------------------------------- signature viz
        st.subheader("The whole decision rule, in one picture")
        st.markdown(
            "Every point is a real 2026 called pitch. **Left–right:** how likely "
            "the call was wrong. **Up–down:** how many runs ride on it (ΔRE). "
            "The dashed line is the break-even **p\\* = C / (ΔRE + C)** — "
            "challenge above it, hold below it. Notice the shape: when the "
            "stakes are high (top of the chart) a call only ~20–30% likely "
            "wrong is already worth challenging; when the stakes are tiny "
            "(bottom) even a near-certain miss isn't. **Low confidence + huge "
            "stakes beats high confidence + nothing at stake.**"
        )
        CAT_COLORS = {
            "Challenged — justified": COLOR_OPTIMAL,
            "Challenged — low value": "#DC2626",
            "Should have challenged": COLOR_CEILING,
            "Correctly held": "#CBD5E1",
        }
        _ch = decision_scatter[decision_scatter.was_challenged]
        _low_value_pct = (_ch.category == "Challenged — low value").mean() * 100
        _y_max = 1.5
        dsc = decision_scatter[decision_scatter.dre <= _y_max].copy()
        _bnd = decision_boundary[(decision_boundary.p_star <= 0.95)
                                 & (decision_boundary.dre <= _y_max)]
        boundary_line = alt.Chart(_bnd).mark_line(
            color="#0F172A", strokeDash=[6, 4], size=2).encode(
            x=alt.X("p_star:Q", title="P(the call was wrong)",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("dre:Q", title="Runs at stake if the call flips (ΔRE)",
                    scale=alt.Scale(domain=[0, _y_max])))
        pts = alt.Chart(dsc).mark_circle(opacity=0.45).encode(
            x=alt.X("p_success:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("dre:Q", scale=alt.Scale(domain=[0, _y_max])),
            size=alt.Size("was_challenged:N", scale=alt.Scale(
                domain=[True, False], range=[38, 14]), legend=None),
            color=alt.Color("category:N", scale=alt.Scale(
                domain=list(CAT_COLORS), range=list(CAT_COLORS.values())),
                legend=alt.Legend(title=None, orient="top", columns=2)),
            tooltip=[alt.Tooltip("category:N", title="Class"),
                     alt.Tooltip("p_success:Q", title="P(wrong)", format=".0%"),
                     alt.Tooltip("dre:Q", title="ΔRE (runs)", format=".2f"),
                     alt.Tooltip("challenger:N", title="Could challenge"),
                     alt.Tooltip("was_challenged:N", title="Challenged?"),
                     alt.Tooltip("overturned:N", title="Overturned?")],
        )
        st.altair_chart((pts + boundary_line).properties(height=430),
                        width="stretch")
        st.caption(
            "Non-challenged pitches are the small dots (a 4,000-point sample "
            "each of *should have challenged* and *correctly held*, since there "
            "are hundreds of thousands of routine takes); every actually-"
            "challenged pitch is a large dot. Each point is classed against the "
            "break-even it truly faced — C(0) for its own half-inning — while "
            "the drawn line uses the start-of-game C(0) = "
            f"{decision_boundary.C_ref.iloc[0]:.3f} runs for one clean "
            "reference; late-game boundaries sit lower. \"P(wrong)\" is the "
            "0.5-inch-blur geometry read, the same basis as the case studies. "
            f"About **{_low_value_pct:.0f}%** of real challenges land below their "
            "break-even line — *challenged, low value*: won often enough, but "
            "not on calls worth the risk. A few pitches with ΔRE above "
            f"{_y_max:g} runs (deep bases-loaded full counts) are off the top "
            "of this view."
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
        ).properties(height=190,
                     title="A lower challenge success rate can produce more value")
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

        with st.expander("Method note: how the policy is solved, and what \"arithmetic\" means here"):
            st.markdown(
                "- **The decision rule is arithmetic.** Challenge when "
                "P(call was wrong) exceeds **p\\* = C / (ΔRE + C)**, where ΔRE "
                "is the run swing if the call flips and C is the option value "
                "of one incorrect-challenge token. Because a correct challenge "
                "is returned, the cost side carries (1 − p), which is what "
                "pushes p\\* below 50%. This follows directly from the challenge "
                "rules and run expectancy — no fitted parameter, no p-value.\n"
                "- **The magnitude (~8–10 runs/team-season) is a fitted model "
                "output.** It depends on the fitted perceptual σ, the posterior "
                "challenge-success probabilities, the 2026 run-expectancy "
                "table, the empirical opportunity distribution, and the "
                "simulation. The point estimate has moved by a run or two "
                "across data refreshes.\n"
                "- **Continuation value is summarized at the half-inning "
                "level.** The backward induction solves V over (half-inning t, "
                "opportunity j within t, challenges spent k), but the option "
                "value C exported to this app and to the calculator is "
                "C(t, k) — evaluated at the *start* of each half-inning and "
                "applied to every opportunity within it, not a distinct "
                "C(t, j, k) per pitch. Almost all of a token's value comes "
                "from future half-innings, so the position-within-inning "
                "correction is second order; still, this is a "
                "half-inning-level continuation-value approximation, not a full "
                "per-state lookup. Making C fully position-dependent is on the "
                "v2 list.\n"
                "- **Extra-inning rule.** A challenge is restored at the start "
                "of an extra inning only for a team that enters it with none "
                "left (k: 0→0, 1→1, 2→1).\n"
                "- **Not the same question as Statcast's \"Expected "
                "Challenges.\"** That model is *descriptive* — how likely is a "
                "player to challenge, given the context? This is *prescriptive* "
                "— given the state and P(call wrong), *should* the team "
                "challenge to maximize value? The gap between the two is the "
                "decision gap."
            )

        st.markdown(
            f"**Decision gap: +{decision_gap:.0f} runs per team-season.** This is the part "
            "a team can capture by changing policy alone. Unlike the "
            "information gap below, it uses only the *fitted* perceptual noise "
            "and makes no assumption about how accurate the tracking cameras "
            "are. It is still a fitted-model output — its size also reflects "
            "the 2026 run-expectancy table, the empirical opportunity "
            "distribution, the half-inning-level continuation value, and the "
            "simulation — so treat it as an **8–10 run range**; only the "
            "decision *rule* it rests on is arithmetic."
        )
        st.warning(
            "The remaining gap to the **high-precision benchmark** (σ = 0.5 in — "
            "not perfect information; it still carries half an inch of tracking "
            "noise) is **not** coachable, and its size depends entirely on an "
            "assumed tracking precision that this dataset cannot measure — we "
            "only ever observe Hawk-Eye's own output (MLB's camera-based tracking "
            "system, the same one that draws the strike zone on TV), never "
            "independent ground truth. Shown below as a sensitivity curve rather "
            "than a single number."
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

        st.subheader("Does perceptual noise vary by location, not just by role?")
        lr_stat = zone_interaction.lr_stat.iloc[0]
        p_val = zone_interaction.p_value.iloc[0]
        swing_pp = zone_interaction.swing_pp.iloc[0]
        move = zone_sigma_sensitivity.set_index("label")["decision_gap_vs_observed_per_season"]
        move_season = (move.get("optimal @ zone-region sigma (sensitivity)", float("nan"))
                       - move.get("optimal @ player sigma (role-only, canonical)", float("nan")))
        boot_lo = zone_sigma_bootstrap.move_runs_per_season.quantile(0.025)
        boot_hi = zone_sigma_bootstrap.move_runs_per_season.quantile(0.975)
        st.markdown(
            f"**The model above assumes one σ per role, everywhere in the zone. "
            f"That assumption is measurably wrong — but how much it costs the "
            f"headline number isn't pinned down by one season of data.** "
            f"Splitting the same challenges into a 3×3 grid (in/middle/away × low/middle/high, "
            f"relative to the batter) and testing whether the role gap in success rate "
            f"varies by location: it does. Under the null of no location interaction, a "
            f"likelihood-ratio statistic at least this large would be very unlikely "
            f"(p {'< 0.0001' if p_val < 0.0001 else f'= {p_val:.4f}'}); "
            f"the success-rate gap swings **{swing_pp:.0f} percentage points** across "
            f"well-populated regions. "
            f"Refitting σ separately for each of the 9 regions and re-running the full "
            f"decision model moves the headline decision gap by "
            f"**{move_season:+.2f} runs per team-season** on the actual 2026 data — but "
            f"each region is fit on as few as ~200 challenged pitches, so we "
            f"bootstrapped it (resampling just the challenged pitches within each "
            f"region 150 times, refitting, and re-solving the model each time). "
            f"The result: a 95% interval of **{boot_lo:+.1f} to {boot_hi:+.1f} runs "
            f"per team-season** — wide enough to cross zero. The location effect "
            f"itself is real (that likelihood-ratio test doesn't depend on this "
            f"resampling), but this dataset can't yet say whether accounting for "
            f"it in the decision model is worth a little, a lot, or possibly points "
            f"the other way. Read {move_season:+.2f} as one plausible draw from a "
            f"wide distribution, not a precise correction.\n\n"
            f"**This is a robustness check on a modeling assumption, not the "
            f"headline.** The +{decision_gap:.0f}-run decision gap uses the "
            f"role-level σ only and does not move with it. An interval that "
            f"crosses zero *here* means \"a finer σ model might nudge the number "
            f"up or down\" — not \"the effect might be nothing.\""
        )

        order_v = ["high", "middle", "low"]
        order_h = ["away", "middle", "in"]
        zh = zone_heatmap[zone_heatmap.n >= 30].copy()
        zh["Role"] = zh.challenger.map(ROLE_LABELS)
        zh["success_pct"] = zh.success_rate * 100
        heat_chart = alt.Chart(zh).mark_rect().encode(
            x=alt.X("horiz_region:N", sort=order_h, title="Horizontal (batter-relative)"),
            y=alt.Y("vert_region:N", sort=order_v, title="Vertical (zone-relative)"),
            color=alt.Color("success_rate:Q", title="Success rate",
                            scale=alt.Scale(scheme="blues", domain=[0.35, 0.70]),
                            legend=alt.Legend(format="%")),
            tooltip=["Role", "vert_region", "horiz_region", "n",
                     alt.Tooltip("success_rate:Q", format=".1%")],
            facet=alt.Facet("Role:N", title=None),
        ).properties(width=220, height=220)
        text_chart = alt.Chart(zh).mark_text(fontWeight="bold").encode(
            x=alt.X("horiz_region:N", sort=order_h),
            y=alt.Y("vert_region:N", sort=order_v),
            text=alt.Text("success_pct:Q", format=".0f"),
            color=alt.condition("datum.success_rate > 0.55", alt.value("white"), alt.value("black")),
            facet=alt.Facet("Role:N", title=None),
        ).properties(width=220, height=220)
        st.altair_chart(heat_chart + text_chart, width='stretch')
        st.caption(
            "\"In\" = the horizontal third closest to the batter's body, \"away\" = "
            "farthest, mirrored by batter handedness and verified against real data "
            "(hit-by-pitch location averages −1.93 ft for right-handed batters and "
            "+1.99 ft for left-handed batters — confirming which physical side is "
            "\"inside\" for each). Cells with too few challenges to trust (the dead "
            "center of the zone, where almost nobody challenges) are omitted."
        )

        st.markdown(
            "**The standout cell: high-middle.** Everywhere else, catchers and "
            "pitchers out-read batters by 6–17 points, matching the vantage-point "
            "story above. On pitches up and over the heart of the plate, that "
            "flips — batters succeed 69% of the time there versus 49% for the "
            "battery, a genuine reversal, not noise (n=252 and 367). A pitch up in "
            "the zone is arguably an easier read from the side (batters see it "
            "rising out of the pitcher's hand) than from a crouch behind the plate "
            "looking up through it."
        )
        st.markdown(
            "*What this means: the pooled model is still the right headline number, "
            "but it's a worse guide for any single high-middle call specifically — "
            "there, if anything, trust the batter's read over the battery's, which "
            "is the opposite of the general pattern.*"
        )
        st.caption(
            "Breaking balls vs. fastballs showed no comparable split (batters: "
            "48.0% on fastballs vs. 50.0% on breaking/offspeed; battery: 58.2% vs. "
            "57.3%) — pitch type doesn't appear to be a meaningful blind spot for "
            "either role, unlike location. Full per-region figures and the sigma "
            "refit are in scripts/zone_analysis.py, scripts/zone_sigma_refit.py, "
            "and data/zone_sigma.parquet."
        )

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

    # ---------------------------------------------------------------- real games
    with tab_games, guarded_section("Real games from 2026"):
        st.markdown(
            "##### In short\n"
            "The rest of this site is averages. This tab is the specific "
            "at-bats behind them — real 2026 games, with names, dates, the "
            "situation, and what the model saw. Every case is a single called "
            "ball or strike, scored two ways: **how likely the call was wrong**, "
            "and **how many runs were riding on it**."
        )
        st.divider()

        st.warning(
            "**How to read these — two things that are easy to get wrong.**\n\n"
            "**1. A missed challenge is a mistake at the moment of the "
            "decision, not because of what happened afterward.** Each case is "
            "framed on the *expected* runs at stake and the break-even "
            "confidence needed to challenge — both knowable before the next "
            "pitch. What actually happened next is shown too, but only as "
            "colour: a call that was 90% likely wrong in a spot with a 4% "
            "break-even was worth challenging *whether or not* the batter went "
            "on to score. Don't let a quiet outcome argue you out of the "
            "decision, or a dramatic one into it.\n\n"
            "**2. These are the tail, and they're labelled as such.** Each "
            "list states how many comparable situations exist in 2026 and "
            "where the shown case ranks. The examples are extreme *by "
            "construction* — that is the point of showing them — not a claim "
            "that a typical missed challenge looks like this."
        )
        st.caption(
            "\"P(call was wrong)\" is the model's geometry read: where the "
            "pitch crossed the middle of the plate versus this batter's ABS "
            "zone, with a half-inch of tracking blur. \"Break-even\" is "
            "C / (ΔRE + C) — the confidence at which challenging is worth the "
            "risk of losing a challenge token — the same formula as the "
            "\"Should I challenge?\" tab."
        )

        def render_case(row, kind):
            half_word = "Top" if row.half == "Top" else "Bottom"
            with st.container(border=True):
                st.markdown(
                    f"#### #{int(row['rank'])} — {row.game_date}, "
                    f"{row.away_team} @ {row.home_team}")
                L, R = st.columns([3, 2])
                with L:
                    st.markdown(
                        f"- **{half_word} {_ord(row.inning)}**, "
                        f"{row.count_label} count, {row.base_out_label}\n"
                        f"- **{row.batter_name}** batting, "
                        f"**{row.pitcher_name}** pitching\n"
                        f"- Call on the field: **{row.call_was}** · the "
                        f"**{row.challenging_team}** {row.challenger_desc} "
                        f"could have challenged "
                        f"(**{int(row.challenges_remaining)}** challenge"
                        f"{'s' if row.challenges_remaining != 1 else ''} left)"
                    )
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Model: P(call was wrong)", f"{row.p_wrong:.0%}")
                    m2.metric("Runs at stake (ΔRE)", f"{row.dre:.2f}")
                    m3.metric("Break-even to challenge", f"{row.p_star:.0%}")

                    if kind == "endorsed_win":
                        st.markdown(
                            f"✅ **Challenged — and won.** The read was "
                            f"genuinely close (only **{row.p_wrong:.0%}** "
                            f"likely wrong), but the break-even here was just "
                            f"**{row.p_star:.0%}**, so the optimal policy says "
                            f"fire. Expected runs secured: **{row.ev_net:.2f}**."
                        )
                    elif kind == "burn":
                        st.markdown(
                            f"🔴 **Nothing left to challenge with.** Expected "
                            f"runs forfeited on this call: **{row.ev_net:.2f}**, "
                            f"against a break-even of only **{row.p_star:.0%}**."
                        )
                        st.markdown(
                            f"**What the {row.challenging_team} spent their two "
                            f"challenges on instead:**\n\n{row.burn_detail}")
                        st.caption(
                            f"Leverage gap — this call versus the average of "
                            f"the two they burned: **{row.leverage_gap:.2f} "
                            f"runs**.")
                    else:
                        st.markdown(
                            f"🔴 **No challenge was made.** Expected runs "
                            f"forfeited: **{row.ev_net:.2f}** — the model's "
                            f"read (**{row.p_wrong:.0%}** wrong) cleared the "
                            f"**{row.p_star:.0%}** break-even with room to "
                            f"spare."
                        )

                    st.markdown(
                        f"*What happened next — colour only, it does not change "
                        f"whether the decision was right: {row.narrative}*")
                with R:
                    st.altair_chart(case_zone_chart(row), width="content",
                                    key=f"czone_{kind}_{int(row['rank'])}")
                    side_word = ("winning" if kind == "endorsed_win"
                                 else "wrong")
                    st.caption(
                        f"Where the pitch crossed the middle of the plate, on "
                        f"{row.batter_name}'s ABS zone. Solid box: the rulebook "
                        f"zone. Dashed box: the real ABS boundary, one "
                        f"ball-radius further out. The pitch was **"
                        f"{abs(row.miss_in):.1f} in** onto the {side_word} side "
                        f"of that boundary.")

        g1, g2, g3 = st.tabs([
            "① Biggest missed opportunities",
            "② Costly early burns",
            "③ Model-endorsed successes"])

        with g1:
            st.markdown(
                "Called pitches where P(wrong) was high, the runs at stake "
                "were large, and the side that could have challenged still had "
                "a challenge in hand — and didn't use it. Ranked by **expected "
                "runs forfeited** (P(wrong) × ΔRE, net of the cost of spending "
                "a challenge token)."
            )
            sub = cases[cases.category == "missed"].sort_values("rank")
            ms = miss_summary

            st.markdown(f"##### How big are these {int(ms.n):,}, really?")
            st.markdown(
                f"That works out to **{ms.per_game:.1f} model-endorsed "
                f"unchallenged pitches per game** — a number that only means "
                f"something once you see the sizes. Half forfeit under "
                f"**{ms.ev_net_median:.2f} runs**; nine in ten under "
                f"**{ms.ev_net_p90:.2f}**. Broken out by expected runs "
                f"forfeited:\n\n"
                f"- **{ms.frac_hair:.0%}** barely clear the bar — under 0.05 runs\n"
                f"- **{ms.frac_modest:.0%}** are modest — 0.05 to 0.20 runs\n"
                f"- **{ms.frac_comfortable:.0%}** are comfortable — 0.20 to "
                f"0.50 runs (**{int(ms.n_comfortable_plus):,}** pitches at "
                f"0.20+, about **{ms.n_comfortable_plus_per_game:.0f} a game**)\n"
                f"- **{ms.frac_large:.1%}** are large — half a run or more "
                f"(**{int(ms.n_large)}** all season, one every "
                f"~{ms.games_per_large:.0f} games)\n\n"
                f"The honest version: **most missed opportunities are small "
                f"change.** The part worth acting on is the ~"
                f"{ms.n_comfortable_plus_per_game:.0f} a game worth at least a "
                f"fifth of a run, and the handful worth much more."
            )
            hh = miss_hist.copy()
            hh["bucket"] = [f"{lo:g}–{hi:g}" if hi < 3 else f"{lo:g}+"
                            for lo, hi in zip(hh.lo, hh.hi)]
            st.altair_chart(
                alt.Chart(hh).mark_bar(color=COLOR_OPTIMAL).encode(
                    x=alt.X("bucket:N", sort=list(hh.bucket),
                            title="Expected runs forfeited on the missed call"),
                    y=alt.Y("n:Q", title="Missed opportunities"),
                    tooltip=["bucket", "n"],
                ).properties(height=200), width='stretch')
            st.caption(
                f"\"Small\" here means low *stakes*, not a *close decision*. "
                f"The median missed opportunity still clears its break-even by "
                f"**{ms.conf_margin_median * 100:.0f} points of confidence** — "
                f"the model is quite sure the call was wrong; there just isn't "
                f"much riding on most of them."
            )

            st.markdown("##### The one situation to watch: the borderline full count")
            st.markdown(
                f"Across *all* {int(ms.n):,} endorsed misses, 3-2 counts are "
                f"just **{ms.full_count_share_all:.0%}** — most are low-stakes "
                f"takes early in the count. Among the ones that actually cost "
                f"something, it flips hard:\n\n"
                f"- of the **{int(ms.n_large)}** misses worth half a run or "
                f"more, **{ms.full_count_share_large:.0%}** were full counts\n"
                f"- of the **100 largest**, **{ms.full_count_share_top100:.0%}** were\n\n"
                f"The mechanism is clean: on 3-2 the call *is* the plate "
                f"appearance — ball four or strike three, no third outcome — so "
                f"the run-expectancy swing is as large as a single call ever "
                f"gets. An endorsed full-count miss averages "
                f"**{ms.full_count_mean_dre:.2f} runs** at stake versus "
                f"**{ms.non_full_mean_dre:.2f}** everywhere else, which drops "
                f"the break-even to about **{ms.full_count_mean_p_star:.0%}** — "
                f"you should challenge a borderline 3-2 pitch on a hunch. The "
                f"case studies below land on 3-2 by themselves, pulled from "
                f"real games by expected-runs rank alone — independent "
                f"confirmation the model flags the right moments."
            )
            st.markdown(
                "**Takeaway:** if a team puts extra attention on exactly one "
                "situation, make it the borderline full count. That's where "
                "the unclaimed runs are concentrated."
            )
            _order = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2",
                      "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
            _scope_lbl = {"all": "All endorsed misses",
                          "large": "≥ 0.5 runs forfeited", "top100": "100 largest"}
            mb = miss_by_count.copy()
            mb["Scope"] = mb.scope.map(_scope_lbl)
            st.altair_chart(
                alt.Chart(mb).mark_bar().encode(
                    x=alt.X("count_label:N", sort=_order, title="Count (balls-strikes)"),
                    xOffset=alt.XOffset("Scope:N", sort=list(_scope_lbl.values())),
                    y=alt.Y("pct:Q", axis=alt.Axis(format="%"),
                            title="Share of that set"),
                    color=alt.Color("Scope:N", sort=list(_scope_lbl.values()),
                                    scale=alt.Scale(domain=list(_scope_lbl.values()),
                                                    range=[COLOR_OBSERVED, COLOR_OPTIMAL,
                                                           COLOR_CEILING]),
                                    legend=alt.Legend(title=None)),
                    tooltip=["Scope", "count_label",
                             alt.Tooltip("pct:Q", format=".0%"), "n"],
                ).properties(height=240), width='stretch')
            st.caption(
                f"The same {int(miss_summary['n']):,} pitches under three cuts: "
                "everything, the ≥0.5-run tail, and the 100 largest. 3-2 is a "
                "rounding error in the first and the plurality in the last.")

            st.divider()
            st.markdown("##### The cases")
            if len(sub):
                st.caption("Where this set sits: " + sub.comparable_desc.iloc[0])
                for _, row in sub.iterrows():
                    render_case(row, "missed")
            else:
                st.info("No model-endorsed missed opportunities in the current data.")

        with g2:
            st.markdown(
                "One team, one game, the whole thesis: both challenges spent "
                "early — and lost — on calls worth almost nothing, then a call "
                "worth **more than a run** and clearly wrong, with nothing left "
                "to challenge it. This is \"challenge *different*, not "
                "*more*\" in a single box score."
            )
            sub = cases[cases.category == "burn"].sort_values("rank")
            if len(sub):
                st.caption("Where this set sits: " + sub.comparable_desc.iloc[0])
                for _, row in sub.iterrows():
                    render_case(row, "burn")
            else:
                st.write("No qualifying games this season.")

        with g3:
            st.markdown(
                "The model isn't only a critic. These were genuinely close "
                "calls — the kind you would not be *sure* about — that the "
                "optimal policy still says to challenge, because the leverage "
                "is high and the break-even tiny. All were **made, and won**."
            )
            sub = cases[cases.category == "endorsed_win"].sort_values("rank")
            cs = coinflip_summary
            # Whichever team makes the most genuinely-uncertain endorsed
            # challenges this refresh, and how well it does on them. Data-driven
            # so a change in the season standings can't break the page.
            cf_leader = (coinflip_team.loc[coinflip_team["cf_attempts"].idxmax()]
                         if len(coinflip_team) else None)

            st.markdown("##### Do the coin-flip wins cluster — or is everyone equally good at them?")
            st.markdown(
                f"They cluster. Across the 30 teams, the number of "
                f"genuinely-uncertain endorsed challenges a team makes per game "
                f"correlates with its total runs gained from challenges at "
                f"**r = {cs.r_attempts_runs:.2f}** "
                f"(p {'< 0.001' if cs.p_attempts_runs < 0.001 else f'= {cs.p_attempts_runs:.3f}'}). "
                f"Part of that is mechanical — a won coin-flip challenge *is* "
                f"runs — so the cleaner test is the same rate against runs "
                f"gained on a team's **other**, non-coin-flip challenges. It "
                f"still holds: **r = {cs.r_attempts_runs_other:.2f}** "
                f"(p = {cs.p_attempts_runs_other:.3f}). Teams that pull the "
                f"trigger on close, high-leverage calls also do better on the "
                f"routine ones — it reads as a general challenging skill, not a "
                f"lucky run of 50/50s."
            )
            _leader_txt = (
                f" The team that makes the most of these genuinely-uncertain "
                f"endorsed challenges is **{cf_leader.team_abbr}**, at a "
                f"**{cf_leader.cf_win_rate:.0%} win rate on "
                f"{int(cf_leader.cf_attempts)}** of them — the same clubs that "
                f"show up with individually sharp catchers in the *Runs left on "
                f"the table* tab." if cf_leader is not None else "")
            st.markdown(
                f"**{int(cs.top5_overlap)} of the top 5** runs-gained teams "
                f"({cs.top5_runs_teams}) are also top 5 in coin-flip rate "
                f"({cs.top5_by_cf_rate_teams}).{_leader_txt} Same finding, "
                f"different angle: a sharp battery lets a team act on calls "
                f"other teams have to let go."
            )
            ct = coinflip_team.copy()
            # Highlight the 5 teams with the most total runs gained from
            # challenges. `runs_rank` is exactly rank(runs_gained, desc) from
            # scripts/team_decomposition.py, so this tracks the data on every
            # refresh -- no literal team list to go stale. Falls back to a
            # threshold on runs_gained if the rank column is ever absent.
            if "runs_rank" in ct.columns:
                _is_top5 = ct.runs_rank <= 5
            else:
                _is_top5 = ct.runs_gained >= ct.runs_gained.nlargest(5).min()
            ct["Highlight"] = np.where(_is_top5, "Top-5 runs gained", "Other teams")
            enc = dict(
                x=alt.X("cf_attempts_per_game:Q",
                        title="Genuinely-uncertain endorsed challenges per game"),
                y=alt.Y("runs_other_per_game:Q",
                        title="Runs gained per game on all OTHER challenges"))
            pts = alt.Chart(ct).mark_circle(size=110, opacity=0.8).encode(
                color=alt.Color("Highlight:N", scale=alt.Scale(
                    domain=["Top-5 runs gained", "Other teams"],
                    range=[COLOR_OPTIMAL, COLOR_OBSERVED]),
                    legend=alt.Legend(title=None)),
                tooltip=["team_abbr", alt.Tooltip("cf_attempts_per_game:Q", format=".2f"),
                         alt.Tooltip("runs_gained:Q", title="Runs gained (all)", format=".1f"),
                         alt.Tooltip("runs_gained_other:Q", title="Runs gained (other)", format=".1f")],
                **enc)
            trend = alt.Chart(ct).transform_regression(
                "cf_attempts_per_game", "runs_other_per_game").mark_line(
                color="#334155", strokeDash=[5, 3]).encode(**enc)
            labels = alt.Chart(ct[ct.Highlight == "Top-5 runs gained"]).mark_text(
                dx=9, align="left", fontWeight="bold", color=COLOR_OPTIMAL).encode(
                text="team_abbr", **enc)
            st.altair_chart((pts + trend + labels).properties(height=320),
                            width='stretch')

            cr = coinflip_role.set_index("challenger")["share_of_role_challenges"]
            f_share = cr.get("fielding")
            b_share = cr.get("batting")
            if f_share is not None and b_share is not None:
                st.markdown(
                    f"**By role:** catchers and pitchers make more of them — "
                    f"**{f_share:.0%}** of fielding-side challenges are "
                    f"coin-flip-endorsed versus **{b_share:.0%}** of batting-side "
                    f"ones. Consistent with the vantage-point result elsewhere on "
                    f"this page: reading location from behind the plate lets the "
                    f"battery act on closer calls than a batter can from the side."
                )
            st.caption(
                f"Two caveats. It's the *volume* of good borderline attempts "
                f"that tracks with run production, not the win rate on them "
                f"(win rate vs. other-challenge runs: "
                f"r = {cs.r_winrate_runs_other:+.2f}, "
                f"p = {cs.p_winrate_runs_other:.2f} — not significant). And the "
                f"raw success rate on these looks high "
                f"({cs.cf_win_rate_overall:.0%}) because players only challenge "
                f"the band-pitches they already feel good about — the model "
                f"would tell you to fire at 45%."
            )

            st.divider()
            st.markdown("##### The cases")
            if len(sub):
                st.caption("Where this set sits: " + sub.comparable_desc.iloc[0])
                for _, row in sub.iterrows():
                    render_case(row, "endorsed_win")
            else:
                st.info("No model-endorsed coin-flip wins in the current data.")

        st.divider()
        st.caption(
            "Cases regenerated by scripts/build_case_studies.py from the same "
            "per-pitch scoring as every other number on this site; \"what "
            "happened next\" is pulled from the MLB Stats API play-by-play.")

    # ---------------------------------------------------------------- tab 2
    PRESETS = {
        "Full count, bases loaded, 2 outs": dict(
            dt_inning=9, dt_half="Bottom", dt_balls=3, dt_strikes=2, dt_outs=2,
            dt_r1=True, dt_r2=True, dt_r3=True, dt_k="0", dt_role="Batter"),
        "0-0, bases empty, 0 outs": dict(
            dt_inning=1, dt_half="Top", dt_balls=0, dt_strikes=0, dt_outs=0,
            dt_r1=False, dt_r2=False, dt_r3=False, dt_k="0", dt_role="Batter"),
        "Same full count — but you've already blown one challenge": dict(
            dt_inning=9, dt_half="Bottom", dt_balls=3, dt_strikes=2, dt_outs=2,
            dt_r1=True, dt_r2=True, dt_r3=True, dt_k="1", dt_role="Batter"),
    }


    def _apply_preset(name):
        for k, v in PRESETS[name].items():
            st.session_state[k] = v
        st.session_state["dt_confidence"] = 50
        st.session_state["dt_picked"] = None


    with tab2, guarded_section("Should I challenge?"):
        st.markdown(
            "##### In short\n"
            "**Set up a real game situation below and get an actual recommendation** "
            "— not just a curve. The tool tells you the break-even confidence needed "
            "to challenge, what's actually at stake, and a plain verdict: challenge "
            "or hold. Click a spot in the strike zone (or drag the slider) for the "
            "model's own read on how likely the call was wrong."
        )
        st.divider()

        st.markdown("**Try a preset, or set up your own situation below:**")
        pcols = st.columns(len(PRESETS))
        for pcol, name in zip(pcols, PRESETS):
            pcol.button(name, on_click=_apply_preset, args=(name,), width='stretch')

        st.subheader("Game situation")
        c1, c2, c3 = st.columns(3)
        with c1:
            inning = st.number_input("Inning", 1, 15, 1, key="dt_inning")
            half_display = st.radio("Half", ["Top", "Bottom"], horizontal=True, key="dt_half")
            half = "Bot" if half_display == "Bottom" else "Top"
        with c2:
            balls = st.selectbox("Balls", [0, 1, 2, 3], key="dt_balls")
            strikes = st.selectbox("Strikes", [0, 1, 2], key="dt_strikes")
            outs = st.selectbox("Outs", [0, 1, 2], key="dt_outs")
        with c3:
            r1 = st.checkbox("Runner on 1st", key="dt_r1")
            r2 = st.checkbox("Runner on 2nd", key="dt_r2")
            r3 = st.checkbox("Runner on 3rd", key="dt_r3")

        c4, c5 = st.columns(2)
        with c4:
            k_display = st.radio("Incorrect challenges already used this game",
                                 ["0", "1", "2 (exhausted)"], horizontal=True, key="dt_k",
                                 help="Only INCORRECT challenges cost you — a correct one "
                                 "is returned immediately and doesn't count against you.")
        with c5:
            role_display = st.radio("Who's challenging", ["Batter", "Catcher / pitcher"],
                                    horizontal=True, key="dt_role",
                                    help="Batters only challenge called strikes (hoping it "
                                    "was actually a ball); catchers and pitchers only "
                                    "challenge called balls (hoping it was actually a strike).")
        role = "batting" if role_display == "Batter" else "fielding"

        dre = compute_dre(balls, strikes, outs, r1, r2, r3)
        t = half_inning_index(inning, half)
        C0, C1 = option_value_at(t)

        st.divider()

        if k_display.startswith("2"):
            st.error(
                "**Rights exhausted.** Two incorrect challenges are gone — there is "
                "no option value left to protect, and no challenge is available "
                "regardless of confidence."
            )
        elif dre is None:
            st.warning("That exact combination of count/outs/bases doesn't occur in the "
                       "2026 data (e.g. 4 balls) — pick a valid count.")
        else:
            k = int(k_display)
            C = C0 if k == 0 else C1
            p_star = break_even_probability(dre, C)

            st.markdown(
                "**How this tool decides.** Challenge when your confidence the "
                "call was wrong exceeds a break-even number: "
                "**p\\* = C / (ΔRE + C)**. Two pieces feed that formula, both "
                "computed below for the situation you set up:\n"
                "- **ΔRE ('runs at stake')** — how much *run expectancy* "
                "(the average runs a team can expect to score from this point "
                "in the inning onward) changes if the call flips. A full-count "
                "walk with the bases loaded is worth a lot; a first-pitch take "
                "with nobody on is worth almost nothing.\n"
                "- **Option value, C(k)** — what one *incorrect* challenge is "
                "worth to hold onto, right now, for the rest of the game. It "
                "isn't the value of this one pitch — it's the value of still "
                "having the right to challenge later, which is why it depends "
                "on how many incorrect challenges (k) are already spent.\n\n"
                "Because a correct challenge is free, the cost side of the "
                "decision carries a factor of (1 − confidence), not 1 — that "
                "asymmetry is why the break-even number below is usually well "
                "under 50%, not at it."
            )
            st.caption(
                "\"Confidence\" throughout this tool means the probability the "
                "call was actually wrong — a number, not a feeling. The model "
                "knows exactly where the pitch crossed the plate; a player "
                "doesn't. That's the whole reason the zone plot below exists: "
                "it converts \"where you think the pitch was\" into a "
                "probability, using how precisely a player in this role "
                "actually reads location (measured from real 2026 challenges)."
            )

            st.subheader("Where do you think the pitch was?")
            st.caption(
                "Click a spot in the zone for the model's own estimate of P(the call "
                "was wrong), based on how precisely a player in this role actually "
                "reads pitch location (measured from real 2026 challenges, never "
                "assumed). Or skip straight to the confidence slider below. Drawn to "
                "true scale — 17 inches wide, and the vertical extent set by the ABS "
                "rule (53.5% to 27% of a league-average batter's height) — so the "
                "panel is genuinely a strike zone, not a stretched square."
            )

            height_ft = REPRESENTATIVE_HEIGHT_FT
            grid = zone_click_grid(role)
            click = alt.selection_point(name="pt", fields=["xc", "zc"], nearest=True,
                                        on="click", empty=False)
            x_scale = alt.Scale(domain=[PLOT_X_LO, PLOT_X_HI])
            z_scale = alt.Scale(domain=[PLOT_Z_LO, PLOT_Z_HI])
            heat = alt.Chart(grid).mark_rect().encode(
                x=alt.X("x0:Q", title="Horizontal location (inches from the plate's center)",
                        scale=x_scale),
                x2="x1:Q",
                y=alt.Y("z1:Q", title="Height off the ground (inches)", scale=z_scale),
                y2="z0:Q",
                color=alt.Color("p_wrong:Q", title="Model's P(wrong)",
                                scale=alt.Scale(scheme="oranges", domain=[0, 1]),
                                legend=alt.Legend(format="%")),
                tooltip=[alt.Tooltip("xc:Q", title="Horizontal (in)", format=".1f"),
                         alt.Tooltip("zc:Q", title="Height (in)", format=".1f"),
                         alt.Tooltip("p_wrong:Q", title="P(wrong)", format=".0%")],
            ).add_params(click).properties(
                width=ZONE_PLOT_WIDTH_PX, height=ZONE_PLOT_HEIGHT_PX)
            rule_zone = alt.Chart(pd.DataFrame([
                {"x0": -ZONE_WIDTH_IN / 2, "x1": ZONE_WIDTH_IN / 2,
                 "z0": ZONE_BOT_IN, "z1": ZONE_TOP_IN}
            ])).mark_rect(fill=None, stroke="#0F172A", strokeWidth=2).encode(
                x=alt.X("x0:Q", scale=x_scale), x2="x1:Q",
                y=alt.Y("z1:Q", scale=z_scale), y2="z0:Q")
            abs_boundary = alt.Chart(pd.DataFrame([
                {"x0": -ZONE_WIDTH_IN / 2 - BALL_RADIUS_IN, "x1": ZONE_WIDTH_IN / 2 + BALL_RADIUS_IN,
                 "z0": ZONE_BOT_IN - BALL_RADIUS_IN, "z1": ZONE_TOP_IN + BALL_RADIUS_IN}
            ])).mark_rect(fill=None, stroke="#0F172A", strokeWidth=1.5, strokeDash=[5, 3]).encode(
                x=alt.X("x0:Q", scale=x_scale), x2="x1:Q",
                y=alt.Y("z1:Q", scale=z_scale), y2="z0:Q")
            plate = alt.Chart(home_plate_outline(PLATE_TOP_Z)).mark_line(
                color="#0F172A", strokeWidth=2, fill="#E2E8F0", fillOpacity=0.9).encode(
                x=alt.X("x:Q", scale=x_scale), y=alt.Y("z:Q", scale=z_scale), order="order:O")

            event = st.altair_chart(
                heat + abs_boundary + rule_zone + plate,
                on_select="rerun", key=f"zone_chart_{role}", width="content")
            st.caption(
                "Solid box: the rulebook strike zone (17 in × the ABS top/bottom "
                "percentages). Dashed box: the actual ABS boundary, one ball-radius "
                "(~1.45 in) further out on every side — a pitch whose center misses "
                "the solid box can still be a strike if the ball's edge reaches it. "
                "Home plate is drawn to scale for a width reference, positioned just "
                "below the zone for compactness — not at its true height (the real "
                "plate sits at the ground, well below this view)."
            )
            picked = None
            try:
                sel = event.selection.get("pt") if hasattr(event, "selection") else \
                    event["selection"].get("pt")
                if sel:
                    picked = (sel[0]["xc"], sel[0]["zc"])
            except Exception:
                picked = None

            if picked is not None and st.session_state.get("dt_picked") != picked:
                st.session_state["dt_picked"] = picked
                model_p_wrong = p_wrong_given_click(picked[0] / 12, picked[1] / 12, role, height_ft)
                st.session_state["dt_confidence"] = round(model_p_wrong * 100)
                st.rerun()

            st.subheader("How confident are you the call was wrong?")
            confidence = st.slider("Your confidence (%)", 0, 100, 50, key="dt_confidence")
            p_conf = confidence / 100.0

            st.divider()
            st.subheader("Recommendation")
            st.caption(
                "This tool optimizes **run expectancy** — the average runs "
                "scored from here to the end of the half-inning. It does not "
                "know the score or the inning's win-probability leverage (see "
                "*Runs, not wins* in the limitations). Everything below is in "
                "runs."
            )

            decision, p_star_chk, ev_net = recommend(p_conf, dre, C, k)
            ev_cost = (1.0 - p_conf) * C   # expected run cost of firing here

            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated overturn probability", f"{confidence}%",
                      help="Your confidence the call was wrong — set by the "
                           "zone click or the slider.")
            m2.metric("Run-value swing if overturned (ΔRE)", f"+{dre:.3f}")
            m3.metric("Break-even probability", fmt_pct(p_star),
                      help="p* = C / (ΔRE + C). Challenge when your overturn "
                           "probability exceeds this.")
            m4, m5, m6 = st.columns(3)
            m4.metric(f"Challenge opportunity cost, C({k})", f"{C:.3f} runs",
                      help="Option value of one incorrect-challenge token — "
                           "what holding it is worth for the rest of the game.")
            m5.metric("Expected cost if you fire", f"−{ev_cost:.3f} runs",
                      help="(1 − overturn probability) × C — the chance you're "
                           "wrong times the token you'd lose.")
            m6.metric("Expected challenge value", f"{ev_net:+.3f} runs",
                      help="overturn probability × ΔRE − (1 − overturn "
                           "probability) × C. Positive → challenging beats "
                           "holding.")

            st.caption(
                f"How the opportunity cost changes with challenges remaining: "
                f"right now (0 used) an incorrect challenge costs "
                f"**C(0) = {C0:.3f} runs**; if you'd already used one, the same "
                f"mistake would cost **C(1) = {C1:.3f} runs** — worse, since a "
                f"second incorrect challenge leaves nothing in reserve."
            )

            if decision == "CHALLENGE":
                st.success(
                    f"### CHALLENGE\n"
                    f"Expected challenge value **{ev_net:+.3f} runs**. Your "
                    f"**{confidence}%** read clears the **{fmt_pct(p_star)}** "
                    f"break-even. Although the call is not certain to be "
                    f"overturned, the potential run-value swing is large enough "
                    f"to justify using a challenge here."
                )
            else:
                st.warning(
                    f"### HOLD\n"
                    f"Expected challenge value **{ev_net:+.3f} runs**. Your "
                    f"**{confidence}%** read falls short of the **{fmt_pct(p_star)}** "
                    f"break-even. The call may well be wrong, but preserving the "
                    f"challenge for a higher-value spot later in the game has "
                    f"greater expected value than spending it now."
                )

            if picked is not None:
                st.caption(
                    f"Model's own read at the point you clicked "
                    f"({picked[0]:.1f} in horizontal, {picked[1]:.1f} in high): "
                    f"P(call was wrong) ≈ {p_wrong_given_click(picked[0]/12, picked[1]/12, role, height_ft):.0%}. "
                    f"Assumes a league-typical batter (6'0\"); an individual batter's "
                    f"actual zone shifts this by a few inches."
                )

    # ---------------------------------------------------------------- tab 3
    with tab3, guarded_section("Runs left on the table"):
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

        # ------------------------------------------------ whose skill is it?
        st.divider()
        st.subheader("Whose skill is it — the team's, or the player's?")
        st.markdown(
            "##### In short\n"
            "**The evidence leans toward challenge accuracy being a *player* "
            "skill — most clearly with catchers — rather than a front-office "
            "one, but one strand of it is fragile.** Three pieces:\n\n"
            "1. **The role effect replicates team by team.** 28 of 30 teams "
            "individually read fielding (catcher / pitcher) challenges more "
            "precisely than batting ones — the exact league-wide pattern, found "
            "28 separate times. Stable across every data refresh.\n"
            "2. **A team's quality edge tracks its own catcher's accuracy** "
            "(**r = 0.48, p = 0.007**, across all 30 teams) — the clubs that "
            "gain runs by challenging *well* rather than just *often* are the "
            "ones with an individually sharp catcher. Also stable across "
            "refreshes.\n"
            "3. **The split-half reliability test — the fragile strand.** "
            "Splitting the season in half, individual challengers' success "
            "rates correlate across halves at **r ≈ 0.27** (≥8 challenges/half) "
            "to **r ≈ 0.38** (≥10); the team-level version is **r ≈ 0.28**, "
            "interval crossing zero. The player numbers still clear zero, but "
            "five days of new games moved *both* the player and team figures "
            "~0.05 toward each other — so treat split-half reliability as "
            "poorly estimated with one partial season, and lean on (1) and (2).\n\n"
            "So the team table above is a snapshot of who was on the roster in "
            "2026, not a standing ranking of front offices. What this season "
            "*can't* settle is whether the player skill is one a front office "
            "can reliably acquire or develop. **None of this feeds the "
            "headline:** the decision-gap number is arithmetic, and holds "
            "whether or not accuracy turns out to be a coachable trait."
        )

        st.markdown("**Perceptual precision, by team and role**")
        st.markdown(
            "Fitted the same way as the league-wide estimate in the Decomposition "
            "tab — σ (sigma) is how far off a player's read of the pitch tends to "
            "be, in inches; smaller is more precise, and it's fit from *where* "
            "players challenge, never from success rate. Every one of 28 of 30 "
            "teams reads fielding challenges (catcher/"
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
        st.markdown(
            "*What this means: each dot is one team. Points below the dashed "
            "line read fielding more precisely than batting — 28 of 30 teams "
            "do.*"
        )
        st.caption(
            "The 28-of-30 replication stands on its own. A further probe — "
            "whether teams with sharper fielding reads also *win* more of their "
            "challenges — is only weak, not-quite-significant "
            "(r = −0.35, p = 0.06): suggestive, and nothing the conclusion "
            "leans on either way. Full per-team figures, bootstrap confidence "
            "intervals, and the underlying tests are in "
            "scripts/team_skill_test.py and data/team_skill_test.parquet."
        )
        with st.expander("What do \"r\", \"p\", and \"confidence interval\" mean? (this page uses them a lot below)"):
            st.markdown(
                "Three numbers show up throughout the rest of this tab. None of "
                "them are as precise as they look — they're all here to answer "
                "one question: **how much should you trust this pattern?**\n\n"
                "- **Correlation (r)** runs from −1 to +1 and says how tightly "
                "two things move together. Near 0 means no relationship; near "
                "±1 means one reliably predicts the other. The r = −0.35 above "
                "is a mild lean, not a strong link.\n"
                "- **p-value (p)** is the probability of seeing a pattern at "
                "least this strong *if the null hypothesis were true* (i.e. if "
                "there were really nothing going on). It is **not** the "
                "probability that the null is true. Small (well under 0.05) "
                "means the data are hard to explain by chance alone; p = 0.06 "
                "above is right on that edge — suggestive, not confirmed.\n"
                "- **Confidence interval (CI)** is the range the true number "
                "probably falls in, given how much data went into it. A wide "
                "CI means the point estimate is shakier than it looks; a CI "
                "that spans zero means the data can't rule out \"no effect at "
                "all.\" When that lands on a *supporting* check — like the "
                "team-level reliability test just below — a null still carries "
                "information: it points toward the skill being a player's "
                "rather than a team's, though (see below) that reading now "
                "rests more on the catcher-quality and role-effect results "
                "than on the split-half contrast."
            )

        st.divider()
        st.markdown("##### Why the team-level test washes out")
        st.caption(
            "This is the supporting analysis the reframing above turns on — a "
            "null result here is the *evidence*, not a loose end. The headline "
            "decision gap does not depend on it."
        )

        sh = split_half.merge(team_sig_test[["team", "z", "p_bonferroni"]], on="team")
        r_val = np.corrcoef(sh.h1_rate, sh.h2_rate)[0, 1]

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("**Split-half reliability — team level**")
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
                f"*What this means: a team's first-half challenge success only "
                f"loosely predicts its second half — the dots don't hug the "
                f"diagonal (r = {r_val:.2f}, 95% CI crosses zero). The "
                f"player-level version below is somewhat stronger, but both "
                f"estimates have moved ~0.05 between data refreshes, so this "
                f"contrast is suggestive rather than settled.*"
            )
        with col2:
            st.markdown("**But the spread itself is real**")
            _outlier = (team_sig_test.loc[team_sig_test["z"].idxmax()]
                        if len(team_sig_test) else None)
            _outlier_txt = (
                f" The strongest single outlier, **{_outlier.team}**, sits "
                f"about {_outlier.z:.1f} standard deviations above the league "
                f"rate — well outside the team-to-team wobble that null "
                f"produces — with a Bonferroni-corrected p ≈ "
                f"{_outlier.p_bonferroni:.2f} for having checked all 30 teams."
                if _outlier is not None else "")
            st.markdown(
                f"Simulating 30 league-average teams at each team's real attempt "
                f"count: under that null, the *spread* in success rate at least "
                f"as large as the one actually observed is rare (p = 0.004), and "
                f"the same holds for runs gained (p < 0.0001).{_outlier_txt}"
            )
            st.markdown(
                "*So real variation exists in 2026 — the open question is only "
                "whether it's a **team** property or a **player** property. The "
                "next two analyses settle that: player, not team.*"
            )

        st.markdown("**The same test at the player level**")
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
            "Same split-half design, on individual challengers instead of "
            "teams. The load-bearing rows are the **8-** and **10-per-half** "
            "thresholds: **r ≈ 0.27–0.38**, both 95% intervals clearing zero — "
            "positive, but only the 10-per-half row is unambiguous, and the "
            "8-per-half row now sits about where the team-level estimate does. "
            "The 5-per-half row (r ≈ 0, too lax a bar) and the 15-per-half row "
            "(underpowered) bracket it as diagnostics.\n\n"
            "*The catch: both the player and team split-half numbers moved "
            "~0.05 toward each other on five days of additional games, so "
            "split-half reliability is poorly estimated with one partial "
            "season. It leans toward the skill being the player's, but the "
            "weight of the personnel reading is on the catcher-quality "
            "correlation and the 28/30 role-effect replication — not this "
            "contrast. What no single season settles: whether a front office "
            "can systematically build the skill in.*"
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
            f"caveat, not a cleared one. It's a caveat on the *ranking* only — "
            f"the player-skill reading rests mainly on the quality correlation "
            f"below and the role-effect replication, which survive it."
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

        # Data-driven: describe whichever catchers are actually the top-5
        # teams' primaries this refresh, widest CI first. No hardcoded names.
        _t5 = top5_catchers.dropna(subset=["ci_lo", "ci_hi"]).copy()
        if len(_t5):
            _t5["ci_span_pct"] = [
                _pct_equiv(r.ci_hi) - _pct_equiv(r.ci_lo) for r in _t5.itertuples()]
            _t5 = _t5.sort_values("ci_span_pct", ascending=False)
        _spans = "; ".join(
            f"{r.name} spans the {_pct_equiv(r.ci_lo):.0f}th–"
            f"{_pct_equiv(r.ci_hi):.0f}th percentile"
            for r in _t5.head(3).itertuples()) or "n/a in the current data"
        _firm = ", ".join(_t5.tail(2).name.tolist()) if len(_t5) >= 2 else "the rest"
        st.markdown(
            f"**How much to trust one catcher's rank:** a primary catcher gets "
            f"roughly 40–160 challenges of his own in a season — enough to see a "
            f"real signal, not enough to pin down a precise number. Converting "
            f"each top-5 team catcher's 95% CI back into where it would land in "
            f"the league distribution: {_spans} — several of these are wide "
            f"enough to include a below-average catcher. {_firm} sit on firmer "
            f"ground, but even the firmest low end is not always clearly above "
            f"average. This fits the player-level split-half reliability above "
            f"(r ≈ 0.27–0.38, and itself unstable across refreshes): individual "
            f"accuracy looks like real repeatable signal, but a noisy one — this "
            f"season's exact ranking of any one catcher would likely move some "
            f"by next season."
        )

        st.markdown("**Does a team's quality edge line up with its own catcher's accuracy? — the analysis that pins it to the catcher**")
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
        _q = cc_all[cc_all.top_team].sort_values("quality_ratio", ascending=False)
        _quality_led = _q[_q.quality_ratio > 1.05]
        _volume_led = _q[_q.quality_ratio <= 1.05]
        _ql = ", ".join(f"{r.name} ({r.team})" for r in _quality_led.itertuples()) or "none this refresh"
        _vl = ", ".join(f"{r.name} ({r.team})" for r in _volume_led.itertuples()) or "none this refresh"
        st.markdown(
            f"Every dot is one of the 30 primary catchers, not just the top 5: x is "
            f"how much of his team's edge is *quality* (success rate, and leverage — "
            f"how many runs are riding on the calls it wins — with attempts held "
            f"fixed) rather than volume; y is that same catcher's own "
            f"individual challenge success rate. They're correlated leaguewide "
            f"(r = {qual_r:.2f}, p = {qual_p:.3f}, n = {qual_n} teams) — not just a "
            f"pattern eyeballed in 5 points. Among the current top-5 runs-gained "
            f"teams, the ones whose edge is mostly **challenge quality** "
            f"(quality ratio > 1) are: {_ql}. The ones whose edge is mostly "
            f"**volume** (near-league quality, more attempts): {_vl}. The "
            f"distinction matters: a volume edge is a lineup-construction choice "
            f"a new front office could keep next season, while a quality edge "
            f"tied to one catcher is associated with that individual and would "
            f"move with him in a trade."
        )

        st.markdown(
            "##### What this section means\n"
            "The evidence leans toward challenge accuracy being a repeatable "
            "**player** skill — most clearly for **catchers** — rather than a "
            "front-office one. The strongest strands: a team's edge in "
            "challenging *well* tracks its own catcher's individual accuracy "
            "(r = 0.48, p = 0.007), and the role effect replicates in 28 of 30 "
            "teams. The split-half reliability test points the same way — "
            "positive on individuals, null on teams — but both halves of that "
            "contrast shifted materially on five days of new data, so it is "
            "treated here as support, not proof. That "
            "reframes the question a team should ask: not just *when* to "
            "challenge (the optimal-policy question this whole page answers) but "
            "*who* calls for one. A club with a sharp catcher has real signal "
            "to trust on close calls; one without is better off leaning on the "
            "same policy logic as everyone else.\n\n"
            "**None of this changes the headline.** The decision-gap number is "
            "arithmetic from the challenge rules and run expectancy; it holds "
            "whether or not challenge accuracy turns out to be a coachable team "
            "trait."
        )

    st.divider()
    st.caption(
        "Method: the live decision model uses a **2026-only** run-expectancy "
        "table (built from our own Statcast data, not `delta_run_exp`, which is "
        "count-based and strips base-out context), to price 2026 calls against "
        "the current run environment; the RE build script also pools 2024–2026 "
        "for validation against published RE24, and the geometry / measured-"
        "height work uses 2024–2026 games where a cross-season check helps. ABS "
        "zone geometry validated against MLB's own `edge_distance` at "
        "R² = 1.0000. Policy solved by backward induction over (half-inning × "
        "pitch × challenges spent); the continuation value exported to the app "
        "is summarized at the half-inning level (see the Decomposition method "
        "note). Caveat: the fitted σ absorbs any real variation in players' "
        "thresholds across counts, which inflates the information gap and "
        "deflates the decision gap — so the decision gap is a floor."
    )

    _policy_v = dec.model_version.iloc[0] if "model_version" in dec.columns else "unstamped"
    _policy_t = dec.generated_at.iloc[0] if "generated_at" in dec.columns else "unknown"
    _sigma_v = sigma.model_version.iloc[0] if "model_version" in sigma.columns else "unstamped"
    _sigma_t = sigma.generated_at.iloc[0] if "generated_at" in sigma.columns else "unknown"
    st.caption(
        f"Policy model: `{_policy_v}` (generated {_policy_t}) · "
        f"Perceptual-σ fit: `{_sigma_v}` (generated {_sigma_t})"
    )
    _rf = load_reported_figures()
    if _rf.get("data_through"):
        _run = str(_rf.get("generated_at", ""))[:10]
        st.caption(
            f"Data through **{_rf['data_through']}** (2026 regular season). "
            + (f"Pipeline last refreshed {_run}; " if _run else "")
            + f"{_rf.get('n_figures', 0)} headline figures are re-checked "
            "against the text of this page and the README on every refresh. "
            "The 2026 season is nearly over, but the pipeline is built to "
            "resume for 2027."
        )
    st.caption(
        "Built and analysed by **Kaushal Namuduri** — "
        "[source on GitHub](https://github.com/Cowshal/mlb-abs-challenge-analysis)"
    )
except Exception:
    # Shared setup (data loads, page header, tab creation, footer) failed --
    # nothing renders, so a page-level message is appropriate. The per-tab
    # guarded_section() handles isolated failures inside a single tab. Either
    # way the FULL traceback is logged to stderr / Streamlit Cloud logs.
    logger.exception("App failed to render (shared setup)")
    traceback.print_exc()
    st.error(
        "**Something didn't load correctly.** Please refresh the page. If the "
        "issue continues, the application may be temporarily unavailable."
    )
    with st.expander("Technical detail"):
        st.code(traceback.format_exc())
    st.stop()
