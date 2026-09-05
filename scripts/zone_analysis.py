"""
Does perceptual accuracy vary by pitch location, and does it vary DIFFERENTLY
by role? The policy model assumes one sigma per role, everywhere in the
zone -- a location-independent read. If batters and the battery (catchers +
pitchers) have different spatial blind spots, that assumption is wrong, and
it's worth knowing whether it's wrong by a little or a lot.

Location is expressed in two batter-relative axes, not raw plate_x/plate_z:
  - horizontal: "in" (near the batter's body) vs "away", derived by mirroring
    plate_x on batter stand. Verified against real data, not assumed: hit-by-pitch
    events (which by definition hit the batter's body) average plate_x = -1.93 ft
    for right-handed batters and +1.99 ft for left-handed batters, confirming
    inside = negative plate_x for a RHB and positive for a LHB.
  - vertical: fraction of the way from zone bottom to zone top (can be <0 or
    >1 for pitches outside the rule zone, which is most of this sample --
    these are all borderline challenges by construction).

Both are binned into thirds (in/middle/away x low/middle/high), the same 3x3
convention as a standard zone plot, giving 9 regions.

Test: does (fielding success rate - batting success rate) vary across the 9
regions more than sampling noise would produce? Fit as logistic regression
(overturned ~ role * region) via IRLS, likelihood-ratio test of the 8
interaction terms against a main-effects-only model.

Run: python scripts/zone_analysis.py
Output: data/zone_heatmap.parquet     (n, success rate, by role x region)
        data/zone_interaction.parquet (LR test result, one row)
        data/zone_sigma.parquet       (per-region-per-role sigma, IF the
                                        interaction test says it's worth it --
                                        see the printed decision at the end)
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from geometry import center_distance_to_zone, ball_edge_distance, HALF_WIDTH
from perception import fit_sigma

MODEL_VERSION = "zone_analysis_v1"
# "large enough to refit" bar: if the interaction test is significant AND the
# role gap swings by more than this many percentage points across regions,
# the single-sigma-per-role assumption is worth revisiting with real numbers
# rather than a footnote.
LARGE_SWING_PP = 10.0


def add_zone_regions(df, x_col="x_mid", z_col="z_mid", height_col="height_ft"):
    df = df.copy()
    top = 0.535 * df[height_col]
    bot = 0.270 * df[height_col]
    signed_inside = np.where(df.stand.values == "L", df[x_col].values, -df[x_col].values)
    frac_height = (df[z_col].values - bot.values) / (top.values - bot.values)

    df["signed_inside_ft"] = signed_inside
    df["frac_height"] = frac_height
    third = HALF_WIDTH / 3.0 * 1.5  # thirds of the zone width, in feet
    df["horiz_region"] = np.select(
        [signed_inside > third, signed_inside < -third],
        ["in", "away"], default="middle")
    df["vert_region"] = np.select(
        [frac_height > 2 / 3, frac_height < 1 / 3],
        ["high", "low"], default="middle")
    df["zone_region"] = df.vert_region + "-" + df.horiz_region
    return df


def fit_logistic(X, y, max_iter=100, tol=1e-9):
    """IRLS (Newton-Raphson) logistic regression. Returns (beta, log-likelihood)."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    return beta, ll


def interaction_test(df):
    """LR test: does the role gap in success rate vary by zone region?"""
    y = df.overturned.astype(float).values
    role = (df.challenger == "fielding").astype(float).values
    regions = sorted(df.zone_region.unique())
    region_dummies = np.column_stack(
        [(df.zone_region == r).astype(float).values for r in regions[1:]])  # drop 1 baseline

    intercept = np.ones(len(df))
    X_reduced = np.column_stack([intercept, role, region_dummies])
    interaction_cols = region_dummies * role[:, None]
    X_full = np.column_stack([X_reduced, interaction_cols])

    _, ll_reduced = fit_logistic(X_reduced, y)
    _, ll_full = fit_logistic(X_full, y)
    lr_stat = 2 * (ll_full - ll_reduced)
    df_diff = X_full.shape[1] - X_reduced.shape[1]
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    generated_at = datetime.now(timezone.utc).isoformat()

    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    act = opp[opp.was_challenged].copy()

    ch = pd.read_parquet("data/abs_challenges.parquet")
    ch["game_pk"] = ch.game_pk.astype(np.int64)
    ch["ab_number"] = ch.ab_number.astype(np.int64)
    ch["pitch_number"] = ch.pitch_number.astype(np.int64)

    act["game_pk"] = act.game_pk.astype(np.int64)
    act["at_bat_number"] = act.at_bat_number.astype(np.int64)
    act["pitch_number"] = act.pitch_number.astype(np.int64)
    act = act.merge(
        ch.rename(columns={"ab_number": "at_bat_number"})[
            ["game_pk", "at_bat_number", "pitch_number", "stand", "pitch_type"]],
        on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    print(f"{len(act):,} challenges with stand + pitch_type matched "
          f"(of {opp.was_challenged.sum():,} in the opportunity set)")

    act = add_zone_regions(act)
    FASTBALLS = {"FF", "SI", "FC"}
    act["pitch_group"] = np.where(act.pitch_type.isin(FASTBALLS), "fastball", "breaking/offspeed")

    print("\n=== SUCCESS RATE BY ROLE x ZONE REGION ===")
    heat = act.groupby(["challenger", "vert_region", "horiz_region"]).agg(
        n=("overturned", "size"), success_rate=("overturned", "mean")).reset_index()
    print(heat.sort_values(["challenger", "vert_region", "horiz_region"]).to_string(index=False))

    piv = heat.pivot_table(index=["vert_region", "horiz_region"],
                           columns="challenger", values="success_rate")
    piv["gap_fielding_minus_batting"] = piv.fielding - piv.batting
    n_piv = heat.pivot_table(index=["vert_region", "horiz_region"],
                             columns="challenger", values="n")
    piv["n_batting"], piv["n_fielding"] = n_piv.batting, n_piv.fielding
    print("\n=== ROLE GAP BY REGION (fielding - batting success rate) ===")
    print(piv.round(3).to_string())

    # The dead-center "middle-middle" cell has n=1-2 -- nobody challenges an
    # obviously-correct pitch, so its "gap" is pure noise, not signal. Report
    # the swing only over regions with enough data in BOTH roles to trust.
    MIN_N = 30
    robust = piv[(piv.n_batting >= MIN_N) & (piv.n_fielding >= MIN_N)]
    swing_pp = (robust.gap_fielding_minus_batting.max()
                - robust.gap_fielding_minus_batting.min()) * 100
    excluded = piv[(piv.n_batting < MIN_N) | (piv.n_fielding < MIN_N)]
    if len(excluded):
        print(f"\nexcluded {len(excluded)} region(s) with <{MIN_N} challenges in either "
              f"role from the swing calculation (unreliable, not evidence):")
        print(excluded[["n_batting", "n_fielding", "gap_fielding_minus_batting"]]
              .round(3).to_string())
    print(f"\nswing in the role gap across well-populated regions "
          f"(n>={MIN_N} each): {swing_pp:.1f} percentage points")

    print("\n=== SUCCESS RATE BY ROLE x PITCH GROUP ===")
    pg = act.groupby(["challenger", "pitch_group"]).agg(
        n=("overturned", "size"), success_rate=("overturned", "mean")).reset_index()
    print(pg.to_string(index=False))

    print("\n=== INTERACTION TEST: does the role gap vary by zone region? ===")
    robust_regions = set(zip(robust.index.get_level_values("vert_region"),
                             robust.index.get_level_values("horiz_region")))
    act_robust = act[act.apply(
        lambda r: (r.vert_region, r.horiz_region) in robust_regions, axis=1)]
    print(f"restricted to the {len(robust_regions)} well-populated regions "
          f"({len(act_robust):,} of {len(act):,} challenges)")
    lr_stat, df_diff, p_value = interaction_test(act_robust)
    print(f"LR statistic = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.4f}")

    heat["model_version"] = MODEL_VERSION
    heat["generated_at"] = generated_at
    heat.to_parquet("data/zone_heatmap.parquet", index=False)

    interaction = pd.DataFrame([{
        "lr_stat": lr_stat, "df": df_diff, "p_value": p_value,
        "swing_pp": swing_pp, "n_challenges": len(act),
        "model_version": MODEL_VERSION, "generated_at": generated_at,
    }])
    interaction.to_parquet("data/zone_interaction.parquet", index=False)
    print("\nsaved data/zone_heatmap.parquet, data/zone_interaction.parquet")

    is_large = (p_value < 0.05) and (swing_pp > LARGE_SWING_PP)
    print(f"\n=== DECISION: is this large enough to refit sigma by zone? ===")
    print(f"significant (p<0.05): {p_value < 0.05}   "
          f"swing > {LARGE_SWING_PP}pp: {swing_pp > LARGE_SWING_PP}")
    print("-> " + ("YES: proceeding to per-region sigma fit and decomposition re-run "
                   "(see scripts/zone_sigma_refit.py)" if is_large else
                   "NO: reporting as a location-independence limitation, not refitting. "
                   "One sigma per role stands as the headline model."))


if __name__ == "__main__":
    main()
