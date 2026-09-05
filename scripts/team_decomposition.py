"""
Multiplicative decomposition of each team's observed runs gained (2026) against
league baseline: runs_gained = attempts x success_rate x mean_stake_per_overturn.

    attempts_ratio  = team_attempts / league_avg_attempts_per_team
    success_ratio   = team_success_rate / league_success_rate
    leverage_ratio  = team_mean_stake_per_overturn / league_mean_stake_per_overturn
    total_ratio     = attempts_ratio * success_ratio * leverage_ratio
                     = team_actual_runs / (league_avg_attempts_per_team * league_success * league_stake)

total_ratio compares a team to a league-average team with league-average
volume. quality_ratio (= success_ratio * leverage_ratio, i.e. total_ratio with
attempts_ratio divided back out) compares a team to a league-average team
GIVEN THIS TEAM'S OWN ATTEMPT COUNT -- it isolates "were the challenges good"
from "were there a lot of them."

Run: python scripts/team_decomposition.py
Output: data/team_decomposition.parquet
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)


def main():
    pt = pd.read_parquet("app/data/per_team.parquet").copy()
    # Inherit per_team.parquet's own stamp rather than mint a new one: this
    # script is a pure re-derivation of per_team's numbers (ratios against a
    # league baseline), not an independent model, so it should carry the same
    # provenance as its one input rather than look like a separate pipeline.
    model_version = pt.model_version.iloc[0] if "model_version" in pt.columns else "unstamped"
    generated_at = pt.generated_at.iloc[0] if "generated_at" in pt.columns else None

    league_attempts_per_team = pt.actual_challenges.sum() / len(pt)
    league_success = pt.actual_correct.sum() / pt.actual_challenges.sum()
    league_stake = pt.actual_runs.sum() / pt.actual_correct.sum()
    print(f"league: {league_attempts_per_team:.2f} attempts/team, "
          f"{league_success:.4f} success rate, {league_stake:.4f} runs/overturn")

    pt["team_stake"] = pt.actual_runs / pt.actual_correct
    pt["attempts_ratio"] = pt.actual_challenges / league_attempts_per_team
    pt["success_ratio"] = pt.actual_success / league_success
    pt["leverage_ratio"] = pt.team_stake / league_stake
    pt["total_ratio"] = pt.attempts_ratio * pt.success_ratio * pt.leverage_ratio
    pt["quality_ratio"] = pt.success_ratio * pt.leverage_ratio  # attempts-neutral
    baseline_at_volume = pt.actual_challenges * league_success * league_stake
    pt["baseline_runs_at_volume"] = baseline_at_volume

    # log-additive share of the *quality* edge (success vs leverage) -- how
    # much of "why is this team's quality ratio not 1.0" is success vs stake
    log_succ = np.log(pt.success_ratio)
    log_lev = np.log(pt.leverage_ratio)
    denom = log_succ.abs() + log_lev.abs()
    pt["success_share"] = np.where(denom > 0, log_succ.abs() / denom, np.nan)
    pt["leverage_share"] = np.where(denom > 0, log_lev.abs() / denom, np.nan)

    out = pt[["team", "actual_challenges", "actual_success", "team_stake",
              "attempts_ratio", "success_ratio", "leverage_ratio", "total_ratio",
              "quality_ratio", "baseline_runs_at_volume", "actual_runs",
              "success_share", "leverage_share"]].sort_values(
        "total_ratio", ascending=False)
    out["model_version"] = model_version
    out["generated_at"] = generated_at
    out.to_parquet("data/team_decomposition.parquet", index=False)

    print("\n=== CIN check against the manual read ===")
    cin = out[out.team == "CIN"].iloc[0]
    print(f"attempts={cin.actual_challenges:.0f}  attempts_ratio={cin.attempts_ratio:.3f}")
    print(f"success={cin.actual_success:.3f}  success_ratio={cin.success_ratio:.3f}")
    print(f"stake={cin.team_stake:.3f}  leverage_ratio={cin.leverage_ratio:.3f}")
    print(f"quality_ratio (success x leverage) = {cin.quality_ratio:.3f}")
    print(f"baseline_runs_at_volume = {cin.baseline_runs_at_volume:.1f}")
    print(f"success_share={cin.success_share:.1%}  leverage_share={cin.leverage_share:.1%}")
    print(f"actual_runs={cin.actual_runs:.1f}")

    print("\n=== full table, sorted by total_ratio ===")
    disp = out.copy()
    for c in ("actual_success", "success_ratio", "leverage_ratio", "attempts_ratio",
              "total_ratio", "quality_ratio", "success_share", "leverage_share"):
        disp[c] = disp[c].round(3)
    disp["team_stake"] = disp.team_stake.round(3)
    disp["baseline_runs_at_volume"] = disp.baseline_runs_at_volume.round(1)
    disp["actual_runs"] = disp.actual_runs.round(1)
    print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
