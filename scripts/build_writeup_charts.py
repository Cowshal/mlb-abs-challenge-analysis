"""
Render static chart images for docs/writeup.md from the same precomputed data
the app uses, styled with the same color encoding (gray=observed, blue=optimal,
amber=ceiling) so the writeup and the app tell a visually consistent story.

Run: python scripts/build_writeup_charts.py
Output: docs/images/*.png
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from abs_policy import reach_probabilities, run_scenario, threshold
from geometry import center_distance_to_zone, ball_edge_distance

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

COLOR_OBSERVED = "#64748B"
COLOR_OPTIMAL = "#2563EB"
COLOR_CEILING = "#F59E0B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CBD5E1",
    "text.color": "#0F172A",
    "axes.labelcolor": "#0F172A",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def fig1_decomposition():
    dec = pd.read_parquet("app/data/policy_decomposition.parquet")
    labels = {"observed 2026": "Observed\n(2026)",
              "optimal @ player sigma": "Optimal\n(same information)",
              "ceiling @ sigma=0.5in": "Ceiling\n(perfect information)"}
    colors = {"observed 2026": COLOR_OBSERVED, "optimal @ player sigma": COLOR_OPTIMAL,
              "ceiling @ sigma=0.5in": COLOR_CEILING}
    order = ["ceiling @ sigma=0.5in", "optimal @ player sigma", "observed 2026"]
    dec = dec.set_index("label").loc[order]

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bars = ax.barh([labels[i] for i in order], dec.runs_per_team_game,
                   color=[colors[i] for i in order], height=0.6)
    for bar, val in zip(bars, dec.runs_per_team_game):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=11, color="#0F172A")
    ax.set_xlabel("Runs per team, per game")
    ax.set_xlim(0, 0.72)
    ax.set_title("Where the gap comes from", fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "decomposition_bars.png", dpi=200)
    plt.close(fig)


def fig2_threshold_curve():
    con = duckdb.connect("data/baseball.duckdb")
    allp = con.execute("SELECT game_pk, inning, inning_topbot FROM statcast WHERE game_year=2026").df()
    q = reach_probabilities(allp)
    opp = pd.read_parquet("data/challenge_opportunities.parquet")
    opp["d"] = ball_edge_distance(center_distance_to_zone(
        opp.x_mid.values, opp.z_mid.values, opp.height_ft.values))
    opp["t"] = (opp.inning - 1) * 2 + np.where(opp.inning_topbot == "Bot", 2, 1)
    sig = pd.read_parquet("data/perception_sigma.parquet")
    player_sigma = dict(zip(sig.side, sig.sigma_ft))
    _, ov, _, _ = run_scenario(opp, player_sigma, q, label="optimal @ player sigma")
    start = ov[ov.t == 1].iloc[0]

    dre_grid = np.linspace(0.01, 1.2, 300)
    p_star = threshold(start.C_k0, dre_grid)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(dre_grid, p_star * 100, color=COLOR_OPTIMAL, linewidth=3)
    for dre, label in ((0.62, "Full count,\nrunners on"), (0.05, "0-0 take,\nbases empty")):
        p = threshold(start.C_k0, dre) * 100
        note = f"{label}\n~{p:.0f}%"
        ax.plot([dre], [p], "o", color=COLOR_OPTIMAL, markersize=8, zorder=5)
        ax.annotate(note, (dre, p), textcoords="offset points",
                    xytext=(12, 14) if dre > 0.3 else (12, 10),
                    fontsize=10, color="#0F172A")
    ax.set_xlabel("Value of the call, in runs (ΔRE)")
    ax.set_ylabel("Minimum confidence to challenge")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x)}%")
    ax.set_title("The more a call is worth, the less sure you need to be",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_curve.png", dpi=200)
    plt.close(fig)


def fig3_sigma_by_role():
    sig = pd.read_parquet("app/data/perception_sigma.parquet")
    labels = {"batting": "Batters", "fielding": "Catchers &\npitchers"}
    sig = sig.set_index("side").loc[["batting", "fielding"]]
    colors = [COLOR_OBSERVED, COLOR_OPTIMAL]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar([labels[i] for i in sig.index], sig.sigma_in, color=colors, width=0.55)
    for bar, val in zip(bars, sig.sigma_in):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f} in",
                ha="center", fontsize=11, color="#0F172A")
    ax.set_ylabel("Read precision, σ (inches)")
    ax.set_ylim(0, 3.2)
    ax.set_title("Catchers read the pitch more precisely",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "sigma_by_role.png", dpi=200)
    plt.close(fig)


def fig4_runs_left_teams():
    pt = pd.read_parquet("app/data/per_team.parquet").sort_values(
        "runs_left_on_table", ascending=False).head(10)
    pt = pt.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(pt.team, pt.runs_left_on_table, color=COLOR_OPTIMAL, height=0.6)
    ax.set_xlabel("Runs left on the table (2026 season)")
    ax.set_title("Ten teams with the most to gain from challenging smarter",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "runs_left_teams.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_decomposition()
    fig2_threshold_curve()
    fig3_sigma_by_role()
    fig4_runs_left_teams()
    for f in sorted(OUT.glob("*.png")):
        print(f"wrote {f} ({f.stat().st_size/1024:.0f} KB)")
