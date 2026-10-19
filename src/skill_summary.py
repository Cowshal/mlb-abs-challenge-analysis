"""App copy derived from the same reliability results as the displayed table."""


def interval_description(lo: float, hi: float) -> str:
    if lo > 0:
        return "is above zero"
    if hi < 0:
        return "is below zero"
    return "includes zero"


def player_reliability_summary(results) -> str:
    rows = results.set_index("min_challenges_per_half")
    sentences = []
    for threshold in (8, 10):
        row = rows.loc[threshold]
        sentences.append(
            f"At **≥{threshold} challenges per half**, **r ≈ {row.r:.2f}** "
            f"(n = {int(row.n_players)}, 95% CI [{row.ci_lo:.2f}, "
            f"{row.ci_hi:.2f}]); the interval "
            f"{interval_description(row.ci_lo, row.ci_hi)}."
        )
    return " ".join(sentences)
