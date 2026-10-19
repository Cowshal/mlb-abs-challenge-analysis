"""Narrative claims must follow refreshed confidence intervals."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from skill_summary import interval_description, player_reliability_summary


@pytest.mark.parametrize("lo,hi,expected", [
    (-0.0418, 0.3157, "includes zero"),
    (0.0302, 0.4164, "is above zero"),
    (-0.4, -0.1, "is below zero"),
    (0.0, 0.2, "includes zero"),
    (-0.2, 0.0, "includes zero"),
])
def test_interval_description(lo, hi, expected):
    assert interval_description(lo, hi) == expected


def test_october_refresh_does_not_claim_both_player_intervals_clear_zero():
    # Logged by refresh run 39151946750, 2026-10-18.
    results = pd.DataFrame([
        dict(min_challenges_per_half=8, n_players=116, r=0.1416,
             ci_lo=-0.0418, ci_hi=0.3157),
        dict(min_challenges_per_half=10, n_players=93, r=0.2324,
             ci_lo=0.0302, ci_hi=0.4164),
    ])
    text = player_reliability_summary(results)
    assert "**r ≈ 0.14** (n = 116, 95% CI [-0.04, 0.32]); the interval includes zero" in text
    assert "**r ≈ 0.23** (n = 93, 95% CI [0.03, 0.42]); the interval is above zero" in text


def test_summary_updates_when_stronger_evidence_arrives():
    results = pd.DataFrame([
        dict(min_challenges_per_half=n, n_players=100, r=0.4, ci_lo=0.2, ci_hi=0.6)
        for n in (8, 10)
    ])
    text = player_reliability_summary(results)
    assert text.count("is above zero") == 2
    assert "0.40" in text
    assert "includes zero" not in text
