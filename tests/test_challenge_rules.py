"""
Unit tests for the centralized challenge-rule helpers.

Run: pytest -q
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from challenge_rules import (  # noqa: E402
    MAX_CHALLENGES_START,
    break_even_probability,
    challenges_remaining,
    expected_challenge_value,
    extra_inning_k_transition,
    recommend,
)


# --------------------------------------------------------------------------- #
# 1. Extra-inning challenge reset                                             #
#    MLB rule: a team gets one challenge back at the start of an extra inning #
#    ONLY if it enters with zero remaining. k = incorrect challenges spent.   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k_in, k_out", [(0, 0), (1, 1), (2, 1)])
def test_extra_inning_transition(k_in, k_out):
    assert extra_inning_k_transition(k_in) == k_out


def test_extra_inning_transition_is_idempotent_within_an_inning():
    # Applying it twice for the same inning boundary must not restore twice.
    assert extra_inning_k_transition(extra_inning_k_transition(2)) == 1


def test_extra_inning_never_creates_a_third_challenge():
    for k in (0, 1, 2):
        assert challenges_remaining(extra_inning_k_transition(k)) <= MAX_CHALLENGES_START


# --------------------------------------------------------------------------- #
# 4. Challenges remaining is bounded to [0, MAX]                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k, remaining", [(0, 2), (1, 1), (2, 0), (3, 0), (5, 0)])
def test_challenges_remaining_bounds(k, remaining):
    assert challenges_remaining(k) == remaining
    assert 0 <= challenges_remaining(k) <= MAX_CHALLENGES_START


# --------------------------------------------------------------------------- #
# 2. Break-even probability  p* = C / (dre + C)                               #
# --------------------------------------------------------------------------- #
def test_break_even_formula_matches_closed_form():
    assert break_even_probability(0.20, 0.05) == pytest.approx(0.05 / 0.25)


def test_break_even_is_below_half_when_cost_below_stake():
    # A correct challenge is free, so whenever the run swing exceeds the token
    # cost the break-even sits under 50%.
    assert break_even_probability(0.30, 0.10) < 0.5


def test_break_even_decreases_as_stake_grows():
    p_low_stake = break_even_probability(0.05, 0.10)
    p_high_stake = break_even_probability(0.60, 0.10)
    assert p_high_stake < p_low_stake


def test_break_even_degenerate_input_is_nan():
    assert math.isnan(break_even_probability(0.0, 0.0))
    assert math.isnan(break_even_probability(-0.10, 0.05))


# --------------------------------------------------------------------------- #
# 3. Recommendation classification                                           #
# --------------------------------------------------------------------------- #
def test_recommend_high_p_high_stake_challenges():
    decision, p_star, ev = recommend(p_wrong=0.85, dre=0.40, C=0.10, k=0)
    assert decision == "CHALLENGE"
    assert ev > 0


def test_recommend_low_p_low_stake_holds():
    decision, p_star, ev = recommend(p_wrong=0.20, dre=0.05, C=0.10, k=0)
    assert decision == "HOLD"
    assert ev < 0


def test_recommend_large_stake_justifies_lower_confidence():
    # Same modest confidence; only the stake changes. Low stake -> HOLD,
    # high stake -> CHALLENGE.
    low = recommend(p_wrong=0.35, dre=0.04, C=0.10, k=0)
    high = recommend(p_wrong=0.35, dre=1.00, C=0.10, k=0)
    assert low[0] == "HOLD"
    assert high[0] == "CHALLENGE"


def test_recommend_holds_unconditionally_when_rights_exhausted():
    decision, p_star, ev = recommend(p_wrong=0.99, dre=5.0, C=0.10,
                                     k=MAX_CHALLENGES_START)
    assert decision == "HOLD"
    assert ev == 0.0


def test_expected_value_sign_agrees_with_break_even():
    C, dre = 0.08, 0.25
    p_star = break_even_probability(dre, C)
    assert expected_challenge_value(p_star + 0.05, dre, C) > 0
    assert expected_challenge_value(p_star - 0.05, dre, C) < 0
