"""
Regression tests for the challenge-token dynamics in src/abs_policy.py that the
extra-inning-rule fix touches: successful challenges are free, failed ones cost
exactly one token, rights are gone at two, and the extra-inning restore only
helps a team that enters exhausted.

Run: pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from abs_policy import simulate, solve_half_inning, threshold  # noqa: E402
from challenge_rules import break_even_probability  # noqa: E402


def _ov(n_t=20, c0=0.10, c1=0.18):
    """Flat option-value table: C is constant across half-innings so these
    tests isolate the token accounting, not the DP's continuation values."""
    return pd.DataFrame(
        {"t": list(range(1, n_t + 1)), "C_k0": c0, "C_k1": c1}
    )


def _opp_row(t, inning, half, ab, pitch, won, p_post=0.99, dre=1.0):
    return {
        "game_pk": 1, "inning": inning, "inning_topbot": half,
        "at_bat_number": ab, "pitch_number": pitch,
        "challenger": "batting", "p_post": p_post, "dre": dre,
        "won_if_challenged": won, "batter": 100, "t": t,
    }


def _simulate(rows):
    opp = pd.DataFrame(rows)
    sim, fires = simulate(opp, _ov(), seed=0)
    assert len(sim) == 1
    return sim.iloc[0], fires


# --------------------------------------------------------------------------- #
# 5. A successful (correct) challenge does not consume a token.               #
# --------------------------------------------------------------------------- #
def test_correct_challenges_are_free():
    rows = [_opp_row(t=1, inning=1, half="Top", ab=i, pitch=1, won=True)
            for i in range(1, 8)]
    s, _ = _simulate(rows)
    assert s.used == 7 and s.correct == 7
    assert s.runs == pytest.approx(7.0)  # dre=1.0 each, all won


# --------------------------------------------------------------------------- #
# 6. A failed challenge consumes exactly one token; rights end at two.        #
# --------------------------------------------------------------------------- #
def test_failed_challenges_cost_one_each_and_rights_end_at_two():
    rows = [_opp_row(t=1, inning=1, half="Top", ab=i, pitch=1, won=False)
            for i in range(1, 6)]  # five wrong firing chances in regulation
    s, _ = _simulate(rows)
    assert s.used == 2          # third through fifth are blocked (k >= 2)
    assert s.correct == 0
    assert s.runs == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 1 / 7. Extra-inning restore only helps a team that enters exhausted.        #
#    A team that burned exactly ONE challenge in regulation must still be at  #
#    one remaining in extras -- not two. The old max(k-1, 0) rule handed it   #
#    a second one back, which this test pins against.                         #
# --------------------------------------------------------------------------- #
def test_extra_inning_does_not_refund_a_team_that_still_holds_one():
    rows = [
        _opp_row(t=1, inning=1, half="Top", ab=1, pitch=1, won=False),   # burn #1 -> k=1
        _opp_row(t=19, inning=10, half="Top", ab=2, pitch=1, won=False),  # extras
        _opp_row(t=19, inning=10, half="Top", ab=3, pitch=1, won=False),
        _opp_row(t=19, inning=10, half="Top", ab=4, pitch=1, won=False),
    ]
    s, _ = _simulate(rows)
    # correct rule: enters 10th with 1 in hand (no refund) -> exactly one more
    # firing, total 2. buggy max(k-1,0) rule: refunded to k=0 -> two more, total 3.
    assert s.used == 2


def test_extra_inning_does_refund_a_team_that_enters_exhausted():
    rows = [
        _opp_row(t=1, inning=1, half="Top", ab=1, pitch=1, won=False),    # k=1
        _opp_row(t=3, inning=2, half="Top", ab=2, pitch=1, won=False),    # k=2 (exhausted)
        _opp_row(t=5, inning=3, half="Top", ab=3, pitch=1, won=False),    # blocked
        _opp_row(t=19, inning=10, half="Top", ab=4, pitch=1, won=False),  # refund -> k=1, fires
        _opp_row(t=19, inning=10, half="Top", ab=5, pitch=1, won=False),  # k=2 again, blocked
    ]
    s, _ = _simulate(rows)
    assert s.used == 3  # two in regulation + one enabled by the extra-inning refund


# --------------------------------------------------------------------------- #
# 7. solve_half_inning boundary + monotonicity.                              #
# --------------------------------------------------------------------------- #
def test_solve_half_inning_empty_returns_continuation_unchanged():
    W_next = np.array([0.30, 0.19, 0.0])
    out = solve_half_inning([], W_next)
    assert out[0] == pytest.approx(0.30)
    assert out[1] == pytest.approx(0.19)
    assert out[2] == pytest.approx(0.0)


def test_solve_half_inning_value_is_nondecreasing_in_challenges_held():
    # More challenges in hand is never worth less: V(k=0) >= V(k=1) >= V(k=2).
    W_next = np.array([0.25, 0.15, 0.0])
    opps = np.array([[0.6, 0.3], [0.4, 0.5], [0.55, 0.2]])
    out = solve_half_inning(opps, W_next)
    assert out[0] >= out[1] >= out[2]


# --------------------------------------------------------------------------- #
# threshold() must stay array-safe: it is called with a numpy grid of dre     #
# values to plot the break-even curve (scripts/build_writeup_charts.py).      #
# It agrees with the scalar break_even_probability for well-posed inputs.     #
# --------------------------------------------------------------------------- #
def test_threshold_accepts_a_numpy_array_of_dre():
    C = 0.10
    dre = np.array([0.02, 0.10, 0.25, 0.60, 1.20])
    out = threshold(C, dre)
    assert out.shape == dre.shape
    assert np.all((out > 0) & (out <= 1))
    # monotone decreasing in dre
    assert np.all(np.diff(out) < 0)


def test_threshold_matches_break_even_probability_for_scalars():
    for C, dre in [(0.10, 0.20), (0.05, 0.60), (0.18, 0.03)]:
        assert threshold(C, dre) == pytest.approx(break_even_probability(dre, C))
