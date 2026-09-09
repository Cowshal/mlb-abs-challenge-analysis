"""
Centralized ABS challenge-rule state transitions and the break-even decision
arithmetic.

Every part of this project that reasons about "how many challenges does this
team still have" or "is firing here worth it" goes through these helpers, so
the rule can never again be implemented two different ways in two files (it
was: see the extra-inning note below).

State convention used everywhere in the codebase:

    k = number of INCORRECT challenges already spent this game, in {0, 1, 2}
    challenges_remaining = MAX_CHALLENGES_START - k     (2, 1, 0)

Only an *incorrect* challenge costs a token; a correct one is returned
immediately. Rights are gone at k = 2 (two incorrect challenges).

This module is intentionally pure Python (no numpy / pandas) so it is cheap
to import anywhere, including the deployed Streamlit app.
"""

MAX_CHALLENGES_START = 2


def challenges_remaining(k):
    """Challenges the team can still use, given k incorrect ones spent."""
    return max(0, MAX_CHALLENGES_START - int(k))


def extra_inning_k_transition(k):
    """
    MLB rule (2026): a team receives ONE additional challenge at the start of
    an extra inning ONLY if it enters that inning with zero challenges
    remaining. A team that still holds one or two challenges receives nothing.

    In k-space (k = incorrect challenges already spent):

        k = 0  (2 remaining) -> 0     no bonus
        k = 1  (1 remaining) -> 1     no bonus
        k = 2  (0 remaining) -> 1     one challenge restored

    So k is pulled back to 1 iff the team was exhausted, and is otherwise
    unchanged.

    The earlier implementation used ``max(k - 1, 0)``, which also (wrongly)
    handed a token back to a team sitting on k = 1, giving it a second
    challenge it is not entitled to. This helper is the single source of
    truth for the transition; call it, do not re-derive it.
    """
    k = int(k)
    return 1 if k >= MAX_CHALLENGES_START else k


def break_even_probability(dre, C):
    """
    Minimum confidence that the call is wrong which justifies challenging:

        p* = C / (dre + C)

    Derived from the indifference condition  p*.dre == (1 - p*).C, i.e. the
    expected run gain from a correct overturn equals the expected run cost of
    burning a token on a wrong one. ``C`` is the option value of holding one
    incorrect-challenge token (runs); ``dre`` is the run swing if the call
    flips. Because a correct challenge is free, the cost side carries the
    factor (1 - p*), which is why p* usually sits well below 0.5.

    Returns nan for a degenerate (dre + C <= 0) input.
    """
    denom = dre + C
    if denom <= 0:
        return float("nan")
    return C / denom


def expected_challenge_value(p_wrong, dre, C):
    """
    Expected runs from firing the challenge now versus holding it:

        EV = p_wrong * dre - (1 - p_wrong) * C

    Positive means challenging beats holding in expectation.
    """
    return p_wrong * dre - (1.0 - p_wrong) * C


def recommend(p_wrong, dre, C, k=0):
    """
    Convenience wrapper returning (decision, break_even, ev_net):

        decision   : "CHALLENGE" if p_wrong > p*, else "HOLD"
        break_even : p* = C / (dre + C)
        ev_net     : expected_challenge_value(p_wrong, dre, C)

    When rights are exhausted (k >= MAX_CHALLENGES_START) the decision is
    "HOLD" unconditionally and there is no option value left to risk.
    """
    if int(k) >= MAX_CHALLENGES_START:
        return "HOLD", float("nan"), 0.0
    p_star = break_even_probability(dre, C)
    ev = expected_challenge_value(p_wrong, dre, C)
    decision = "CHALLENGE" if (p_wrong > p_star) else "HOLD"
    return decision, p_star, ev
