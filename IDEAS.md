# v2 ideas

Deliberately out of scope for v1. v1 ships first.

## Modelling

**Count-varying challenge thresholds.** The perceptual sigma is currently fitted
with a single cutoff per role. Real players almost certainly vary their threshold
by count and leverage, and that heterogeneity is indistinguishable from noise to
the current estimator — so it inflates sigma. Because a larger sigma enlarges the
information gap and shrinks the decision gap, **the reported ~10 runs/team-season
decision gap is a floor and the information gap a ceiling.** Letting the cutoff
vary by count would tighten both bounds. This is the single highest-value
refinement.

**Win probability instead of run expectancy (top item).** The value function is
run-denominated, so it is indifferent to score and inning. A challenge in a
9-run blowout scores identically to one in a tie game, which is right in runs and
wrong in wins. Swapping the RE surface for a WP surface changes the objective,
not the machinery — the backward induction and the `p* = C / (Δ + C)` threshold
carry over unchanged, and `src/challenge_rules.py` is already written against a
generic objective delta Δ (runs today; win-probability later) rather than ΔRE
specifically. A WP version would condition on inning, score differential, outs,
runners, count, and challenges remaining. The blocker is having a *validated*
win-probability model — not the decision engine, which is ready for it.

**Position-dependent continuation value.** Today the option value `C(t, k)` is
evaluated once per half-inning and reused for every opportunity in it (a
half-inning-level approximation; see README Method). A cleaner version would
carry `C(t, j, k)` — or at least condition on outs / opportunities remaining in
the current half-inning. Second order (most of a token's value is future
half-innings), but worth doing so the deployed model matches the `V(t, j, k)`
description exactly rather than approximately.

**Two-player game.** Currently each team's challenge budget is an independent
single-agent MDP. In reality a team might challenge more freely once the opponent
is out of challenges. Probably small, but untested.

**Bound the tracking-precision ceiling.** The information gap depends entirely on
an assumed ceiling sigma, and this dataset cannot measure it — we only ever
observe Hawk-Eye's own output, never independent ground truth (our R² = 1.0000
fit against `edge_distance` measures agreement of algebra, not tracking accuracy).
Published Hawk-Eye specs or a calibration study would collapse the sensitivity
curve to a point.

**Per-pitcher framing effects.** Does a catcher challenge differently behind a
pitcher with a reputation for edge command?

## Data

**Extend measured heights.** 364 batters have a backed-out measured height, 203
with >= 3 challenges behind them. Another season of challenges would cover most
of the league and remove the listed-height fallback entirely.

**Backfill 2026 games before 2026-03-26.** The ingest window starts at the season
opener but the very first ABS challenge (2026-03-25, Caballero) sits outside it.

**Catcher and pitcher identity on fielding challenges.** Currently fielding
challenges are attributed to the team, not the individual. `challenging_player_id`
is in the feed — the per-player view could cover catchers, which is where the
skill story probably is.

## App

**Per-player leaderboard for catchers**, not just batters.

**Pitch-level explorer**: pick a game, see every called pitch with its `p`, its
`ΔRE`, the threshold, and whether the optimal policy would have fired.

**Nightly refresh** via GitHub Actions (`--incremental --days 3`, since Statcast
revises recent data).

**Port to Shiny.** Explicitly named in Mariners baseball-ops postings.
