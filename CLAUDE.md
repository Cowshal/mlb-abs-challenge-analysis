# CLAUDE.md
## ⚠️ Rules for editing this file

This file is project memory. It is read at the start of every session and
is expensive to reconstruct.

- NEVER write this file in full. Use targeted edits that change only the
  lines you intend to change.
- NEVER shorten, summarize, condense, or "clean up" existing sections.
  Length is not a problem here.
- To add information, append a new section or add lines to an existing one.
- Before editing, `git diff CLAUDE.md` afterward and confirm the change is
  additive. If the diff shows removed lines you did not intend to remove,
  revert with `git checkout -- CLAUDE.md` and redo the edit.
- Commit this file after every change: `git add CLAUDE.md && git commit -m "context: <what changed>"`
## Project

MLB ABS (Automated Ball-Strike) challenge policy analysis. 2026 is the first
season with the challenge system. Goal: build a model of the *optimal* challenge
policy, compare it to how players actually behave, and quantify how many runs
teams leave on the table. Deploy as a public web app.

This is a portfolio project for baseball operations internship applications.
That means: clean repo, readable code, a real README, and a deployed link matter
as much as the analysis being correct.

## Current status

- [ ] Part 0: Statcast pull + DuckDB load
- [x] Step 1.1: locate pitch-level ABS challenge data (Savant `gf` feed confirmed
      primary source; see Data sources below)
- [ ] Step 1.2: plate-midpoint geometry + ABS zone model (ball-radius correction
      confirmed and implemented in `src/geometry.py`; measured-height back-out
      done for 364 batters via `scripts/verify_ball_radius.py` ->
      `data/measured_heights.parquet`; still need to wire this into the full
      pitch-level classification pipeline)
- [ ] Step 1.3: run expectancy by count and base-out state
- [ ] Step 1.4: decision model (backward induction)
- [ ] Step 1.5: findings + charts
- [ ] Part 4: deploy

Update this checklist as things land.

## Key domain facts (don't re-derive these)

**ABS zone geometry:**
- Width: 17 inches, same as home plate
- Top: 53.5% of the batter's *measured* height without cleats
- Bottom: 27% of measured height
- Location captured as the ball crosses the **middle** of the plate, NOT the front
- **Ball-radius rule (confirmed 2026-09-04):** ABS applies "any part of the ball
  over the zone." A trajectory solve gives the ball's CENTER; MLB's edge_distance
  is measured from the ball's EDGE. The two differ by exactly one ball radius,
  `BALL_RADIUS_FT = 0.1208` (2.9" diameter ball). Verified via regression on
  1,136 side-bound (height-independent) real challenges from 2026-04-01 to
  2026-05-15: slope = 0.9999, intercept = 0.1208 ft = 1.450 in, R^2 = 1.0000,
  mean residual = 0.000 in. Without this correction, naive center-based zone
  classification misclassifies inside/outside on ~27% of borderline challenges
  (3 of 11 in the first small sample); with it, 11/11. Implemented as
  `center_distance_to_zone()` + `ball_edge_distance()` in `src/geometry.py`
  (kept as two separate named outputs — do not collapse them).

**Statcast coordinate system:**
- `y = 0` is the back point of home plate
- `plate_x` / `plate_z` are reported at the FRONT of the plate (`y = 17/12 ft`)
- Middle of plate is `y = 8.5/12 ft` — must re-solve the trajectory to get there
- Use `x0,y0,z0 / vx0,vy0,vz0 / ax,ay,az` with constant-acceleration kinematics

**CORRECTION (2026-09-05) — the two bullets above about `plate_x` are WRONG.**
Measured, not assumed: solving the trajectory to `y = 8.5/12` reproduces
`plate_x`/`plate_z` to **0.0000 inches** over 940 pitches, while `y = 17/12`
(front) is off by 0.25 in horizontally and 0.98 in vertically, and `y = 0`
(back) is off by a similar amount. So **`plate_x`/`plate_z` are already the
MIDPLATE location** — exactly where ABS judges. Also true on 2024 games, so
it is a long-standing convention, not an ABS-era change. Consequences:
- The trajectory re-solve (`location_at_midplate`) is correct but redundant;
  `plate_x`/`plate_z` can be used directly. It was still worth building, since
  it is what proved the equivalence and it validated at R^2 = 1.0000 against
  MLB's own `edge_distance`.
- This matters practically: the bulk Statcast CSV (`pybaseball.statcast()`)
  exports `vx0/vy0/vz0` and `ax/ay/az` but **no `x0/y0/z0` anchor**, so the
  re-solve is impossible there. `plate_x`/`plate_z` are the only route, and
  they are the right answer anyway.
- The CSV's `plate_x`/`plate_z` are byte-identical to the Savant `gf` feed's,
  and CSV `at_bat_number` matches gf `ab_number` with no offset.

**Challenge rules (these drive the decision model):**
- Two challenges per team to start the game
- A CORRECT challenge is retained — it costs nothing
- Rights are lost after TWO INCORRECT challenges
- One challenge restored at the start of each extra inning
- Only batters, pitchers, and catchers may initiate
- **Not permitted when a position player is pitching** — exclude these PAs from
  the challengeable denominator

**2026 league baselines (for sanity-checking):**
- Overall overturn rate: 53%
- Batters: 45% · Fielders (C/P): 59%
- ~1,730 attempts as of the reference snapshot

The correct/incorrect asymmetry is the core insight. The expected cost of a
challenge is `P(wrong) x (value of one incorrect-challenge token)`, NOT
`P(anything) x cost`. If optimal thresholds come out well below 50%, the ~53%
observed success rate means players are under-challenging.

## Data sources

1. **Statcast pitch data** — `pybaseball.statcast()`, cached to
   `data/statcast_{year}.parquet`, loaded into `data/baseball.duckdb`.
   NOTE: Statcast revises data retroactively. Re-pulling must be idempotent.

2. **ABS challenge records** — NOT exposed in the Statcast CSV export (verified).
   Candidate sources, in order:
   - `baseballsavant.mlb.com/gf?game_pk={id}` — Savant game feed JSON
   - `statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live` — Stats API play-by-play
   - Savant + Baseball Reference aggregate leaderboards (fallback)

   Ground-truth anchor for field discovery: first regular-season challenge was
   2026-03-25, Yankees @ Giants, 4th inning, Jose Caballero vs Logan Webb,
   first-pitch called strike, call UPHELD.

   **Confirmed: Savant `gf` is the primary source.** Every pitch object in
   `team_home`/`team_away` (etc.) carries `is_abs_challenge` (bool); when true,
   an `abs_challenge` object with `is_overturned`, `challenge_team_id`,
   `challenging_player_id`/`type`, and `edge_distance`/`edge_distance_calc`.
   It's already joined to full trajectory data (`x0,y0,z0,vx0,vy0,vz0,ax,ay,az`,
   `plate_x`, `plate_z`, `call_name`), so loop `game_pk` over this feed rather
   than Stats API. Stats API `feed/live` is a cross-check/backfill: challenged
   `playEvents` carry a `reviewDetails` object (`isOverturned`, `reviewType`,
   `challengeTeamId`, `player`), absent on non-challenged pitches; `boxscore.info`
   also has a human-readable `"ABS Challenge"` summary line per game.

   **Confirmed join keys** (verified against the same real pitch across both
   feeds): Savant `play_id` is byte-identical to Stats API `playId`. Savant
   `ab_number` = Stats API `atBatIndex` + 1 (Savant is 1-indexed, Stats API
   0-indexed). `(game_pk, at_bat_number, pitch_number)` also works as a join key.

   **`edge_distance` coverage: challenge-only.** Checked 93 called (ball/strike)
   pitches in one game that were NOT challenged — zero of them carry an
   `abs_challenge` key at all (not even a null placeholder). So `edge_distance`
   cannot be a model feature for scoring unchallenged pitches (that's the whole
   point of the project — conditioning on a field that only exists where a
   challenge happened is selection on the dependent variable). Use it as a
   validation set only, for pitches that were actually challenged.

   **Gotcha: Savant's `call_name` is the FINAL (post-review) call, not the
   original on-field call.** On an overturned pitch, `call_name` already
   reflects the new ruling — e.g. a challenge that flips a strike to a ball
   shows `call_name: "Ball"`, `is_overturned: true`. Confirmed via
   `challenging_player_type`: every `call_name="Strike", is_overturned=true`
   row was challenged by a catcher/pitcher (they only challenge balls, hoping
   for a strike — so the original call was Ball); every
   `call_name="Strike", is_overturned=false` row was challenged by a batter
   (batters only challenge strikes). To recover the original on-field call:
   same as `call_name` if `is_overturned=false`, the opposite if `true`.

   **The bulk CSV's `description` has the SAME problem** (learned the hard way
   2026-09-05, after documenting it for `call_name` and then not applying it).
   `description` in `pybaseball.statcast()` output is also post-review, so a
   challenged pitch shows the flipped call. Symptom: role assignment and win
   condition both invert for every overturned challenge, producing a ~0.6%
   measured success rate against a real 53.6%. Fix: join the challenge records
   and recover `original_call` before assigning which side the opportunity
   belonged to (see `scripts/build_challenge_opportunities.py`). Only ~3% of
   called pitches are affected, but they are exactly the ones any
   challenge-behaviour analysis is about.

   Useful side effect: our geometry's sign agrees with the FINAL call on 99.4%
   of challenged pitches, which independently confirms both that ABS resolves
   to ball-edge geometry and that our zone model reproduces its decisions.

   **`edge_distance` has no sign convention — it's an unsigned magnitude.**
   Checked 13 real challenges spanning all 4 combinations of
   original-call x overturned/confirmed: `edge_distance` was positive in
   every single case regardless of whether the pitch was actually inside or
   outside the zone. It's `|distance to boundary|`, not a signed inside/outside
   indicator. Units are feet (not inches) — confirmed by regression, see the
   ball-radius entry above.

3. **Batter heights** — `statsapi.mlb.com/api/v1/people/{id}`. Listed height may
   differ from ABS "measured height without cleats". **Update:** the gap is
   much smaller than assumed. Backed out actual measured height per batter by
   solving the radius-corrected zone equation against real challenged pitches
   (`scripts/verify_ball_radius.py` -> `data/measured_heights.parquet`, 364
   batters with >=1 usable vertical challenge, 203 with >=3). Listed-vs-measured
   diff for batters with >=3 pitches: mean 0.006 in, median 0.043 in, p95 0.465
   in. Listed height is a fine proxy for this project; don't over-invest here.

   **Correction (2026-09-05): the mean-signed number above understated the
   real per-batter error.** Mean SIGNED diff ~0 just means errors are
   symmetric and cancel across batters, not that any individual batter's
   error is small. Robust set (n>=3 pitches, 203 batters), diff = measured -
   listed, in inches:
   - mean absolute = 0.244, median absolute = 0.234, std (signed) = 0.281, p95 = 0.465, max = 0.510
   - Confirmed: every listed height in the dataset is an exact whole number
     of inches (max fractional part = 0.0). KS test of (measured - listed)
     against Uniform(-0.5, +0.5) inches: D=0.0577, p=0.49 — does not reject.
     Conclusion: **listed height is true height rounded to the nearest inch,
     and the back-out recovers the unrounded value.** `data/measured_heights.parquet`
     is genuinely novel per-batter data, not a diagnostic byproduct.
   - Circularity ruled out explicitly: the back-out formula
     (`true_edge = z_mid ± (signed_edge_distance - BALL_RADIUS_FT)`, then
     `/0.535` or `/0.270`) never references listed height — listed height only
     picks which bucket (top/bottom/side/corner) a challenge falls into, not
     the computed value. Verified empirically: for the 178 batters with both a
     top-bound and a bottom-bound challenge, `zone_top/0.535` and
     `zone_bot/0.270` are two fully independent derivations (different
     pitches, different formulas) yet agree with EACH OTHER to a median of
     0.012 in (p95 0.030 in) — ~20x tighter than either one's ~0.24 in
     departure from listed height.
   - **Data precision floor, not a bug:** raw `edge_distance` (and the
     trajectory fields behind our `z_mid`) appear to be published at ~0.001 ft
     (~0.01 in) resolution — confirmed by inverting the boundary equation
     pre-division-by-0.270/0.535 and finding the unique values sit on a clean
     0.001 ft grid. This is why unrelated batters occasionally produce
     bit-identical implied heights (heights cluster naturally + narrow
     near-boundary z-range + shared grid) — verified NOT a pandas
     indexing/alignment bug by hand-recomputing one such row from raw inputs
     and matching the stored value exactly. ~20x below the ~0.24 in signal, so
     it doesn't threaten the finding, but don't report these numbers to more
     than ~2 decimal places of inches — the extra digits aren't real.
   - **Uncertainty model for the ~500 batters without a measured height:**
     `src/geometry.py::p_inside_zone(x, z, height_ft, height_uncertainty_ft)`
     integrates zone membership over a height distribution instead of a
     deterministic call. Use `height_uncertainty_ft=0` (deterministic) for
     batters with a measured height; use
     `HEIGHT_ROUNDING_HALFWIDTH_FT` (0.5/12 ft) for listed-height-only batters
     (true height ~ Uniform(listed-0.5in, listed+0.5in)). On the 979
     vertical-bound challenged pitches for the 203 robust-set batters: using
     listed instead of measured height flips the deterministic in/out call on
     **2.04%** of pitches; **7.97%** fall in a zone where the rounding
     uncertainty alone makes the call genuinely ambiguous (0.05<P<0.95). Put
     both numbers in the writeup limitations section.

4. **Run expectancy** — `src/run_expectancy.py` builds RE(balls, strikes, outs,
   base state) from our own data into `data/run_expectancy.parquet` +
   `data/baseball.duckdb`. Two traps found the hard way (2026-09-05):

   - **Use `MAX(post_bat_score)`, never `MAX(bat_score)`, for the half-inning
     final score.** `bat_score` is the score BEFORE the play resolves, so runs
     scored on the play that ENDS the half-inning appear in no later row and
     get silently dropped. This understated every state (bases-loaded-2-out by
     0.027 runs, bases-empty-0-out by 0.007) and pushed bases-loaded-2-out
     below its published band, which is how it was caught.
   - **Exclude walk-off half-innings** (game's last half-inning, bottom half,
     home team won). They're censored — the home team stops batting the instant
     the winning run scores — which biases exactly the high-RE late-inning
     states the challenge model cares about most.
   - Validated: all 24 base-out cells land in published RE24 bands
     (empty/0out 0.485, loaded/0out 2.359, loaded/2out 0.755).
   - Run environment is drifting by season — bases-loaded-2-out is 0.826 (2024)
     / 0.751 (2025) / 0.721 (2026) on even samples. Use 2026 for the decision
     model rather than the pooled table, or at minimum check sensitivity.

5. **`delta_run_exp` is COUNT-BASED, not base-out aware — do NOT use it as the
   run value of flipping a call.** The walkthrough (docs/) suggests it as a
   shortcut for Step 1.3. That shortcut is wrong for this project. Verified
   three ways: (a) for 0-0 called strikes it ranges only -0.038 (bases empty)
   to -0.044 (bases loaded) — essentially flat where true RE differs by ~8x;
   (b) recovering Statcast's implied RE from inning-ending outs gives ~0.24 for
   ALL eight 2-out base states (true values run 0.099 to 0.755); (c) individual
   3-2 inning-ending strikeouts all carry an identical -0.367 across different
   games. It's a pitch-quality metric that deliberately strips situational
   context. Our own base-out-aware table is required, since the entire value of
   flipping a ball/strike call depends on who's on base.

   Transition-level check (`scripts/validate_re_transitions.py`): correlation
   between our implied RE(next)-RE(current) and mean `delta_run_exp` is 0.89
   unweighted / 0.87 pitch-weighted, median difference -0.006 runs. The 431/576
   cells exceeding 0.02 runs are concentrated exactly in bases-loaded and
   deep-count states — i.e. the divergence is the base-out blindness above, not
   an off-by-one in our count transitions (an off-by-one would scatter).

## Conventions

- Python 3.11, venv at `.venv`
- Analysis code in `src/`, one module per concern
- Notebooks in `notebooks/` are exploration ONLY — nothing load-bearing lives there
- `data/` is gitignored. Never commit parquet files.
- Precomputed app data goes in `app/data/` and must stay under ~20 MB
- Prefer DuckDB SQL over pandas for aggregations (this project doubles as SQL practice)
- Mirror lefty pitchers into a single handedness frame before any modeling
- When you notice a possible systematic explanation for a discrepancy
  (a missing constant, an offset, a unit error), TEST IT IMMEDIATELY.
  Compute the residuals and check whether they cluster. Never file a
  hypothesis as a footnote and move on.
- When reporting numbers that should agree, always report the offset
  or ratio between them, not just the two values side by side.
- Every `requests.get` in this project must go through
  `src/net.py::get_with_retries` (timeout=30, retry-with-backoff). A request
  with no timeout can hang a script indefinitely with no visible symptom.
  Don't name a local module `http.py` / `http/` — it shadows the stdlib
  `http` package that `requests` itself depends on (`http.client`), which
  breaks every request the moment `src/` is ahead of stdlib on `sys.path`.
## Working preferences

- Explain the *why* behind non-obvious choices, especially statistical ones —
  I'm learning this material, not just shipping it
- When there's a modeling judgment call, say what the alternatives were
- Flag leakage risks aggressively (e.g. group CV folds by pitcher, not randomly)
- Don't silently widen scope. New ideas go in `IDEAS.md`; v1 ships first.
- If something can't be verified, say so rather than guessing at field names

## Commands

```bash
source .venv/bin/activate
python src/ingest.py                  # pull Statcast (slow, cached)
python src/build_db.py                # load parquet -> DuckDB
python scripts/build_app_data.py      # precompute app inputs
streamlit run app/streamlit_app.py    # local app
```
