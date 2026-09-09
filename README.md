# Who's Leaving Runs on the Table?

**By [Kaushal Namuduri](https://github.com/Cowshal)**

**Live app: [mlb-abs-challenge-analysis.streamlit.app](https://mlb-abs-challenge-analysis.streamlit.app)**

[![refresh data](https://github.com/Cowshal/mlb-abs-challenge-analysis/actions/workflows/refresh.yml/badge.svg)](https://github.com/Cowshal/mlb-abs-challenge-analysis/actions/workflows/refresh.yml)

**Optimal ABS challenge policy vs. observed behaviour — 2026 MLB season, 9,000+ challenges across 2,100+ games** (the season is live; the app footer shows the data-through date).

2026 is the first season with the automated ball-strike challenge system. Each team
gets two challenges, a **correct** challenge is retained, and rights are lost only
after **two incorrect** ones — so a challenge is not a resource you spend, it is a
resource you spend *only when you're wrong*. That asymmetry puts the break-even
confidence far below a coin flip, and league behaviour (2.1 challenges per
team-game at a 54% success rate) does not look like it has been priced that way.
Solving for the optimal policy — using the same imperfect information players
actually have — says teams leave roughly **8–10 runs per team-season** on the
table (the estimate has moved by a run or two as the season has progressed —
see Limitations).
The mechanism is not volume. **The optimal policy wins a *smaller* share of its
challenges than teams currently do (~43% vs ~54%) and still nets more runs, because
the calls it picks are worth more.** Challenge different, not challenge more.

**The "Should I challenge?" tab in the live app turns this into an actual
decision tool** — set up any game situation (count, outs, bases, inning,
challenges already spent) and role, click where you think the pitch was, and
get a break-even confidence, the runs at stake, the option value being risked,
and a plain challenge-or-hold verdict. Not a description of last season; a tool
for the moment the count runs full.

That policy result rests on getting the zone geometry right, and the geometry
finding is the more surprising number in this repo: ABS rules a pitch a strike if
*any part of the ball* — not its center — crosses the zone. Measuring from the
center instead, which is the natural first approach, gets **two out of every
three genuinely close calls wrong**. Not a rounding error — a coin flip you'd
lose more often than win.

## The decomposition

| | challenges / team-game | success rate | runs / team-game |
|---|---|---|---|
| **Observed 2026** | 2.15 | 53.7% | 0.226 |
| **Optimal, same information** | 2.90 | 42.5% | 0.277 |
| Ceiling (perfect information) | 4.83 | 79.7% | 0.624 |

**Decision gap: ~8–10 runs per team-season.** This is the actionable number. It
depends only on players' measured perceptual noise, not on any assumption about
tracking technology. The point estimate has drifted between roughly 8 and 10
runs across data revisions as the season has filled in; the range, not a single
number, is the honest statement.

The remaining gap to a perfect-information ceiling is **not coachable**, and its
size depends entirely on an assumed tracking precision that this data cannot
measure — we only ever observe Hawk-Eye's own output, never independent ground
truth. So it is reported as a curve, not a number:

| assumed ceiling σ | information gap (runs / team-season) |
|---|---|
| 0.10 in | 75 |
| 0.25 in | 69 |
| 0.50 in | 56 |
| 0.75 in | 44 |
| 1.00 in | 33 |

### The mechanism: leverage, not volume

| policy | mean stake (runs) | median stake | runs per overturn | success |
|---|---|---|---|---|
| observed | 0.215 | 0.142 | 0.196 | 53.7% |
| optimal | 0.259 | 0.201 | 0.225 | 42.5% |

Teams are challenging calls that are *easy to win* rather than calls that are
*worth winning*. A borderline strike three with the bases loaded is worth several
times a borderline take on 0-0 with nobody on, and the break-even confidence moves
accordingly — from roughly 73% on a low-stake call down to 15% on a full-count
flip with two challenges in hand. A single fixed habit cannot be right across that
range.

## Who, specifically

**Teams** — all 30 have ~140 games of sample, so no filtering needed:

| team | actual rate | actual success | optimal rate | optimal success | runs left / season |
|---|---|---|---|---|---|
| SD | 2.01/g | 51.2% | 2.94/g | 48.1% | 19.9 |
| WSH | 2.04/g | 48.8% | 2.99/g | 42.5% | 18.5 |
| TB | 2.05/g | 53.0% | 2.98/g | 47.0% | 18.4 |
| STL | 1.69/g | 50.4% | 2.89/g | 39.6% | 13.4 |
| SEA | 2.02/g | 50.2% | 2.97/g | 45.8% | 13.3 |

**Players** — restricted to `optimal_challenges ≥ 20` (71 of 398 batters clear this
bar) so the list isn't small-sample noise; below it, one lucky or unlucky swing
of the season shuffles the ranking:

| player | actual challenges | actual success | optimal challenges | optimal success | runs left |
|---|---|---|---|---|---|
| Curtis Mead | 10 | 30.0% | 22 | 54.5% | 3.57 |
| Yandy Díaz | 0 | — | 29 | 51.7% | 3.09 |
| Mike Trout | 11 | 63.6% | 33 | 54.5% | 3.02 |
| Steven Kwan | 14 | 50.0% | 34 | 50.0% | 2.51 |
| Francisco Lindor | 1 | 0.0% | 21 | 47.6% | 2.45 |

Two of these (Díaz, and to a lesser extent Lindor) barely challenge at all
in reality — the model isn't saying they challenge badly, it's saying they're
sitting on a right they almost never use. Full tables (`app/data/per_team.parquet`,
`app/data/per_batter.parquet`) are in the app's "Runs left on the table" tab,
where the minimum-sample threshold is adjustable.

Runs gained factors exactly as attempts × success rate × mean stake per
overturn, and teams lead the table for different reasons. Cincinnati's edge is
almost entirely *quality*: attempts are close to league average (+7%), but
success rate is 17% above league and average stake per overturn is 13% above —
together worth about **1.32×** a league-average team's output at Cincinnati's
own volume. Minnesota's edge, by contrast, is almost entirely *volume*: 27%
more attempts than league average at a essentially league-average success
rate. Both end up near the top of the raw runs-left table; only one of them
got there by picking good moments rather than taking more swings. Full
per-team ratios (attempts/success/leverage, against league baseline) are in
`data/team_decomposition.parquet`, built by `scripts/team_decomposition.py`.

**Read the team table as a 2026 snapshot, not a proven skill ranking.**
Splitting each team's season in half and correlating first-half success rate
against second-half gives **r ≈ 0.28** across all 30 teams (95% CI
[−0.09, 0.58], p = 0.13). The interval contains zero, so this on its own can't
call challenge accuracy a stable team trait. What *does* hold up: the spread
in success rate and runs gained across teams is bigger than 30 teams drawing
from the league rate at their own volume would produce by chance (p = 0.004
and p < 0.0001), and Cincinnati's success rate sits 3.4 standard deviations
above the league mean, surviving a correction for having checked all 30 teams
(Bonferroni-adjusted p ≈ 0.02).

**Is it a team trait or a personnel one?** Rosters change mid-season, so a
real player-level skill can fail a team-level reliability check. Re-running
the split-half test on individual challengers: at **≥8 challenges per half**
the individual correlation is **r ≈ 0.27** (n ≈ 108, 95% CI [0.09, 0.44]);
at **≥10** it is **r ≈ 0.38** (n ≈ 84, 95% CI [0.19, 0.55]). Both clear zero,
but only the ≥10 cut is unambiguous — at ≥8 the player estimate now sits
essentially on top of the team-level one.

**This contrast is fragile.** Five days of additional games moved *both*
numbers by about 0.05, toward each other (team 0.22 → 0.28, player-≥8
0.32 → 0.27). Split-half reliability is simply not well estimated with one
partial season at n = 30 teams or n ≈ 108 players, and the gap between the
two levels should not be leaned on. The personnel reading rests instead on
two things that have been **stable across every pipeline refresh**: the
catcher-quality correlation (**r = 0.48, p = 0.007** — a team's challenge-
*quality* edge tracks its own primary catcher's individual accuracy) and the
role-σ effect replicating in **28 of 30 teams**. Cincinnati's quality-driven
lead comes with a primary catcher (Tyler Stephenson) around the 85th
percentile of all catchers on his own challenge success; Minnesota's
volume-driven lead comes with a middle-of-the-pack one. Read the team table
as **who was on the roster in 2026**, not a proven front-office skill
ranking — see `scripts/team_skill_test.py` and the "Whose skill is it?"
section in the app's "Runs left on the table" tab.

## Catchers should challenge more than batters — not fewer

Fitting each role's perceptual noise separately:

| role | fitted σ | observed success | observed rate | optimal rate |
|---|---|---|---|---|
| batters | 2.75 in | 48.8% | 0.98 / game | 1.34 / game |
| catchers & pitchers | **1.99 in** | 57.9% | 1.16 / game | **1.57 / game** |

The known 45%/59% batter-fielder success split is usually read as fielders being
*better* at challenging. The fit says it is mostly **better information** — they
read the pitch about 28% more precisely, consistent with seeing it from behind the
plate rather than from the side. And because they see it better, they should be
challenging *more* than batters, not fewer, despite already succeeding more often.

**This result only appears with role-specific σ. Pooling the two roles into one
average reverses it** — the pooled model says batters should challenge more. Any
analysis that estimates a single league-wide noise parameter will get this
backwards.

**It also replicates team by team.** Fitting σ separately for every team, split
by role: **28 of 30 teams** individually read fielding challenges more precisely
than batting challenges — the same pattern found leaguewide, not an artifact of
pooling everyone together. Unlike the team-skill question above, this doesn't
depend on any reliability test — it's a direct, team-by-team confirmation of
the role effect.

The σ estimates come from *where* players chose to challenge — never from how
often they challenged or how often they were right. That matters: fitting σ to
volume and success rate would be circular, forcing the measured gap to zero by
construction. As an out-of-fit check, the model reproduces the known 45%/59% split
(48.8% / 57.9%) without ever seeing it.

**One σ per role assumes accuracy is location-independent — it isn't, but the
headline number doesn't care.** Splitting the zone into a 3×3 grid (in/middle/away
× low/middle/high, relative to the batter) and testing whether the role gap
varies by location: it does, hard (likelihood-ratio test p < 0.0001, a 38-point
swing across well-populated regions), including a real reversal on pitches up
over the heart of the plate — batters succeed 69% of the time there against 49%
for the battery, the opposite of the pattern everywhere else (n=252 and 367,
not noise). Refitting σ separately per zone region and re-running the whole
decision model end to end moves the headline decision gap by **+0.74
runs/team-season** (from 9.1 to 9.8) — real, about 8% of the headline number,
but not enough to change which policy is better or by roughly how much. Read
that as: don't trust the pooled model for any single high-middle call, and
don't treat the headline decision gap as precise to the tenth of a run either
— but the qualitative story (teams are leaving real runs on the table by
challenging the wrong pitches) doesn't depend on which of these two numbers
you use. See `scripts/zone_analysis.py` and `scripts/zone_sigma_refit.py`.

## Method

- **Zone geometry.** ABS applies the "any part of the ball over the zone" rule,
  so the boundary is the rectangle inflated by a ball radius, not the rectangle
  itself. Restricted to a rigorously defined borderline band (within one ball
  radius of the centre-based boundary — roughly half of all season
  challenges), measuring from the ball's centre instead of its edge disagrees
  with MLB's actual ruling on
  **~67%** of them — worse than a coin flip. Across all challenges
  unconditionally, the centre-only error rate is **~34%**. Validated against
  MLB's own `edge_distance` at R² = 1.0000 (slope 0.9999, intercept 0.1208 ft
  — exactly one ball radius); the corrected model matches MLB's ruling
  **~99.8%** of the time overall, and within the borderline band
  (`scripts/validate_ball_radius_classification.py`). This replaces a figure
  (34.1%/66.9%/99.41%/99.46%) that had no reproducing script and turned out
  to predate the dedup fix. The disagreement rates above reproduce closely;
  the corrected-model match rate did not (it now reproduces around 99.8% vs. a
  previously-quoted 99.41%), which is large enough to chase down rather than
  wave at rounding.
  Two concrete explanations were tested and ruled out: (1) using the raw
  trajectory re-solve instead of `plate_x`/`plate_z` — identical to 14
  decimal places, not the cause; (2) circularity — 1,198 of these season
  challenges are the exact pitches `scripts/verify_ball_radius.py` used to
  back out that batter's own measured height, so testing on them with that
  height is partly testing the back-out formula's ability to reproduce its
  own input. Refit with leave-one-out height (excluding each pitch from its
  own batter's height estimate) for those 1,198 pitches — the number moved
  from 99.77% to 99.75%, confirming the effect exists but is too small to
  explain the gap either. The remaining ~0.3pp difference is not accounted
  for; the original one-off computation's exact method no longer exists to
  compare against. This script's number is the one to trust going forward,
  since it's the one that can be inspected and regenerated.
- **Batter heights** backed out per batter by inverting the zone equation against
  real challenges. Listed heights turn out to be true heights rounded to the
  nearest inch (KS test against Uniform(±0.5 in): p = 0.49). Two independent
  derivations — from top-edge and bottom-edge challenges — agree to 0.012 in.
- **Run expectancy** built from **2026 Statcast alone**, not pooled across
  seasons, and validated against published RE24 (bases empty / 0 out = 0.485;
  loaded / 0 out = 2.359; loaded / 2 out = 0.755). Pooling 2024–2026 roughly
  triples the sample behind each state, but the run environment has drifted
  down monotonically — bases-loaded, two-out run expectancy was 0.814 in 2024,
  0.740 in 2025, 0.711 in 2026 — so a pooled table prices 2026 calls against a
  scoring context that no longer holds. Re-solving the full decision model on
  the pooled table drops the headline gap from ~8.2 to ~7.0 runs/team-season
  (≈15%), which is why 2026-only is chosen explicitly rather than by default.
- **Policy** solved by backward induction over (half-inning × pitch × challenges
  spent), with the two-incorrect absorption as a boundary condition and extra
  innings restoring a challenge. Decisions are made on a simulated noisy
  observation; outcomes are resolved against the true location.

### Two premises worth correcting

Both are commonly assumed, and both are stated explicitly in the project brief
this repo was built from
([`docs/baseball-projects-walkthrough.md`](docs/baseball-projects-walkthrough.md)).
Both are wrong, and both were measured rather than assumed.

1. **[`delta_run_exp` is count-based, not base-out aware](docs/delta_run_exp_is_not_base_out_aware.md)** —
   the brief proposes it as the shortcut for the run value of flipping a call:
   "`delta_run_exp` already gives you the run value of each pitch outcome."
   That's the wrong tool here: it prices a bases-loaded strike three almost
   identically to a bases-empty one — implied run expectancy is ~0.24 for all
   eight 2-out base states, where true values run from 0.099 to 0.755 —
   disqualifying for a decision that turns entirely on situational leverage.

2. **[`plate_x` / `plate_z` are already at the middle of the plate](docs/plate_x_is_already_midplate.md)** —
   the brief states Statcast reports these at the front of the plate ("you
   can't use them directly — you have to re-solve the trajectory"), which was
   true through 2025. Savant's own CSV documentation confirms the field was
   redefined for 2026 "to align with the ABS system," and — worth knowing if
   you work with this data — the historical record was backfilled under the
   new definition, so a 2024 game queried today also returns middle-of-plate.
   Solving to y = 8.5/12 reproduces the reported values to 0.00000 in with
   zero variance across 1,220 pitches: an algebraic identity, not an
   approximation. Matters practically too, since the bulk Statcast export
   ships no position anchor to re-solve from.

## Limitations

- **The decision gap is a floor.** The fitted σ absorbs any real variation
  in players' thresholds across counts and leverage, since varying thresholds look
  like noise to this estimator. That inflates the information gap and deflates the
  decision gap. Letting the cutoff vary by count would tighten both bounds.
- **The ceiling is an assumption, not a measurement.** Hence the sensitivity curve.
- **Sparse run-expectancy cells — checked, and they don't matter.** The
  2026-only RE table has 288 count × base-out states; the sparsest is seen 9
  times and ~7% have fewer than 100 observations. But those cells are
  structurally near-unreachable by a challenge (a 3-0 count with a runner on
  third, nobody out, and so on — nobody challenges a 3-0 count): of the ~9,300
  real challenges, **0.3%** land in a cell with < 100 observations, carrying
  **0.6%** of the total run value at stake. Shrinking the thin cells toward a
  smoother model would, at that weight, change nothing measurable, so it was
  not added — knowing the estimate doesn't lean on those cells is the stronger
  statement.
- **Listed vs. measured height.** 201 batters have a measured height backed out
  from ≥3 challenges; the rest fall back to listed height carrying ±0.5 in of
  rounding uncertainty, which flips the in/out call on 2.1% of near-boundary
  pitches and leaves 7.8% genuinely ambiguous
  (`scripts/measured_height_uncertainty.py`).
- **Runs, not wins.** The model is indifferent to score, so it values a challenge
  in a blowout the same as one in a tie game.
- **One partial season** of a brand-new system, so first-year learning
  effects are unmodelled and the policy may be chasing a moving target. The
  numbers on this page are regenerated daily as games are played; the app
  footer shows the data-through date.
- **Split-half reliability is not well estimated yet, so the team-vs-player
  contrast is fragile.** Team-level split-half reliability is currently
  **r ≈ 0.28** (95% CI [−0.09, 0.58]) and player-level is **r ≈ 0.27** at
  ≥8 challenges/half, **r ≈ 0.38** at ≥10 — but both moved ~0.05 toward each
  other on five days of new data, so treat these as poorly estimated with one
  partial season. The "it's personnel, not front-office skill" reading leans
  on the catcher-quality correlation (r = 0.48, p = 0.007) and the 28/30
  role-σ replication, both stable across refreshes, more than on the
  split-half gap. Treat "runs left on the table" by team as a 2026 snapshot
  of who was on the roster, not a proven ranking of organizations.
- **Perceptual σ is assumed location-independent within a role; it isn't —
  but how much that costs the headline number isn't pinned down by one
  season.** Splitting challenges into a 3×3 zone grid and testing whether the
  role gap in success rate varies by location: it does (likelihood-ratio
  test, p < 0.0001, a 38-point swing across well-populated regions), including
  a genuine reversal on pitches up over the heart of the plate, where batters
  out-read the battery. Refitting σ per zone region and re-running the full
  decision model moves the headline decision gap by +0.74 runs/season on the
  actual 2026 data — but each (role, region) cell is fit on as few as ~200
  challenged pitches, and bootstrap-resampling just the challenged subset
  within each cell (holding everything else fixed, 150 replicates,
  `scripts/zone_sigma_bootstrap.py`) puts a 95% interval of **-1.6 to +3.1
  runs/season** around that move — wide enough to cross zero and to contain
  both this figure and an earlier, much smaller estimate from before a data
  bug was fixed. The honest statement is that region-level σ variation is
  real (the likelihood-ratio test doesn't depend on this resampling and isn't
  in question) but this dataset cannot yet say whether accounting for it in
  the decision model would move the headline gap by a little, a lot, or even
  in the other direction. Treat +0.74 as one plausible point on a wide
  distribution, not a correction to bank. See `scripts/zone_analysis.py`,
  `scripts/zone_sigma_refit.py`, and `scripts/zone_sigma_bootstrap.py`.

## What I'd do with team-internal data

The binding constraint here is not the model, it is knowing what the player knew.
Perceptual σ is currently inferred from revealed behaviour across the whole
league; with team data I would estimate it *per player* from their own challenge
history and, ideally, from tracking where they were looking. That converts a
league-average prescription into a personalised one — and since the fitted noise
already differs by 28% across roles, it very likely differs meaningfully across
individuals within a role. The immediate deliverable would be a per-player
threshold card: given this count, this base-out state, and this many challenges
left, here is *your* break-even confidence. Beyond that, catcher framing and
challenge accuracy are plausibly the same underlying skill measured two ways, and
a club could test that directly against its own receiving data.

## Performance and reliability

**Caching.** Every parquet read goes through one `@st.cache_data`-decorated
`load()` — nothing else in the app touches disk. The decision tool's derived
lookups (the RE dict, the option-value table, the per-role posterior grids,
the clickable zone grid) are each cached too, keyed on their actual inputs
(e.g. the zone grid recomputes only when role changes, not on every click).
Audited the rest of the app for the failure mode this implies checking for —
a widget interaction silently re-reading a file or re-running an aggregate —
and found none: the only uncached per-interaction work left is trivial pandas
filtering on tables of at most 392 rows (the batter-table attempt-count
slider, a 30-row merge in the team-skill section), too cheap to be worth the
complexity of caching.

**Measured latency.** Timed via the browser's own performance API against the
live deployed app (not a guess): a preset-button click on the decision
tool — game situation changes, RE and option-value lookups run, the
recommendation re-renders — takes **~1.0-1.4 seconds** end to end. That's
Streamlit Cloud's websocket round-trip and script rerun, not local compute;
every lookup involved is a cached dict/array access, sub-millisecond on its
own. On a paid/dedicated instance this would likely drop closer to the
websocket latency alone.

**Failing gracefully.** A stale-parquet mismatch after a deploy (a new app
build served against not-yet-refreshed data files) has now caused a raw
`KeyError` stack trace on the live app twice. The whole app body is now
wrapped in one top-level `try/except`: whatever rendered before the failure
stays on screen, followed by a plain "something didn't load right, try
refreshing" message with a collapsed technical-detail expander, instead of a
traceback. Verified by deliberately dropping columns from a live data file
and confirming the friendly message renders in place of the crash. This
doesn't prevent the underlying deploy-sync race — that's Streamlit Cloud's
timing, not this repo's — it just means a visitor who hits that window sees
a sentence instead of a wall of Python.

**Also found and fixed during this pass:** `app/streamlit_app.py` had picked
up an import (`from run_expectancy import flip_value`) whose module did
`import duckdb` at the top level — harmless locally, since duckdb is in the
dev environment, but the deployed app's `requirements.txt` deliberately
excludes duckdb (the app must never touch it), so the live site crashed with
`ModuleNotFoundError` immediately on load. Fixed by moving that import inside
the one function that actually needs it; verified in a from-scratch venv
built from `requirements.txt` alone that the app's imports now succeed
without duckdb installed at all.

**A third instance of the same pattern, found in a correctness audit:**
`data/abs_challenges.parquet` had 121 duplicate rows across 26 games. Root
cause: the MLB schedule API lists a game_pk once per date it appears under,
and 26 games (almost certainly doubleheader/makeup-game listings) appeared
under two dates each in the collection window — `collect_abs_challenges.py`
fetched and saved each of those games twice. Three scripts had grown a
defensive `.drop_duplicates()` to cope with this downstream; the actual fix
is at the source (`final_game_pks()` now deduplicates, and `main()` refuses
to save a file with any duplicate rows), so those three defensive dedups were
removed rather than kept as redundant insurance. Re-ran the full pipeline on
the corrected data — see "What changed when this was fixed" below.

**A structural guard against all three.** The `.gitignore` match, the
`duckdb` import, and this duplicate-rows bug are the same failure shape: a
convention that holds almost everywhere, with nothing checking "everywhere."
`scripts/build_app_data.py` now runs three checks before writing anything —
`data/abs_challenges.parquet` has no duplicate rows, every module the app
imports at load time resolves under `requirements.txt` alone (a static,
transitive check of the app's own import graph, not a guess), and every file
it writes to `app/data/` carries a `model_version`/`generated_at` stamp — and
raises loudly, refusing to write, if any of the three fail. All three were
verified to actually fire (each was deliberately triggered once, on a copy of
the data, and reverted) rather than assumed to work from reading the code.

**What changed when this was fixed.** The 26 affected games' individual
challenge counts roughly halved (e.g. Tyler Stephenson's own challenges,
123 → 119; Hunter Goodman's, 97 → 95) — but since exact duplication doesn't
change a ratio, no previously-published *rate* moved because of the
duplicates themselves. What *did* move, from re-running the full pipeline
end to end on the corrected data (combined with a small amount of ordinary
data revision between the original collection and this one): the decision
gap (9.6 → 9.1 runs/season), the team-level split-half correlation
(r = 0.24 → 0.22), the player-level split-half correlations (r = 0.28–0.37 →
0.32–0.38), the cross-team spread significance test (p = 0.003 → 0.004), the
measured-height robust set (203 → 201 batters, with the associated
height-rounding-ambiguity figures moving from 2.04%/7.97% to 2.1%/7.8% now
that they're computed by a script instead of quoted from an old run), the
ball-radius classification check's borderline-band error rate (66.9% →
67.2%, also now script-backed instead of quoted), and the zone-region sigma
sensitivity check, whose point estimate moved from +0.02 to +0.74
runs/season. That last one got a follow-up check this move itself prompted:
a 1.8% change in the challenge count moving a sensitivity estimate 37x is
the kind of thing that should be verified, not just re-quoted, so it was
bootstrapped (see the Limitations section) — the honest finding is that this
specific number was never well-estimated by one season of data in the first
place, with or without the duplicate games. One figure moved by more than
rounding for reasons that were investigated but not fully pinned down: the
ball-radius corrected model's match rate against the final call came back
at 99.75% against a previously-quoted 99.41%. Both the old height-uncertainty
and ball-radius corrected-match figures had no reproducing script before
this pass; for the latter, two concrete explanations (a trajectory-solve vs.
`plate_x`/`plate_z` discrepancy, and circularity between the classification
check and the measured-height back-out it partly overlaps with) were tested
and ruled out — see the Zone geometry limitations bullet for the full
account. The new, regenerable numbers are the ones to trust in both cases.
Every other headline number in this README reproduced within rounding of what
was published at that point. (Since 2026-09-09 the pipeline regenerates and
re-checks all of them daily against the prose — see "Daily refresh and
keeping the prose honest" — so any later movement, e.g. the catcher-quality
correlation drifting from r = 0.44 to r = 0.48 over the following days, shows
up as a failed build rather than a stale sentence.)

## Running it

`requirements.txt` holds only what the deployed app needs (4 packages);
`requirements-dev.txt` holds the full pipeline.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

scripts/refresh_all.sh          # the whole pipeline, in dependency order
streamlit run app/streamlit_app.py
```

`refresh_all.sh` is the single entry point — it runs every step below in
order, then `build_reported_figures.py`, which **fails the run if any
hardcoded number in `README.md`, `docs/writeup.md`, or the app has drifted
past tolerance** from the freshly rebuilt data (see "Keeping the prose
honest" below). To run steps individually:

```bash
python src/ingest.py                            # Statcast pull (idempotent: 2024-25 cached, 2026 trailing re-pull)
python src/run_expectancy.py                    # RE table -> DuckDB
python scripts/collect_abs_challenges.py        # 2026 challenge records (end date = yesterday)
python scripts/verify_ball_radius.py            # per-batter measured heights (fixed early-season window)
python scripts/build_challenge_opportunities.py
python scripts/estimate_perception_sigma.py
python src/abs_policy.py                        # solve + decompose
python scripts/team_decomposition.py
python scripts/team_skill_test.py               # team split-half + significance + per-team sigma
python scripts/player_skill_test.py             # player split-half + catcher accuracy
python scripts/zone_analysis.py                 # location-dependence test
python scripts/validate_ball_radius_classification.py
python scripts/measured_height_uncertainty.py
python scripts/build_decision_tool_data.py      # RE + posterior lookups for the app
python scripts/build_case_studies.py            # "Real games from 2026" tab
python scripts/build_app_data.py                # precompute for the app (runs its own guards)
python scripts/build_reported_figures.py        # provenance snapshot + prose-drift gate
```

**Frozen, not run by `refresh_all.sh`:** `scripts/zone_sigma_refit.py` and
`scripts/zone_sigma_bootstrap.py`. That sensitivity number is noise-dominated
at one season of challenge data (see the Limitations section) and the
bootstrap is 150 full DP re-solves. Their committed outputs in `app/data/`
are the source of truth; `refresh_all.sh` seeds them back into `data/` so the
downstream copy and provenance checks pass. Re-run them by hand and commit
the new parquet when the season is complete or the method changes.

```
src/        geometry, perception model, run expectancy, policy solver, reported-figures manifest
scripts/    data collection, validation, app precompute, refresh_all.sh orchestrator
docs/       methodological findings
app/        Streamlit app + precomputed parquet (~100 KB)
.github/    scheduled refresh workflow
```

## Daily refresh and keeping the prose honest

The season is live, so the data moves every night. `.github/workflows/refresh.yml`
runs `scripts/refresh_all.sh` once a day (08:00 PT), and if `app/data/`
changed it commits the rebuilt parquet back to `main` — which is what makes
Streamlit Community Cloud redeploy. No servers, no secrets (every source is a
public API).

**Python version.** Development is on 3.14; `requirements-dev.txt` is frozen
from there and several pins (`numpy`, `scipy`, `xgboost`) require ≥ 3.12. CI
runs **3.12** — the newest version for which every pinned package has a
prebuilt wheel (`ruptures==1.1.9`, currently unused, has no wheel past cp312
and would trigger a source build on 3.13+). `requirements.txt` (the app) has
the same ≥ 3.12 floor from `numpy`/`pandas`, so the repo carries a
`.python-version` of `3.12`; the Streamlit Cloud app's Python version must be
set to 3.12 or newer in its advanced settings or a redeploy will fail at
`pip install`.

The risk with automating that is a site that updates its charts but not the
sentences next to them. So `src/reported_figures.py` is a manifest of every
data-derived statistic that appears in prose — the decomposition table, the
σ values, the reliability correlations, the geometry match rates, the counts
— each tied to the parquet it comes from and to a regex for where it's
written. `scripts/build_reported_figures.py` regenerates
`data/reported_figures.json` from the current data and then greps the prose;
if a number has drifted past its tolerance, or an anchoring phrase can no
longer be found, **the pipeline fails and the workflow opens a
`pipeline-drift` issue naming each figure, its location, and its new value**,
rather than pushing an internally inconsistent commit. The app footer shows
the data-through date from the same JSON.

**If the footer's "Data through" date falls behind the latest daily-refresh
commit, the Streamlit Cloud deploy is stuck** (it occasionally is — not a
Python or dependency problem; Cloud runs 3.14 and the pins are fine). A
manual **Reboot app** from the Streamlit Cloud dashboard clears it.

Physical constants (17-inch plate, the ball radius, the 53.5% / 27% zone
edges), rounded restatements of a tracked figure, and the per-team / per-player
snapshot tables are deliberately excluded — see the comment block at the
bottom of `src/reported_figures.py`.

**After the 2026 regular season** (ends 2026-09-28) the workflow becomes a
near-no-op: `ingest.py` stops pulling, nothing changes, nothing is committed.
It is left enabled on purpose so it resumes automatically for 2027 — bump the
season dates in `src/ingest.py` and `scripts/collect_abs_challenges.py` and
un-freeze the zone-sigma scripts when that season starts.

See `IDEAS.md` for the v2 list — count-varying thresholds and a win-probability
objective are the two that would move the number most.

## License

MIT — see [`LICENSE`](LICENSE).
