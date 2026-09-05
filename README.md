# Who's Leaving Runs on the Table?

**Live app: [mlb-abs-challenge-analysis.streamlit.app](https://mlb-abs-challenge-analysis.streamlit.app)**

**Optimal ABS challenge policy vs. observed behaviour — 2026 MLB season, 9,037 challenges across 2,136 games.**

2026 is the first season with the automated ball-strike challenge system. Each team
gets two challenges, a **correct** challenge is retained, and rights are lost only
after **two incorrect** ones — so a challenge is not a resource you spend, it is a
resource you spend *only when you're wrong*. That asymmetry puts the break-even
confidence far below a coin flip, and league behaviour (2.1 challenges per
team-game at a 54% success rate) does not look like it has been priced that way.
Solving for the optimal policy — using the same imperfect information players
actually have — says teams leave roughly **10 runs per team-season** on the table.
The mechanism is not volume. **The optimal policy wins a *smaller* share of its
challenges than teams currently do (43% vs 54%) and still nets more runs, because
the calls it picks are worth more.** Challenge different, not challenge more.

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
| **Optimal, same information** | 2.91 | 43.3% | 0.285 |
| Ceiling (perfect information) | 4.76 | 80.2% | 0.622 |

**Decision gap: ~10 runs per team-season.** This is the actionable number. It
depends only on players' measured perceptual noise, not on any assumption about
tracking technology.

The remaining gap to a perfect-information ceiling is **not coachable**, and its
size depends entirely on an assumed tracking precision that this data cannot
measure — we only ever observe Hawk-Eye's own output, never independent ground
truth. So it is reported as a curve, not a number:

| assumed ceiling σ | information gap (runs / team-season) |
|---|---|
| 0.10 in | 73.6 |
| 0.25 in | 67.4 |
| 0.50 in | 54.5 |
| 0.75 in | 42.4 |
| 1.00 in | 31.8 |

### The mechanism: leverage, not volume

| policy | mean stake (runs) | median stake | runs per overturn | success |
|---|---|---|---|---|
| observed | 0.216 | 0.146 | 0.196 | 53.7% |
| optimal | 0.259 | 0.201 | 0.227 | 43.3% |

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
| SD | 2.01/g | 51.2% | 2.89/g | 47.9% | 16.9 |
| TB | 2.05/g | 53.0% | 2.81/g | 46.1% | 16.1 |
| WSH | 2.04/g | 48.8% | 2.93/g | 40.1% | 14.6 |
| STL | 1.69/g | 50.4% | 2.85/g | 39.6% | 13.2 |
| HOU | 2.37/g | 53.6% | 2.96/g | 45.7% | 12.6 |

**Players** — restricted to `optimal_challenges ≥ 20` (72 of 392 batters clear this
bar) so the list isn't small-sample noise; below it, one lucky or unlucky swing
of the season shuffles the ranking:

| player | actual challenges | actual success | optimal challenges | optimal success | runs left |
|---|---|---|---|---|---|
| Ben Williamson | 5 | 60.0% | 22 | 54.5% | 3.33 |
| Yandy Díaz | 0 | — | 26 | 57.7% | 3.28 |
| Garrett Mitchell | 8 | 25.0% | 20 | 65.0% | 2.41 |
| Mike Trout | 11 | 63.6% | 30 | 56.7% | 2.31 |
| Matt Chapman | 12 | 33.3% | 21 | 52.4% | 2.20 |

Two of these (Díaz, and to a lesser extent Williamson) barely challenge at all
in reality — the model isn't saying they challenge badly, it's saying they're
sitting on a right they almost never use. Full tables (`app/data/per_team.parquet`,
`app/data/per_batter.parquet`) are in the app's "Runs left on the table" tab,
where the minimum-sample threshold is adjustable.

Runs gained factors exactly as attempts × success rate × mean stake per
overturn, and teams lead the table for different reasons. Cincinnati's edge is
almost entirely *quality*: attempts are close to league average (+8%), but
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
against second-half gives r = 0.24 across all 30 teams, 95% CI [-0.14, 0.55]
— too weak and too uncertain to call challenge accuracy a stable, predictable
team trait. At the same time, the actual spread in success rate and runs
gained across teams is bigger than 30 teams drawing from the league rate at
their own volume would produce by pure chance (p = 0.003 and p < 0.0001,
respectively), and Cincinnati's success rate sits 3.4 standard deviations
above the league mean — a result that survives correcting for having checked
all 30 teams (Bonferroni-adjusted p ≈ 0.02). Put plainly: there is more real
variation here than luck alone explains, but the team-level split-half test
can't confirm it's a stable team trait.

**That's because it isn't a team-level trait — it's a personnel one.**
Rosters change mid-season, so a real trait belonging to specific players can
still fail a team-level reliability check if those players move around.
Re-running split-half reliability on individual challengers instead of teams
gives r = 0.28 (p = 0.003, n = 108 players with ≥8 challenges/half) to
r = 0.37 (p < 0.001, n = 84 with ≥10/half) — both clearly above the
team-level 0.24 and clear of zero, unlike it. It lines up with *why* teams
lead: Cincinnati's quality-driven edge comes with a primary catcher (Tyler
Stephenson) who individually ranks in the 85th percentile of all catchers
leaguewide on his own challenge success; Minnesota's volume-driven edge comes
with a merely average one. Read the team table as **who was on the roster in
2026**, not as a proven front-office skill ranking — see
`scripts/team_skill_test.py` and the "Is this a repeatable team skill?"
section in the app's "Runs left on the table" tab for the full walkthrough.

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
(48.9% / 57.9%) without ever seeing it.

## Method

- **Zone geometry.** ABS applies the "any part of the ball over the zone" rule,
  so the boundary is the rectangle inflated by a ball radius, not the rectangle
  itself. Restricted to a rigorously defined borderline band (within one ball
  radius of the centre-based boundary, n=4,654 — half of all 9,197 season
  challenges), measuring from the ball's centre instead of its edge disagrees
  with MLB's actual ruling on **66.9%** of them — worse than a coin flip.
  Across all 9,197 challenges unconditionally, the centre-only error rate is
  34.1%. Validated against MLB's own `edge_distance` at R² = 1.0000 (slope
  0.9999, intercept 0.1208 ft — exactly one ball radius); the corrected model
  matches MLB's ruling 99.4% of the time.
- **Batter heights** backed out per batter by inverting the zone equation against
  real challenges. Listed heights turn out to be true heights rounded to the
  nearest inch (KS test against Uniform(±0.5 in): p = 0.49). Two independent
  derivations — from top-edge and bottom-edge challenges — agree to 0.012 in.
- **Run expectancy** built from 2024–2026 Statcast and validated against published
  RE24 (bases empty / 0 out = 0.485; loaded / 0 out = 2.359; loaded / 2 out = 0.755).
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

- **The 10-run decision gap is a floor.** The fitted σ absorbs any real variation
  in players' thresholds across counts and leverage, since varying thresholds look
  like noise to this estimator. That inflates the information gap and deflates the
  decision gap. Letting the cutoff vary by count would tighten both bounds.
- **The ceiling is an assumption, not a measurement.** Hence the sensitivity curve.
- **Listed vs. measured height.** 203 batters have a measured height backed out
  from ≥3 challenges; the rest fall back to listed height carrying ±0.5 in of
  rounding uncertainty, which flips the in/out call on 2.0% of near-boundary
  pitches and leaves 8.0% genuinely ambiguous.
- **Runs, not wins.** The model is indifferent to score, so it values a challenge
  in a blowout the same as one in a tie game.
- **One partial season** (through 2026-09-03) of a brand-new system, so
  first-year learning effects are unmodelled and the policy may be chasing a
  moving target.
- **Team-level rankings track rosters, not front offices.** Team-level
  split-half reliability is r = 0.24 (95% CI [-0.14, 0.55]) — too weak to
  call challenge accuracy a team trait — but player-level reliability
  (r = 0.28–0.37 for players with enough volume in both halves) is
  meaningfully higher, and clear of zero. Treat "runs left on the table" by
  team as a 2026 snapshot of who was on the roster, not a proven ranking of
  organizations.

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

## Running it

`requirements.txt` holds only what the deployed app needs (4 packages);
`requirements-dev.txt` holds the full pipeline.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python src/ingest.py                            # pull 2024-2026 Statcast
python src/run_expectancy.py                    # RE table -> DuckDB
python scripts/collect_abs_challenges.py        # 2026 challenge records
python scripts/build_challenge_opportunities.py
python scripts/estimate_perception_sigma.py
python src/abs_policy.py                        # solve + decompose
python scripts/build_app_data.py                # precompute for the app

streamlit run app/streamlit_app.py
```

```
src/        geometry, perception model, run expectancy, policy solver
scripts/    data collection, validation, app precompute
docs/       methodological findings
app/        Streamlit app + precomputed parquet (96 KB)
```

See `IDEAS.md` for the v2 list — count-varying thresholds and a win-probability
objective are the two that would move the number most.

## License

MIT — see [`LICENSE`](LICENSE).
