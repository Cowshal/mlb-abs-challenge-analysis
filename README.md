# Who's Leaving Runs on the Table?

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

The σ estimates come from *where* players chose to challenge — never from how
often they challenged or how often they were right. That matters: fitting σ to
volume and success rate would be circular, forcing the measured gap to zero by
construction. As an out-of-fit check, the model reproduces the known 45%/59% split
(48.9% / 57.9%) without ever seeing it.

## Method

- **Zone geometry** validated against MLB's own `edge_distance` at R² = 1.0000
  (slope 0.9999, intercept 0.1208 ft — exactly one ball radius). ABS applies the
  "any part of the ball over the zone" rule, so the boundary is the rectangle
  inflated by a ball radius; measuring from the ball's centre instead
  misclassifies about 27% of borderline calls.
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

Both are standard advice for this kind of project. Both are wrong, and both were
measured rather than assumed.

1. **[`delta_run_exp` is count-based, not base-out aware](docs/delta_run_exp_is_not_base_out_aware.md)** —
   commonly recommended as the shortcut for the run value of flipping a call. It
   prices a bases-loaded strike three almost identically to a bases-empty one:
   implied run expectancy is ~0.24 for all eight 2-out base states, where true
   values run from 0.099 to 0.755. Disqualifying here, because the entire decision
   turns on situational leverage.

2. **[`plate_x` / `plate_z` are already at the middle of the plate](docs/plate_x_is_already_midplate.md)** —
   usually described as front-of-plate, requiring a trajectory re-solve to reach
   the plane ABS actually judges at. Solving to y = 8.5/12 reproduces them to
   0.00000 in with zero variance across 1,220 pitches: an algebraic identity, not
   an approximation. This also matters practically, since the bulk Statcast export
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

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

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
