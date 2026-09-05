# Who's Leaving Runs on the Table?

Optimal ABS challenge policy vs. observed behaviour, 2026 MLB season.

2026 is the first season with the automated ball-strike challenge system. Teams
get two challenges, a **correct** challenge is retained, and rights are lost only
after **two incorrect** ones. So a challenge isn't a resource you spend — it's a
resource you spend *only when you're wrong*. That asymmetry means the break-even
confidence is far below a coin flip, and league-wide behaviour (2.1 challenges
per team-game at a 54% success rate) suggests players haven't priced it that way.

**Headline:** teams leave roughly **10 runs per team-season** on the table through
challenge policy alone — and the optimal fix is not "challenge more," it's
"challenge different": trade hit rate for leverage.

## Findings

| | challenges/team-game | success | runs/team-game |
|---|---|---|---|
| observed 2026 | 2.15 | 53.7% | 0.226 |
| optimal, same information | 2.91 | 43.3% | 0.285 |
| ceiling (perfect information) | 4.76 | 80.2% | 0.622 |

The optimal policy **wins a smaller share of its challenges** (43% vs 54%) while
gaining more runs, because the calls it picks are worth more: mean stake 0.259
runs vs 0.216, and 0.227 runs per overturn vs 0.196.

The gap splits in two:

- **Decision gap: ~10 runs/team-season.** Actionable. Independent of any
  assumption about tracking precision.
- **Information gap: ~30–75 runs/team-season** depending on assumed ceiling
  precision. Not coachable, and not measurable from this data — reported as a
  sensitivity curve, not a point estimate.

Players' perceptual noise is estimated at **2.75 in (batters)** and **1.99 in
(catchers/pitchers)** — fitted from *where* they chose to challenge, using neither
challenge volume nor success rate, so the estimate cannot absorb the decision gap
it exists to measure. As an out-of-fit check it reproduces the known 45%/59%
batter/fielder success split (48.9% / 57.9%).

## Two corrections to the standard approach

Both measured rather than assumed. Details in `docs/`.

1. **[`delta_run_exp` is count-based, not base-out aware](docs/delta_run_exp_is_not_base_out_aware.md).**
   It's widely suggested as a shortcut for the run value of flipping a call. It
   prices a bases-loaded strike three almost identically to a bases-empty one
   (implied run expectancy is ~0.24 for all eight 2-out base states, where true
   values run 0.099 to 0.755). Disqualifying for this problem.

2. **[`plate_x`/`plate_z` are already at the middle of the plate](docs/plate_x_is_already_midplate.md).**
   Commonly described as front-of-plate, requiring a trajectory re-solve to reach
   the plane ABS judges at. Solving to y = 8.5/12 reproduces them to 0.00000 in
   with zero variance across 1,220 pitches — it's an algebraic identity, not an
   approximation. Matters practically: the bulk CSV ships no `x0/y0/z0` anchor, so
   the re-solve is impossible there anyway.

## Method

- **Zone geometry** validated against MLB's own `edge_distance` at R² = 1.0000
  (slope 0.9999, intercept 0.1208 ft = one ball radius). ABS applies the "any part
  of the ball over the zone" rule, so the boundary is the rectangle inflated by a
  ball radius; measuring from the ball's centre instead misclassifies ~27% of
  borderline calls.
- **Measured batter heights** backed out per batter by inverting the zone equation
  against real challenges. Listed heights turn out to be true heights rounded to
  the nearest inch (KS test against Uniform(±0.5in): p = 0.49).
- **Run expectancy** built from 2024–2026 Statcast, validated against published
  RE24 (bases empty/0 out 0.485, loaded/0 out 2.359, loaded/2 out 0.755).
- **Policy** solved by backward induction over (half-inning × pitch × challenges
  spent), with `V(·,·,2) = 0` absorption and extra-inning restoration.

## Layout

```
src/        geometry, perception model, run expectancy, policy solver
scripts/    data collection, validation, app precompute
docs/       methodological findings
app/        Streamlit app + precomputed parquet (80 KB)
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/ingest.py                          # pull 2024-2026 Statcast (~2 min)
python src/run_expectancy.py                  # RE table -> DuckDB
python scripts/collect_abs_challenges.py      # 2026 challenge records
python scripts/build_challenge_opportunities.py
python scripts/estimate_perception_sigma.py
python src/abs_policy.py                      # solve + decompose
python scripts/build_app_data.py              # precompute for the app

streamlit run app/streamlit_app.py
```

## Limitations

- The fitted sigma absorbs any real variation in players' thresholds across
  counts, which inflates the information gap and deflates the decision gap. **The
  ~10-run decision gap is therefore a floor.**
- Runs, not wins: the model is indifferent to score.
- The perfect-information ceiling is an assumption, not a measurement.
- 2026 is a partial season (through 2026-09-03) and the first year of the system,
  so learning effects are unmodelled.

See `IDEAS.md` for the v2 list.
