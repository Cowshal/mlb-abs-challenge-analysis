# `delta_run_exp` is count-based, not base-out aware

**Finding:** Statcast's `delta_run_exp` cannot be used as the run value of flipping
a ball/strike call. It is a pitch-quality metric that deliberately strips
base-out context, so it fails precisely in the high-leverage states that
dominate any challenge-policy result.

This matters because it is a natural shortcut, and it is the one recommended in
the project walkthrough this repo was built from:

> **Shortcut:** `delta_run_exp` already gives you the run value of each pitch
> outcome. Average `delta_run_exp` for called strikes vs. called balls within
> each count, and you have the value of flipping that call without building the
> table from scratch.

That shortcut is wrong for this purpose. Below is the evidence.

## Why it looked fine at first

Aggregated by count, `delta_run_exp` behaves exactly as expected: balls are
positive for the batting team, called strikes negative, and the magnitudes grow
with leverage.

| count | ball | called strike |
|---|---|---|
| 0-0 | +0.036 | −0.041 |
| 1-1 | +0.052 | −0.063 |
| 2-2 | +0.097 | −0.230 |
| 3-2 | +0.289 | −0.325 |

Sign and scaling both look right, which is why this passes a casual check. The
problem only appears once you condition on base state.

## Probe 1: it barely moves across base states

For called strikes on 0-0, `delta_run_exp` by base-out state (2024–2026):

| base state | outs | mean `delta_run_exp` |
|---|---|---|
| bases loaded | 0 | −0.044 |
| 2nd + 3rd | 0 | −0.053 |
| 1st only | 0 | −0.042 |
| bases empty | 0 | −0.039 |

The full range across every base state is roughly −0.038 to −0.061. But the true
run expectancy of those states differs by about 8x (bases empty / 0 out = 0.485;
bases loaded / 0 out = 2.359). A metric that encodes base-out context cannot be
that flat.

## Probe 2: implied run expectancy is constant across all eight 2-out states

For an inning-ending out with no runs scoring, run expectancy after the play is
zero by definition, so `delta_run_exp = 0 − RE_before`, and `RE_before` can be
recovered directly as `−delta_run_exp`.

| outs | 1B | 2B | 3B | n | implied RE | our empirical RE |
|---|---|---|---|---|---|---|
| 2 | Y | Y | Y | 3820 | 0.243 | 0.755 |
| 2 | N | Y | Y | 3606 | 0.239 | 0.525 |
| 2 | Y | N | Y | 4630 | 0.250 | 0.460 |
| 2 | N | N | Y | 4998 | 0.247 | 0.334 |
| 2 | Y | Y | N | 10523 | 0.243 | 0.422 |
| 2 | N | Y | N | 12619 | 0.250 | 0.306 |
| 2 | Y | N | N | 22867 | 0.245 | 0.217 |
| 2 | N | N | N | 51843 | 0.237 | 0.099 |

The implied values are ~0.24 for every state. The true values range from 0.099
to 0.755. Bases loaded and bases empty are assigned the same number.

## Probe 3: identical values across different games

Individual 3-2 inning-ending called-strike strikeouts with the bases loaded:

```
game_pk  inning  balls strikes outs  description    events     delta_run_exp
 745267       7      3       2    2  called_strike  strikeout      -0.367
 745395       2      3       2    2  called_strike  strikeout      -0.367
 745717       6      3       2    2  called_strike  strikeout      -0.367
 745765       3      3       2    2  called_strike  strikeout      -0.367
 745928       2      3       2    2  called_strike  strikeout      -0.367
```

Same value to three decimals across unrelated games. This is a lookup keyed on
count and pitch outcome, not a computed state difference.

## Transition-level comparison

Comparing our own base-out-aware table's implied `RE(next) − RE(current)`
against mean `delta_run_exp`, per (count, base state, outs, call) cell:

- correlation 0.89 unweighted, 0.87 pitch-weighted
- median difference −0.006 runs
- 431 of 576 cells differ by more than 0.02 runs, covering 38% of pitches

The disagreement is not scattered, which is what an off-by-one in the count
transition would produce. It is concentrated in bases-loaded and deep-count
cells, where ours run 2–3x larger:

| count | base state | outs | call | ours | `delta_run_exp` |
|---|---|---|---|---|---|
| 3-2 | loaded | 1 | called strike | −1.066 | −0.356 |
| 3-2 | loaded | 2 | called strike | −0.934 | −0.305 |
| 3-2 | loaded | 0 | ball | +0.922 | +0.307 |
| 2-2 | loaded | 1 | called strike | −0.797 | −0.257 |

## What it actually is

`delta_run_exp` appears to be the count-based pitch run value used for pitch- and
pitcher-quality evaluation, where stripping situational context is the correct
design choice — you don't want a pitcher's grade to depend on whether his
defense put runners on. It is being used as intended. It simply answers a
different question than "what is flipping this call worth right now."

## Implication

Any public model that values a ball/strike call using `delta_run_exp` is
implicitly valuing every call as though the bases were in a league-average
configuration. For an ABS challenge model this is disqualifying, because the
entire decision turns on situational leverage: a borderline strike three with
the bases loaded is worth several times a borderline strike three with the bases
empty, and `delta_run_exp` prices them nearly identically.

We build the base-out-aware table ourselves in `src/run_expectancy.py`, validated
against published RE24 values, and use `delta_run_exp` only as a directional
cross-check.

## Reproducing

```bash
python src/ingest.py                        # pull 2024-2026 Statcast
python src/run_expectancy.py                # build RE table + DuckDB views
python scripts/validate_re_transitions.py   # transition-level comparison
```
