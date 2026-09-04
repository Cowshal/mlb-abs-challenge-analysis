# Baseball Analytics Project Walkthrough

Three projects plus deployment, built on one shared data foundation.

**Recommended order:** Foundation → Project 1 (ABS) → Deploy → Project 3 (swing changes) → Project 2 (pitch design).

Project 1 is the differentiator and it's time-sensitive — 2026 is the first ABS season and the public research is thin. Get it done and shipped before you start anything else.

---

## Part 0 — Foundation (do this first, it serves all three)

### 0.1 Environment

```bash
mkdir baseball && cd baseball
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install pybaseball pandas numpy duckdb pyarrow \
            scikit-learn xgboost matplotlib seaborn requests \
            ruptures streamlit
```

Make a repo on day one. Not at the end. The commit history is part of what someone evaluates.

```
baseball/
├── data/              # gitignored — parquet files live here
├── src/
│   ├── ingest.py
│   ├── geometry.py
│   ├── run_expectancy.py
│   └── abs_policy.py
├── notebooks/         # exploration only, not the deliverable
├── app/               # the deployed thing
├── .gitignore
└── README.md
```

Put `data/` and `.venv/` in `.gitignore` immediately. Committing a 400 MB parquet file is a bad first impression.

### 0.2 Pull the data

```python
# src/ingest.py
import pybaseball
from pybaseball import statcast
from pathlib import Path

pybaseball.cache.enable()

SEASONS = {
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", "2026-09-27"),   # adjust end date to today if mid-season
}

Path("data").mkdir(exist_ok=True)

for year, (start, end) in SEASONS.items():
    out = Path(f"data/statcast_{year}.parquet")
    if out.exists():
        print(f"{year} already pulled, skipping")
        continue
    print(f"pulling {year}...")
    df = statcast(start_dt=start, end_dt=end)
    df.to_parquet(out, index=False)
    print(f"  {len(df):,} pitches")
```

Run it once and go get lunch. Each season is 700,000+ pitches and this takes 20–40 minutes per year. The cache means a re-run won't re-download.

> **Design note:** Statcast revises data retroactively, including for past seasons. Write your loader so re-pulling and overwriting is the normal path, not an exception.

### 0.3 Load into DuckDB

This is where your SQL coursework starts paying rent.

```python
import duckdb

con = duckdb.connect("data/baseball.duckdb")
con.execute("""
    CREATE OR REPLACE TABLE statcast AS
    SELECT * FROM read_parquet('data/statcast_*.parquet')
""")
con.execute("CREATE INDEX IF NOT EXISTS idx_game ON statcast(game_pk)")
print(con.execute("SELECT game_year, COUNT(*) FROM statcast GROUP BY 1 ORDER BY 1").df())
```

From here you can query in SQL directly:

```sql
SELECT pitch_type, COUNT(*) AS n, AVG(release_speed) AS velo
FROM statcast
WHERE game_year = 2026 AND pitch_type IS NOT NULL
GROUP BY 1 ORDER BY n DESC;
```

### 0.4 Columns you'll actually use

| Column | What it is |
|---|---|
| `plate_x`, `plate_z` | Pitch location at the **front** of the plate, in feet. `plate_x = 0` is dead center, from the catcher's view. |
| `x0,y0,z0`, `vx0,vy0,vz0`, `ax,ay,az` | Full trajectory parameters. You need these for Project 1. |
| `sz_top`, `sz_bot` | Umpire-judged zone (stringer-set). **Not** the ABS zone. |
| `description` | `called_strike`, `ball`, `swinging_strike`, `hit_into_play`, etc. |
| `delta_run_exp` | Run expectancy change from this pitch. Enormous time-saver. |
| `balls`, `strikes`, `outs_when_up`, `on_1b/2b/3b` | Game state |
| `pfx_x`, `pfx_z` | Movement in feet (positive `pfx_x` = toward first base side) |
| `release_pos_x/z`, `release_extension`, `arm_angle` | Release geometry |
| `bat_speed`, `swing_length` | Bat tracking — 2024 forward only |
| `attack_angle`, `attack_direction`, `swing_path_tilt` | 2025 forward only |
| `game_pk`, `at_bat_number`, `pitch_number` | Unique pitch key |

---

## Part 1 — Optimal ABS challenge policy

**The question:** MLB's challenge system launched in 2026. League-wide success rate is hovering near a coin flip. Given that a *correct* challenge is free and only *incorrect* ones burn the resource, are players challenging too little? How many runs is that costing each team?

### Step 1.1 — Find the challenge data (do this before anything else)

**I don't know for certain whether the pitch-level Statcast CSV export includes a challenge flag.** Verify this yourself first, because it determines how much work Step 1 is. Three places to look, in order:

1. **The Statcast columns.** Pull one game and dump the column list:
   ```python
   df = statcast('2026-06-15', '2026-06-15')
   print([c for c in df.columns if 'chal' in c.lower() or 'abs' in c.lower() or 'review' in c.lower()])
   ```
2. **Savant's ABS leaderboards.** `baseballsavant.mlb.com/leaderboard/abs-challenges` — the leaderboard pages generally accept `&csv=true` for a download. This gets you player/team aggregates but probably not pitch-level detail.
3. **The MLB Stats API game feed.** `https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live` returns full play-by-play JSON including `playEvents` with review/challenge details. This is the most likely source for pitch-level challenge records. It's free and needs no key. You'd loop over `game_pk` values you already have in your Statcast table.

If you end up on route 3, joining Stats API events back to Statcast pitches is done on `(game_pk, at_bat_number, pitch_number)` or on the `playId` GUID.

**Write down what you find in your README.** "Here's what data exists and here's how I got it" is itself a credential — it demonstrates you can source data that isn't handed to you.

### Step 1.2 — Project the pitch to the middle of the plate

This is the piece of the project that shows real technical depth, and it's a detail most people would miss.

The ABS zone is defined as: 17 inches wide (same as home plate), top at **53.5%** of the batter's measured height without cleats, bottom at **27%**. Critically, **location is captured over the middle of the plate, not the front.**

Statcast's `plate_x` / `plate_z` are reported at the *front* of the plate. So you can't use them directly — you have to re-solve the trajectory.

Statcast coordinates put `y = 0` at the back point of home plate. Home plate is 17 inches deep, so:
- front of plate: `y = 17/12 ≈ 1.417 ft`
- middle of plate: `y = 8.5/12 ≈ 0.708 ft`

Constant-acceleration kinematics gets you there:

```python
# src/geometry.py
import numpy as np

Y_MIDPLATE = 8.5 / 12.0
HALF_WIDTH = (17.0 / 12.0) / 2.0   # 0.708 ft

def solve_t_at_y(y0, vy0, ay, y_target):
    """Time at which the ball crosses y_target. Ball travels in -y direction."""
    # y(t) = y0 + vy0*t + 0.5*ay*t^2  =>  0.5*ay*t^2 + vy0*t + (y0 - y_target) = 0
    a, b, c = 0.5 * ay, vy0, (y0 - y_target)
    disc = b**2 - 4 * a * c
    disc = np.where(disc < 0, np.nan, disc)
    t1 = (-b - np.sqrt(disc)) / (2 * a)
    t2 = (-b + np.sqrt(disc)) / (2 * a)
    # take the smaller positive root
    return np.where(t1 > 0, t1, t2)

def location_at_midplate(df):
    t = solve_t_at_y(df.y0, df.vy0, df.ay, Y_MIDPLATE)
    x = df.x0 + df.vx0 * t + 0.5 * df.ax * t**2
    z = df.z0 + df.vz0 * t + 0.5 * df.az * t**2
    return x, z

def in_abs_zone(x, z, batter_height_ft):
    top = 0.535 * batter_height_ft
    bot = 0.270 * batter_height_ft
    return (np.abs(x) <= HALF_WIDTH) & (z >= bot) & (z <= top)
```

Sanity-check it: `location_at_midplate` should give you values close to but systematically slightly different from `plate_x`/`plate_z`. If they're wildly off, your root selection is wrong.

You'll need batter heights — the MLB Stats API people endpoint (`statsapi.mlb.com/api/v1/people/{id}`) returns height, or pull the full roster in one call. Note the ABS zone uses *measured* height without cleats, which may differ slightly from listed height. Note that limitation in your writeup rather than hiding it; a good analyst flags their own measurement error.

**Validate against reality.** You have actual challenge outcomes from Step 1.1. Run your geometry on every challenged pitch and check whether your predicted overturn matches the real one. If you're at 95%+ agreement, your zone model works. If not, debug before proceeding — everything downstream depends on this.

### Step 1.3 — Build a run expectancy table

You need to know what flipping a call is worth. That means run expectancy by count and base-out state. Pure SQL, and a great portfolio piece on its own.

```sql
-- Runs scored from the current state through the end of the half-inning
WITH inning_ends AS (
  SELECT game_pk, inning, inning_topbot,
         MAX(bat_score + post_bat_score - bat_score) AS final_score  -- adjust to your columns
  FROM statcast GROUP BY 1,2,3
)
SELECT balls, strikes,
       (on_1b IS NOT NULL) AS r1,
       (on_2b IS NOT NULL) AS r2,
       (on_3b IS NOT NULL) AS r3,
       outs_when_up,
       AVG(runs_rest_of_inning) AS run_exp,
       COUNT(*) AS n
FROM ...
GROUP BY 1,2,3,4,5,6;
```

**Shortcut:** `delta_run_exp` already gives you the run value of each pitch outcome. Average `delta_run_exp` for called strikes vs. called balls within each count, and you have the value of flipping that call without building the table from scratch. Do the shortcut first to get a working model, then build the real table as a refinement.

### Step 1.4 — The decision model (this is the actual insight)

Here's the asymmetry that makes this interesting. Per the rules: teams start with two challenges, a **correct** challenge is retained, and challenge rights are lost after **two incorrect** ones. Teams also get a challenge back in each extra inning.

So a challenge isn't a resource you spend — it's a resource you spend *only when you're wrong*. The cost of challenging is:

```
E[cost] = P(wrong) × (value of one "incorrect challenge" token)
```

Not `P(anything) × cost`. That distinction is the whole project. If your model says the optimal threshold is meaningfully below 50%, the league-wide ~53% success rate means players are leaving runs on the table by being too selective — and you can quantify exactly how many.

Set it up as backward induction. Define the state as `(incorrect_challenges_used, inning, outs, base_state)` and compute:

```
V(state) = expected future run value of retaining challenge rights
```

Challenge when:

```
P(overturn) × ΔRE(flip)  >  P(not overturn) × [V(k incorrect used) − V(k+1 incorrect used)]
```

The right-hand side is the option value you're risking. Start simple — a two-state version (0 vs 1 incorrect used, ignoring inning) that you can compute in an afternoon — get an answer, then add states.

Your `P(overturn)` comes from Step 1.2. If your zone model is deterministic, add measurement uncertainty: treat the true location as Gaussian around your estimate with a small sigma (start with ~0.5 inches, then estimate it empirically from cases where your model and the real outcome disagreed).

### Step 1.5 — The findings

Now produce the actual output:

1. **Optimal threshold curve** — how the challenge threshold should shift by count, inning, and challenges remaining.
2. **Actual vs. optimal by player** — who challenges well, who's too aggressive, who's too passive. Note that Savant publishes an *expected challenges* model based on location, remaining challenges, runners on, and count/out state — but that's descriptive (what a typical player *would* do). Yours is prescriptive (what they *should* do). Compare all three: actual, Savant expected, and your optimal. **That three-way comparison is your headline chart.**
3. **Runs left on the table** — per team, per player, over the season. This is the number that gets quoted.
4. **Position asymmetry** — catchers reportedly outperform batters at this. Is that skill, or just better information (they see the pitch from behind, batters from the side)?

---

## Part 2 — Pitch design gap finder

**The question:** given a pitcher's existing arsenal and release point, what pitch shape *doesn't* he throw that would be most valuable to add?

### Step 2.1 — Build a "stuff" model

Train a model to predict run value from pitch shape alone, with **no location features**. Location is command; shape is stuff. Mixing them defeats the purpose.

```python
import xgboost as xgb
from sklearn.model_selection import GroupKFold

FEATURES = [
    "release_speed", "pfx_x_adj", "pfx_z", "release_pos_x_adj",
    "release_pos_z", "release_extension", "release_spin_rate",
    "arm_angle", "velo_diff_from_fastball", "hb_diff_from_fastball",
    "ivb_diff_from_fastball",
]

# Mirror everything for lefties so LHP and RHP live in one feature space
df["pfx_x_adj"] = np.where(df.p_throws == "L", -df.pfx_x, df.pfx_x)
df["release_pos_x_adj"] = np.where(df.p_throws == "L", -df.release_pos_x, df.release_pos_x)

model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8)
```

Two things that will otherwise sink this:

- **Group your CV folds by pitcher.** A random split leaks — the same pitcher's pitches appear in train and test, and the model learns "this is Skubal" rather than "this shape is good." Use `GroupKFold(groups=df.pitcher)`.
- **The differential features matter most.** A 92 mph changeup is unremarkable on its own and excellent behind a 99 mph fastball. Compute each pitch's velo/movement gap from that pitcher's primary fastball.

Validate by training on 2024–2025 and testing on 2026. Report correlation with actual run value at the pitcher-season level. Be honest if it's modest — public stuff models typically are, and saying so is more credible than overselling.

### Step 2.2 — Define the feasible shape space

You can't recommend a shape a pitcher physically can't throw. Constrain candidates by:

- **Velocity band:** within roughly ±3 mph of some pitch he already throws
- **Release point:** fixed at his observed average — this is the real constraint, since arm slot dictates achievable spin axes
- **Movement plausibility:** for a given velo and arm angle, check what movement combinations actually appear in the league. Fit the convex hull of observed `(velo, pfx_x, pfx_z)` for pitchers with similar arm angles and only propose points inside it.

That third constraint is what separates a real recommendation from a model hallucinating a 95 mph pitch with 25 inches of arm-side run.

### Step 2.3 — Score and rank candidates

```python
score = predicted_run_value(candidate) - λ * max_similarity_to_existing_arsenal(candidate)
```

The penalty term is essential. A pitch that grades out well but sits on top of his existing slider adds nothing — the value of a new pitch is partly *separation*. Tune λ by eyeballing whether recommendations look sensible for pitchers you know.

### Step 2.4 — Validate against reality

The best possible check: find pitchers who **actually added a new pitch** between 2024 and 2026. Run your model on their prior-year arsenal. Did it recommend something close to what they added? Did the pitches it liked outperform the ones it didn't?

That backtest turns this from "I built a model" into "I built a model and tested whether it works." Nobody does the second part. Do the second part.

---

## Part 3 — In-season swing change detection

**The question:** can you automatically detect when a hitter changes his swing mid-season, and does the change pay off?

### Step 3.1 — Build swing-level time series

Bat tracking (`bat_speed`, `swing_length`) starts in 2024. Swing path metrics (`attack_angle`, `attack_direction`, `swing_path_tilt`) start in 2025. That's a short history — acknowledge the limitation up front.

```sql
SELECT batter, game_date,
       AVG(bat_speed)     AS bat_speed,
       AVG(swing_length)  AS swing_length,
       AVG(attack_angle)  AS attack_angle,
       COUNT(*)           AS swings
FROM statcast
WHERE bat_speed IS NOT NULL AND game_year >= 2025
GROUP BY 1, 2
HAVING COUNT(*) >= 3
ORDER BY 1, 2;
```

Then smooth with a rolling window (25–50 swings, not games — game samples are too noisy).

### Step 3.2 — Detect changepoints

```python
import ruptures as rpt

signal = hitter_df[["bat_speed", "attack_angle", "swing_length"]].values
algo = rpt.Pelt(model="rbf", min_size=20).fit(signal)
breakpoints = algo.predict(pen=10)
```

Tune `pen` deliberately: too low and every hitter has fifteen "changes," too high and you find nothing. Calibrate against known cases — find two or three hitters who publicly discussed a swing change mid-2025 or 2026, and set the penalty so those get flagged.

### Step 3.3 — Guard against false positives

This is where most versions of this project fall apart. Bat speed drifts with fatigue, weather, and opponent quality. Before you claim a mechanical change:

- **Permutation test:** shuffle the hitter's swing order 1,000 times, re-run detection, and check how often you find a "changepoint" that strong by chance. Keep only detections that beat the 95th percentile of the null.
- **Multiple comparisons:** you're testing ~400 hitters. At p < 0.05 you'd expect ~20 false positives by chance alone. Apply Benjamini-Hochberg.

Doing this correctly and saying so is worth more than a longer list of detections.

### Step 3.4 — Did it work?

For each validated changepoint, compare pre/post windows on `xwOBA` on contact, barrel rate, and whiff rate. Then the interesting cut: **which kinds of changes pay off?** Adding bat speed at the cost of swing length? Steepening attack angle? Group changepoints by their direction in feature space and look for patterns.

---

## Part 4 — Deployment

### 4.1 Pick a framework

| | Streamlit (Python) | Shiny (R) |
|---|---|---|
| Time to working app | ~2 hours | ~1 day if new to R |
| Hosting | Streamlit Community Cloud, free | shinyapps.io, free tier |
| Resume value | Good | **Explicitly named in MLB postings** |

The Mariners' Baseball Projects posting listed Shiny experience as preferred. That's a real, specific reason to use it, and you're taking R this semester anyway.

**Practical suggestion:** build it in Streamlit first to get something live fast, then port to Shiny once the analysis is settled. Having a working link matters more than which framework produced it, and porting a finished app is much easier than learning a new framework while your analysis is still moving.

### 4.2 Precompute — do not ship the pipeline

Your app must not run a Statcast pull. Precompute everything to a small file:

```python
# scripts/build_app_data.py
results = compute_challenge_analysis(con)          # your Part 1 output
results.to_parquet("app/data/challenge_results.parquet")   # aim for < 20 MB
```

The app loads that file and does nothing but filter and plot. Anything the user can do should take under a second.

### 4.3 A minimal Streamlit app

```python
# app/streamlit_app.py
import streamlit as st, pandas as pd, plotly.express as px

st.set_page_config(page_title="ABS Challenge Optimizer", layout="wide")

@st.cache_data
def load():
    return pd.read_parquet("data/challenge_results.parquet")

df = load()

st.title("Who's Leaving Runs on the Table?")
st.caption("Optimal ABS challenge policy vs. observed behavior, 2026 MLB season")

col1, col2 = st.columns([1, 3])
with col1:
    role = st.selectbox("Role", ["Batter", "Catcher", "Pitcher"])
    min_ch = st.slider("Min challenges", 1, 50, 10)

sub = df[(df.role == role) & (df.challenges >= min_ch)]

with col2:
    fig = px.scatter(sub, x="actual_rate", y="optimal_rate",
                     size="challenges", hover_name="player_name",
                     labels={"actual_rate": "Actual challenge rate",
                             "optimal_rate": "Model-optimal rate"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Runs left on the table")
st.dataframe(sub.nlargest(25, "runs_lost")[
    ["player_name", "team", "challenges", "actual_rate", "optimal_rate", "runs_lost"]
])
```

Deploy: push to GitHub → share.streamlit.io → point it at your repo and the app file. You need a `requirements.txt` in the repo root. That's the whole process; it's genuinely about ten minutes.

### 4.4 Shiny + shinyapps.io

```r
# app/app.R
library(shiny); library(arrow); library(dplyr); library(ggplot2)

df <- read_parquet("data/challenge_results.parquet")

ui <- fluidPage(
  titlePanel("ABS Challenge Optimizer"),
  sidebarLayout(
    sidebarPanel(
      selectInput("role", "Role", c("Batter", "Catcher", "Pitcher")),
      sliderInput("min_ch", "Min challenges", 1, 50, 10)
    ),
    mainPanel(plotOutput("scatter"), tableOutput("leaders"))
  )
)

server <- function(input, output) {
  filtered <- reactive({
    df |> filter(role == input$role, challenges >= input$min_ch)
  })
  output$scatter <- renderPlot({
    ggplot(filtered(), aes(actual_rate, optimal_rate, size = challenges)) +
      geom_point(alpha = 0.6) +
      geom_abline(linetype = "dashed") +
      labs(x = "Actual challenge rate", y = "Model-optimal rate") +
      theme_minimal()
  })
  output$leaders <- renderTable({
    filtered() |> arrange(desc(runs_lost)) |> head(25)
  })
}

shinyApp(ui, server)
```

Deploy:

```r
install.packages("rsconnect")
rsconnect::setAccountInfo(name = "...", token = "...", secret = "...")
rsconnect::deployApp("app/")
```

Free tier gives you 5 apps and 25 active hours per month — plenty for a portfolio piece.

### 4.5 Keep it fresh automatically

A GitHub Actions cron that updates the data nightly is a strong signal — it says "production pipeline," not "one-time class project."

```yaml
# .github/workflows/update.yml
name: Update Statcast
on:
  schedule:
    - cron: "0 11 * * *"   # 11:00 UTC daily
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python src/ingest.py --incremental --days 3
      - run: python scripts/build_app_data.py
      - run: |
          git config user.name  "github-actions"
          git config user.email "actions@github.com"
          git add app/data/
          git commit -m "data update $(date -I)" || echo "no changes"
          git push
```

Note `--days 3`, not `--days 1`. Statcast revises recent data, so re-pull a short trailing window rather than just yesterday.

---

## Part 5 — The writeup

The analysis is maybe 70% of the value; the writeup is the rest. Nobody hires off a repo they didn't read.

Structure, roughly 1,500–2,500 words:

1. **The question**, in one paragraph a non-technical scout could follow.
2. **Why it's not obvious** — the correct/incorrect asymmetry, the option value of holding a challenge.
3. **Method**, with the geometry section shown honestly (this is where you demonstrate rigor).
4. **Findings** — lead with the number. "MLB teams left roughly N runs on the table in 2026 by under-challenging."
5. **Limitations** — measured vs. listed height, Hawk-Eye error, first-season learning effects. Volunteering these makes everything else more credible, not less.
6. **What you'd do with team data** — this is the paragraph that reads as "I want to work here."

Post it publicly. A personal site, Medium, or a well-formatted repo README all work. Then link it from your resume and your applications.

---

## Realistic timeline

| Weeks | Work |
|---|---|
| 1 | Part 0 foundation. Data pulled, DuckDB loaded, repo live. |
| 2 | Step 1.1 data sourcing + Step 1.2 geometry, validated against real outcomes. |
| 3–4 | Run expectancy + decision model. First real numbers. |
| 5 | Findings, charts, sanity checks. |
| 6 | Deploy. Link is live. |
| 7 | Writeup published. |
| 8+ | Project 3 as an extension, then Project 2 if there's time. |

Eight weeks to a deployed, written-up project. If you start now you're finished well before winter internship deadlines, with something live to link in every application.

## Where things will go wrong

- **The challenge data isn't where you expect.** Budget real time for Step 1.1. If pitch-level data genuinely doesn't exist publicly, pivot to the Stats API play-by-play route — don't abandon the project.
- **The geometry doesn't validate.** Almost always the quadratic root selection or a units mistake (feet vs. inches). Check a handful of pitches by hand.
- **The model says everyone is optimal.** Probably means your option value term is too large. Sanity-check it: what's one challenge actually worth in runs? If the answer is more than ~0.15 runs, something's off.
- **Scope creep.** You will want to add things. Write them in an `IDEAS.md` and ship v1 first.
