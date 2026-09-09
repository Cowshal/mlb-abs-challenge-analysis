"""
Pull pitch-level Statcast data via pybaseball, cached to
data/statcast_{year}.parquet, loaded into DuckDB downstream by
src/run_expectancy.py.

Idempotency contract (the scheduled refresh in scripts/refresh_all.sh and
.github/workflows/refresh.yml re-runs this every day):

- 2024 / 2025 are finished seasons. Pull once; skip if the parquet exists.
  Delete the file to force a re-pull.
- 2026 is the live season. Statcast revises recent games retroactively for
  days after they are played (pitch classifications, trajectory fits, even
  plate_x/plate_z get recomputed), so a one-time pull goes stale silently.
  Every run therefore:
    * sets the end date to *yesterday* (a fully completed slate by the time
      the 08:00-PT job runs), clamped to the regular-season end, and
    * re-pulls a trailing window (TRAILING_REPULL_DAYS) and OVERWRITES those
      dates in the cached parquet, rather than appending. Everything before
      the window is kept as-is; everything from the window start onward is
      replaced with the fresh pull.
  If data/statcast_2026.parquet does not exist yet, the whole season to
  yesterday is pulled in one shot.

Run: python src/ingest.py
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pybaseball
from pybaseball import statcast

pybaseball.cache.enable()

# Finished seasons: (start, end), pulled once and left alone.
FINISHED_SEASONS = {
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

# Live season.
LIVE_SEASON = 2026
LIVE_SEASON_START = "2026-03-26"
LIVE_SEASON_END = "2026-09-28"  # last day of the 2026 regular season

# How many days back from yesterday to re-pull and overwrite on every run.
# Statcast's retroactive revisions overwhelmingly land within a couple of
# days; 4 is deliberate slack. Widen this (one number) if a revision is ever
# found to have changed a game older than this window.
TRAILING_REPULL_DAYS = 4

# The columns that identify a single pitch, used to drop stale rows before
# splicing the fresh trailing-window pull back in.
PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]


def _pull(start, end):
    print(f"  pulling Statcast {start} .. {end} ...")
    df = statcast(start_dt=start, end_dt=end)
    print(f"    {len(df):,} rows")
    return df


def pull_finished_seasons():
    for year, (start, end) in FINISHED_SEASONS.items():
        out = Path(f"data/statcast_{year}.parquet")
        if out.exists():
            print(f"{year}: already pulled, skipping ({out})")
            continue
        print(f"{year}: pulling full season")
        _pull(start, end).to_parquet(out, index=False)
        print(f"  -> {out}")


def refresh_live_season(today=None):
    """Re-pull yesterday's slate and a trailing window, overwriting those
    dates in data/statcast_{LIVE_SEASON}.parquet. Full-season pull if the
    file is missing."""
    today = today or date.today()
    yesterday = today - timedelta(days=1)
    season_start = date.fromisoformat(LIVE_SEASON_START)
    season_end = date.fromisoformat(LIVE_SEASON_END)
    end = min(yesterday, season_end)
    out = Path(f"data/statcast_{LIVE_SEASON}.parquet")

    if end < season_start:
        print(f"{LIVE_SEASON}: nothing to pull yet (yesterday {yesterday} is "
              f"before the season start {LIVE_SEASON_START})")
        return

    # Once the season is over AND Statcast's revision window for the final
    # slate has closed, there is nothing left to pull. Stop, so the daily
    # workflow is a genuine no-op in the offseason rather than re-fetching
    # the same final week forever.
    if out.exists() and yesterday > season_end + timedelta(days=TRAILING_REPULL_DAYS):
        print(f"{LIVE_SEASON}: season complete ({LIVE_SEASON_END}) and revision "
              f"window closed -- cache left unchanged")
        return

    if not out.exists():
        print(f"{LIVE_SEASON}: no cache, pulling full season to {end}")
        _pull(LIVE_SEASON_START, end.isoformat()).to_parquet(out, index=False)
        print(f"  -> {out}")
        return

    window_start = max(season_start, end - timedelta(days=TRAILING_REPULL_DAYS - 1))
    if window_start > end:
        print(f"{LIVE_SEASON}: season is over ({LIVE_SEASON_END}) and the "
              f"trailing window is empty -- cache left unchanged")
        return

    print(f"{LIVE_SEASON}: refreshing trailing window "
          f"{window_start} .. {end} (overwrite), keeping earlier rows")
    existing = pd.read_parquet(out)
    fresh = _pull(window_start.isoformat(), end.isoformat())

    # Drop every existing row from window_start onward, then splice the fresh
    # pull in. Prefer the date column; fall back to the pitch key if a fresh
    # row somehow predates the window (shouldn't happen, but don't lose it).
    if "game_date" in existing.columns:
        keep = existing[pd.to_datetime(existing.game_date).dt.date < window_start]
    else:
        keep = existing.merge(fresh[PITCH_KEY], on=PITCH_KEY, how="left", indicator=True)
        keep = keep[keep._merge == "left_only"].drop(columns="_merge")

    combined = pd.concat([keep, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset=PITCH_KEY, keep="last")
    combined.to_parquet(out, index=False)
    print(f"  kept {len(keep):,} earlier rows + {len(fresh):,} fresh "
          f"= {len(combined):,} total -> {out}")


def main():
    Path("data").mkdir(exist_ok=True)
    pull_finished_seasons()
    refresh_live_season()


if __name__ == "__main__":
    main()
