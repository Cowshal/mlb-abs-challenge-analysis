"""
Pull full-season Statcast pitch-level data via pybaseball, cached to
data/statcast_{year}.parquet. Statcast revises data retroactively, so
re-pulling and overwriting is the normal path -- delete the parquet for
a season to force a re-pull.

Run: python src/ingest.py
"""
import pybaseball
from pybaseball import statcast
from pathlib import Path

pybaseball.cache.enable()

SEASONS = {
    2024: ("2024-03-28", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", "2026-09-04"),  # partial season, adjust end date as it progresses
}


def main():
    Path("data").mkdir(exist_ok=True)
    for year, (start, end) in SEASONS.items():
        out = Path(f"data/statcast_{year}.parquet")
        if out.exists():
            print(f"{year} already pulled, skipping")
            continue
        print(f"pulling {year} ({start} to {end})...")
        df = statcast(start_dt=start, end_dt=end)
        df.to_parquet(out, index=False)
        print(f"  {len(df):,} pitches -> {out}")


if __name__ == "__main__":
    main()
