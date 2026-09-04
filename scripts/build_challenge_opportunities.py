"""
Build the challenge opportunity set: every called ball/strike in 2026 with
    p     = probability a challenge of that call would succeed
    dre   = runs at stake in flipping it (RE_after_ball - RE_after_strike)

Both sides' opportunities are produced from the same pitch. A called strike is
an opportunity for the BATTING team (wins if the pitch was actually outside);
a called ball is an opportunity for the FIELDING team (wins if actually inside).
dre is the same quantity either way -- the batting team gains it, the fielding
team prevents it.

Run: python scripts/build_challenge_opportunities.py
Output: data/challenge_opportunities.parquet
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries
from geometry import (p_inside_zone,
                      HEIGHT_ROUNDING_HALFWIDTH_FT)
from run_expectancy import load_re_lookup, flip_value

SEASON = 2026
LOCATION_SIGMA_FT = 0.5 / 12.0  # ~0.5 inch, per the walkthrough's guidance


def fetch_people(ids):
    """Listed height (ft) and primary position code for a set of player ids."""
    out = {}
    ids = [int(i) for i in ids]
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = get_with_retries(
            "https://statsapi.mlb.com/api/v1/people",
            params={"personIds": ",".join(str(b) for b in chunk)},
        )
        for p in r.json().get("people", []):
            h = p.get("height", "")
            height = None
            if h:
                feet, inches = h.replace('"', "").split("'")
                height = int(feet) + int(inches.strip()) / 12.0
            out[p["id"]] = {
                "height_ft": height,
                "position": (p.get("primaryPosition") or {}).get("code"),
            }
        print(f"  fetched {min(i+50, len(ids))}/{len(ids)}")
    return out


def main():
    con = duckdb.connect("data/baseball.duckdb")

    df = con.execute(f"""
        SELECT game_pk, game_date, inning, inning_topbot, at_bat_number, pitch_number,
               batter, pitcher, description,
               balls, strikes, outs_when_up AS outs,
               (on_1b IS NOT NULL) AS r1, (on_2b IS NOT NULL) AS r2, (on_3b IS NOT NULL) AS r3,
               plate_x, plate_z
        FROM statcast
        WHERE game_year = {SEASON}
          AND description IN ('called_strike', 'ball')
          AND balls IS NOT NULL AND strikes IS NOT NULL AND outs_when_up IS NOT NULL
          AND plate_x IS NOT NULL AND plate_z IS NOT NULL
    """).df()
    print(f"{len(df):,} called pitches in {SEASON}")

    print("fetching batter heights + pitcher positions...")
    people = fetch_people(pd.unique(np.concatenate([df.batter.values, df.pitcher.values])))

    # Position players pitching: challenges aren't permitted, so these PAs are
    # outside the challengeable denominator entirely.
    pitcher_pos = df.pitcher.map(lambda p: (people.get(int(p)) or {}).get("position"))
    n_before = len(df)
    df = df[pitcher_pos == "1"].reset_index(drop=True)
    print(f"dropped {n_before - len(df):,} pitches thrown by position players")

    # Measured height where we have one (>=3 challenges backing it), else listed.
    mh = pd.read_parquet("data/measured_heights.parquet")
    measured = {int(r.batter): r.measured_height_ft
                for r in mh[mh.n_pitches >= 3].itertuples()}
    listed = df.batter.map(lambda b: (people.get(int(b)) or {}).get("height_ft"))
    df["listed_height_ft"] = listed
    df["has_measured"] = df.batter.map(lambda b: int(b) in measured)
    df["height_ft"] = [measured.get(int(b), l) for b, l in zip(df.batter, listed)]
    df = df.dropna(subset=["height_ft"]).reset_index(drop=True)
    print(f"{df.has_measured.sum():,} pitches use a measured height; "
          f"{(~df.has_measured).sum():,} fall back to listed")

    # plate_x/plate_z ARE the midplate location (verified to 0.0000 in against a
    # trajectory re-solve; see the note in src/geometry.py). No re-solve needed,
    # which is fortunate since the bulk CSV ships no x0/y0/z0 anchor.
    df["x_mid"], df["z_mid"] = df.plate_x.values, df.plate_z.values

    # P(inside zone): measured-height batters carry only location uncertainty;
    # listed-height batters also carry the +/-0.5in rounding uncertainty.
    p_inside = np.empty(len(df))
    for has_m in (True, False):
        m = (df.has_measured == has_m).values
        if not m.any():
            continue
        p_inside[m] = p_inside_zone(
            df.x_mid.values[m], df.z_mid.values[m], df.height_ft.values[m],
            height_uncertainty_ft=0.0 if has_m else HEIGHT_ROUNDING_HALFWIDTH_FT,
            location_sigma_ft=LOCATION_SIGMA_FT)
    df["p_inside"] = p_inside

    print("computing flip values from the 2026 RE table...")
    re = load_re_lookup(con, season=SEASON)
    df["dre"] = [flip_value(re, int(b), int(s), int(o), (bool(a), bool(c), bool(d)))
                 for b, s, o, a, c, d in
                 zip(df.balls, df.strikes, df.outs, df.r1, df.r2, df.r3)]
    df = df.dropna(subset=["dre"]).reset_index(drop=True)

    # challenger + success probability
    is_strike_call = df.description == "called_strike"
    df["challenger"] = np.where(is_strike_call, "batting", "fielding")
    df["p_success"] = np.where(is_strike_call, 1.0 - df.p_inside, df.p_inside)

    keep = ["game_pk", "game_date", "inning", "inning_topbot", "at_bat_number",
            "pitch_number", "batter", "pitcher", "description", "challenger",
            "balls", "strikes", "outs", "r1", "r2", "r3",
            "x_mid", "z_mid", "height_ft", "has_measured",
            "p_inside", "p_success", "dre"]
    out = df[keep]
    out.to_parquet("data/challenge_opportunities.parquet", index=False)
    print(f"\nsaved {len(out):,} opportunities -> data/challenge_opportunities.parquet")

    print("\n=== summary ===")
    print(out.groupby("challenger").agg(
        n=("p_success", "size"),
        mean_p_success=("p_success", "mean"),
        mean_dre=("dre", "mean"),
        p_success_gt_50=("p_success", lambda s: (s > 0.5).mean()),
    ).to_string())

    hi = out.groupby(["game_pk", "inning", "inning_topbot", "challenger"]).size()
    print("\nopportunities per half-inning, by challenger:")
    print(hi.groupby("challenger").agg(["mean", "median", "max"]).to_string())


if __name__ == "__main__":
    main()
