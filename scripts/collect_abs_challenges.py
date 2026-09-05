"""
Pull every ABS-challenged pitch from the 2026 season out of Savant's gf feed
and save it as a flat table. This is the pitch-level challenge dataset the
rest of Part 1 (geometry validation, measured heights, decision model) builds on.

Run: python scripts/collect_abs_challenges.py
Output: data/abs_challenges.parquet
"""
import json
import sys
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries

START_DATE = "2026-03-26"
END_DATE = "2026-09-03"
N_WORKERS = 8

PITCH_FIELDS = [
    "game_pk", "play_id", "ab_number", "pitch_number", "inning", "half_inning",
    "batter", "batter_name", "pitcher", "pitcher_name", "stand", "p_throws",
    "call", "call_name", "pitch_type", "pitch_name",
    "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "plate_x", "plate_z", "sz_top", "sz_bot",
]


def final_game_pks():
    """Every completed game_pk in the window, deduplicated.

    The schedule API lists a game_pk once per date it appears under, and a
    single game_pk can appear under more than one date entry (confirmed
    2026-09-05: 26 game_pks each appeared twice in this exact window, almost
    certainly doubleheader/makeup-game listings carried on both their
    original and rescheduled dates). Without deduplicating here, each such
    game gets fetched and appended twice, silently doubling its challenges
    in the saved output -- this bit us for real (121 duplicate rows across
    those exact 26 games) before this fix. Dedupe at the source rather than
    downstream: every script that reads data/abs_challenges.parquet is
    entitled to assume it has no duplicate games in it.
    """
    r = get_with_retries(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "startDate": START_DATE, "endDate": END_DATE},
    )
    pks = []
    seen = set()
    for day in r.json().get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            pk = g["gamePk"]
            if pk in seen:
                continue
            seen.add(pk)
            pks.append(pk)
    return pks


def fetch_challenges(game_pk):
    url = f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
    try:
        r = get_with_retries(url)
        blob = r.json()
    except Exception as e:
        return game_pk, [], str(e)

    rows = []
    seen_play_ids = set()
    for side in ("team_home", "team_away"):
        for p in blob.get(side) or []:
            if not p.get("is_abs_challenge"):
                continue
            pid = p.get("play_id")
            if pid in seen_play_ids:
                continue  # same pitch can appear in both batter/pitcher-indexed arrays
            seen_play_ids.add(pid)
            rec = {f: p.get(f) for f in PITCH_FIELDS}
            ac = p.get("abs_challenge") or {}
            rec.update({
                "is_overturned": ac.get("is_overturned"),
                "challenge_team_id": ac.get("challenge_team_id"),
                "challenging_player_id": ac.get("challenging_player_id"),
                "challenging_player_type": ac.get("challenging_player_type"),
                "edge_distance": ac.get("edge_distance"),
                "edge_distance_calc": ac.get("edge_distance_calc"),
            })
            rows.append(rec)
    return game_pk, rows, None


def main():
    pks = final_game_pks()
    print(f"{len(pks)} final games between {START_DATE} and {END_DATE}")

    all_rows = []
    n_errors = 0
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(fetch_challenges, pk): pk for pk in pks}
        for fut in as_completed(futures):
            game_pk, rows, err = fut.result()
            done += 1
            if err:
                n_errors += 1
                print(f"  [{done}/{len(pks)}] game {game_pk} FAILED: {err}")
            else:
                all_rows.extend(rows)
            if done % 200 == 0:
                print(f"  progress: {done}/{len(pks)} games, "
                      f"{len(all_rows)} challenged pitches so far, {n_errors} errors")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal challenged pitches collected: {len(df)}")
    print(f"Games with fetch errors: {n_errors}")
    if len(df):
        dup_play_id = df.duplicated("play_id").sum()
        dup_key = df.duplicated(["game_pk", "ab_number", "pitch_number"]).sum()
        if dup_play_id or dup_key:
            raise RuntimeError(
                f"data/abs_challenges.parquet would contain duplicate rows "
                f"({dup_play_id} by play_id, {dup_key} by (game_pk, ab_number, "
                f"pitch_number)) -- refusing to save. This is the exact failure "
                f"mode fixed in final_game_pks() (duplicate game_pks from the "
                f"schedule API); a duplicate surviving that fix means a new "
                f"cause, not a rerun of the old one. Investigate before saving.")
        df.to_parquet("data/abs_challenges.parquet", index=False)
        print("Saved to data/abs_challenges.parquet (no duplicate rows)")
        print(df["call_name"].value_counts())
        print(df["is_overturned"].value_counts())


if __name__ == "__main__":
    main()
