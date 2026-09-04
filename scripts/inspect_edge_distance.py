"""
Scan recent 2026 games for overturned ABS challenges, classify by original
call (strike vs ball), and report edge_distance sign for each. Also checks
whether edge_distance/abs_challenge appears on non-challenged called pitches
(coverage check).

Run: python scripts/inspect_edge_distance.py
"""
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries


def games_for_date(d):
    url = "https://statsapi.mlb.com/api/v1/schedule"
    r = get_with_retries(url, params={"sportId": 1, "date": d.isoformat()})
    out = []
    for day in r.json().get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                out.append(g["gamePk"])
    return out


def gf(game_pk):
    url = f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
    r = get_with_retries(url)
    return r.json()


def all_pitches(blob):
    # team_home/team_away contain every pitch of the game, no duplicates across the two
    for side in ("team_home", "team_away"):
        for p in blob.get(side) or []:
            yield p


def main():
    found_strike = None
    found_ball = None
    coverage_checked = False
    n_called_total = 0
    n_called_with_key = 0
    n_games_scanned = 0

    d = date(2026, 9, 3)
    end = date(2026, 3, 26)
    while d >= end and (found_strike is None or found_ball is None):
        try:
            pks = games_for_date(d)
        except Exception as e:
            print(f"{d}: schedule fetch failed: {e}")
            d -= timedelta(days=1)
            continue

        for pk in pks:
            n_games_scanned += 1
            try:
                blob = gf(pk)
            except Exception as e:
                print(f"  game {pk}: gf fetch failed: {e}")
                continue

            for p in all_pitches(blob):
                call_name = p.get("call_name")
                is_called = call_name in ("Called Strike", "Ball")

                # coverage check: sample the first ~500 called pitches seen
                if is_called and n_called_total < 2000:
                    n_called_total += 1
                    if "abs_challenge" in p:
                        n_called_with_key += 1

                if not p.get("is_abs_challenge"):
                    continue
                ac = p.get("abs_challenge") or {}
                if not ac.get("is_overturned"):
                    continue

                rec = {
                    "game_pk": pk,
                    "call_name": call_name,
                    "edge_distance": ac.get("edge_distance"),
                    "edge_distance_calc": ac.get("edge_distance_calc"),
                    "batter_name": p.get("batter_name"),
                    "pitcher_name": p.get("pitcher_name"),
                    "play_id": p.get("play_id"),
                    "plate_x": p.get("plate_x"),
                    "plate_z": p.get("plate_z"),
                    "sz_top": p.get("sz_top"),
                    "sz_bot": p.get("sz_bot"),
                }
                if call_name == "Called Strike" and found_strike is None:
                    found_strike = rec
                    print("FOUND overturned called strike:", json.dumps(rec, indent=2))
                elif call_name == "Ball" and found_ball is None:
                    found_ball = rec
                    print("FOUND overturned called ball:", json.dumps(rec, indent=2))

            time.sleep(0.15)

            if found_strike is not None and found_ball is not None:
                break
        print(f"{d}: scanned {len(pks)} games (cumulative games={n_games_scanned}, "
              f"called_pitches_sampled={n_called_total}, with_abs_challenge_key={n_called_with_key})")
        d -= timedelta(days=1)

    print("\n=== SUMMARY ===")
    print("Overturned called strike example:", json.dumps(found_strike, indent=2))
    print("Overturned called ball example:", json.dumps(found_ball, indent=2))
    print(f"\nCoverage check: of {n_called_total} called (ball/strike) pitches sampled, "
          f"{n_called_with_key} had an 'abs_challenge' key present "
          f"({100*n_called_with_key/max(n_called_total,1):.2f}%).")
    print(f"Games scanned: {n_games_scanned}")


if __name__ == "__main__":
    main()
