"""
Locate ABS challenge fields in MLB's JSON feeds.

Strategy: use a known ground-truth event. The first regular-season ABS
challenge was March 25, 2026 -- Yankees at Giants, 4th inning, Jose Caballero
challenging a first-pitch called strike from Logan Webb. Call was UPHELD.

Pull that game from both feeds and dump every key/value that smells like
a challenge, then look for the one attached to Caballero's plate appearance.

Run:  python find_abs_fields.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from net import get_with_retries

KEYWORDS = ("challenge", "abs", "review", "overturn", "confirm", "upheld")


def walk(obj, path=""):
    """Yield (path, key, value) for every leaf in a nested JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, (dict, list)):
                yield from walk(v, p)
            else:
                yield p, k, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            if isinstance(v, (dict, list)):
                yield from walk(v, p)
            else:
                yield p, str(i), v


def hits(blob):
    """Find leaves whose key OR string value mentions a challenge keyword."""
    out = []
    for path, key, val in walk(blob):
        hay = f"{key} {val}".lower()
        if any(kw in hay for kw in KEYWORDS):
            out.append((path, val))
    return out


def find_game_pk(date="2026-03-25", team_name="Yankees"):
    url = "https://statsapi.mlb.com/api/v1/schedule"
    r = get_with_retries(url, params={"sportId": 1, "date": date})
    for day in r.json().get("dates", []):
        for g in day.get("games", []):
            away = g["teams"]["away"]["team"]["name"]
            home = g["teams"]["home"]["team"]["name"]
            print(f"  {g['gamePk']}  {away} @ {home}")
            if team_name in away or team_name in home:
                return g["gamePk"]
    return None


def probe_statsapi(game_pk):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    r = get_with_retries(url)
    blob = r.json()

    with open(f"raw_statsapi_{game_pk}.json", "w") as f:
        json.dump(blob, f, indent=2)

    print("\n=== STATS API: challenge-flavored leaves ===")
    seen = set()
    for path, val in hits(blob):
        # collapse array indices so we see field shapes, not 300 duplicates
        shape = "".join(c if not c.isdigit() else "N" for c in path)
        if shape in seen:
            continue
        seen.add(shape)
        print(f"  {path}\n     -> {str(val)[:110]}")

    # Zoom in on Caballero specifically
    print("\n=== Caballero plate appearances ===")
    for play in blob.get("liveData", {}).get("plays", {}).get("allPlays", []):
        batter = play.get("matchup", {}).get("batter", {}).get("fullName", "")
        if "Caballero" not in batter:
            continue
        inning = play.get("about", {}).get("inning")
        print(f"  inning {inning}: {play.get('result', {}).get('description', '')[:80]}")
        for ev in play.get("playEvents", []):
            keys = set(ev.keys()) | set(ev.get("details", {}).keys())
            flagged = [k for k in keys if any(kw in k.lower() for kw in KEYWORDS)]
            if flagged:
                print(f"     playEvent keys of interest: {flagged}")
                print(f"     {json.dumps(ev, indent=6)[:600]}")


def probe_savant(game_pk):
    url = f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
    r = get_with_retries(url)
    blob = r.json()

    with open(f"raw_savant_{game_pk}.json", "w") as f:
        json.dump(blob, f, indent=2)

    print("\n=== SAVANT gf: challenge-flavored leaves ===")
    seen = set()
    for path, val in hits(blob):
        shape = "".join(c if not c.isdigit() else "N" for c in path)
        if shape in seen:
            continue
        seen.add(shape)
        print(f"  {path}\n     -> {str(val)[:110]}")

    # Savant pitch objects usually live under team_home / team_away
    print("\n=== Sample Savant pitch object (all available fields) ===")
    for side in ("team_home", "team_away"):
        arr = blob.get(side) or []
        if arr:
            print(f"  [{side}] {sorted(arr[0].keys())}")
            break


if __name__ == "__main__":
    print("Finding game_pk for 2026-03-25 Yankees @ Giants...")
    pk = find_game_pk()
    print(f"\nUsing game_pk = {pk}\n")

    try:
        probe_statsapi(pk)
    except Exception as e:
        print(f"statsapi probe failed: {e}")

    try:
        probe_savant(pk)
    except Exception as e:
        print(f"savant probe failed: {e}")

    print("\nRaw JSON saved to disk -- grep it by hand if nothing surfaced.")
