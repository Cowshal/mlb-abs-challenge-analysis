"""
Generate data/reported_figures.json from the pipeline's parquet outputs, then
gate the build on the prose being consistent with it.

  data/reported_figures.json      -- the machine-readable snapshot: every
                                     tracked statistic's current value, the
                                     tolerance it's checked at, and which
                                     prose files it appears in.
  app/data/reported_figures.json  -- same file, for the app footer's
                                     "Data through <date>" line.

Then src.reported_figures.check() re-derives each value and greps the prose
(README.md, docs/writeup.md, app/streamlit_app.py) for the number as written.
If anything has drifted past its per-figure tolerance -- or an anchoring
phrase can no longer be found -- this script prints every offender and exits
non-zero, which fails scripts/refresh_all.sh and the scheduled workflow
BEFORE a self-contradicting commit is pushed.

Usage:
  python scripts/build_reported_figures.py              # write JSON + check (default)
  python scripts/build_reported_figures.py --check-only # check against existing data, write nothing
  python scripts/build_reported_figures.py --write-only # write JSON, skip the gate (not for CI)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import reported_figures as rf  # noqa: E402

OUT_PIPE = ROOT / "data" / "reported_figures.json"
OUT_APP = ROOT / "app" / "data" / "reported_figures.json"


def write_snapshot() -> dict:
    snap = rf.collect()
    OUT_PIPE.parent.mkdir(parents=True, exist_ok=True)
    OUT_APP.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(snap, indent=2, sort_keys=True) + "\n"
    OUT_PIPE.write_text(body)
    OUT_APP.write_text(body)
    print(f"wrote {OUT_PIPE.relative_to(ROOT)} and {OUT_APP.relative_to(ROOT)} "
          f"-- {snap['n_figures']} figures, data through {snap['data_through']}")
    return snap


def run_gate(snap: dict | None) -> int:
    values = ({k: v["value"] for k, v in snap["figures"].items()} if snap else None)
    problems = rf.check(values)
    if not problems:
        print(f"prose-consistency OK: all {len(rf.FIGURES)} tracked figures "
              f"match README.md / docs/writeup.md / app/streamlit_app.py "
              f"within tolerance")
        return 0

    drift = [p for p in problems if p.line is not None]
    missing = [p for p in problems if p.line is None]
    print(f"\nPROSE IS OUT OF SYNC WITH THE DATA -- {len(problems)} issue(s):\n")
    if drift:
        print(f"  {len(drift)} figure(s) have drifted past tolerance:")
        for p in sorted(drift, key=lambda p: (p.path, p.line)):
            print(f"    {p}")
        print()
    if missing:
        print(f"  {len(missing)} anchor(s) could not be located "
              f"(prose reworded, or a tracked number was deleted):")
        for p in sorted(missing, key=lambda p: p.path):
            print(f"    {p}")
        print()
    print("Fix: update the number(s) in the named file(s) to the value shown, "
          "or -- if a figure was intentionally removed/reworded -- update its "
          "Occ pattern(s) in src/reported_figures.py.")
    return 1


def main(argv: list[str]) -> int:
    check_only = "--check-only" in argv
    write_only = "--write-only" in argv
    if check_only and write_only:
        print("--check-only and --write-only are mutually exclusive", file=sys.stderr)
        return 2

    snap = None
    if not check_only:
        snap = write_snapshot()
    if write_only:
        return 0
    return run_gate(snap)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
