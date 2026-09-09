#!/usr/bin/env bash
#
# The single entry point for a full data + app-data refresh. Runs the exact
# same steps locally and in CI (.github/workflows/refresh.yml calls this
# script and nothing else); the workflow only adds the git checkout/commit/
# push around it, so a local run never touches git.
#
# What it does, in dependency order:
#   1. re-pull Statcast (trailing window; src/ingest.py is idempotent)
#   2. rebuild the DuckDB statcast table + run expectancy
#   3. collect ABS challenge records (end date = yesterday)
#   4. back out per-batter measured heights (fixed early-season window)
#   5. build challenge opportunities -> perception sigma -> the policy DP
#   6. the team / player / catcher skill follow-ups
#   7. zone location-dependence test + the two data-provenance checks
#   8. decision-tool lookups + case studies
#   9. build_app_data.py  (consolidates -> app/data/, runs its own guards)
#  10. build_reported_figures.py  (writes reported_figures.json AND fails the
#      run if any prose figure has drifted past tolerance)
#
# FROZEN, deliberately NOT re-run here: scripts/zone_sigma_refit.py and
# scripts/zone_sigma_bootstrap.py. Per CLAUDE.md that sensitivity number is
# noise-dominated at one season of challenge data and the bootstrap is 150
# full DP re-solves. Their committed outputs in app/data/ are the source of
# truth; this script seeds them back into data/ so the downstream copy and
# provenance checks pass unchanged. Re-run them by hand when the season is
# complete (or the method changes) and commit the new parquet.
#
# Usage:
#   scripts/refresh_all.sh                 # full run
#   scripts/refresh_all.sh --skip-ingest   # reuse the cached Statcast pull
#   ABS_END_DATE=2026-08-31 scripts/refresh_all.sh   # reproducible back-fill
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# Use the repo venv locally if one exists and we're not already inside a venv.
# In CI, python is already set up by the workflow and .venv is absent.
if [[ -z "${VIRTUAL_ENV:-}" && -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi
PYTHON="${PYTHON:-python}"

SKIP_INGEST=0
for arg in "$@"; do
  case "$arg" in
    --skip-ingest) SKIP_INGEST=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

step() { echo; echo "=== $* ==="; }
run()  { step "$*"; "$PYTHON" "$@"; }

mkdir -p data app/data

# --- seed the frozen artifacts from their committed copies -------------------
step "seed frozen artifacts (zone-sigma sensitivity) from app/data/ -> data/"
for f in zone_sigma zone_sigma_sensitivity zone_sigma_sensitivity_bootstrap; do
  if [[ -f "app/data/${f}.parquet" ]]; then
    cp "app/data/${f}.parquet" "data/${f}.parquet"
    echo "  seeded data/${f}.parquet"
  else
    echo "  WARNING: app/data/${f}.parquet missing -- frozen artifact unavailable" >&2
  fi
done

# --- 1-2. raw data + DuckDB + run expectancy --------------------------------
if [[ "$SKIP_INGEST" -eq 0 ]]; then
  run src/ingest.py
else
  step "src/ingest.py (SKIPPED: --skip-ingest)"
fi
run src/run_expectancy.py

# --- 3-4. challenge records + measured heights ------------------------------
run scripts/collect_abs_challenges.py
run scripts/verify_ball_radius.py

# --- 5. opportunities -> sigma -> policy DP --------------------------------
run scripts/build_challenge_opportunities.py
run scripts/estimate_perception_sigma.py
run src/abs_policy.py

# --- 6. team / player / catcher skill follow-ups --------------------------
run scripts/team_decomposition.py
run scripts/team_skill_test.py
run scripts/player_skill_test.py

# --- 7. zone test + provenance checks -----------------------------------
run scripts/zone_analysis.py
run scripts/validate_ball_radius_classification.py
run scripts/measured_height_uncertainty.py

# --- 8. decision-tool lookups + case studies ---------------------------
run scripts/build_decision_tool_data.py
run scripts/build_case_studies.py

# --- 9. consolidate into app/data/ (runs its own guards) --------------
run scripts/build_app_data.py

# --- 10. provenance snapshot + prose-drift gate ----------------------
run scripts/build_reported_figures.py

step "refresh complete"
echo "app/data/ is rebuilt and every tracked prose figure matches the data."
echo "Review 'git status' / 'git diff' before committing."
