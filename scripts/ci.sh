#!/usr/bin/env bash
#
# Local CI gate for Ghost Agent. No hosted CI runs against this repo (the
# canonical git remote lives on another machine), so this script is the
# pre-push check.
#
# The full pytest suite is the HARD gate (non-zero exit on any failure).
# black runs in --check mode but is ADVISORY: the existing tree was never
# black-formatted (hundreds of files would reorder), so a hard format gate
# would fail on legacy code unrelated to your change. It reports drift so
# new files can be kept clean; pass --strict-format to make it fatal once
# the tree has had a one-time repo-wide `black src interface tests scripts`.
#
# Usage:
#   scripts/ci.sh                 # black --check (advisory) + full pytest suite
#   scripts/ci.sh --fix           # auto-format with black, then run the suite
#   scripts/ci.sh --fast          # skip black + lint, tests only
#   scripts/ci.sh --no-lint       # skip the ~50s error-class lint gate
#   scripts/ci.sh --strict-format # make the black --check drift fatal
#
# The pylint error-class gate (§4GJ) is HARD: zero-tolerance symbols and any
# finding absent from tests/lint_baseline.json fail the build. pylint being
# MISSING also fails — an instrument that cannot run is not a pass.
#
# Exit status is non-zero if the test suite fails (or formatting fails under
# --strict-format), so it is safe to chain (e.g. `scripts/ci.sh && git push`).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the project venv if present (deps are NOT installed in the base
# interpreter); fall back to whatever python3 is on PATH.
VENV_PY="/Users/vasilis/Data/AI/.agent.venv/bin/python"
if [[ -x "$VENV_PY" ]]; then
    PY="$VENV_PY"
else
    PY="$(command -v python3)"
fi

# interface/server.py raises at import if GHOST_API_KEY is unset, which
# breaks test collection — provide a throwaway value for the run.
export GHOST_API_KEY="${GHOST_API_KEY:-test-key}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

RUN_BLACK=1
BLACK_MODE="--check"
RUN_TESTS=1
RUN_LINT=1
STRICT_FORMAT=0

for arg in "$@"; do
    case "$arg" in
        --fix)   BLACK_MODE="" ;;          # rewrite in place instead of checking
        --fast)  RUN_BLACK=0; RUN_LINT=0 ;;  # tests only
        --no-tests) RUN_TESTS=0 ;;
        --no-lint) RUN_LINT=0 ;;           # skip the ~50s error-class gate
        --strict-format) STRICT_FORMAT=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

status=0

if [[ "$RUN_BLACK" == "1" ]]; then
    echo "==> black ${BLACK_MODE:-(format)} src interface tests scripts"
    # shellcheck disable=SC2086
    "$PY" -m black $BLACK_MODE src interface tests scripts
    rc=$?
    if [[ $rc -ne 0 ]]; then
        if [[ "$BLACK_MODE" == "--check" && "$STRICT_FORMAT" != "1" ]]; then
            echo "!! formatting drift detected (advisory — not failing the build)." >&2
            echo "   run 'scripts/ci.sh --fix' to apply, or --strict-format to enforce." >&2
        else
            status=$rc
            [[ "$BLACK_MODE" == "--check" ]] && \
                echo "!! formatting check failed (--strict-format)." >&2
        fi
    fi
fi

# Error-class lint gate (§4GJ). HARD for the zero-tolerance symbols, for any
# error-class finding that is not already in tests/lint_baseline.json, and —
# since §4GK round 4 — for SLACK: an entry that outlives its finding is a
# failure, not advice. It used to be advisory, which meant the gate did not
# ratchet at all: fix a finding, leave the baseline alone, and the same defect
# could be re-introduced for free forever. Regenerate downward to clear it.
# Delegates to
# scripts/lint.py so the gate and tests/test_lint_gate.py cannot disagree
# about what "clean" means — the suite pins that they share this runner.
if [[ "$RUN_LINT" == "1" ]]; then
    echo "==> pylint (error class, gated)"
    "$PY" scripts/lint.py
    rc=$?
    if [[ $rc -eq 2 ]]; then
        echo "!! pylint is not installed — the lint gate DID NOT RUN." >&2
        echo "   This is not a pass: install pylint in the venv." >&2
        status=1
    elif [[ $rc -ne 0 ]]; then
        status=$rc
        echo "!! error-class lint findings (see above)." >&2
    fi
fi

if [[ "$RUN_TESTS" == "1" ]]; then
    echo "==> pytest (full suite)"
    "$PY" -m pytest -q
    rc=$?
    [[ $rc -ne 0 ]] && status=$rc
fi

if [[ $status -eq 0 ]]; then
    echo "==> CI PASSED"
else
    echo "==> CI FAILED (exit $status)" >&2
fi
exit $status
