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
#   scripts/ci.sh --fast          # skip black, tests only
#   scripts/ci.sh --strict-format # make the black --check drift fatal
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
STRICT_FORMAT=0

for arg in "$@"; do
    case "$arg" in
        --fix)   BLACK_MODE="" ;;          # rewrite in place instead of checking
        --fast)  RUN_BLACK=0 ;;            # tests only
        --no-tests) RUN_TESTS=0 ;;
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
