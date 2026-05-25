#!/usr/bin/env bash
# Schedule-friendly wrapper around the selfhood falsification probes.
#
# Runs scripts/consciousness_probe.py and
# scripts/introspective_consistency.py, captures their output, and
# writes a dated summary so drift in the selfhood substrate becomes
# observable rather than anecdotal.
#
# Intended to be invoked from cron, systemd timer, or the agent's own
# /schedule plumbing. Stdout is the human-readable summary; the
# per-run JSON / text artefacts live under $REPORTS.
#
# Exit code: 0 when both probes ran. Non-zero when the probes
# couldn't be located, the Python module path is unset, etc. The
# probes themselves are diagnostic — a failed probe is a finding, not
# an error, so we don't fail the wrapper on a non-zero probe exit.

set -u

REPO="${SELFHOOD_REPO:-/Users/vasilis/Data/AI/Agent}"
GHOST_HOME_DEFAULT="${HOME}/ghost_llamacpp"
GHOST_HOME="${GHOST_HOME:-$GHOST_HOME_DEFAULT}"
REPORTS="${SELFHOOD_PROBES_DIR:-$GHOST_HOME/system/selfhood/probes}"

mkdir -p "$REPORTS"

STAMP=$(date +%Y%m%d-%H%M%S)
SUMMARY="$REPORTS/probes-$STAMP.txt"

PROBE_CONSCIOUSNESS="$REPO/scripts/consciousness_probe.py"
PROBE_INTROSPECTIVE="$REPO/scripts/introspective_consistency.py"

for probe in "$PROBE_CONSCIOUSNESS" "$PROBE_INTROSPECTIVE"; do
    if [ ! -f "$probe" ]; then
        echo "missing probe: $probe" | tee -a "$SUMMARY"
        exit 2
    fi
done

{
    echo "=== Selfhood probe run $STAMP ==="
    echo "GHOST_HOME=$GHOST_HOME"
    echo
    echo "--- consciousness_probe.py ---"
    if python3 "$PROBE_CONSCIOUSNESS" 2>&1; then
        echo "[consciousness_probe] exit=0"
    else
        echo "[consciousness_probe] exit=$?"
    fi
    echo
    echo "--- introspective_consistency.py ---"
    if python3 "$PROBE_INTROSPECTIVE" 2>&1; then
        echo "[introspective_consistency] exit=0"
    else
        echo "[introspective_consistency] exit=$?"
    fi
    echo
    echo "=== end of run ==="
} | tee "$SUMMARY"

echo "summary: $SUMMARY"
