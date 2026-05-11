#!/usr/bin/env bash
# Orchestrate the full selfhood functional test:
#
#   1. Kill any running agent, wipe state.
#   2. Start agent (default config), run sections A-D, F-H.
#   3. Kill agent, restart (state persists), run section E (cross-restart).
#   4. Kill agent, restart with --no-self-model, run disable-path check.
#   5. Print combined summary.
#
# Each phase tee's output to a per-phase log so the harness output can
# be inspected after the fact.

set -u

REPO=/Users/vasilis/Data/AI/Agent
LOG=/Users/vasilis/Data/AI/Logs/ghost-agent.log
SANDBOX=/Users/vasilis/Data/AI/Data/sandbox
MEMDIR=/Users/vasilis/Data/AI/Data/system/memory
SELFHOOD=/Users/vasilis/Data/AI/Data/system/selfhood
TRAJ=/Users/vasilis/Data/AI/Data/system/trajectories
REPORTS=/Users/vasilis/Data/AI/Logs/selfhood_func
mkdir -p "$REPORTS"

cyan() { printf "\033[36m%s\033[0m\n" "$1"; }
red()  { printf "\033[31m%s\033[0m\n" "$1"; }
green(){ printf "\033[32m%s\033[0m\n" "$1"; }

kill_agent() {
    local pid
    pid=$(ps auxw | grep "ghost_agent.main" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null
        for i in $(seq 1 20); do
            if ! ps -p "$pid" > /dev/null 2>&1; then
                cyan "  killed pid=$pid after ${i}s"
                return 0
            fi
            sleep 1
        done
        red "  WARN: pid=$pid did not exit after 20s; sending SIGKILL"
        kill -9 "$pid" 2>/dev/null
    fi
}

start_agent() {
    local extra_flags="${1:-}"
    cyan "Starting agent (extra_flags='$extra_flags')..."
    # Inline-substitute the extra flag into a fresh wrapper script so the
    # exec line in start-ghost-agent.sh stays unchanged.
    local wrapper=/tmp/selfhood_wrapper_$$.sh
    sed "s|exec /Users/vasilis/Data/AI/.agent.venv/bin/python -m src.ghost_agent.main|exec /Users/vasilis/Data/AI/.agent.venv/bin/python -m src.ghost_agent.main $extra_flags|" \
        "$REPO/../bin/start-ghost-agent.sh" > "$wrapper"
    chmod +x "$wrapper"

    GHOST_HOME=/Users/vasilis/Data/AI/Data/ "$wrapper" >> "$LOG" 2>&1 &
    local script_pid=$!
    cyan "  wrapper pid=$script_pid (will exec into python)"

    # Wait up to 90s for "system ready" since the LAST kill_agent (use
    # log tail to avoid false positives from earlier boots).
    local marker_count_before
    marker_count_before=$(grep -c "system ready" "$LOG" 2>/dev/null | head -1)
    : "${marker_count_before:=0}"
    for i in $(seq 1 90); do
        local now
        now=$(grep -c "system ready" "$LOG" 2>/dev/null | head -1)
        if [ "${now:-0}" -gt "${marker_count_before:-0}" ]; then
            green "  agent ready after ${i}s"
            sleep 2  # extra settle
            return 0
        fi
        sleep 1
    done
    red "  ERROR: agent did not signal ready within 90s"
    tail -40 "$LOG"
    return 1
}

wipe_all() {
    cyan "Wiping sandbox, memory, selfhood, trajectories, log..."
    rm -rf "$SANDBOX"/* 2>/dev/null
    rm -rf "$MEMDIR"/* 2>/dev/null
    rm -rf "$SELFHOOD"/* 2>/dev/null
    rm -rf "$TRAJ"/* 2>/dev/null
    rm -f "$LOG"
    touch "$LOG"
}

print_header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

cd "$REPO"

# ----------------------------------------------------------------
# Phase 1: main suite (A-D, F-H — every section except E)
# ----------------------------------------------------------------
print_header "PHASE 1: Main suite (wipe, start, sections A-D + F-H)"
kill_agent
wipe_all
start_agent "" || { red "phase1 boot failed"; exit 1; }

PYTHONPATH=src python scripts/selfhood_functional_test.py --mode main 2>&1 \
    | tee "$REPORTS/phase1_main.log"
PHASE1_RC=${PIPESTATUS[0]}

# ----------------------------------------------------------------
# Phase 2: cross-restart recall
# ----------------------------------------------------------------
print_header "PHASE 2: Kill + restart, run section E (cross-restart recall)"
kill_agent
# DO NOT wipe — that's the whole point of this test.
cyan "preserving on-disk selfhood across restart"
ls -la "$SELFHOOD"
start_agent "" || { red "phase2 boot failed"; exit 1; }
PYTHONPATH=src python scripts/selfhood_functional_test.py --mode recall_only 2>&1 \
    | tee "$REPORTS/phase2_recall.log"
PHASE2_RC=${PIPESTATUS[0]}

# ----------------------------------------------------------------
# Phase 3: --no-self-model disable check
# ----------------------------------------------------------------
print_header "PHASE 3: Restart with --no-self-model"
kill_agent
wipe_all   # wipe to make the test clean
start_agent "--no-self-model" || { red "phase3 boot failed"; exit 1; }
PYTHONPATH=src python scripts/selfhood_functional_test.py --mode disabled_check 2>&1 \
    | tee "$REPORTS/phase3_disabled.log"
PHASE3_RC=${PIPESTATUS[0]}

# ----------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------
print_header "FINAL"
echo "Phase 1 (main suite):       rc=$PHASE1_RC"
echo "Phase 2 (cross-restart):    rc=$PHASE2_RC"
echo "Phase 3 (--no-self-model):  rc=$PHASE3_RC"
echo ""
echo "Per-phase logs in $REPORTS/"
echo "Agent log in $LOG"

if [ "$PHASE1_RC" -eq 0 ] && [ "$PHASE2_RC" -eq 0 ] && [ "$PHASE3_RC" -eq 0 ]; then
    green "ALL FUNCTIONAL TESTS PASSED"
    exit 0
else
    red "ONE OR MORE PHASES FAILED"
    exit 1
fi
