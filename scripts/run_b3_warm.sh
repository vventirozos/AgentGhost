#!/usr/bin/env bash
# Warm-store B3 idle-loop ablation.
#
# WHY: the standard B3 run gives each arm an EMPTY GHOST_HOME, so the idle
# loops have nothing to work on — the 2026-08-08 cold run logged "Auto-memory
# pool thin (0)" / "Synthesized 0 new meta-memories", and reflection, PRM and
# router never fired at all (they need a trajectory corpus). A cold result
# therefore cannot separate "these loops produce no value" from "we gave them
# nothing to chew on". This variant seeds every arm from a snapshot of the REAL
# store so the measured question becomes the one worth asking: do the idle
# loops add value to THIS agent, with ITS history?
#
# SAFETY: the live GHOST_HOME is only ever READ, and only via a snapshot taken
# once up front. Each arm still runs in its own tempdir on its own port; no arm
# can write to the live store.
#
# Usage:  scripts/run_b3_warm.sh <report-dir> [repeats]
set -euo pipefail

LIVE_HOME="${GHOST_HOME:-/Users/vasilis/Data/AI/Data}"
REPORT_DIR="${1:?usage: run_b3_warm.sh <report-dir> [repeats]}"
REPEATS="${2:-3}"
VENV="/Users/vasilis/Data/AI/.agent.venv/bin/python"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$REPORT_DIR"
SNAP="$REPORT_DIR/seed-snapshot"

echo "[warm-b3] live home : $LIVE_HOME  (READ ONLY)"
echo "[warm-b3] snapshot  : $SNAP"

# --- 1. Snapshot the live store ONCE -----------------------------------------
# Taken up front and reused by all arms so (a) every arm starts from a byte
# identical corpus — otherwise arms aren't comparable — and (b) we never read
# chroma.sqlite3 concurrently 9 separate times while the live agent writes to
# it, which can capture a torn page.
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP/system"
  for item in "$LIVE_HOME"/system/*; do
    base="$(basename "$item")"
    case "$base" in
      llm_recordings|sandbox|logs) continue ;;   # not read by any idle loop
      *.log|*.bak)                 continue ;;
    esac
    cp -R "$item" "$SNAP/system/" 2>/dev/null || echo "[warm-b3] skip $base"
  done
  echo "[warm-b3] snapshot size: $(du -sh "$SNAP" | cut -f1)"
else
  echo "[warm-b3] reusing existing snapshot"
fi

# --- 2. Integrity-check the snapshot before spending hours on it --------------
CHROMA="$SNAP/system/memory/chroma.sqlite3"
if [ -f "$CHROMA" ]; then
  RES="$(sqlite3 "$CHROMA" "PRAGMA integrity_check;" 2>&1 | head -1)"
  echo "[warm-b3] snapshot chroma integrity: $RES"
  if [ "$RES" != "ok" ]; then
    echo "[warm-b3] ABORT: snapshot is torn — re-run when the agent is quiet." >&2
    exit 1
  fi
  echo "[warm-b3] seeded rows: $(sqlite3 "$CHROMA" 'SELECT COUNT(*) FROM embedding_metadata WHERE key="type";' 2>/dev/null)"
fi

# --- 3. Run the ablation against the snapshot --------------------------------
cd "$HERE"
export GHOST_B3_SEED_HOME="$SNAP"
export GHOST_MODEL="${GHOST_MODEL:-qwen-3.6-35b-a3}"
echo "[warm-b3] starting: repeats=$REPEATS time-scale=15 idle-epochs=6 epoch-sleep=250"
exec "$VENV" scripts/ablation_trackb3.py \
  --repeats "$REPEATS" \
  --time-scale 15 \
  --idle-epochs 6 \
  --epoch-sleep 250 \
  --base-port 8056 \
  --report-dir "$REPORT_DIR"
