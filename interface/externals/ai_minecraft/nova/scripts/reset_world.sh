#!/bin/bash
# Wipe Eva's World back to nothing: the world, every session, her journal,
# the prompt archive and the body's memory. NO backup (operator, 10-26).
# Refuses unless the world is stopped (press Stop / `eva-world stop` first),
# because the server and the body run detached and hold those files.
# The next Start generates a new world from a new random seed.
#
#   reset_world.sh --yes
set -euo pipefail

WORLD="${WORLD:-$HOME/Data/AI/World}"
RUN="$WORLD/run"
[ "${1:-}" = "--yes" ] || { echo "reset_world: deletes the world and the whole record; run with --yes" >&2; exit 2; }

phase="$(/usr/bin/python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("phase",""))
except FileNotFoundError: print("stopped")' "$RUN/state.json")"
[ "$phase" = "stopped" ] || { echo "reset_world: the world is $phase; stop it first" >&2; exit 1; }
for p in server mindcraft; do
  pid="$(cat "$RUN/$p.pid" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "reset_world: $p (pid $pid) is still running; stop the world first" >&2; exit 1
  fi
done

rm -rf "$WORLD/server/evaworld" "$WORLD/server/evaworld_nether" "$WORLD/server/evaworld_the_end" \
       "$WORLD/data/sessions" "$WORLD/data/eva" "$WORLD/mindcraft/bots/Eva"
# the hub creates these only at startup
mkdir -p "$WORLD/data/sessions" "$WORLD/data/eva/prompts"
echo "reset_world: wiped world, sessions, journal, prompts, body memory"

# a fresh hub: no attention or watchdog state from the old world
${HUB_RESTART_CMD:-sudo -n launchctl kickstart -k system/com.local.eva-world-hub}
echo "reset_world: hub restarted"
