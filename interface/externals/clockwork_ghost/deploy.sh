#!/bin/bash
# Deploy the uConsole client to the ClockworkPi.
#
# Things this script exists to prevent, each learned the hard way:
#   1. The LIVE client is ~/bin/client.py on the device — NOT ~/clockwork_ghost/,
#      which does not exist there. Deploying to the obvious-looking path is a
#      silent no-op.
#   2. webface/matrix_graph.js must be a COPY of the canonical web-UI file
#      (interface/static/matrix_graph.js). Re-copying on every deploy is what
#      keeps the handheld's face identical to the browser's instead of slowly
#      drifting into a fork.
#   3. (2026-10-01) client.py cannot be imported anywhere but on the device —
#      PyQt6 lives there — so a NameError in it used to be discovered by the
#      operator, on the panel, after the old client had already been killed.
#      The new build is now copied to a STAGING directory, compiled and
#      PROBED there (device_probe.py drives the real window offscreen against
#      a fake agent), and installed only if that passes. A failed probe leaves
#      the running client and ~/bin exactly as they were.
#
# Usage:  ./deploy.sh [host]     (default host: 100.91.39.52)
set -euo pipefail

HOST="${1:-100.91.39.52}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CANON="$HERE/../../static/matrix_graph.js"
# Every module the client imports. tests/test_clockwork_ux.py walks the
# imports and fails if one is missing from this line — a module that is not
# deployed is an ImportError on the device and a client that will not start.
MODULES="client.py webface.py chatlog.py turnstatus.py facestate.py markup.py speech.py devstatus.py agentapi.py commands.py device_probe.py"
STAGE=/home/vasilis/bin/.stage

echo "→ syncing canonical face module"
cp "$CANON" "$HERE/webface/matrix_graph.js"

echo "→ staging the new build (the live client is untouched)"
ssh "$HOST" "rm -rf $STAGE && mkdir -p $STAGE/webface"
# shellcheck disable=SC2086
(cd "$HERE" && scp -q $MODULES launch_ghost.sh "$HOST:$STAGE/")
scp -qr "$HERE/webface/." "$HOST:$STAGE/webface/"

echo "→ compile check + device probe, in staging (venv python — the system one has no httpx)"
# `timeout`: a probe that hangs must fail the deploy, not hang it (and be left
# running on the device when the operator gives up and presses Ctrl-C).
ssh "$HOST" "source ~/gui_env/bin/activate && cd $STAGE && python3 -m py_compile $MODULES && echo '  compile OK' && timeout 180 python3 device_probe.py"

echo "→ installing"
ssh "$HOST" "bash -s" <<EOS
set -e
cp ~/bin/client.py ~/bin/client.py.bak-\$(date +%Y%m%d-%H%M%S) 2>/dev/null || true
# Keep the three newest backups. Every deploy used to leave one behind
# forever — 22 of them by 2026-10-01.
ls -1t ~/bin/client.py.bak-* 2>/dev/null | tail -n +4 | xargs -r rm -f
cp $STAGE/*.py ~/bin/
# The launcher is a RUNNING bash script (the old client's supervisor). bash
# reads a script as it goes, so copying over it in place makes the running
# copy continue at its old offset in the new text. Write beside it and rename:
# the running one keeps its file, the next start gets the new one.
cp $STAGE/launch_ghost.sh ~/bin/.launch_ghost.sh.new && chmod +x ~/bin/.launch_ghost.sh.new
mv -f ~/bin/.launch_ghost.sh.new ~/bin/launch_ghost.sh
mkdir -p ~/bin/webface && cp -a $STAGE/webface/. ~/bin/webface/
rm -rf $STAGE
EOS

echo "→ restarting"
# NOTE: driven over STDIN (bash -s). `ssh host "pkill -f bin/client.py; …"`
# self-matches — the pattern appears in the remote shell's own cmdline, so it
# kills itself before the relaunch runs and leaves the client dead.
ssh "$HOST" "bash -s" <<'EOS'
# The LAUNCHER first: it restarts a client that crashes, and although it does
# not restart one that was signalled, killing it first means the old client
# cannot come back whatever killed it.
pkill -f 'launch_ghost.sh' || true
# Pattern is 'python3.*client\.py', NOT 'bin/client.py' (2026-08-03): a client started
# by hand from ~/bin has the cmdline `python3 client.py`, which the narrower
# pattern does not match. One such instance survived FIVE deploys, and two
# clients on this device is not a cosmetic problem — they compete for the CM4's
# single GPU, the second face never gets a WebGL context, and the operator sees
# a stale build's window on top of the new one.
pkill -f 'python3.*client\.py' || true
sleep 2
# WAYLAND_DISPLAY / XDG_RUNTIME_DIR: a client started by the session has them;
# one started from here did not, and the panel wake (wlopm) needs both.
setsid env DISPLAY=:0 XAUTHORITY=/home/vasilis/.Xauthority \
  XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=wayland-0 \
  /home/vasilis/bin/launch_ghost.sh > /dev/null 2>&1 < /dev/null &
sleep 6
running=$(pgrep -fc 'python3.*client\.py' || true)
case "$running" in
  0|"") echo "  CLIENT DID NOT START — log:"; tail -20 /tmp/ghost_ui.log; exit 1 ;;
  1)    : ;;
  *)    echo "  WARNING: $running clients running — they will fight over the GPU:"
        pgrep -fa 'python3.*client\.py' ;;
esac
# "One client is alive" is not "the client started". The launcher restarts a
# crash, so a build that dies a few seconds in is alive again whenever anyone
# looks — and then gives up, leaving a bare desktop after the deploy said OK.
# So: the SAME pid six seconds apart, and no crash in the launcher's log.
pid_a=$(pgrep -f 'python3.*client\.py' | head -1)
sleep 6
pid_b=$(pgrep -f 'python3.*client\.py' | head -1)
if grep -q 'CRASHED' /tmp/ghost_ui.log || [ "$pid_a" != "$pid_b" ]; then
  echo "  CLIENT IS CRASHING ON START — log:"; tail -30 /tmp/ghost_ui.log; exit 1
fi
echo "  client running (pid $pid_b, stable)"
# The face applies the remembered form only once its WebGL context exists; a
# panel that swayidle powered off stalls that indefinitely, so a missing line
# here means "screen asleep", not "broken build".
grep '\[face\]' /tmp/ghost_ui.log || echo "  (face not ready yet — panel asleep? wlopm --on '*')"
if grep -q '\[face\] ERROR' /tmp/ghost_ui.log; then
  echo "  THE FACE FAILED TO START (see the [face] ERROR line above) — the client runs, on a black background"
  exit 1
fi
EOS
echo "✓ deployed"
