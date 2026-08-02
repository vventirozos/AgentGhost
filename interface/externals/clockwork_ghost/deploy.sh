#!/bin/bash
# Deploy the uConsole client to the ClockworkPi.
#
# Two things this script exists to prevent, both learned the hard way:
#   1. The LIVE client is ~/bin/client.py on the device — NOT ~/clockwork_ghost/,
#      which does not exist there. Deploying to the obvious-looking path is a
#      silent no-op.
#   2. webface/matrix_graph.js must be a COPY of the canonical web-UI file
#      (interface/static/matrix_graph.js). Re-copying on every deploy is what
#      keeps the handheld's face identical to the browser's instead of slowly
#      drifting into a fork.
#
# Usage:  ./deploy.sh [host]     (default host: 100.91.39.52)
set -euo pipefail

HOST="${1:-100.91.39.52}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CANON="$HERE/../../static/matrix_graph.js"

echo "→ syncing canonical face module"
cp "$CANON" "$HERE/webface/matrix_graph.js"

echo "→ backing up the current device copy"
STAMP=$(date +%Y%m%d-%H%M%S)
ssh "$HOST" "cp ~/bin/client.py ~/bin/client.py.bak-$STAMP 2>/dev/null || true"

echo "→ copying client + face"
scp -q "$HERE/client.py" "$HERE/webface.py" "$HERE/chatlog.py" "$HOST:/home/vasilis/bin/"
ssh "$HOST" "mkdir -p ~/bin/webface"
scp -qr "$HERE/webface/." "$HOST:/home/vasilis/bin/webface/"

echo "→ compile check (venv python — the system one has no httpx)"
ssh "$HOST" "source ~/gui_env/bin/activate && python3 -m py_compile ~/bin/client.py ~/bin/webface.py ~/bin/chatlog.py && echo '  OK'"

echo "→ restarting"
# NOTE: driven over STDIN (bash -s). `ssh host "pkill -f bin/client.py; …"`
# self-matches — the pattern appears in the remote shell's own cmdline, so it
# kills itself before the relaunch runs and leaves the client dead.
ssh "$HOST" "bash -s" <<'EOS'
pkill -f 'bin/client.py' || true
pkill -f 'launch_ghost.sh' || true
sleep 2
setsid env DISPLAY=:0 XAUTHORITY=/home/vasilis/.Xauthority \
  /home/vasilis/bin/launch_ghost.sh > /tmp/ghost_ui.log 2>&1 < /dev/null &
sleep 6
pgrep -f 'bin/client.py' >/dev/null && echo "  client running (pid $(pgrep -f 'bin/client.py' | head -1))" \
  || { echo "  CLIENT DID NOT START — log:"; tail -20 /tmp/ghost_ui.log; exit 1; }
EOS
echo "✓ deployed"
