#!/bin/bash
# Start the uConsole client, keep a log of it, and bring it back if it crashes.
#
# Two things this did not do before 2026-10-01, both found on the device:
#
#   1. NO LOG. The autostart entry runs this script with its output going
#      nowhere, so every diagnostic the client prints — the [tls] resolve
#      check, the [face] lines, every voice error — existed only when the
#      client had been restarted by deploy.sh, which redirects by hand. After
#      an ordinary boot /tmp/ghost_ui.log did not exist. The script now owns
#      its log, so every start has one.
#   2. NO SUPERVISOR. A crashed client left a bare desktop until someone
#      noticed. The loop below restarts it — but only on a CRASH:
#        rc 0          the operator typed /exit           -> stop
#        rc 130 / 143  SIGINT / SIGTERM (deploy.sh, pkill) -> stop
#        anything else                                     -> restart
#      and it gives up after 5 crashes that each came within a minute, so a
#      client that cannot start does not spin forever. The wait between tries
#      doubles (2, 4, 8, 16 s): five tries span half a minute, not ten
#      seconds — long enough for a display or a network that was merely not
#      up yet when the session started.
#
# ⚠ deploy.sh kills THIS script before the client, in that order — killing
# only the client reads as a signal exit here and is not restarted, but the
# order keeps that from depending on which signal was used.

LOG="${GHOST_UI_LOG:-/tmp/ghost_ui.log}"
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.1" 2>/dev/null   # keep the previous run's log
# /tmp is sticky: a log left there by another user (a `sudo` run) can be
# neither moved nor appended to. Fall back rather than run with no log.
if ! ( : >>"$LOG" ) 2>/dev/null; then
    LOG="$HOME/ghost_ui.log"
fi
exec >>"$LOG" 2>&1
echo "[launch] $(date '+%F %T') starting (pid $$)"

# X11 screensaver switches. They do nothing under labwc — the panel is
# blanked by swayidle (~/.config/labwc/autostart) — and are kept only for the
# case where this is run in a plain X session.
xset s off 2>/dev/null
xset -dpms 2>/dev/null
xset s noblank 2>/dev/null

source /home/vasilis/gui_env/bin/activate
export QT_QPA_PLATFORM=xcb
export PYTHONUNBUFFERED=1                   # prints reach the log as they happen

crashes=0
while :; do
    started=$(date +%s)
    python3 /home/vasilis/bin/client.py
    rc=$?
    case "$rc" in
        0|130|143) echo "[launch] $(date '+%F %T') client stopped (rc=$rc)"; break ;;
    esac
    ran=$(( $(date +%s) - started ))
    [ "$ran" -ge 60 ] && crashes=0
    crashes=$((crashes + 1))
    echo "[launch] $(date '+%F %T') client CRASHED rc=$rc after ${ran}s (crash $crashes of 5)"
    if [ "$crashes" -ge 5 ]; then
        echo "[launch] giving up — 5 crashes in a row; see the traceback above"
        break
    fi
    sleep $((1 << crashes))
done
