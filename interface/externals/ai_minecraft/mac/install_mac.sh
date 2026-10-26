#!/bin/bash
# Install Eva's mind on the Mac as a launchd service (system domain, runs as
# you). It runs eva_mind.py from THIS directory, so a code change here takes
# effect on `sudo launchctl kickstart -k system/com.local.eva-world-mind`.
#
# Needs: ~/Data/AI/.world_hub_key (deploy.sh copies it from nova) and the
# agent key ~/Data/AI/.ghost_api_key (already there).
set -euo pipefail

[ "$(id -u)" -ne 0 ] || { echo "run this as yourself, not with sudo: it asks for sudo where it needs it" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME=com.local.eva-world-mind
[ -s "$HOME/Data/AI/.world_hub_key" ] || { echo "missing ~/Data/AI/.world_hub_key (run deploy.sh first)" >&2; exit 1; }
[ -s "$HOME/Data/AI/.ghost_api_key" ] || { echo "missing ~/Data/AI/.ghost_api_key" >&2; exit 1; }
mkdir -p "$HOME/Data/AI/Logs"

# the eva-world command (status/start/stop/pause/resume/journal/reset)
mkdir -p "$HOME/.local/bin"
ln -sf "$HERE/eva-world" "$HOME/.local/bin/eva-world"

sed -e "s|@DIR@|$HERE|g" -e "s|@HOME@|$HOME|g" -e "s|@USER@|$(id -un)|g" "$HERE/$NAME.plist" > "/tmp/$NAME.plist"
sudo install -m 644 -o root -g wheel "/tmp/$NAME.plist" "/Library/LaunchDaemons/$NAME.plist"
sudo launchctl bootout "system/$NAME" 2>/dev/null || true
sudo launchctl bootstrap system "/Library/LaunchDaemons/$NAME.plist"
sleep 3
sudo launchctl print "system/$NAME" | grep -E "state =|pid =" | head -2
tail -3 "$HOME/Data/AI/Logs/eva_world_mind.log" 2>/dev/null || true
