#!/bin/bash
# Deploy Eva's World from this directory (the source of truth) to nova.
#
#   ./deploy.sh            copy the app, install hub deps from the lockfile,
#                          restart the hub and (needs passwordless sudo on
#                          the Mac) Eva's mind
#   ./deploy.sh --install  full install on nova (tools, Mindcraft, Paper,
#                          launchd) + the Mac side; needs ACCEPT_MINECRAFT_EULA=yes
#                          the first time
#
# The hub restart does NOT stop a running world: the hub adopts it again.
set -euo pipefail

HOST="${NOVA_HOST:-100.83.184.117}"
HERE="$(cd "$(dirname "$0")" && pwd)"
APP='~/Data/AI/World/app'

echo "→ copying app to $HOST"
ssh "$HOST" "mkdir -p ~/Data/AI/World/app"
rsync -a --delete --exclude node_modules --exclude 'config/world.json' --exclude '.DS_Store' \
  "$HERE/nova/" "$HOST:Data/AI/World/app/"

if [ "${1:-}" = "--install" ]; then
  echo "→ installing on $HOST"
  ssh "$HOST" "ACCEPT_MINECRAFT_EULA='${ACCEPT_MINECRAFT_EULA:-}' /bin/bash $APP/scripts/install_nova.sh"
else
  echo "→ hub deps + restart"
  ssh "$HOST" "cd $APP/hub && PATH=/opt/homebrew/opt/node@20/bin:\$PATH npm ci --no-fund --no-audit >/dev/null && sudo -n launchctl kickstart -k system/com.local.eva-world-hub"
fi

echo "→ hub key to the Mac"
scp -q "$HOST:Data/AI/World/.hub_key" "$HOME/Data/AI/.world_hub_key"
chmod 600 "$HOME/Data/AI/.world_hub_key"

if [ "${1:-}" = "--install" ]; then
  echo "→ installing Eva's mind on the Mac"
  /bin/bash "$HERE/mac/install_mac.sh"
else
  if sudo -n launchctl kickstart -k system/com.local.eva-world-mind 2>/dev/null; then
    echo "→ Eva's mind restarted"
  else
    echo "→ Eva's mind NOT restarted (no passwordless sudo here): run" >&2
    echo "    sudo launchctl kickstart -k system/com.local.eva-world-mind" >&2
  fi
fi

sleep 2
code=$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:8090/api/status" || true)
echo "hub: HTTP $code (401 = up, key required)"
echo "monitor: http://$HOST:8090/"
