#!/bin/bash
# Bring Mindcraft and the server in line with Eva's World config.
# Run by the hub before EVERY world start (and by install_nova.sh), so it must
# be idempotent and it must FAIL LOUDLY: a patch that silently stopped
# applying (an `npm install` re-ran, upstream changed a line) would leave the
# viewer open to the network or the bot on stock settings.
#
#   1. copy our settings / body profile / keys over Mindcraft's
#   2. viewer: listen on 127.0.0.1 only, serve under /viewer, follow camera;
#      mineflayer: never send a non-finite position (server kicks for it);
#      the viewer draws the body with Eva's skin, not Steve's, and can
#      re-centre on her (the monitor's "find Eva" button);
#      Mindcraft: a craft is reported only once the server has confirmed it
#   3. Paper: bStats metrics off (no outbound reporting)
set -euo pipefail

WORLD="${WORLD:-$HOME/Data/AI/World}"
APP="$WORLD/app"
MC="$WORLD/mindcraft"
BIND_IP="$(/usr/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("bind",""))' "$APP/config/world.json")"
[ -n "$BIND_IP" ] || { echo "apply_patches: no bind address in $APP/config/world.json" >&2; exit 1; }

# 1. config
sed "s/@BIND_IP@/$BIND_IP/" "$APP/config/mindcraft_settings.js" > "$MC/settings.js"
cp "$APP/config/eva_body.json" "$MC/eva_body.json"
cp "$APP/config/keys.json" "$MC/keys.json"
grep -q "\"host\": \"$BIND_IP\"" "$MC/settings.js" || { echo "apply_patches: settings.js host not set" >&2; exit 1; }

# 2a. viewer listens on localhost only
PV="$MC/node_modules/prismarine-viewer/lib/mineflayer.js"
if ! grep -q "http.listen(port, '127.0.0.1'" "$PV"; then
  sed -i '' "s/http.listen(port, () =>/http.listen(port, '127.0.0.1', () =>/" "$PV"
fi
grep -q "http.listen(port, '127.0.0.1'" "$PV" || { echo "apply_patches: viewer localhost patch did not apply ($PV)" >&2; exit 1; }

# 2b. viewer under /viewer with a follow camera (you can orbit around the body)
BV="$MC/src/agent/vision/browser_viewer.js"
if ! grep -q "prefix: '/viewer'" "$BV"; then
  sed -i '' "s|mineflayerViewer(bot, { port: 3000+count_id, firstPerson: true, });|mineflayerViewer(bot, { port: 3000+count_id, firstPerson: false, prefix: '/viewer' });|" "$BV"
fi
grep -q "prefix: '/viewer'" "$BV" || { echo "apply_patches: viewer prefix patch did not apply ($BV)" >&2; exit 1; }

# 2c. mineflayer: never send a non-finite position (see patch_physics.cjs)
PH="$MC/node_modules/mineflayer/lib/plugins/physics.js"
"${NODE:-/opt/homebrew/opt/node@20/bin/node}" "$APP/scripts/patch_physics.cjs" "$PH"
grep -q "EVA-WORLD-GUARD v2" "$PH" || { echo "apply_patches: physics guard did not apply ($PH)" >&2; exit 1; }

# 2d. the viewer draws the body as a 1.16.4 "player", i.e. textures/1.16.4/
# entity/steve.png. Give it Noor (wide-arm copy, to fit that model): the
# default skin a real 1.21 client shows for an offline player named "Eva".
PVT="$MC/node_modules/prismarine-viewer/public/textures"
SKIN="$PVT/1.21.4/entity/player/wide/noor.png"
STEVE="$PVT/1.16.4/entity/steve.png"
[ -f "$SKIN" ] && [ -f "$STEVE" ] || { echo "apply_patches: viewer skin files missing ($PVT)" >&2; exit 1; }
[ -f "$STEVE.orig" ] || cp "$STEVE" "$STEVE.orig"
cmp -s "$SKIN" "$STEVE" || cp "$SKIN" "$STEVE"
cmp -s "$SKIN" "$STEVE" || { echo "apply_patches: viewer skin patch did not apply ($STEVE)" >&2; exit 1; }

# 2e. viewer: a re-centre hook for the monitor's "find Eva" button (see patch_viewer.cjs)
PVJ="$MC/node_modules/prismarine-viewer/public/index.js"
"${NODE:-/opt/homebrew/opt/node@20/bin/node}" "$APP/scripts/patch_viewer.cjs" "$PVJ"
grep -q "EVA-WORLD-RECENTER v2" "$PVJ" || { echo "apply_patches: viewer recenter hook did not apply ($PVJ)" >&2; exit 1; }

# 2f. Mindcraft: confirm a craft with the server before reporting it (see patch_craft.cjs)
SK="$MC/src/agent/library/skills.js"
"${NODE:-/opt/homebrew/opt/node@20/bin/node}" "$APP/scripts/patch_craft.cjs" "$SK"
grep -q "EVA-WORLD-CRAFT-CHECK v1" "$SK" || { echo "apply_patches: craft check did not apply ($SK)" >&2; exit 1; }

# 3. Paper metrics off
mkdir -p "$WORLD/server/plugins/bStats"
cat > "$WORLD/server/plugins/bStats/config.yml" <<'EOF'
# Eva's World: no metrics reporting.
enabled: false
serverUuid: 00000000-0000-0000-0000-000000000000
logFailedRequests: false
logSentData: false
logResponseStatusText: false
EOF

echo "apply_patches: ok (bind $BIND_IP)"
