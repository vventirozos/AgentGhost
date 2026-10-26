#!/bin/bash
# One-time (and re-runnable) install of Eva's World on nova.
# deploy.sh copies the app to ~/Data/AI/World/app and runs this there.
#
# Pinned: the Mindcraft commit, mineflayer (Mindcraft's own patches target
# 4.33.0 exactly; a newer one makes `patch-package` fail and the bot lose its
# digging/placing fixes), the Paper build by SHA-256, the hub's whole
# dependency tree (package-lock.json, `npm ci`). Mindcraft's other
# dependencies resolve from its own package.json.
#
# Re-running keeps what you changed: server.properties is created once (only
# the bind address and RCON password are refreshed), world.json keeps its
# other keys.
#
# Usage: ACCEPT_MINECRAFT_EULA=yes ./install_nova.sh [bind-ip]
#   bind-ip defaults to this machine's Tailscale IPv4 address.
set -euo pipefail

WORLD="$HOME/Data/AI/World"
APP="$WORLD/app"
MINDCRAFT_REPO="https://github.com/kolbytn/mindcraft.git"
MINDCRAFT_COMMIT="f6a9556"
MINEFLAYER_VERSION="4.33.0"
PAPER_VERSION="1.21.6"
PAPER_BUILD="48"
PAPER_SHA256="35e2dfa66b3491b9d2f0bb033679fa5aca1e1fdf097e7a06a80ce8afeda5c214"
PAPER_URL="https://fill-data.papermc.io/v1/objects/$PAPER_SHA256/paper-$PAPER_VERSION-$PAPER_BUILD.jar"
NODE20=/opt/homebrew/opt/node@20/bin
JAVA=/opt/homebrew/opt/openjdk@21/bin/java
PLIST_NAME=com.local.eva-world-hub
export HOMEBREW_NO_AUTO_UPDATE=1

step() { printf '\n→ %s\n' "$*"; }

BIND_IP="${1:-}"
if [ -z "$BIND_IP" ]; then
  BIND_IP="$( (tailscale ip -4 2>/dev/null || /Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4 2>/dev/null) | head -1)"
fi
[ -n "$BIND_IP" ] || { echo "no bind address: pass the Tailscale IP as the first argument" >&2; exit 1; }

step "tools (Java 21, Node 20 for Mindcraft, git)"
for f in openjdk@21 node@20 git; do brew list --versions "$f" >/dev/null 2>&1 || brew install "$f"; done
"$JAVA" -version 2>&1 | head -1
"$NODE20/node" -v

step "Mindcraft @ $MINDCRAFT_COMMIT"
if [ ! -d "$WORLD/mindcraft/.git" ]; then
  git clone "$MINDCRAFT_REPO" "$WORLD/mindcraft"
fi
cd "$WORLD/mindcraft"
if [ "$(git rev-parse --short=7 HEAD)" != "$MINDCRAFT_COMMIT" ]; then
  git fetch origin && git -c advice.detachedHead=false checkout "$MINDCRAFT_COMMIT"
fi
git checkout -- package.json settings.js 2>/dev/null || true
"$NODE20/node" -e '
const fs=require("fs"); const p=JSON.parse(fs.readFileSync("package.json"));
p.dependencies.mineflayer=process.argv[1]; p.overrides=p.overrides||{}; p.overrides.mineflayer=process.argv[1];
fs.writeFileSync("package.json", JSON.stringify(p,null,4)+"\n");' "$MINEFLAYER_VERSION"
PATH="$NODE20:$PATH" npm install --no-fund --no-audit 2>&1 | tail -4
PP="$(PATH="$NODE20:$PATH" npx patch-package 2>&1)" || { echo "$PP" >&2; echo "a Mindcraft patch failed to apply" >&2; exit 1; }
case "$PP" in *ERROR*) echo "$PP" >&2; echo "a Mindcraft patch failed to apply" >&2; exit 1;; esac
MFV="$("$NODE20/node" -p 'require("./node_modules/mineflayer/package.json").version')"
[ "$MFV" = "$MINEFLAYER_VERSION" ] || { echo "mineflayer is $MFV, not $MINEFLAYER_VERSION" >&2; exit 1; }

step "Paper $PAPER_VERSION build $PAPER_BUILD"
mkdir -p "$WORLD/server"
cd "$WORLD/server"
JAR="paper-$PAPER_VERSION-$PAPER_BUILD.jar"
if ! echo "$PAPER_SHA256  $JAR" | shasum -a 256 -c --status 2>/dev/null; then
  curl -fsSL --max-time 300 -o "$JAR.part" "$PAPER_URL"
  echo "$PAPER_SHA256  $JAR.part" | shasum -a 256 -c --status || { echo "Paper checksum mismatch" >&2; rm -f "$JAR.part"; exit 1; }
  mv "$JAR.part" "$JAR"
fi
# the hub starts the first paper-*.jar it finds: only this build may be here
find . -maxdepth 1 -name 'paper-*.jar' ! -name "$JAR" -delete
[ -s .rcon_password ] || (umask 077; openssl rand -hex 16 > .rcon_password)
chmod 600 .rcon_password
if [ ! -s server.properties ]; then
  (umask 077; sed -e "s/@BIND_IP@/$BIND_IP/" -e "s/@RCON_PASSWORD@/$(cat .rcon_password)/" \
    "$APP/config/server.properties.template" > server.properties)
else
  # keep the operator's settings (difficulty set from the monitor, ...)
  sed -i '' -e "s/^server-ip=.*/server-ip=$BIND_IP/" -e "s/^rcon.password=.*/rcon.password=$(cat .rcon_password)/" server.properties
fi
chmod 600 server.properties
if ! grep -q '^eula=true' eula.txt 2>/dev/null; then
  if [ "${ACCEPT_MINECRAFT_EULA:-}" = "yes" ]; then
    echo "eula=true" > eula.txt
  else
    echo "The Minecraft server needs Mojang's EULA accepted (https://aka.ms/MinecraftEULA)." >&2
    echo "Re-run with ACCEPT_MINECRAFT_EULA=yes once you have read it." >&2
    exit 1
  fi
fi

step "hub"
[ -s "$WORLD/.hub_key" ] || (umask 077; openssl rand -hex 24 > "$WORLD/.hub_key")
chmod 600 "$WORLD/.hub_key"
# set the bind address, keep any other overrides already in world.json
"$NODE20/node" -e '
const fs=require("fs"); const f=process.argv[1]; let c={};
try { c=JSON.parse(fs.readFileSync(f,"utf8")); } catch {}
c.bind=process.argv[2]; fs.writeFileSync(f, JSON.stringify(c,null,2)+"\n");' "$APP/config/world.json" "$BIND_IP"
cd "$APP/hub" && PATH="$NODE20:$PATH" npm ci --no-fund --no-audit 2>&1 | tail -2
mkdir -p "$WORLD/run" "$WORLD/data"

step "patches"
WORLD="$WORLD" /bin/bash "$APP/scripts/apply_patches.sh"

step "launchd service $PLIST_NAME"
sed -e "s|@HOME@|$HOME|g" -e "s|@USER@|$(id -un)|g" -e "s|@NODE@|$NODE20/node|g" "$APP/scripts/$PLIST_NAME.plist" > "/tmp/$PLIST_NAME.plist"
sudo install -m 644 -o root -g wheel "/tmp/$PLIST_NAME.plist" "/Library/LaunchDaemons/$PLIST_NAME.plist"
sudo launchctl bootout "system/$PLIST_NAME" 2>/dev/null || true
sudo launchctl bootstrap system "/Library/LaunchDaemons/$PLIST_NAME.plist"
for _ in $(seq 1 20); do
  curl -s -o /dev/null -w '%{http_code}' "http://$BIND_IP:8090/api/status" 2>/dev/null | grep -q 401 && break
  sleep 1
done
curl -s -o /dev/null -w 'hub answers: HTTP %{http_code} (401 = up, key required)\n' "http://$BIND_IP:8090/api/status"

echo
echo "Done. Monitor: http://$BIND_IP:8090/   key: $WORLD/.hub_key"
