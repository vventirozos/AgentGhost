#!/usr/bin/env bash
# update-youtube-stack.sh — the remedy the YouTube route canary names (§4KH).
#
# When YouTube rotates its BotGuard challenge, yt-dlp and the PO-token helper
# (bgutil-ytdlp-pot-provider: the pip plugin AND the node server) must move
# together. This script does the four steps and PROVES the result with one
# live resolve over Tor. It is run BY THE OPERATOR on purpose: it upgrades
# third-party code in the agent's venv and restarts a daemon — and it fetches
# from PyPI, GitHub and npm over the CLEAR from this host (operator egress,
# not the agent's; the agent's own traffic stays Tor-only).
#
#   ./bin/update-youtube-stack.sh            # upgrade + verify
#   ./bin/update-youtube-stack.sh --verify   # only the live check
#
# Env: GHOST_VENV (default ~/Data/AI/.agent.venv), POT_DIR (default
# ~/Data/AI/PotProvider), POT_URL (default http://127.0.0.1:4416),
# TOR_PROXY (default socks5h://127.0.0.1:9050).
set -euo pipefail

VENV="${GHOST_VENV:-$HOME/Data/AI/.agent.venv}"
POT_DIR="${POT_DIR:-$HOME/Data/AI/PotProvider}"
POT_URL="${POT_URL:-http://127.0.0.1:4416}"
TOR_PROXY="${TOR_PROXY:-socks5h://127.0.0.1:9050}"
UPSTREAM="https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git"
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/bgutil-pot-build}"
CANARY="https://www.youtube.com/watch?v=jNQXAC9IVRw"
PY="$VENV/bin/python"

say() { printf '\n== %s\n' "$*"; }

verify() {
  say "helper /ping"
  curl -fsS -m 5 "$POT_URL/ping" && echo
  say "live resolve over Tor (up to 6 fresh circuits; ~4 in 10 pass with a good token)"
  # Per-attempt SOCKS username = a fresh Tor circuit (IsolateSOCKSAuth); the
  # scheme is forced to socks5h so DNS stays inside Tor.
  local base scheme hostport i tag
  base="${TOR_PROXY/socks5:\/\//socks5h://}"
  scheme="${base%%//*}//"; hostport="${base##*@}"; hostport="${hostport##*//}"
  for i in 1 2 3 4 5 6; do
    tag="upd${i}$RANDOM"
    if "$PY" -m yt_dlp --no-playlist --socket-timeout 30 \
         --proxy "${scheme}${tag}:x@${hostport}" \
         --js-runtimes "deno:$(command -v deno || echo /opt/homebrew/bin/deno)" \
         --extractor-args "youtubepot-bgutilhttp:base_url=$POT_URL" \
         --skip-download --print "%(title)s|%(duration)s" "$CANARY" 2>/tmp/upd_yt_err.log \
         | sed "s/^/   circuit $i: OK  /"; then
      say "VERIFIED: the route resolves with the current yt-dlp + helper"
      return 0
    fi
    printf '   circuit %s: %s\n' "$i" "$(grep -m1 '^ERROR' /tmp/upd_yt_err.log | cut -c1-110)"
  done
  echo "NOT VERIFIED: every circuit refused. If the errors say 'not a bot', YouTube's challenge is ahead of" >&2
  echo "the helper — check the upstream repo for a newer release; otherwise inspect /tmp/upd_yt_err.log." >&2
  return 1
}

if [ "${1:-}" = "--verify" ]; then verify; exit $?; fi

say "0/4 preflight"
sudo -n true 2>/dev/null || { echo "passwordless sudo is required (the daemon restart in step 3); nothing was changed." >&2; exit 2; }
command -v node >/dev/null && command -v npm >/dev/null && command -v git >/dev/null || { echo "node, npm and git are required; nothing was changed." >&2; exit 2; }

say "1/4 upgrading yt-dlp + the PO-token plugin in $VENV"
"$VENV/bin/pip" install -q -U yt-dlp bgutil-ytdlp-pot-provider
PLUGIN_VER="$("$PY" -c 'import importlib.metadata as m; print(m.version("bgutil-ytdlp-pot-provider"))')"
"$PY" -c 'import yt_dlp, importlib.metadata as m; print("   yt-dlp", yt_dlp.version.__version__, "| plugin", m.version("bgutil-ytdlp-pot-provider"))'

say "2/4 rebuilding the helper server at the release matching the plugin ($PLUGIN_VER)"
rm -rf "$BUILD_DIR"; git clone -q "$UPSTREAM" "$BUILD_DIR"
( cd "$BUILD_DIR" && { git checkout -q "$PLUGIN_VER" 2>/dev/null || git checkout -q "v$PLUGIN_VER" 2>/dev/null \
    || echo "   (no tag $PLUGIN_VER / v$PLUGIN_VER upstream — building the default branch; the plugin and server protocol are versioned together, so verify below is the arbiter)"; } )
( cd "$BUILD_DIR/server" && npm install --no-audit --no-fund --loglevel=error && npx tsc )
mkdir -p "$POT_DIR"
rsync -a --delete "$BUILD_DIR/server/build" "$BUILD_DIR/server/node_modules" "$BUILD_DIR/server/package.json" "$POT_DIR/"
( cd "$BUILD_DIR" && git rev-parse --short HEAD ) > "$POT_DIR/UPSTREAM_COMMIT"
echo "   helper at commit $(cat "$POT_DIR/UPSTREAM_COMMIT") → $POT_DIR"

say "3/4 restarting com.local.ghost-pot"
sudo -n launchctl kickstart -k system/com.local.ghost-pot
sleep 3

say "4/4 verifying"
verify
echo
echo "Done. Ask the agent's canary to confirm now (it will post RECOVERED if it had reported a failure):"
echo "  curl -s -X POST -H \"X-Ghost-Key: \$(cat ~/Data/AI/.ghost_api_key)\" http://127.0.0.1:8000/api/youtube-canary/run"
echo "  then: curl -s -H \"X-Ghost-Key: ...\" http://127.0.0.1:8000/api/health | jq .youtube_route"
