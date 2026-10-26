#!/usr/bin/env bash
# update-youtube-stack.sh — the remedy the YouTube route canary names (§4KH).
#
# When YouTube rotates its BotGuard challenge, yt-dlp and the PO-token helper
# (bgutil-ytdlp-pot-provider: the pip plugin AND the node server) must move
# together. This script does the steps IN A SAFE ORDER and PROVES the result
# with one live resolve over Tor. It is run BY THE OPERATOR on purpose: it
# upgrades third-party code in the agent's venv and restarts a daemon — and it
# fetches from PyPI, GitHub and npm over the CLEAR from this host (operator
# egress, not the agent's; the agent's own traffic stays Tor-only).
#
# §4PI (2026-10-15, ops review): the old order was pip FIRST, then clone/build
# (a build failure left the plugin new and the helper old), `rsync --delete`
# over the RUNNING daemon's build, a restart BEFORE the verify, and no
# rollback. Now: build into a scratch dir → stage the helper beside the live
# one → pip → atomic swap (the previous helper is kept as <POT_DIR>.prev) →
# restart → wait for /ping → verify; a failed build or pip changes nothing
# live, and a failed verify prints the rollback.
#
#   ./bin/update-youtube-stack.sh            # upgrade + verify
#   ./bin/update-youtube-stack.sh --verify   # only the live check
#   ./bin/update-youtube-stack.sh --rollback # put the previous helper + venv pins back
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
CANARY="https://www.youtube.com/watch?v=jNQXAC9IVRw"
PY="$VENV/bin/python"
LABEL="system/com.local.ghost-pot"

say() { printf '\n== %s\n' "$*"; }

wait_ping() {   # the helper takes a few seconds to bind after a restart (§4PI R3-m6)
  local i
  for i in $(seq 1 30); do
    if curl -fsS -m 2 "$POT_URL/ping" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

verify() {
  say "helper /ping"
  if ! wait_ping; then echo "   the helper at $POT_URL does not answer /ping (waited 30 s)" >&2; return 1; fi
  curl -fsS -m 5 "$POT_URL/ping" && echo
  say "live resolve over Tor (up to 6 fresh circuits; ~4 in 10 pass with a good token)"
  # Per-attempt SOCKS username = a fresh Tor circuit (IsolateSOCKSAuth); the
  # scheme is forced to socks5h so DNS stays inside Tor. The error log is a
  # private temp file, not a shared /tmp name.
  local base scheme hostport i tag errlog
  errlog="$(mktemp -t upd_yt_err)"
  base="${TOR_PROXY/socks5:\/\//socks5h://}"
  scheme="${base%%//*}//"; hostport="${base##*@}"; hostport="${hostport##*//}"
  for i in 1 2 3 4 5 6; do
    tag="upd${i}$RANDOM"
    if "$PY" -m yt_dlp --no-playlist --socket-timeout 30 \
         --proxy "${scheme}${tag}:x@${hostport}" \
         --js-runtimes "deno:$(command -v deno || echo /opt/homebrew/bin/deno)" \
         --extractor-args "youtubepot-bgutilhttp:base_url=$POT_URL" \
         --skip-download --print "%(title)s|%(duration)s" "$CANARY" 2>"$errlog" \
         | sed "s/^/   circuit $i: OK  /"; then
      say "VERIFIED: the route resolves with the current yt-dlp + helper"
      rm -f "$errlog"
      return 0
    fi
    printf '   circuit %s: %s\n' "$i" "$(grep -m1 '^ERROR' "$errlog" | cut -c1-110)"
  done
  echo "NOT VERIFIED: every circuit refused. With a healthy stack this happens by chance about 5% of the time" >&2
  echo "(≈0.6^6) — run '$0 --verify' once more before concluding. If the errors say 'not a bot', YouTube's" >&2
  echo "challenge is ahead of the helper — check the upstream repo for a newer release; otherwise inspect $errlog." >&2
  return 1
}

rollback() {
  say "rollback"
  if [ ! -d "$POT_DIR.prev" ]; then
    echo "   no previous helper build at $POT_DIR.prev — nothing to roll back to (the venv is left as it is," >&2
    echo "   so the helper and the plugin stay a matched pair)." >&2
    return 1
  fi
  sudo -n launchctl kill SIGTERM "$LABEL" 2>/dev/null || true
  rm -rf "$POT_DIR.rolledback"; mv "$POT_DIR" "$POT_DIR.rolledback"; mv "$POT_DIR.prev" "$POT_DIR"
  echo "   helper: previous build restored (the rejected one is at $POT_DIR.rolledback)"
  # the pins saved beside the previous build (now POT_DIR again) are the ones that matched it
  if [ -f "$POT_DIR/PINS.prev" ]; then
    "$VENV/bin/pip" install -q --force-reinstall --no-deps -r "$POT_DIR/PINS.prev" && echo "   venv: previous yt-dlp + plugin pins restored"
  else
    echo "   no PINS.prev beside the previous build — reinstall the matching plugin by hand: pip install bgutil-ytdlp-pot-provider==<its version>" >&2
  fi
  sudo -n launchctl kickstart -k "$LABEL"
  wait_ping && echo "   helper up" || echo "   helper not answering after the rollback — check: sudo launchctl print $LABEL" >&2
}

plan() {   # the order of operations, for the operator and for the pin that guards it
  cat <<'PLAN'
1 resolve the newest plugin release (nothing installed)
2 clone + build the helper in a private scratch dir, stage it beside the live one
3 pip upgrade yt-dlp + the plugin (previous pins saved beside the previous helper for --rollback)
4 swap the helper atomically (previous build kept as <POT_DIR>.prev), restart, wait for /ping
5 verify over 6 fresh circuits; on failure: re-run --verify once, then --rollback
PLAN
}

case "${1:-}" in
  --verify) verify; exit $? ;;
  --rollback) rollback; exit $? ;;
  --plan) plan; exit 0 ;;
esac

say "0/5 preflight"
sudo -n true 2>/dev/null || { echo "passwordless sudo is required (the daemon restart in step 4); nothing was changed." >&2; exit 2; }
command -v node >/dev/null && command -v npm >/dev/null && command -v git >/dev/null || { echo "node, npm and git are required; nothing was changed." >&2; exit 2; }
BUILD_DIR="$(mktemp -d -t bgutil-pot-build)"
trap 'rm -rf "$BUILD_DIR"' EXIT

say "1/5 resolving the newest plugin release (nothing installed yet)"
PLUGIN_VER="$( { "$PY" -m pip index versions bgutil-ytdlp-pot-provider 2>/dev/null || true; } | sed -n 's/^bgutil-ytdlp-pot-provider (\([^)]*\)).*/\1/p' | head -1)"
[ -n "$PLUGIN_VER" ] || { echo "   could not read the newest plugin version from PyPI; nothing was changed." >&2; exit 3; }
echo "   newest plugin: $PLUGIN_VER (installed: $("$PY" -c 'import importlib.metadata as m; print(m.version("bgutil-ytdlp-pot-provider"))' 2>/dev/null || echo none))"

say "2/5 building the helper server at the release matching the plugin ($PLUGIN_VER) — in $BUILD_DIR"
git clone -q "$UPSTREAM" "$BUILD_DIR"
( cd "$BUILD_DIR" && { git checkout -q "$PLUGIN_VER" 2>/dev/null || git checkout -q "v$PLUGIN_VER" 2>/dev/null \
    || echo "   (no tag $PLUGIN_VER / v$PLUGIN_VER upstream — building the default branch; the plugin and server protocol are versioned together, so verify below is the arbiter)"; } )
( cd "$BUILD_DIR/server" && npm install --no-audit --no-fund --loglevel=error && npx tsc )
STAGE="$POT_DIR.new"; rm -rf "$STAGE"; mkdir -p "$STAGE"
rsync -a "$BUILD_DIR/server/build" "$BUILD_DIR/server/node_modules" "$BUILD_DIR/server/package.json" "$STAGE/"
( cd "$BUILD_DIR" && git rev-parse --short HEAD ) > "$STAGE/UPSTREAM_COMMIT"
echo "   staged helper at commit $(cat "$STAGE/UPSTREAM_COMMIT") → $STAGE (the live helper is untouched)"

say "3/5 upgrading yt-dlp + the PO-token plugin in $VENV"
# the previous pins live with the PREVIOUS helper (POT_DIR is still the old
# build here; it becomes <POT_DIR>.prev at the swap, which is what --rollback restores)
mkdir -p "$POT_DIR"
"$PY" -c 'import yt_dlp, importlib.metadata as m
print("yt-dlp==" + yt_dlp.version.__version__); print("bgutil-ytdlp-pot-provider==" + m.version("bgutil-ytdlp-pot-provider"))
try: print("yt-dlp-ejs==" + m.version("yt-dlp-ejs"))
except m.PackageNotFoundError: pass' > "$POT_DIR/PINS.prev" 2>/dev/null || true
"$VENV/bin/pip" install -q -U yt-dlp "bgutil-ytdlp-pot-provider==$PLUGIN_VER"
# §4RR: the JS-challenge solver scripts, at the version THIS yt-dlp pins (its
# `default` extra) — without them every fetch warns "n challenge solving
# failed … Some formats may be missing". Not `yt-dlp[default]`: that extra
# would also move requests/urllib3/websockets under the agent.
EJS_VER="$("$PY" -c 'import importlib.metadata as m, re
print(next((re.search(r"==([^;\s]+)", r).group(1) for r in (m.requires("yt-dlp") or []) if r.replace("_", "-").startswith("yt-dlp-ejs") and "==" in r), ""))')"
if [[ -n "$EJS_VER" ]]; then
  "$VENV/bin/pip" install -q "yt-dlp-ejs==$EJS_VER"
else
  echo "   yt-dlp names no yt-dlp-ejs pin — solver scripts left as they are" >&2
fi
"$PY" -c 'import yt_dlp, importlib.metadata as m
try: ejs = m.version("yt-dlp-ejs")
except m.PackageNotFoundError: ejs = "not installed"
print("   yt-dlp", yt_dlp.version.__version__, "| plugin", m.version("bgutil-ytdlp-pot-provider"), "| ejs", ejs)'

say "4/5 swapping the helper and restarting $LABEL"
sudo -n launchctl kill SIGTERM "$LABEL" 2>/dev/null || true
rm -rf "$POT_DIR.prev"
if [ -d "$POT_DIR" ]; then mv "$POT_DIR" "$POT_DIR.prev"; fi
mv "$STAGE" "$POT_DIR"
sudo -n launchctl kickstart -k "$LABEL"

say "5/5 verifying"
if verify; then
  echo
  echo "Done. Ask the agent's canary to confirm now (it will post RECOVERED if it had reported a failure):"
  echo "  curl -s -X POST -H \"X-Ghost-Key: \$(cat ~/Data/AI/.ghost_api_key)\" http://127.0.0.1:8000/api/youtube-canary/run"
  echo "  then: curl -s -H \"X-Ghost-Key: ...\" http://127.0.0.1:8000/api/health | jq .youtube_route"
  echo "The previous helper build stays at $POT_DIR.prev until the next update."
else
  echo
  echo "The new stack did not verify. Re-run '$0 --verify' once (a 0/6 is ~5% luck on a healthy stack);" >&2
  echo "if it fails again: '$0 --rollback' restores the previous helper and the previous venv pins." >&2
  exit 1
fi
