#!/usr/bin/env bash
# yt_tor_download.sh — download a URL's audio over Tor, ROTATING THE TOR EXIT
# NODE on every attempt and retrying when an exit is rate-limited / blocked
# (HTTP 429 and friends).
#
# Why rotation: YouTube aggressively rate-limits Tor exit IPs. Retrying on the
# SAME circuit just earns another 429 — the only useful retry rides a DIFFERENT
# exit. We fold the attempt number into the SOCKS username so Tor's
# IsolateSOCKSAuth (on by default) hands each attempt its own circuit — the same
# control-port-free trick tools/search.py uses for search reachability. No Tor
# control port, no global NEWNYM, no identity/egress beyond Tor.
#
# The URL comes from $1 OR, if absent, from a file (default .yt_url). The macro
# passes it via the file — written with a QUOTED heredoc so a hostile URL
# (`$(...)`, backticks, quotes) is stored LITERALLY and never shell-evaluated.
#
# Usage:  yt_tor_download.sh <url>        # or write the url to .yt_url
# Output: <basename>.<ext> (m4a preferred) in the current directory. The
#         transcribe step resolves the real extension by stem, so a non-m4a
#         fallback (opus/webm) still transcribes — but ONLY because we pre-clean
#         stale <basename>.* first, so exactly one such file ever exists.
# Exit:   0 success; 1 every exit blocked/unavailable; the yt-dlp status on a
#         NON-retryable error (stops early); 2 bad/missing url.
#
# Env overrides (mainly for tests):
#   YT_TOR_ATTEMPTS      max attempts             (default 8)
#   YT_TOR_BACKOFF_BASE  sleep = attempt# * base  (default 3; 0 disables sleep)
#   YT_TOR_BACKOFF_MAX   cap on the per-try sleep (default 8s) — keeps a long
#                        rotation bounded instead of growing 3,6,9,12,…
#   YT_TOR_PROXY_HOST    host:port of Tor SOCKS   (default 127.0.0.1:9050)
#   YT_TOR_OUT           output basename          (default yt_audio)
#   YT_TOR_URL_FILE      file to read the url from (default .yt_url)
set -u

URL="${1:-}"
if [ -z "$URL" ]; then
  URL_FILE="${YT_TOR_URL_FILE:-.yt_url}"
  [ -f "$URL_FILE" ] && URL="$(head -n1 "$URL_FILE" 2>/dev/null || true)"
fi
# Strip any stray whitespace/newline, then require an http(s) URL. Because the
# value may have arrived from an untrusted source, it is only ever used as a
# QUOTED variable ("$URL") — never re-evaluated — and rejected if it is not a
# plain web URL.
URL="$(printf '%s' "$URL" | tr -d '[:space:]')"
if [ -z "$URL" ]; then
  echo "usage: yt_tor_download.sh <url>   (or write the url to .yt_url)" >&2
  exit 2
fi
case "$URL" in
  http://*|https://*) : ;;
  *) echo "refusing to download a non-http(s) URL" >&2; exit 2 ;;
esac

# More attempts = more distinct exits tried before giving up. YouTube blocks a
# large, VARIABLE share of Tor exits with 429, so 8 tries materially beats 5 at
# drawing one it hasn't rate-limited — while the backoff cap keeps a fully-
# blocked run from dragging (3,6,8,8,8,… instead of 3,6,9,12,15,…).
ATTEMPTS="${YT_TOR_ATTEMPTS:-8}"
BACKOFF_BASE="${YT_TOR_BACKOFF_BASE:-3}"
BACKOFF_MAX="${YT_TOR_BACKOFF_MAX:-8}"
PROXY_HOST="${YT_TOR_PROXY_HOST:-127.0.0.1:9050}"
OUT="${YT_TOR_OUT:-yt_audio}"

# Give yt-dlp a SUPPORTED JavaScript runtime. Without one, recent yt-dlp falls
# back to a limited client that YouTube serves "Video unavailable" for — even for
# public videos — the real wall behind Tor once the 429s are dodged (rotation
# can't help, every exit hits it). yt-dlp's EJS supports ONLY deno here: node
# (even v20), bun and quickjs are all reported "unavailable"/"unsupported"
# (verified 2026-08-12), while deno is recognised. Ensure deno is present
# (install to /usr/local via the official installer when absent), then hand
# yt-dlp its path. Best-effort and NON-fatal — a failed install just falls back
# as before. Gated on apt-get so it never fires on a non-Debian host; the
# container persists, so it installs at most once per container.
# `YT_TOR_NO_JS=1` disables the whole block (tests / debugging).
JS_RT=()
if [ -z "${YT_TOR_NO_JS:-}" ]; then
  _deno="$(command -v deno 2>/dev/null || true)"
  if [ -z "$_deno" ] && command -v curl >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1 \
     && command -v apt-get >/dev/null 2>&1; then
    echo "installing deno for yt-dlp's JS runtime — one-time per container…"
    command -v unzip >/dev/null 2>&1 || sudo apt-get install -y unzip >/dev/null 2>&1
    curl -fsSL https://deno.land/install.sh 2>/dev/null \
      | sudo env DENO_INSTALL=/usr/local sh >/dev/null 2>&1 || true
    _deno="$(command -v deno 2>/dev/null || true)"
    [ -z "$_deno" ] && [ -x /usr/local/bin/deno ] && _deno=/usr/local/bin/deno
  fi
  [ -n "$_deno" ] && JS_RT=(--js-runtimes "deno:${_deno}")
fi

# Pre-clean: a stale <basename>.* from a PRIOR video would (a) make yt-dlp skip
# the download ("already downloaded", exit 0) and (b) make the transcribe step's
# stem-resolver pick the wrong/old file — silently transcribing stale audio.
# Removing it first guarantees exactly one <basename>.* exists after a success.
rm -f "${OUT}".* 2>/dev/null || true

# A failure a DIFFERENT exit node may not hit — worth rotating and retrying.
# ("this video is unavailable" over Tor is almost always a block symptom, so it
# is treated as retryable; a genuinely-removed video simply exhausts attempts.)
BLOCK_RE='429|too many requests|temporarily unavailable|rate.?limit|sign in to confirm|not a bot|video (is )?unavailable|video is not available|content isn.?t available|blocked|http error 403|forbidden'

success=0
last_tail=""
for i in $(seq 1 "$ATTEMPTS"); do
  tag="ytdl_${i}_${RANDOM}"
  proxy="socks5h://${tag}:x@${PROXY_HOST}"
  echo "-- attempt ${i}/${ATTEMPTS} on a fresh Tor circuit (socks tag=${tag}) --"
  # ${JS_RT[@]+"${JS_RT[@]}"} = the js-runtimes flag if set, nothing if empty —
  # the guarded form so an empty array is safe under `set -u` on bash 3.2 too.
  yt-dlp ${JS_RT[@]+"${JS_RT[@]}"} --proxy "$proxy" --no-playlist --force-overwrites \
    -f "ba[ext=m4a]/ba" -o "${OUT}.%(ext)s" "$URL" >yt_attempt.log 2>&1
  rc=$?
  last_tail="$(tail -n 15 yt_attempt.log 2>/dev/null)"
  if [ "$rc" -eq 0 ] && ls "${OUT}".* >/dev/null 2>&1; then
    success=1
    echo "download succeeded on attempt ${i}"
    break
  fi
  if grep -qiE "$BLOCK_RE" yt_attempt.log 2>/dev/null; then
    back=$(( i * BACKOFF_BASE ))
    [ "$back" -gt "$BACKOFF_MAX" ] && back="$BACKOFF_MAX"
    echo "attempt ${i}: exit looks rate-limited/blocked (rc=${rc}) — rotating Tor circuit (backoff ${back}s)"
    [ "$back" -gt 0 ] && sleep "$back"
    continue
  fi
  # Non-retryable: a real problem (bad URL, unsupported site, no audio). Rotating
  # exits won't help — stop now and surface the ACTUAL yt-dlp status/message.
  code=$rc
  [ "$code" -eq 0 ] && code=1   # rc=0 with no file is still a failure
  echo "attempt ${i}: non-retryable yt-dlp error (rc=${rc}) — stopping early:"
  echo "$last_tail"
  exit "$code"
done

if [ "$success" -eq 1 ]; then
  ls -la "${OUT}".*
  exit 0
fi
echo "DOWNLOAD FAILED after ${ATTEMPTS} attempt(s): every Tor exit was blocked/rate-limited or the video is unavailable."
echo "$last_tail"
exit 1
