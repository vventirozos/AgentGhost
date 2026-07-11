#!/bin/bash
# Launcher for the OWNER-LOCKED Ghost Agent Slack bot (see main.py).
#
# Fixed 2026-07-11 alongside the bot rewrite:
#   * .env now loads via `set -a` + source (the old export-of-cat-through-
#     word-splitting pattern mangled any value containing spaces or quotes),
#     and is resolved relative to THIS script — not whatever directory you
#     happened to invoke it from;
#   * runs on the agent venv's python (the bare homebrew python3 doesn't have
#     slack_bolt/httpx — same class of problem as the uvicorn gotcha in the
#     agent's own launcher); override with GHOST_BOT_PYTHON;
#   * pre-flight checks now cover everything main.py hard-requires, including
#     the owner lock (the bot REFUSES to start unlocked), so a misconfigured
#     launch fails here with instructions instead of a traceback;
#   * exec's python so signals from a supervisor reach the bot directly;
#   * dropped the PYTHONPATH export — the bot imports nothing from src/.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present — script dir first, then cwd as a fallback.
for envfile in "$SCRIPT_DIR/.env" ".env"; do
    if [ -f "$envfile" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$envfile"
        set +a
        break
    fi
done

# The agent venv has slack_bolt/httpx; bare python3 usually doesn't.
PYTHON="${GHOST_BOT_PYTHON:-/Users/vasilis/Data/AI/.agent.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "Warning: $PYTHON not found — falling back to python3 on PATH" >&2
    PYTHON="$(command -v python3)"
fi

fail=0
require() {
    if [ -z "${!1:-}" ]; then
        echo "Error: $1 is not set. $2" >&2
        fail=1
    fi
}
require SLACK_BOT_TOKEN "Bot token (xoxb-…) for Slack API calls."
require SLACK_APP_TOKEN "App-level token (xapp-…) for Socket Mode."

# GHOST_API_KEY checks SET-ness, not non-emptiness: an explicitly EMPTY key
# (GHOST_API_KEY=) is valid and means the agent runs with auth disabled
# (prod uses --api-key ""). Do NOT pad it with a space to satisfy a check —
# whitespace reaches httpx as an illegal header value and breaks every call.
if [ -z "${GHOST_API_KEY+set}" ]; then
    echo "Error: GHOST_API_KEY is not set. Use the agent's --api-key value," >&2
    echo "or set it explicitly empty (GHOST_API_KEY=) if the agent runs" >&2
    echo "with auth disabled." >&2
    fail=1
fi

# Owner lock: main.py exits without one of these — fail here with the how-to.
if [ -z "${GHOST_SLACK_OWNER:-}" ] && [ -z "${GHOST_SLACK_OWNER_EMAIL:-}" ]; then
    echo "Error: owner lock unresolved — set GHOST_SLACK_OWNER to your Slack" >&2
    echo "member ID (U…; profile → ⋯ → 'Copy member ID'), or" >&2
    echo "GHOST_SLACK_OWNER_EMAIL to your Slack email (needs the" >&2
    echo "users:read.email scope)." >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/main.py" "$@"
