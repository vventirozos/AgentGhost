#!/bin/bash
# Single-iteration eval harness for the Ghost Agent.
#
#   1. Wipes the sandbox, memory, and log file (unless --no-reset).
#   2. Kills the agent so launchd restarts it (unless --no-reset).
#   3. Waits for the agent to be responsive on :8000.
#   4. POSTs the prompt (arg $1 or default Nginx one) and waits up to 35 min.
#   5. Writes LOG_START / LOG_END markers so callers can slice the
#      per-iteration log out of the combined file.
#
# Usage:
#   scripts/run_eval_loop.sh                                    # default Nginx prompt, resets
#   scripts/run_eval_loop.sh "Your prompt here"                 # custom prompt, resets
#   scripts/run_eval_loop.sh --no-reset "Follow-up prompt"      # custom prompt, no reset (for multi-turn tests)
#
# Exit codes: 0 on HTTP 200 response, non-zero on transport error.

set -u  # catch typos in var names; don't use -e because kill/xargs can exit 1 harmlessly

LOG=/Users/vasilis/Data/AI/Logs/ghost-agent.log
SANDBOX=/Users/vasilis/Data/AI/Data/sandbox
MEMDIR=/Users/vasilis/Data/AI/Data/system/memory
DEFAULT_PROMPT='Start a new project called "Build a log parser for Nginx access logs" — kind CODING, goal "Parse 1GB of access logs and emit per-endpoint latency stats".'

RESET=true
if [ "${1:-}" = "--no-reset" ]; then
    RESET=false
    shift
fi
PROMPT="${1:-$DEFAULT_PROMPT}"

if $RESET; then
    echo "=== RESET $(date '+%H:%M:%S') ==="
    rm -rf "$SANDBOX"/* 2>/dev/null
    rm -rf "$MEMDIR"/* 2>/dev/null
    rm -f "$LOG"
    touch "$LOG"
    # Kill the python agent process on :8000. launchd will restart it.
    OLD_PID=$(ps auxw | grep python | grep 8000 | grep -v grep | awk '{print $2}')
    if [ -n "$OLD_PID" ]; then
        echo "Killing PID(s): $OLD_PID"
        echo "$OLD_PID" | xargs kill 2>/dev/null
        # Wait until the old PID actually exits. Without this, /api/version
        # keeps returning 200 from the dying process for ~2s after SIGTERM
        # and the subsequent POST races against a half-killed server.
        for i in $(seq 1 20); do
            if ! ps -p $OLD_PID > /dev/null 2>&1; then
                echo "Old PID $OLD_PID exited after ${i}s"
                break
            fi
            sleep 1
        done
    fi

    echo "=== WAIT FOR BOOT $(date '+%H:%M:%S') ==="
    # Now the new process needs to (a) get spawned by launchd, (b) import
    # all modules, (c) boot sandbox/memory/skills, (d) start listening.
    # Poll "system ready" in the log — that's the last line of boot and
    # guarantees the new process is the one we're talking to. The
    # `/api/version` endpoint is too early: FastAPI binds before memory
    # and sandbox finish initializing.
    for i in $(seq 1 180); do
        if grep -q "system ready" "$LOG" 2>/dev/null; then
            echo "Agent fully booted after ${i}s"
            break
        fi
        sleep 1
        if [ "$i" -eq 180 ]; then
            echo "ERROR: agent did not boot within 180s"
            tail -30 "$LOG"
            exit 1
        fi
    done

    # One extra settle second to let the first request-handling coroutine
    # finish registering.
    sleep 2
else
    echo "=== SKIPPING RESET (--no-reset) ==="
fi

echo "=== SEND PROMPT $(date '+%H:%M:%S') ==="
echo "PROMPT: $PROMPT"
echo "LOG_START $(date '+%H:%M:%S') PROMPT=${PROMPT:0:80}" >> "$LOG"

# Build request body with proper JSON escaping.
REQ_BODY=$(python3 -c '
import json, sys
prompt = sys.argv[1]
body = {
    "model": "qwen-3.6-35b-a3",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False,
}
print(json.dumps(body))
' "$PROMPT")

# 35-minute max timeout — some past traces have taken up to 18 min.
HTTP_CODE=$(curl -s -o /tmp/ghost_eval_response.json -w "%{http_code}" \
    -m 2100 \
    --connect-timeout 10 \
    -H "Content-Type: application/json" \
    -X POST http://127.0.0.1:8000/api/chat \
    -d "$REQ_BODY")

echo "LOG_END $(date '+%H:%M:%S')" >> "$LOG"
echo "=== DONE $(date '+%H:%M:%S') | HTTP $HTTP_CODE ==="

# Preview the agent's response (first 500 chars of the assistant message).
python3 -c '
import json, sys
try:
    with open("/tmp/ghost_eval_response.json") as f:
        data = json.load(f)
    msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("--- ASSISTANT RESPONSE PREVIEW (first 500 chars) ---")
    print(msg[:500])
    if len(msg) > 500:
        print(f"... [{len(msg) - 500} more chars]")
except Exception as e:
    print(f"(could not parse response: {e})")
' 2>/dev/null

exit 0
