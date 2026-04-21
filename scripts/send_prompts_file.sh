#!/bin/bash
# Read a text file with one prompt per line and send each to the agent
# sequentially. Waits for each response to complete before sending the next.
#
# Usage:
#   scripts/send_prompts_file.sh [prompts_file]
#
# Default prompts file: init_personal_data.prompts (in cwd).

set -u

PROMPTS_FILE="${1:-init_personal_data.prompts}"
AGENT_URL="http://127.0.0.1:8000/api/chat"
MODEL="qwen-3.6-35b-a3"
TIMEOUT=2100

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "ERROR: prompts file not found: $PROMPTS_FILE"
    exit 1
fi

TOTAL=$(grep -cve '^\s*$' "$PROMPTS_FILE")
echo "=== SENDING $TOTAL PROMPT(S) FROM $PROMPTS_FILE ==="

IDX=0
while IFS= read -r PROMPT || [ -n "$PROMPT" ]; do
    # Skip blank lines and comments.
    [ -z "${PROMPT// }" ] && continue
    case "$PROMPT" in \#*) continue ;; esac

    IDX=$((IDX + 1))
    echo ""
    echo "=== [$IDX/$TOTAL] SEND $(date '+%H:%M:%S') ==="
    echo "PROMPT: $PROMPT"

    REQ_BODY=$(python3 -c '
import json, sys
body = {
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "stream": False,
}
print(json.dumps(body))
' "$MODEL" "$PROMPT")

    HTTP_CODE=$(curl -s -o /tmp/ghost_prompt_response.json -w "%{http_code}" \
        -m "$TIMEOUT" \
        --connect-timeout 10 \
        -H "Content-Type: application/json" \
        -X POST "$AGENT_URL" \
        -d "$REQ_BODY")

    echo "=== [$IDX/$TOTAL] DONE $(date '+%H:%M:%S') | HTTP $HTTP_CODE ==="

    python3 -c '
import json
try:
    with open("/tmp/ghost_prompt_response.json") as f:
        data = json.load(f)
    msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("--- RESPONSE PREVIEW (first 500 chars) ---")
    print(msg[:500])
    if len(msg) > 500:
        print(f"... [{len(msg) - 500} more chars]")
except Exception as e:
    print(f"(could not parse response: {e})")
' 2>/dev/null
done < "$PROMPTS_FILE"

echo ""
echo "=== ALL DONE: $IDX PROMPT(S) SENT ==="
exit 0
