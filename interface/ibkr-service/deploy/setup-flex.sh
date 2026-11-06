#!/bin/bash
# Stores the Flex token and query id on this machine. The token is read hidden, so it never lands in
# shell history, in a process listing or in a chat transcript. Run in a real terminal on the service machine.
set -eu
TOKEN_FILE="$HOME/Data/AI/.ibkr_flex_token"
CONFIG="$HOME/Data/AI/ibkr-service/data/config.json"
read -r -s -p "Flex token (hidden): " token; echo
read -r -p "Flex query id: " query
[ -n "$token" ] && [ -n "$query" ] || { echo "both values are needed; nothing changed" >&2; exit 1; }
umask 077
mkdir -p "$(dirname "$CONFIG")"
printf '%s\n' "$token" > "$TOKEN_FILE"
"$HOME/Data/AI/ibkr-service/venv/bin/python" - "$CONFIG" "$query" <<'PY'
import json, os, sys
path, query = sys.argv[1], sys.argv[2]
try:
    config = json.load(open(path))
except FileNotFoundError:
    config = {}
config["flex_query_id"] = query
descriptor = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump(config, handle, indent=2)
os.replace(path + ".tmp", path)
PY
echo "Saved. The service picks this up within a minute."
