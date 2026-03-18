#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Load .env if it exists
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

if [ -z "$SLACK_BOT_TOKEN" ]; then
    echo "Error: SLACK_BOT_TOKEN is not set."
    exit 1
fi

if [ -z "$SLACK_APP_TOKEN" ]; then
    echo "Error: SLACK_APP_TOKEN is not set."
    exit 1
fi

python3 interface/slack_bot/main.py "$@"
