# Ghost Agent

Ghost Agent is a self-hosted AI agent. You send it a message — through the web UI, Slack, voice, or a plain HTTP API — and it plans, recalls relevant memories, runs whatever tools it needs (code, shell, web) inside a Docker sandbox, checks its own answer with a second model, and replies.

## What it does

- **Chats and executes tasks** — answers questions and carries out multi-step work using tools that run in isolated Docker containers.
- **Remembers** — a six-tier memory (facts, relationships, your profile, learned skills, journal, past episodes) is retrieved and fused into every turn.
- **Verifies itself** — a second-model verifier judges each final answer and repairs it in-loop if it doesn't hold up.
- **Improves itself while idle** — a dream/self-play loop consolidates experience into reusable skills and tunes its own prompts, entirely locally.
- **Stays anonymous** — every outbound network request goes through Tor; the agent refuses to start without it.

## Install

You need: Python 3, a running Docker daemon, a Tor daemon, and an upstream LLM server (Ghost is tuned for Qwen 3.6 35B-A3).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Required environment (full list in the Install & Run docs)
export GHOST_API_KEY="..."
export GHOST_HOME="$HOME/ghost"
export GHOST_MODEL="qwen-3.6-35b-a3"
export GHOST_SANDBOX_DIR="$GHOST_HOME/sandbox"
export TOR_PROXY="socks5h://127.0.0.1:9050"   # required — agent fails closed without it

# Start the agent
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --host 0.0.0.0 --port 8000 --verbose
```

Companion processes (web UI, Slack bot, voice, image-gen) are covered in [Install & Run](https://vventirozos.github.io/AgentGhost/installation.html).

## Documentation

**Full reference: <https://vventirozos.github.io/AgentGhost/>** — architecture, capabilities, CLI flags, request lifecycle, memory internals, and a page for every module. HTML sources live in [`docs/`](docs/).
