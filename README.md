# Ghost Agent

An autonomous FastAPI-based AI agent service with multi-tier memory, Docker-isolated tool execution, swarm inference, and biological-rhythm self-play.

> **Runtime stance.** Ghost Agent is designed and tuned around an **uncensored Qwen 3.6 35B-A3** upstream model, and every outbound network request the agent issues is mandated through **Tor**. The agent fails closed if `TOR_PROXY` is unset or the Tor daemon is unreachable — a silently-cleartext agent is worse than a stalled one.

## Documentation

Full reference is published on GitHub Pages: **<https://vventirozos.github.io/AgentGhost/>**

Every module in `src/ghost_agent/` and `interface/` has a dedicated page. The HTML sources live in [`docs/`](docs/) on the default branch.

| Entry point | What's there |
| --- | --- |
| [Home](https://vventirozos.github.io/AgentGhost/) | Overview, runtime stance, source map, conceptual model |
| [Capabilities](https://vventirozos.github.io/AgentGhost/capabilities.html) | What the agent can do, with example prompts |
| [System Architecture](https://vventirozos.github.io/AgentGhost/architecture.html) | End-to-end diagram of interfaces, API, core, memory, sandbox |
| [Install & Run](https://vventirozos.github.io/AgentGhost/installation.html) | Prerequisites, env vars, boot sequence, process commands |
| [CLI Reference](https://vventirozos.github.io/AgentGhost/cli_reference.html) | Every flag on `python -m src.ghost_agent.main` |
| [Anonymity & Tor routing](https://vventirozos.github.io/AgentGhost/#anonymity) | Fetch pipeline, identity rotation, routing table |
| [Request lifecycle](https://vventirozos.github.io/AgentGhost/algorithms/request_lifecycle.html) | What happens when a message hits `/api/chat` |
| [Memory hydration (RRF)](https://vventirozos.github.io/AgentGhost/algorithms/memory_hydration.html) | How the six memory tiers are fused per turn |
| [Dream / self-play](https://vventirozos.github.io/AgentGhost/algorithms/dream_cycle.html) | Idle-time consolidation and skill extraction |
| [Docker sandbox](https://vventirozos.github.io/AgentGhost/sandbox/docker.html) | Container lifecycle, mounts, resource limits |

Treat this `README.md` and `CLAUDE.md` as orienting documents. The published site (sourced from `docs/`) is the authoritative reference.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Required environment (see https://vventirozos.github.io/AgentGhost/installation.html for the full list)
export GHOST_API_KEY="..."
export GHOST_HOME="$HOME/ghost"
export GHOST_MODEL="qwen-3.6-35b-a3"
export GHOST_SANDBOX_DIR="$GHOST_HOME/sandbox"
export TOR_PROXY="socks5h://127.0.0.1:9050"   # required — agent fails closed without it

# Core agent (needs a running upstream LLM, Docker daemon, and Tor)
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --host 0.0.0.0 --port 8000 --verbose
```

Companion processes (web UI, Slack bot, voice, image-gen) are documented in [Install & Run](https://vventirozos.github.io/AgentGhost/installation.html#process-commands).

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/ghost_agent/core/` | Reasoning loop, planning, dream cycle, MCTS, swarm router |
| `src/ghost_agent/memory/` | Vector + graph + profile + skill + journal + episodic stores |
| `src/ghost_agent/tools/` | Tool registry and per-tool implementations |
| `src/ghost_agent/sandbox/` | Docker container manager |
| `src/ghost_agent/api/` | FastAPI routes |
| `src/ghost_agent/utils/` | Logging, sanitiser, token counter, Tor helpers |
| `interface/` | Web UI, Slack bot, voice / image servers, desktop client |
| `docs/` | **HTML source for the published reference** (served at [vventirozos.github.io/AgentGhost](https://vventirozos.github.io/AgentGhost/)) |
| `tests/` | Behaviour-organised pytest suite (`asyncio_mode=auto`) |

## Tests & lint

```bash
pytest                                   # full suite
pytest tests/test_agent_planning.py      # single file
pytest -k "memory and not slack"         # filter
black src interface tests
pylint src/ghost_agent
```

## Telemetry

Telemetry from PostHog, ChromaDB, and Hugging Face Hub is disabled at import time. Don't reintroduce libraries that re-enable it without matching opt-outs.
