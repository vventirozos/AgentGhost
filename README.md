# Ghost Agent

A self-hosted AI agent. It runs on your own hardware, remembers what you tell it across sessions, does its work inside a Docker sandbox, and checks its own answer with a second model before replying.

Ghost is not a model — it's the program around one. You point it at any OpenAI-compatible LLM server (llama.cpp, vLLM, Ollama) and it adds memory, tools, a sandbox, a self-check, and a set of background passes that learn from its own transcripts. Everything runs locally; the only traffic that leaves your machine goes out through Tor.

## What it does

- **Runs tasks, not just chat.** Writes and executes code, reads and writes files, drives a real browser, queries Postgres, generates PDFs — all inside a container that shares exactly one directory with your machine.
- **Remembers across sessions.** Six stores — facts, relationships, your profile, lessons from its own mistakes, a journal, and full past episodes — searched in parallel and merged by rank rather than score.
- **Refuses its own answers.** A second model reads the final answer against the evidence the tools actually returned, and asks whether it answers the question *before* asking whether it's correct. A high-confidence refusal sends Ghost back to fix it.
- **Learns while idle.** Given a quiet stretch it consolidates memory, reviews failed turns into lessons, extracts repeated procedures into reusable skills, retrains two small local models, and practises against problems it invents for itself. None of it interrupts you.
- **Tunes its own prompts.** An optimizer rewrites three internal prompts from Ghost's own successful turns, using the same local model — no teacher, no external service. Capped at 16 rounds, because longer optimization drifts toward gaming the metric, and a tuned prompt only ships if it beats the hand-written one.
- **Searches Tor hidden services.** Ordinary search engines don't index `.onion` sites, so there's a separate path through the dedicated onion engines, with each retry forced onto a fresh circuit.
- **Fails closed on the network.** A boot-time Tor probe plus a process-wide block on direct connections to public addresses. If Tor is unreachable, Ghost doesn't start.

It holds no API keys for third-party services and has no ability to authenticate as you to anything.

## Requirements

| | |
| --- | --- |
| **An LLM server** | Anything speaking the OpenAI chat API. Tuned around **Qwen 3.6 35B-A3**; the prompts assume roughly that capability. |
| **Docker** | Daemon running. Every piece of executed code goes into a container. |
| **Tor** | A local daemon on `9050`. Ghost refuses to start without it. |
| **Python 3.10+** | Plus `requirements.txt` (ChromaDB, PyTorch and sentence-transformers make the first install large). |

## Install and run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export GHOST_HOME="$HOME/ghost"                # everything Ghost owns lives here
export GHOST_API_KEY="..."                     # required for a non-loopback bind
export GHOST_MODEL="qwen-3.6-35b-a3"           # must match what your LLM server serves
export TOR_PROXY="socks5://127.0.0.1:9050"

python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --host 127.0.0.1 --port 8000 --verbose
```

Boot is deliberately noisy — it prints each subsystem as it loads, so a hang tells you which one is stuck. First start takes a minute while the embedding model loads.

> **On a fresh install, boot once with `--no-mandatory-tor`.** The embedding model hasn't been downloaded yet and the Tor egress guard blocks that first fetch. Every run afterwards is normal.

### Check it actually works

Starting and working are different things — Ghost can boot degraded, answering HTTP with no memory at all.

```bash
curl -s -H "X-Ghost-Key: $GHOST_API_KEY" \
     http://127.0.0.1:8000/api/health | python -m json.tool
```

`memory_system_loaded` and `biological_watchdog_alive` must both be `true`. The first means memory loaded; the second means the background learning loop is alive. Nothing else will tell you if either is wrong.

### Talk to it

The agent is an HTTP service with no interface of its own. The web UI is a separate process:

```bash
python interface/server.py     # then open http://localhost:8080
```

Slack, voice, a desktop client and a CLI are also available — see [Interfaces](https://vventirozos.github.io/AgentGhost/interfaces.html).

## Documentation

**<https://vventirozos.github.io/AgentGhost/>** — sources in [`docs/`](docs/).

Two tiers. The **guide** is written for a beginner-to-intermediate reader:

| Page | What's in it |
| --- | --- |
| [What Ghost Agent is](https://vventirozos.github.io/AgentGhost/) | The whole thing in five steps, and whether it's a good fit. |
| [Install and run it](https://vventirozos.github.io/AgentGhost/getting-started.html) | Prerequisites in order, environment variables, verifying the install. |
| [Interfaces](https://vventirozos.github.io/AgentGhost/interfaces.html) | Web UI, Slack, voice, desktop client, CLI, HTTP API. |
| [One message, end to end](https://vventirozos.github.io/AgentGhost/how-it-works.html) | The turn loop, and which step to blame when it goes wrong. |
| [What it remembers](https://vventirozos.github.io/AgentGhost/memory.html) | The six stores, how they're merged, how to correct one. |
| [What it can do](https://vventirozos.github.io/AgentGhost/tools.html) | The toolbox, and how Ghost chooses between tools. |
| [How it improves itself](https://vventirozos.github.io/AgentGhost/self-improvement.html) | The self-check, the idle-time passes, and prompt tuning. |
| [Safety and privacy](https://vventirozos.github.io/AgentGhost/safety.html) | What leaves your machine, what can't, how it's enforced. |
| [Settings that matter](https://vventirozos.github.io/AgentGhost/configuration.html) | The twenty flags worth knowing, out of sixty-four. |
| [When something breaks](https://vventirozos.github.io/AgentGhost/troubleshooting.html) | Sorted by symptom. |
| [Glossary](https://vventirozos.github.io/AgentGhost/glossary.html) | Plain-English definitions for this project's shorthand. |

The **[deep reference](https://vventirozos.github.io/AgentGhost/reference.html)** is a page per module for changing the code — dense, and explicit about which parts of `src/ghost_agent/` it doesn't yet cover.

## Repository layout

```
src/ghost_agent/
  main.py        CLI entrypoint and FastAPI lifespan
  api/           HTTP routes (Ollama- and OpenAI-compatible)
  core/          turn loop, planning, verifier, memory bus, idle-time passes
  memory/        the six stores, plus scratchpad, sessions and projects
  tools/         the toolbox and its registry
  sandbox/       Docker container lifecycle and supervised services
  selfhood/      first-person continuity across sessions
  workspace/     the outward-facing model of your files and tasks
  distill/       trajectory recording — what every learning pass reads
  reflection/    failure review and the post-mortem engine
  skills_auto/   automatic skill extraction and re-verification
  prm/ router/   the two small self-trained models
  optim/         prompt optimization and tool-choice evaluation
  eval/          the evaluation suite and benches
  utils/         logging, redaction, the Tor egress guard, tokenisation
interface/       web UI, Slack bot, voice, desktop client, CLI
docs/            this documentation set
tests/           the test suite
```

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

788 test files. `GHOST_HOME` is unset for every test by an autouse fixture, so a
run in a shell that exports it cannot reach — or write to — a live agent's
memory. Tests needing a home set one explicitly.
