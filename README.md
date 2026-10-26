# Ghost Agent

A self-hosted AI agent for one owner. It runs on your own hardware, remembers what you tell it across sessions, does its work inside a Docker sandbox, and sends all its network traffic through Tor.

Ghost is not a model — it's the program around one. You point it at any OpenAI-compatible LLM server (llama.cpp, vLLM, Ollama) and it adds memory, tools, a sandbox, after-reply checks, and a learning loop that you approve step by step. Everything runs locally; the only traffic that leaves your machine goes out through Tor.

## What it does

- **Runs tasks, not just chat.** Writes and executes code, reads and writes files, drives a real browser, queries Postgres, generates PDFs and images — all inside a container that shares exactly one directory with your machine.
- **Remembers you across sessions.** Your profile, the facts you've told it about yourself, and the rules you've approved go into every prompt. Documents and older notes are stored too, and searched when a question needs them.
- **Checks what it can check mechanically.** After a reply, it checks the things a program can test: does the page it built load without errors, are the files it says it wrote really on disk, is the reply an answer rather than a description of its work, does a generated image show what the reply says it shows. A page that throws gets one short repair round. Other failures are recorded for the learning loop. These checks do **not** judge whether a factual answer is true.
- **Learns only what you approve.** A 👎 or a correction from you starts an idle-time replay of that question. The replay ends in *one* proposed rule and a regression test. Neither takes effect until you say so, and kept tests re-run after every deploy.
- **Searches Tor hidden services.** Ordinary search engines don't index `.onion` sites, so there's a separate path through the dedicated onion engines, with each retry forced onto a fresh circuit. Downloads go to a quarantine and are never run.
- **Fails closed on the network.** A boot-time Tor probe plus a process-wide block on direct connections to public addresses. If Tor is unreachable, Ghost doesn't start.

It holds no API keys for third-party services and has no ability to authenticate as you to anything.

## What it doesn't do (any more)

Earlier versions did a lot more on their own. Each of these was measured on real turns, didn't help, and was switched off or deleted:

- **A second model judging every answer.** Off since 2026-10-14. It was wrong on 9 of the 17 answers it rejected.
- **Self-teaching while idle.** Skill extraction, self-written lessons, reflection, post-mortems, practice on invented problems, and retraining two small local models were all removed on 2026-10-11.
- **Prompt tuning.** No tuned prompt ever beat the hand-written one, so the optimizer was removed. Every prompt is hand-written.

## What to expect

Ghost runs a local model (Qwen3.8-27B today), so answers are less capable than a frontier hosted assistant's. In a graded sample of real owner turns from October 2026, about 43% of answers were right and 14% wrong. Facts the model makes up are the biggest open failure. The median reply takes about ten seconds; long tool-heavy turns can take minutes. That's the price of keeping everything on your own hardware and off the open network. If you need frontier-quality answers more than you need that privacy, this is the wrong tool.

## Requirements

| | |
| --- | --- |
| **An LLM server** | Anything speaking the OpenAI chat API. Runs on **Qwen3.8-27B** (developed on Qwen 3.6 35B-A3 before that); the prompts assume roughly that capability. A second, small model on another machine (Gemma 4 E4B) handles research extraction and image checks. |
| **Docker** | Daemon running. Every piece of executed code goes into a container. |
| **Tor** | A local daemon on `9050`. Ghost refuses to start without it. |
| **Python 3.10+** | Plus `requirements.txt` (ChromaDB, PyTorch and sentence-transformers make the first install large). |

## Install and run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export GHOST_HOME="$HOME/ghost"                # everything Ghost owns lives here
export GHOST_API_KEY="..."                     # required for a non-loopback bind
export GHOST_MODEL="qwen3.8-27b"           # must match what your LLM server serves
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

`memory_system_loaded` and `biological_watchdog_alive` must both be `true`. The first means memory loaded. The second means the idle scheduler is alive: that's what runs memory upkeep, failure replays and your regression tests (the key keeps an old name). Nothing else will tell you if either is wrong.

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
| [What it remembers](https://vventirozos.github.io/AgentGhost/memory.html) | What's stored, what goes into every prompt, how to correct it. |
| [What it can do](https://vventirozos.github.io/AgentGhost/tools.html) | The toolbox, and how Ghost chooses between tools. |
| [Checking and learning](https://vventirozos.github.io/AgentGhost/self-improvement.html) | The after-reply checks, the learning loop you approve, and idle-time upkeep. |
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
  core/          turn loop, after-reply checks, memory bus, failure replay, idle scheduler
  memory/        the stores, plus scratchpad, sessions and projects
  tools/         the toolbox and its registry
  sandbox/       Docker container lifecycle and supervised services
  selfhood/      first-person continuity across sessions
  workspace/     the outward-facing model of your files and tasks
  distill/       turn recording, reaction judging and correction detection
  optim/         tool descriptions rendered from their signatures
  eval/          the evaluation suite and the verifier's test bench
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

## License

Ghost is released under the [BSD 3-Clause License](LICENSE). Copyright (c) 2026 Vasilis Ventirozos.
Use it, change it, sell it, embed it; keep the notice and the disclaimer.

Bundled third-party code keeps its own licence, all MIT: [three.js](interface/static/vendor/three/),
[marked](interface/static/vendor/marked.min.js), [mermaid](interface/static/vendor/mermaid.min.js).
Python dependencies are listed in `requirements.txt` under their own licences.
