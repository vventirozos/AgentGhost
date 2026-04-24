# Self-Improvement Pipeline

Ghost Agent's Stage 1 self-improvement substrate. All six modules run
fully local — no external teacher, no hosted embedder, no outbound
API. The reasoning chain: **measure → log → route → optimise →
acquire → reflect**, each step producing signal the next needs.

**Status:** wired end-to-end in `main.py`, verified against the live
agent. A seeded failure produces a reflection whose diagnosis + plan
is persisted into `SkillMemory`, retrieved by the memory bus on a
fresh-but-similar user turn, and visibly changes the agent's first
action — all without any weight update.

## Module map

| Module | Purpose | Wires into |
|---|---|---|
| `ghost_agent._env` | Telemetry hardening (single source of truth) | `main.py` import + `probe:telemetry_disabled` |
| `ghost_agent.eval` | Offline outcome eval harness + network egress guard | CLI `scripts/eval_baseline.py` (`--suite {default,post_learning} --timeout N`) |
| `ghost_agent.distill` | Trajectory JSONL logs + N-sample self-consistency | `$GHOST_HOME/system/trajectories/` |
| `ghost_agent.router` | Hand-crafted features → numpy logistic regression → dispatch | `core/agent.py::handle_chat` (body\["_router_decision"\]) |
| `ghost_agent.optim` | DSPy/GEPA prompt optimisation (scope-gated) | Tunes planning / tool-selection / reflection prompts |
| `ghost_agent.skills_auto` | Passive skill extraction from validator-passing trajectories | Biological phase 2.6 (logs candidates) |
| `ghost_agent.reflection` | Self-critique biological phase on FAILED trajectories | Biological phase 2.5 → composite sink (JSONL + SkillMemory) |

## The flow (as wired)

```
       user request
            │
            ▼
     ┌──────────────┐
     │   router     │  decision stashed on body["_router_decision"]
     └──────────────┘   (fail-safe: escalates to full swarm when unsure)
            │
            ▼
       agent turn ─────────────────┐
            │                      │
            ▼                      ▼
  _record_turn_trajectory    response to user
  → TrajectoryCollector
    .append (redacts, day-
     partitioned JSONL)
            │
            ▼
   ┌──────────────┐
   │ biological   │  idle-time phases:
   │   watchdog   │   1.   journal drain       (>120s)
   └──────────────┘   2.   REM dream           (600-3600s, cooldown 30m)
            │         2.5  reflection          (900-3600s, cooldown 40m)
            │         2.6  skills auto-extract (900-3600s, cooldown 2h)
            │         3.   self-play           (>3600s,   cooldown 60m)
            ▼
  FAILED trajectories ─→ Reflector.run ─→ (diagnosis, plan)
                                          │
                                          ▼
                         composite sink ──┬──→ JSONL (task_kind=reflection)
                                          └──→ SkillMemory.learn_lesson
                                               (retrieved by memory bus
                                                on next similar user turn)

   on-demand:
            │
            ▼
     eval/suite.run → SuiteResult → diff vs baseline.json
       (--suite default        → regression + curated + template bank)
       (--suite post_learning  → 5 file-read-shape prompts that score
                                 "discover before reading" behaviour)
```

## Privacy guarantees (strict local-only)

* `ghost_agent._env` sets every telemetry opt-out at import time
  (`ANONYMIZED_TELEMETRY`, `POSTHOG_DISABLED`, `TELEMETRY_IMPL`,
  `CHROMA_TELEMETRY_IMPL`, `HF_HUB_DISABLE_TELEMETRY`,
  `DISABLE_VERSION_CHECK`). `check_disabled()` is the probe's verdict
  — adding a new required flag in `_REQUIRED_FLAGS` is picked up
  automatically by the regression probe.
* `eval/network_guard.no_external_network()` — opt-in context manager
  that raises `NetworkEgressError` on any non-loopback socket connect.
  `scripts/eval_baseline.py` wraps the whole suite run in it.
* `distill/redact.redact_trajectory` — runs inside every `collector.append`.
  Strips API keys (OpenAI, Anthropic, Slack, GitHub, AWS), bearer
  headers (HTTP and JSON-quoted), `.onion` addresses, emails,
  `/Users/<name>` / `/home/<name>` paths, and non-loopback IPv4.
  Idempotent and order-preserving. Verified end-to-end in the live
  test: a prompt containing `sk-liveABCDEFGH...` and `/Users/alice/...`
  landed on disk with `<REDACTED_API_KEY>` / `/Users/<user>`.
* `optim/run_gepa` uses **Ghost's own upstream** as the optimizer LM
  via the `_GhostLMAdapter` wrapper — no teacher endpoint anywhere.
  `dspy-ai>=3.2.0` is listed in `requirements.txt`; the wrapper
  defers `import dspy` to call sites via `_require_dspy()` so a
  broken install surfaces a clear error instead of a cryptic
  `ImportError` during module load.
* `router/` uses hand-crafted features; the (optional) embedding
  augment is downloaded once and then runs offline.

## Running the eval

```bash
# Default suite (mixed regression + curated + template bank) against
# a running Ghost Agent on 127.0.0.1:8000:
python -m scripts.eval_baseline freeze \
    --suite default \
    --runner http --base-url http://127.0.0.1:8000 \
    --api-key "$GHOST_API_KEY" \
    --model qwen-3.6-35b-a3 \
    --timeout 300 \
    --output "$GHOST_HOME/system/eval/baseline.json"

# Post-learning suite: 5 file-read-shape prompts that score the
# "discover before reading" lesson the Reflector has been producing.
# A live-agent compare against a pre-seeding baseline shows whether
# the lesson is generalising:
python -m scripts.eval_baseline freeze \
    --suite post_learning \
    --runner http --base-url http://127.0.0.1:8000 \
    --api-key "$GHOST_API_KEY" \
    --model qwen-3.6-35b-a3 \
    --timeout 300 \
    --output "$GHOST_HOME/system/eval/post_learning.json"

# Compare a later run to the frozen baseline:
python -m scripts.eval_baseline compare \
    --suite default \
    --runner http --base-url http://127.0.0.1:8000 \
    --timeout 300 \
    --baseline "$GHOST_HOME/system/eval/baseline.json"
# Exit code 0 on no regressions, 1 on any top-level pass_rate drop.
```

Flag notes:

* **`--runner stub`** — the default; echoes prompts and makes
  non-regression tasks fail. Exists so CI can exercise the pipeline
  without a live upstream.
* **`--runner http`** — POSTs to a running agent over loopback.
  The network guard permits only `127.0.0.1` / `localhost`, so this
  is the only shape of real-agent eval that stays privacy-safe.
* **`--timeout N`** — applied to BOTH the httpx client AND
  `EvalSuite.per_task_timeout_s`. Default 300 s. Template tasks that
  multi-turn against a local Qwen-class model commonly run 80–250 s;
  the earlier 60 s default produced spurious timeouts that made the
  baseline `pass_rate` look worse than the agent's actual behaviour.
* **`--suite post_learning`** — small targeted bank used to
  demonstrate reflection lessons are being picked up by the memory
  bus on fresh turns. Passing means the agent's response contains a
  discovery signal (`list / find / search / locate / verify /
  workspace` keywords); failing means it blindly fabricated a result
  without verifying the file exists.

## Wiring the reflection phase

Wired automatically in `main.py` during `lifespan`. Minimum config
for a custom entry point:

```python
from ghost_agent.distill import TrajectoryCollector
from ghost_agent.reflection import Reflector

async def critique(prompt: str) -> str:
    # max_tokens=4096 is deliberate — Qwen 3.6 is a reasoning model
    # whose hidden `reasoning_content` often exceeds 2000 tokens.
    # A short cap leaves the `content` field empty and the reflector
    # logs "unparseable reflection response".
    res = await llm_client.chat_completion({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": False,
    })
    return res["choices"][0]["message"]["content"]

ctx.trajectory_collector = TrajectoryCollector()  # writes to $GHOST_HOME/system/trajectories
ctx.reflector = Reflector(critique_fn=critique, model=args.model)

# The important part — the composite sink is what CLOSES the loop.
# Without it, reflections just land in JSONL and nothing reads them.
def reflection_sink(traj):
    ctx.trajectory_collector.append(traj)
    ctx.skill_memory.learn_lesson(
        task=traj.user_request[:400],
        mistake=traj.extra.get("source_failure_reason", "failure")[:400],
        solution=(traj.planning_output or traj.final_response)[:1200],
        memory_system=ctx.memory_system,
    )
ctx.reflection_sink = reflection_sink
```

With those set, the watchdog fires reflection every ~40 min on recent
FAILED trajectories. Each reflection is persisted to JSONL AND to
`SkillMemory`; on any future user turn whose semantic-neighbourhood
retrieval surfaces the lesson, the agent enters the turn already
primed with the corrected plan.

## Cooldown anchor discipline

All idle-triggered phases mirror the same pattern — fail to follow
it and the phase re-fires every 60 s on exception until the idle
window naturally expires:

```python
# 1. Set anchor BEFORE await — a crash mid-run still advances it.
self._last_reflection_at = datetime.datetime.now()
try:
    await reflector.run(...)
finally:
    # 2. Re-affirm in finally — belt AND braces.
    self._last_reflection_at = datetime.datetime.now()
```

`_last_dream_at`, `_last_reflection_at`, `_last_skills_auto_at`, and
`_last_selfplay_at` all follow this shape. The `test_reflection_
biological_tick` integration test exercises this explicitly — the
anchor must advance even when the inner call raises.

## Verified end-to-end (2026-04-24)

Direct functional test against the live agent (upstream Qwen 3.6 35B-A3):

1. **Seeded trajectories:** 2 FAILED (`FileNotFoundError: access.log`,
   `awk: can't open file emails.txt`).
2. **Reflection produced specific diagnoses:**
   * "The file `access.log` is missing from the workspace, so the directory must be listed to identify the correct name"
   * "The `emails.txt` file was not present in the sandbox workspace when the awk command was executed"
3. **Plans were actionable** (3-step sequences starting with `file_system(action=list)`).
4. **`SkillMemory` playbook grew from 1 → 3 lessons.** The skill_mem's
   "🎓 skill acquired — Lesson learned: ..." log fired for each.
5. **Retrieval works on unseen similar prompts.** User sent *"I need
   to parse a logfile and count errors. Just tell me your FIRST step
   in 1 sentence."* — the memory bus hydrated the lesson, and the
   agent replied with **"I'll search for the log file in your workspace
   so I can locate and analyze it"**. That is the corrective behaviour
   the Reflector learned, applied without any weight update.
6. **Post-learning eval:** 3/5 targeted prompts scored as
   discover-first, exposing a measurable generalisation delta.
7. **Default eval suite:** 15/15 (`pass_rate=1.000`) with
   `--timeout 300`. All 8 template clusters (`algo, bash, concurrency,
   data_analysis, python_general, regex_parse, sql, web_automation`)
   completed in 84–246 s — the earlier 0.400 pass-rate was a pure
   timeout artifact, not an agent regression.

## Sandbox image (prerequisite for template tasks & self-play)

Template-cluster tasks (`data_analysis`, `regex_parse`, `algo`, `bash`,
`sql`, `concurrency`, `python_general`, `web_automation`) and the
self-play harness all run LLM-emitted code inside a Docker container.
The container image is `ghost-agent-base:latest`, built from
`sandbox/Dockerfile`. Build it **once per Ghost version**:

```bash
scripts/build_sandbox_image.sh
# → builds ghost-agent-base:latest (~2 GB first run; ~5 min on a warm
#   docker cache) and runs a Chromium smoke test.
```

The Dockerfile bakes apt system packages, the deep-learning pip stack,
and `playwright install --with-deps chromium` at image build time —
self-play can launch browser tasks immediately without burning agent
turns on runtime re-installs.

If the image is missing, the runtime wrapper falls back to installing
everything inside a fresh container on first boot, committing to
`ghost-agent-base:latest` when done. Both paths converge on the
`/root/.supercharged.v2` marker; older images without it are treated
as un-provisioned.

Diagnostic: if the self-play log shows `playwright install chromium`
firing as an agent tool call, the container's Chromium install is
broken. The runtime gate now detects this (marker present + binary
absent) and forces a re-install on next `ensure_running`; if the
behaviour persists, rebuild the image: `scripts/build_sandbox_image.sh`.

## Stage 2 hook (future work)

The trajectory log is the ingredient Stage 2 (local SFT via rejection
sampling) needs. `distill.self_consistency.pairwise_pass_fail()`
produces the (failed, succeeded) pairs; `optim.trainset.build_trainset`
consolidates them per signature. Training itself needs GPU and is
out of Stage 1 scope. The skills_auto phase currently LOGS candidates
only — promoting them into `memory/skills.py` or
`tools/acquired_skills.py` is a deliberate follow-up step because
persisting auto-extracted sequences without human review can poison
retrieval.
