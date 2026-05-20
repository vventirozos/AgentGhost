# Self-Improvement Pipeline

Ghost Agent's Stage 1 self-improvement substrate. All six modules run
fully local — no external teacher, no hosted embedder, no outbound
API. The reasoning chain: **measure → log → route → optimise →
acquire → reflect**, each step producing signal the next needs.

**Status:** wired end-to-end in `main.py`, verified against the live
agent on **two complementary timescales**:

* **real-time** (post-turn) — a user-correction-shaped follow-up
  promotes the prior trajectory to FAILED, schedules
  `Reflector.reflect_one` as an `asyncio.create_task`, and the
  composite sink writes the lesson to `SkillMemory` before the user
  types their next message. No idle window required.
* **idle backstop** (biological watchdog phase 2.5) — the original
  path. Trajectories the user never returned to correct (or whose
  correction was missed by the heuristic gate) get reflected on
  during the 15-60 min idle window.

Both paths share the composite sink and the `_reflected_trajectory_ids`
dedup set, so a trajectory reflected by either path is skipped by
the other. Verified live: a seeded failure produces a reflection
whose diagnosis + plan is persisted into `SkillMemory`, retrieved by
the memory bus on a fresh-but-similar user turn, and visibly changes
the agent's first action — all without any weight update.

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
| `ghost_agent.prm` | Per-step value model — scores `(state, action)` for MCTS lookahead | Biological phase 2.7 (retrain) → `core.mcts.MCTSReasoner` (fast scoring) |
| `ghost_agent.selfhood` | First-person autobiographical diary + self-state + recognition / wake-up + narrative consolidation | Biological phase 2.8 (narrative regen) → wake-up prefix on every `handle_chat` |

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
  → stash (response_fp → traj)        ◄─ enables real-time path
    on ctx._recent_trajectories
            │
            ▼
   NEXT user turn arrives
            │
            ▼
   _maybe_promote_prior_turn_via_user_correction:
     (A) anchored correction-phrase regex
         on the new user message AND
     (B) Jaccard token-overlap of new
         user vs prior user ≥ 0.40
     ──── BOTH must fire ────
                │
                ▼
        update_outcome → corrections.jsonl  (overlay, audit-safe)
        traj.outcome   = FAILED              (in-memory)
        asyncio.create_task(
            Reflector.reflect_one(traj, sink))    ◄─ fire-and-forget,
                                                     real-time path
            │
            ▼
   ┌──────────────┐
   │ biological   │  idle-time phases (backstop):
   │   watchdog   │   1.   journal drain       (>120s)
   └──────────────┘   2.   REM dream           (600-3600s, cooldown 30m)
            │         2.5  reflection          (900-3600s, cooldown 40m)
            │         2.6  skills auto-extract (900-3600s, cooldown 2h)
            │         2.7  PRM retrain         (900-3600s, cooldown 3h)
            │         2.8  selfhood narrative  (900-3600s, cooldown 1h)
            │         3.   self-play           (>3600s,   cooldown 60m)
            ▼
  FAILED trajectories ─→ Reflector.run ─→ (diagnosis, plan)
                                          │
                  ─────── BOTH paths converge here ────────
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
ctx.reflector = Reflector(
    critique_fn=critique,
    # 120s ceiling: Qwen 3.6 is a reasoning model whose
    # `reasoning_content` phase regularly burns 30-60s before any
    # visible content, AND the post-turn reflect_one path competes
    # with the user-facing turn for the same upstream. 45s was too
    # tight in practice — observed silent timeout on the post-turn
    # path even though the structural promotion fired correctly.
    per_call_timeout_s=120.0,
    model=args.model,
)

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

With those set:

* **The biological watchdog** fires reflection every ~40 min on
  recent FAILED trajectories that the user never returned to (or
  whose correction the heuristic gate missed).
* **The real-time post-turn path** (`_maybe_promote_prior_turn_via_user_correction`)
  is automatically active once `ctx.reflector` and
  `ctx.trajectory_collector` are wired — `handle_chat` invokes it
  on every user turn with no extra opt-in. A correction-shaped
  follow-up promotes the prior trajectory and schedules
  `reflect_one` immediately; the lesson typically lands within
  ~10 s of the correction message returning, on a warm upstream.

Each reflection is persisted to JSONL AND to `SkillMemory`; on any
future user turn whose semantic-neighbourhood retrieval surfaces the
lesson (planner pre-fetch at `agent.py:2260`, execution-stage
fetch at `agent.py:2402`), the agent enters the turn already primed
with the corrected plan.

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

`_last_dream_at`, `_last_reflection_at`, `_last_skills_auto_at`,
`_last_prm_train_at`, and `_last_selfplay_at` all follow this shape.
The `test_reflection_biological_tick` and `test_prm_biological_phase`
integration tests exercise this explicitly — the anchor must advance
even when the inner call raises.

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

## Closing the loop on interactive-session failures (2026-04-26)

The Reflector iterates only `outcome=FAILED` trajectories. Chat turns
ship with `outcome=UNKNOWN` because there's no validator on free-form
chat — only self-play and self-consistency batches produce explicit
FAILED. That made the self-improvement loop *blind* to interactive
sessions: a 70-minute thrash on a misdiagnosed UI bug never produced
a lesson because the per-turn trajectories all stayed UNKNOWN.

`distill/outcome_heuristics.py::classify_chat_outcome` looks at a
just-recorded chat trajectory and promotes UNKNOWN → FAILED when one
of four signals fires:

1. **Runtime abort markers** — `[ATTEMPT_ABORTED_*]` substrings in
   `final_response` (cross-turn loop, thinking-budget cap, n-gram
   loop, …). These markers fire only when an in-band guard has
   already determined the turn was non-productive, so they're a
   strong signal.
2. **Browser selector thrash** — the same selector appears in ≥ 4
   browser tool-call invocations within one turn (atomic ops + every
   `interact` sub-action are counted). This is the exact shape of the
   2026-04-26 webOS incident: identical click selectors fired across
   8 nested `interact` calls.
3. **Repeated identical tool errors** — the same `(tool, normalized
   error message)` pair appears ≥ 3 times in one turn. Errors are
   normalised (whitespace squash, lowercase, leading "Error:"
   prefix stripped) so two textually-similar errors hash to one key.
4. **Browser sequence aborted** — the result text contains
   `⚠ SEQUENCE ABORTED` (set by `op_interact` when a goto fails and
   cascades through the rest of the action list).

The classifier is **conservative**: existing PASSED / FAILED outcomes
are never overruled, three repeats of the same selector is below
threshold, and a single tool error doesn't fire. False positives
flood the lesson store with bad reflections, so the bar stays high.

`apply_chat_outcome_heuristics(traj)` is the in-place wrapper called
from `core/agent.py::_record_turn_trajectory` just before
`collector.append`. It runs after the trajectory is fully assembled;
classification failure is logged at debug and never blocks the turn.
Cross-turn signals (e.g. "the same misdiagnosis appears across 5
turns") need session-scoped state and are deliberately out of scope —
they belong in a future `session_telemetry.py` keyed by `session_id`.

Coverage: `tests/test_trajectory_failure_heuristic.py` (signal
matrix, threshold knobs, no-op on healthy turns, end-to-end
integration with the Reflector).

## Real-time loop closure: user-correction promotion (2026-04-28)

`outcome_heuristics` (above) catches *mechanically-stuck* failures —
selector thrash, repeated tool errors, abort markers. It misses the
dominant interactive-chat failure mode: the agent confidently
produces an answer that's *wrong*, the user pushes back, and we
want to learn from that exchange before the user's next message.

The user's next message **is** the cheapest, most reliable supervisor
for free-form chat. If they're correcting us, the prior turn was
FAILED — by the user's own verdict, no validator required. Two
mechanisms make that signal usable:

### 1. The classifier (`distill/user_correction.py`)

Pure-Python, two-signal predicate. **Promotion requires BOTH** to
fire:

* **Signal A — anchored correction phrase.** Regex anchored at the
  start of `current_user_text`: `no`, `nope`, `wrong`, `actually`,
  `that's not right`, `I meant`, `you misunderstood`, `try again`,
  `redo`, `didn't work`, … A "no" deep inside a sentence does *not*
  count (anchored start guards against discourse-marker false
  positives).
* **Signal B — semantic rephrase.** Token-overlap Jaccard between
  the prior user message and the current one, computed over
  content tokens (stopwords stripped — articles, pronouns, common
  quantifiers, common modal verbs). Threshold ≥ `0.40`. The
  intuition: if the user is re-asking the same question, that's
  strong evidence the prior assistant answer was inadequate.

A single signal alone has too many false positives. *"No, I think
you're right"* is phrase-without-rephrase. *"… and also, what about
X?"* is rephrase-without-phrase. Both signals together catch the
genuine corrections while leaving prosaic follow-ups alone. The
classifier is purely lexical — no LLM call, no embeddings.

Knobs (module-level constants for runtime tuning):

| Constant | Default | Meaning |
|---|---|---|
| `JACCARD_REPHRASE_THRESHOLD` | `0.40` | Minimum content-token overlap for Signal B |
| `MIN_CURRENT_TOKENS_FOR_REPHRASE` | `2` | Floor on current-message content tokens (a bare "no" can't fire B) |

Coverage: `tests/test_user_correction.py` (24 cases — phrase
coverage, single-signal guards, anchored start, defensive
normalisation of None / non-string inputs, threshold pinning,
verdict-shape contract).

### 2. The wiring (`core/agent.py`)

Two new helpers on `GhostAgent`:

* `_stash_trajectory_for_correction_lookup(traj)` — called inside
  `_record_turn_trajectory` right after `collector.append`. Builds
  a stable md5 fingerprint of the response prefix (whitespace-
  collapsed first 500 chars, lowered) and stores
  `{fingerprint: traj}` on `ctx._recent_trajectories_for_correction`.
  Bounded LRU at 32 entries — enough for several concurrent
  conversations without unbounded growth.

* `_maybe_promote_prior_turn_via_user_correction(messages, current_user_text)` —
  called from `handle_chat` immediately after `last_user_content`
  is set. Walks `messages[:-1]` to find the prior assistant +
  prior user, fingerprints the assistant content, looks up the
  cached trajectory. If the classifier returns
  `is_correction=True`, it:
  1. mutates the cached trajectory's `outcome` and `failure_reason`
     **in memory** (so the immediate `reflect_one` call sees them);
  2. appends a record to the corrections sidecar (durable);
  3. drops the cache entry (one promotion per stashed trajectory);
  4. schedules `Reflector.reflect_one(traj, sink, already_reflected)`
     via `loop.create_task` — fire-and-forget, the user turn
     doesn't block on the LLM critique.

The `_pending_reflection_tasks` set on the context tracks in-flight
tasks (each adds itself, removes on done). A done-callback logs
the result via `pretty_log("Post-Turn Reflection", …)` — `ok` with
the diagnosis preview, `no lesson` with the error reason
(`timeout after Ns`, `unparseable reflection response`), or
`failed` with the exception type. **Without that callback the
async task's result is invisible**: a critique timeout silently
produced no lesson and operators couldn't tell the difference
between "loop misfired" and "LLM was slow".

The shared `_reflected_trajectory_ids` set is honoured by both
`Reflector.reflect_one` (real-time) and `Reflector.run` (biological
backstop) so a trajectory reflected via the real-time path is
skipped by the watchdog and vice versa.

### 3. The corrections sidecar (`distill/collector.py`)

Outcome promotions discovered AFTER the original JSONL write land
in **`corrections.jsonl`** at the trajectory tree root, NOT by
rewriting the original line. `update_outcome(trajectory_id, outcome,
reason, source=…)` appends a JSON line; `iter_trajectories` overlays
the latest correction per id on read. Properties:

* The original JSONL line stays byte-identical — the audit trail is
  preserved.
* Last-write-wins on repeat updates for the same id.
* Malformed sidecar lines are skipped without poisoning the overlay.
* Orphan corrections (id not in any JSONL) are silently ignored on
  read.
* `update_outcome` is a no-op when the collector is `enabled=False`
  (mirrors `append`).
* The sidecar is a single file (NOT day-partitioned) — the workload
  is tiny (one record per failed turn) and a single growing file
  lets readers apply corrections in O(corrections) instead of
  scanning every day's directory.

Coverage: `tests/test_trajectory_corrections_sidecar.py` (12 cases),
`tests/test_post_turn_reflection_wiring.py` (12 cases), and the
end-to-end ratchet `tests/test_self_improvement_loop_e2e.py` (2
cases).

### 4. `Reflector.reflect_one(traj, sink, already_reflected)`

Single-trajectory entrypoint that bypasses the iterator path used
by `run()`. Honours the same `already_reflected` dedup set, and
adds the trajectory id to the set **before** the await — so a
concurrent biological-tick `run()` can't double-reflect on the same
trajectory while `reflect_one`'s critique is pending. Sink contract
matches `run()`: invoked once per ok reflection; sink exceptions
are logged at WARNING and swallowed.

Coverage: `tests/test_reflect_one.py` (7 cases).

### Live verification (2026-04-28)

Multi-turn against the running agent on `:8000`:

```
turn 1 user: "Reply with three words exactly: kangaroo trampoline lighthouse..."
turn 1 ai  : "kangaroo trampoline lighthouse"
   → trajectory id=46df2dfe... outcome=unknown
   → fingerprint stashed on ctx cache

turn 2 user: "no, reply with three words exactly: kangaroo trampoline metronome..."
   → classifier verdict: is_correction=True, signals=[phrase, rephrase(jaccard=0.82)]
   → corrections.jsonl record:
     {trajectory_id: "46df2dfe…", outcome: "failed",
      reason: "user-correction signal: phrase + rephrase(jaccard=0.82)",
      source: "user_correction"}
   → TrajectoryCollector overlay yields outcome=FAILED ✓
   → asyncio.create_task(reflector.reflect_one(...))
   → done-callback logs:
     post-turn reflection: ok (traj=46df2dfe): diagnosis='The previous response
     likely included extra text...'
   → SkillMemory.learn_lesson → playbook 22 → 23 lessons
   → similar query "respond with three words exactly" surfaces the new lesson
```

The first attempt of this exact test failed cleanly with
`no lesson (traj=…): timeout after 45.0s` (the model was busy
processing turn 2 on the same upstream). Bumped
`Reflector.per_call_timeout_s` from 45s → 120s in `main.py` —
Qwen 3.6 35B-A3 is a reasoning model whose hidden
`reasoning_content` regularly burns 30-60s before emitting visible
content, and 45s left no headroom when the user-facing turn was
saturating the upstream. After the bump, `reflect_one` completes in
~9s on average and the lesson lands within seconds of the user's
correction message returning.

## Wrong-question detection: verifier alignment + lesson retraction (2026-04-28)

A user trace exposed a triple failure: user asked *"how can I see how many lines of code is a project? just give me the code"*; the agent ran `wc -l` in its own sandbox and replied *"The project has **1,623 lines of code**"*; the verifier returned **CONFIRMED (100%)**; the Perfection-Protocol then saved an **"Optimization Analysis"** lesson into `SkillMemory` based on that wrong answer. Three layers failed in sequence — the model misread the user, the verifier rubber-stamped it, and the opt-prot baked the misread into long-term memory.

The fix touches all three layers.

### 1. Verifier audits user-request alignment

Before: `verify_code_output(code, output, intent)` asked the LLM "does the OUTPUT contain the information the user asked for?" — a check that's true whenever the printed claim is internally consistent with the tool output, regardless of whether the agent answered a different question than the user asked.

After: `verify_code_output(code, output, intent, *, response="")` takes the agent's user-facing reply as a fourth slot. The prompt rubric leads with **constraint satisfaction**:

> Does the user's wording include explicit constraints on the form of the answer? Examples: "just give me the code", "in one sentence", "without using X", "list only the names", "as JSON". If yes, does the AGENT'S RESPONSE satisfy those constraints? **If the user asked for code and the agent returned a number / prose / a result, that is a REFUTED — the agent answered a different question than the one asked, even if the tool output is internally consistent.**

The prompt enumerates the failure shapes explicitly (user asks for code → agent returns a result; user asks for format X → agent ignores it; tool output is a sandbox-internal artefact the user can't actually use) so the verifier LLM has concrete patterns to match. A CONFIRMED verdict requires BOTH the tool output to be sound AND the response to match what the user asked for. The verifier callsite in `core/agent.py` passes `response=final_ai_content` so the rubric has the agent's reply to audit.

Coverage: `tests/test_verifier_user_intent_alignment.py` (10 cases — prompt content invariants, the response slot rendering, back-compat sentinel for callers that don't pass `response`, response-slot truncation, and a rubric-following stub that pins the exact 12:04 failure shape gets REFUTED under the new prompt format).

### 2. Lesson provenance via `source_trajectory_id`

Every `learn_lesson` write now records the trajectory id of the turn that produced the lesson. Persisted on:

* the JSON playbook entry (`source_trajectory_id` field, populated via `build_lesson` and surfaced through `_normalize_lesson` so legacy lessons read back as `""`);
* the vector-store metadata (so `collection.delete(where={"source_trajectory_id": ...})` is the one-liner that scrubs the embedding tier).

Two production writers thread the id:

* **Perfection-Protocol** (`core/agent.py::handle_chat` → `learn_lesson(...)`) uses the current turn's pre-allocated `current_trajectory_id`. The id is allocated at the **start of `handle_chat`** with `uuid.uuid4().hex` because the opt-prot fires BEFORE `_record_turn_trajectory` writes the trajectory to disk; both callsites must use the same id or retraction can't link them.
* **Composite reflection sink** (`main.py`) uses `reflected_trajectory.extra["reflected_from"]` — the *original failed trajectory's* id, not the reflection trajectory's own id. Rationale: the reflection's lesson IS the corrective behaviour for that source failure, so provenance unifies under one id per source-of-failure.

Legacy lessons (written before the schema change) read back as `source_trajectory_id=""`. The empty-string-id case is a **deliberately protected sentinel**: `retract_lessons_from_trajectory("")` returns 0 without touching disk, so a buggy caller passing an empty string can't accidentally scrub every legacy lesson at once.

Coverage: `tests/test_skill_provenance_and_retraction.py` (16 cases across schema, persistence, retraction matching/idempotency, legacy protection, vector-delete `where`-filter shape, error swallowing, and the full poison→correction→retraction integration).

### 3. Retraction on FAILED promotion

`SkillMemory.retract_lessons_from_trajectory(trajectory_id, memory_system=None) -> int` is the scrub primitive. JSON pass under the lock, atomic write of the surviving entries; vector pass via `collection.delete(where={"source_trajectory_id": ...})` (best-effort — JSON is the canonical store). Idempotent. Returns the count removed from the playbook. Logged via `pretty_log("Skill Retracted", …)` so a tail of the agent log makes scrubs visible.

Two callsites in `core/agent.py`:

* **Verifier-driven retraction (preventive)** — when the verifier returns REFUTED with confidence ≥ 0.7, the gate appends the verifier note to `final_ai_content` AND immediately calls `retract_lessons_from_trajectory(current_trajectory_id, memory_system=ctx.memory_system)`. This catches the dominant case at source: the Perfection-Protocol's lesson is on disk, the verifier just disagreed with the response, scrub before the user even sees the reply.
* **User-correction-driven retraction (recovery)** — `_maybe_promote_prior_turn_via_user_correction` calls retract on the prior turn's id immediately after writing the sidecar correction record and BEFORE scheduling reflect_one. The reflection then writes the corrective lesson with the same `source_trajectory_id` (because `reflected_from` is the prior turn's id), so the playbook ends up with the right entry rather than both. Without retraction, the previous-turn's poisoned lesson and the reflection's corrective lesson would coexist in the playbook with no demotion mechanism, and BM25 / vector ranking would still surface the wrong one for some queries.

Both retraction paths are wrapped in `try/except logger.debug` so a retraction failure can never break the user turn. The verifier-driven path runs synchronously inside `handle_chat`; the user-correction path runs synchronously inside the next-turn classifier helper.

### Live verification against the running agent (2026-04-28)

Re-issued the original failure prompt. The agent's response now leads with the command — *"Here's the command: \`find . -type f \\(-name "\*.js" -o -name "\*.html" -o -name "\*.css" -o -name "\*.py" \\) -exec cat {} + | wc -l\`. For this sandbox, the result is **1,601 lines of code**."* — and the verifier returned CONFIRMED (correctly: the user got the command they asked for). The Perfection-Protocol's eager-write gate didn't fire because the response is now > 50 chars (the gate guards against empty replies, not against verbose ones), so no poisoned lesson was written for this turn. The polluted entry from the original 12:04 trace remains in the playbook with `source_trajectory_id=""` (legacy, pre-schema-change) — the protection sentinel keeps it safe from accidental bulk retraction; future opt-prot writes carry provenance and can be scrubbed cleanly.

The non-reproducibility of the failure is itself a partial validation: the agent improved its answer between runs because the polluted lesson surfaced its previously-cached find/wc one-liner in the system-prompt context, and the agent's own planner used it. We can't tell from this alone whether the verifier alignment fix would have caught the original wrong response, so the prompt-rubric audit lives in the unit test (`test_wrong_question_shape_can_be_refuted` exercises a stub LLM that follows the rubric literally on the exact failure-trace inputs and asserts REFUTED).

## Browser `interact` abort semantics

The `browser.interact` op runs a list of sub-actions inside a single
Chromium context. Under the default `stop_on_error=False`, a failed
per-action step (e.g. a click on a missing selector) is recorded and
the loop continues — useful for "try all these selectors, tell me
which ones matched" exploratory flows.

**Navigation failures are the one exception: they always abort the
sequence, regardless of `stop_on_error`.** A `page.goto(...)` that
raises (ERR_FILE_NOT_FOUND, ERR_CONNECTION_REFUSED, DNS failure, …)
leaves Chromium on an error page; every subsequent click/fill/
extract_text would just wait the full per-action timeout for elements
that don't exist. Before the fix a 54-action sequence whose first
goto 404'd hung for ~108 minutes (54 × 120 s) before the outer
subprocess timeout fired.

The fix: `op_interact` in the runner catches the `goto` exception,
records `aborted_sequence: True` on the result, and breaks out of
the loop immediately. The agent-facing output now shows
`⚠ SEQUENCE ABORTED: goto_failed` as a banner so the next-turn
planner reads the failure as "bad URL, retry the whole interact"
rather than "53 mysterious click failures".

Covered by `tests/test_browser_interact_abort.py` — the tests exec
the runner source inline (with a stubbed Playwright import) so the
production code path itself is under test, not a reimplementation.

## Process Reward Model (`ghost_agent.prm`, 2026-04-29)

The PRM is the third inference-time learner in the pipeline (after
`router/` for request difficulty and `skills_auto/` for tool
sequences). It scores per-step `(state, action)` tuples in
microseconds against a numpy logistic regression model trained on the
same trajectory store the rest of the pipeline reads — closing the
loop between past tool-call outcomes and future plan-candidate
evaluation.

Mechanism in one paragraph: terminal `Outcome.PASSED` / `FAILED` is
back-propagated to per-step values via the AlphaZero-style γ-discount
trick (`V(step_i) = γ^(N-i-1) · terminal_value`); features are
hand-crafted (request shape + plan progress + action shape + tool
bucket + cross signals); the model is the same numpy LR shape as
`router/`, with a versioned JSON checkpoint format
(`ghost.prm.logreg.v1`). Loaded once at startup via
`PRMScorer.load(--prm-model)`, hot-swapped via `scorer.set_model(...)`
after each idle retrain pass without an agent restart.

Module layout, training pipeline, and integration details: see
[`docs/algorithms/prm.md`](algorithms/prm.md).

The PRM is **opt-in but always-attached**: `ctx.prm_scorer` is set
unconditionally in lifespan (no-op pass-through when no checkpoint is
loaded), so call sites can score `(state, action)` unconditionally
without branching on availability. MCTS engages the fast path only
when (a) `prm_scorer` is attached, (b) `has_model is True`, and (c)
the caller passes `prm_state=` — falling back to the existing
LLM-simulation path when any of those conditions miss. Existing
callers continue working unchanged; no regression to the 15/15 eval.

CLI:

```bash
# Production: load a previously-trained checkpoint at startup.
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --prm-model "$GHOST_HOME/system/prm/checkpoint.json"

# Bootstrap: omit the flag entirely. The biological retrain phase
# (2.7) will produce a first-ever checkpoint after enough
# trajectories accumulate (defaults: ≥5 trajectories, ≥20 step
# samples, ≥5% per class) and hot-swap it into the live scorer.

# Faster retrain cadence for development:
python -m src.ghost_agent.main --prm-train-cooldown 600
```

Coverage: `tests/test_prm_*.py` (195 tests across features, labels,
model, trainer, MCTS integration, biological phase, corner cases, and
adversarial fuzz/stress). Numerical hardening: NaN/inf inputs are
neutralised at `_vectorize` (inputs) and `_to_arrays` (labels) so a
single bad value can't poison the whole gradient or prediction; MCTS
defensively clamps any scorer return to [0, 1] regardless of the
scorer implementation; concurrent `score()` during `set_model()` was
exercised under 4-reader/1-swapper thread thrash. Full agent suite
remains green at **3248 passing**.

## Frontier-aware self-play (closes the PRM → self-play loop)

The PRM produces a per-step confidence signal; the trajectory store
records per-cluster coverage. Frontier-aware self-play (default on,
`--frontier-selfplay`) combines them to choose which cluster the
biological-watchdog phase-3 self-play pass should target:

```
cluster weight  =  PRM_uncertainty(cluster)  ×  trajectory_rarity(cluster)
                   └─ 1 − 2·|p − 0.5| ─┘     └─ 1/(1 + log1p(count)) ─┘
```

Saturated clusters (per `FrontierTracker.list_saturated_clusters()`)
are excluded with weight 0. The math lives in
`core/frontier_selection.py` as pure functions; the integration is on
`FrontierTracker.pick_frontier_seed`, which mirrors the dict shape of
the legacy `pick_seed` so call sites in `core/dream.py` need no
schema branching.

**Why it matters.** The brittle-pool scoring in `pick_seed` sees
outcomes but not coverage — a cluster the agent has barely tried
looks identical to a cluster it solves first-try (both have no recent
failures). Frontier weighting surfaces the under-explored quiet ones.
That matters because the PRM is itself trained on the trajectories
self-play produces: if self-play keeps targeting the same handful of
well-trodden clusters, the PRM's opinion of the others stays stuck at
neutral, and the brittle-pool picker never gets a reason to rotate
to them. Frontier weighting breaks the loop.

**Engagement gate (strict).** `isinstance(ctx.prm_scorer, PRMScorer)
and ctx.prm_scorer.has_model and isinstance(ctx.trajectory_collector,
TrajectoryCollector)`. MagicMock-backed test contexts fail closed at
both checks, so legacy tests continue exercising the old path
unchanged. Cold-boot agents (no PRM model yet, no trajectories yet)
also fall through cleanly to `pick_seed`.

**Sanity floor.** `--frontier-uniform-sample-prob` (default 0.2)
bypasses frontier weighting on a per-tick dice roll and falls back to
the legacy `pick_seed`. Without this floor, a systematically-wrong
PRM could self-reinforce onto one cluster and starve the others of
training signal — keeping the PRM wrong about them in perpetuity.
20% uniform sampling breaks the feedback loop without losing the
benefit of frontier targeting on the other 80%.

CLI:

```bash
# Default — frontier weighting on with 20% sanity floor:
python -m src.ghost_agent.main --upstream-url "http://127.0.0.1:8080"

# A/B comparison — explicitly revert to legacy brittle-pool pick:
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --no-frontier-selfplay

# Aggressive — drop sanity floor to 5% if the PRM is well-trained:
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --frontier-uniform-sample-prob 0.05
```

Coverage: `tests/test_prm_uncertainty.py` (10) +
`tests/test_frontier_selection.py` (32) +
`tests/test_frontier_pick_frontier_seed.py` (9) +
`tests/test_dream_frontier_weighted.py` (4) = 55 new tests, all
green. Existing `tests/test_dream_synthetic_curiosity.py`,
`tests/test_frontier_tracker.py`, and all `tests/test_selfplay_*.py`
continue passing — no regression to the legacy path.

End-to-end walkthrough:
[`docs/core/frontier_selection.html`](core/frontier_selection.html)
and the new section in
[`docs/algorithms/dream_cycle.html`](algorithms/dream_cycle.html).

## Meaningful self-play redesign (2026-05-17)

A post-mortem of the 2026-05-17 self-play log found that hundreds of
cycles produced **zero** lessons. The agent was solving every cycle
first-try, so:

* the compression-delta metric (tool-call count vs. prior best) stayed
  pinned at `+0.000` — no gradient for the scorer;
* the write gate (`struggled-then-won`, `new_cluster`,
  `first-failure`) never opened because every cluster had been
  seen and no cluster struggled;
* mastery (5-streak first-try wins with `delta > 0.05`) was unreachable
  because `delta` never moved;
* the reflector only fired on `outcome == FAILED`, so passing-but-
  boring cycles never reached it;
* journal mining wrote a generic `input.txt` and a lenient validator
  for every entry regardless of what the original user task was;
* PRM training only ran from the biological watchdog's 15-60 min idle
  window — but a busy self-play loop never reaches that window, so
  `PRMScorer.has_model` stayed `False` and the frontier-weighted
  picker silently fell back to the brittle pool;
* the LLM challenge generator had no incentive to produce *hard*
  challenges since it shared weights with the solver.

The redesign closes all eight gaps. Modules:

| File | Change |
|---|---|
| `core/solution_novelty.py` | **new** — AST-canonical hash + Jaccard novelty against prior winning solutions for a cluster. |
| `core/self_play_scoring.py` | multi-signal score: `passed*(1 + α·Δ + γ·novelty + δ·attempts_efficiency) − β·errors`. Defaults preserve back-compat. |
| `core/challenge_templates.py` | qualitative tier twists (`na_rows`, `negative_values`, `duplicate_ids`, `schema_drift`, …) — tier = K-combination of twists, not just N× rows. |
| `core/journal_challenges.py` | shape-aware fixtures (`input.csv` / `input.json` / `input.log` / `input.db` / `input.txt`) + shape-specific validator rubrics. |
| `core/adversarial_generator.py` | **new** — per-prompt-fingerprint solver pass-rate tracker; `suggest_bias()` injects guidance into the next challenge-gen prompt. |
| `memory/frontier.py` | per-template saturation (proposal H); ring buffer of recent winning `solution.py` sources for novelty scoring; `record_run()` now consumes `solution_source`, `template_key`, `novelty`. |
| `reflection/loop.py` | opt-in `accept_low_novelty_passes` admits self-play passes with `extra.solution_novelty < threshold` into the reflection batch. |
| `tools/memory.py` | self-play loop calls `_maybe_retrain_prm()` every 20 cycles → PRM model stays fresh without waiting for idle. |
| `core/dream.py` | wires all of the above: reads winning `solution.py`, computes novelty, passes it to scorer + tracker + reflector; opens write gate on `novel-shape first-try pass`; appends adversarial bias to generator prompt. |

### The new score

```
if passed:
    base = 1.0 + α·compression_delta + γ·novelty + δ·attempts_efficiency
else:
    base = 0.0
score = base − β·tool_errors
```

Defaults: α=0.4, β=0.1, γ=0.6, δ=0.3. `attempts_efficiency` is
`{1→1.0, 2→0.5, 3→0.2}`. `novelty ∈ [0, 1]` is the Jaccard distance
between the new solution's canonical AST shape bigrams and the
cluster's stored prior winning shapes (cluster cold start → 1.0;
exact AST duplicate → 0.0).

Concrete swing observed live on a fresh cold-start regex_parse cycle:
the old score reported `+1.000`; the new one reported `+1.900`
(1.0 base + 0.6 novelty + 0.3 first-try) — a real gradient that
discriminates among passes that the old score collapsed onto a single
binary outcome.

### New write-gate path

Added between `struggled-then-won` and `new-failure`:

> `passed and attempt == 0 and novelty ≥ 0.5` →
> *first-try pass with novel shape → write lesson*

A boring first-try pass with `novelty < 0.5` no longer suppresses
silently; it's reported as *"defer to reflector"* and the reflector
(when constructed with `accept_low_novelty_passes=True`) admits it
into the batch and asks *why was this boring?* — the meta-lesson
either grows the curriculum or stays the same.

### Tier twists (qualitative difficulty)

Each cluster declares an axis set in `_TWIST_AXES`. The tier-to-twist
map is:

| tier | twists picked |
|---|---|
| basic | 0 |
| intermediate | 1 |
| advanced | 2 |
| expert | 3 |

Twists are sampled deterministically from the seed so the setup and
the validator agree. `data_analysis` declares
`{na_rows, negative_values, duplicate_ids, schema_drift}` — a solver
that aced the basic shape gets four qualitatively different harder
versions to learn from, not just a 4× larger one.

### Per-template saturation (proposal H)

In addition to cluster-level saturation, each template within a
cluster gets its own outcome history. A template that earns two
consecutive first-try wins with `novelty ≤ 0.05` is marked saturated
(`saturated_at` timestamp). `list_saturated_templates()` returns
`(cluster, template)` pairs so the dreamer can rotate to a different
template within the same cluster instead of rotating the whole
cluster out.

### PRM scheduler inside the loop (proposal E)

`tools/memory._run_self_play_loop` now calls `_maybe_retrain_prm`
every 20 cycles. Trainer bails out cleanly (with a logged reason)
when there aren't enough trajectories yet; on success it hot-swaps
the new `StepValueModel` into the live `PRMScorer`. The frontier-
weighted picker (`pick_frontier_seed`) can then engage on the next
cycle instead of falling back to the brittle pool every time.

### Adversarial generator (proposal G)

`AdversarialGeneratorTracker` is keyed by a hash of the variable part
of the challenge-gen prompt (the frontier hint). It records solver
pass/fail per fingerprint, exposes `worst_fingerprints(limit)`, and
synthesises a short `suggest_bias()` block that the dreamer appends
to the system prompt for the next LLM challenge generation. Result:
the generator gets a quiet incentive to produce more challenges in
families the solver is failing on rather than rotating to easier
ones.

### Tests

New: `tests/test_self_play_meaningful.py` (44 cases) covers the
score combiner, AST novelty, twist resolver, journal shape detector,
reflector opt-in admission, adversarial tracker, write-gate inputs,
per-template saturation, and PRM scheduler safety.

Pre-existing self-play tests were updated where they pinned the old
contract:

* `tests/test_self_play_structured_lessons.py` — journal mining now
  asserts shape-appropriate fixture names instead of `input.txt`.
* `tests/test_tier_aware_templates.py` — `data_analysis` reference
  solution updated to handle the new twists; setup-script assertion
  loosened from literal `random.random() < 0.0` to
  `na_fraction = 0.0`.

Full suite: **3670 passed, 11 skipped, 0 failed.** No regressions.

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

The PRM lands as a Stage-1.5 capability: it doesn't fine-tune weights
(stays inside the no-GPU constraint) but it does close a measurable
loop — every validator-passing or user-correction-promoted trajectory
becomes a labelled training example, and the model retrains every 3
hours of idle time. Watch `pretty_log("PRM Retrain", …)` lines in the
agent log for visible improvement over time.
