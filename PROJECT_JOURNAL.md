# Ghost Agent — Project Journal

The single source of truth for the agent's hardening history, architecture
decisions, operational conventions, and open work. **Supersedes and replaces**
`BUGHUNT.md`, `BUGHUNT_FUNCTIONAL.md`, `COGNITIVE_LAYER_REDESIGN.md`, and
`IMPROVEMENTS.md` (all folded in here 2026-07-07).

Structure:
- **1. Current state** — one-screen summary.
- **2. Operational reference** — how to run/restart/test the live agent (load-bearing).
- **3. Cognitive-layer redesign** — the toggles + re-enable criteria (cited by `core/agent.py`).
- **4. WHAT REMAINS TO DO** — the consolidated open work (start here for the next session).
- **5. Completed ledgers** — the 27-item improvement board + the two bug-hunt unit tables.
- **6. Session history** — chronological log of notable fixes.

---

## 1. Current state (2026-07-07)

- **Static bug hunt (source review): COMPLETE.** All 28 units CLEAR (`utils` → `core/agent.py`
  → `scripts`), 2026-07-03/04. Every subsystem reviewed for concrete-failure bugs; confirmed
  bugs fixed with regression tests + HTML docs. Residual/uncertain findings live in §4.
- **Functional bug hunt (live agent on :8000): COMPLETE.** All 14 subsystems CLEAR, 2026-07-04.
  Real `/api/chat` requests drove each subsystem; ~10 real fixes shipped (turn-loop parse,
  insert_fact hang, native tool-call repair ×3, MCTS flat-score, etc.). Residuals in §4.
- **Cognitive-layer redesign: APPLIED + deployed** (2026-06-28). A paired ablation showed the
  in-session cognitive stack ≈ a stripped baseline at ~1.8× latency; advisory/ungrounded layers
  are default-OFF via module constants (§3). Cross-session memory DOES earn its keep (Track B).
- **6-agent improvement review (2026-07-07): 24 of 27 items DONE.** Full unit suite **6508
  passed / 11 skipped / 0 failed** (+110 tests this cycle). Partials/blocked (§4): #5-big,
  #6, #7-descriptions, #27b.
- **Live validations (2026-07-07):** KV pin confirmed holding in prod (byte-identical
  stable-prefix hash across a request's turns); B3 idle-loop ablation ran its first live pass —
  the pure-idle self-play loop is **proven productive** (treatment produced a lesson, control 0).

Deployment: single long-running asyncio process on macOS (Darwin), served on :8000, upstream
local Qwen3.6-35B-A3B "heretic" on llama-server :8088. RAM-tight (36GB box, llama ~22.6GB
wired). Repo is versioned on another server (local git intentionally absent).

---

## 2. Operational reference (live agent)

**Process / flags.** `python -m src.ghost_agent.main --port 8000` under a **root launchd job**
`/Library/LaunchAgents/com.local.ghost-agent.plist` (**KeepAlive=true**). Live flags (2026-07-07):
`--verbose --deep-reason --smart-memory 0.9 --max-context 240000 --api-key "" --mandatory-tor
--autoadvance-idle --enable-metacog --metacog-mem-high 98 --metacog-mem-floor-mb 300
--visual-nodes http://127.0.0.1:8088|Eva --image-gen-nodes http://100.122.46.101:8000|Jetson`.
Env: `GHOST_HOME=/Users/vasilis/Data/AI/Data/`, `GHOST_CRITIC_ASYNC=1`, `GHOST_CRITIC_NO_THINK=0`,
`GHOST_PIN_TOOL_SCHEMAS=1`. cwd `/Users/vasilis/Data/AI/Agent`. Verifier is ENABLED; postmortem is
deliberately OFF. The launcher exec line (out-of-repo `bin/start-ghost-agent.sh`) is the only
flag truth — this list can drift; check `GET /api/health` `config` for the resolved reality.

**⚠️ Supervisor gotcha.** A plain `kill` of prod is undone within ~9s by launchd (parent pid 1).
To actually stop prod for an isolated run, the operator must disable the launchd service
(`sudo launchctl bootout …` / unload the plist). Re-enable it afterward to restore auto-restart.

**Restart WITHOUT the launchd service** (must use the **venv python** — the bare homebrew python
lacks `uvicorn`; Tor on :9050 must be up for the `--mandatory-tor` boot gate):
```bash
cd /Users/vasilis/Data/AI/Agent
export GHOST_HOME=/Users/vasilis/Data/AI/Data/ GHOST_CRITIC_ASYNC=1 GHOST_PIN_TOOL_SCHEMAS=1
/Users/vasilis/Data/AI/.agent.venv/bin/python -m src.ghost_agent.main --port 8000 \
  --upstream-url http://127.0.0.1:8088 --visual-nodes 'http://127.0.0.1:8088|Eva' \
  --image-gen-nodes 'http://100.122.46.101:8000|Jetson' --verbose --deep-reason \
  --smart-memory 0.9 --max-context 240000 --api-key '' --mandatory-tor --autoadvance-idle \
  --enable-metacog --metacog-mem-high 98 --metacog-mem-floor-mb 300 \
  >> /Users/vasilis/Data/AI/Logs/ghost-agent.log 2>&1 &
```
A manually-started prod is **unsupervised**; kill it before re-enabling the launchd service to
avoid a :8000 bind conflict.

**Drive a request** (model name validated — must be `qwen-3.6-35b-a3`):
```bash
curl -s -m 180 -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"qwen-3.6-35b-a3","messages":[{"role":"user","content":"…"}],"stream":false}'
```
Reply at `choices[0].message.content`. **Introspect health:** `GET /api/health` (X-Ghost-Key)
returns rss/uptime/tasks/foreground counters/`biological_watchdog_alive`/`memory_system_loaded`/
scheduler jobs + the resolved config. Two silent-failure detectors: `memory_system_loaded=false`
= degraded boot (all biological phases dead); `biological_watchdog_alive=false` = daemon died.

**Logs.** Live pretty-stream (ANSI, appends across restarts):
`/Users/vasilis/Data/AI/Logs/ghost-agent.log` (errors: `…/ghost-agent.err`). App logger
(WARN/ERROR, non-pretty): `$GHOST_HOME/system/ghost-agent.log`. Boot dumps a "Resolved Config"
block + writes `$GHOST_HOME/system/last_config.json`. `kill -USR2 <pid>` dumps live asyncio
tasks + await stacks (hunts silently-parked coroutines).

**Test suite** (~6500 tests / ~3 min, must be green):
`GHOST_API_KEY=test-key PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python -m pytest tests/ -q`.
`FORCE_COLOR` unset for two env-sensitive thinking-loop tests. Never wipe memory / `$GHOST_HOME`.
`run_selfhood_functional.sh` is DESTRUCTIVE (`rm -rf` live selfhood) — never on live GHOST_HOME.

**RAM reality.** 36GB box; llama-server ~22.6GB (wired/mlocked → protected from the OOM killer).
With prod up only ~150–660MB physical free + ~1GB swap. A throwaway agent boots to ~630MB RSS in
~12s; ONE throwaway + prod fits, but sustained multi-agent idle work is risky — **stop prod first**
for ablations. Swap stays ~950MB free once a throwaway is up; abort a run if swap_free < 250MB.
`malloc_trim` is Linux-only (no-op on Darwin). RSS watchdog (#3) is opt-in via `GHOST_MAX_RSS_MB`
(default off).

**Conventions.** Prefer closing loops over new modules. A "bug" needs a concrete failure scenario;
an "inefficiency" needs measured before/after. Any change adds tests in `tests/` + updates HTML
docs in `docs/`. Flag/env changes need a manual relaunch. Logging: `pretty_log` + distinct icons;
the operator watches the live stream; `logger.warning`/`error` auto-render.

---

## 3. Cognitive-layer redesign (2026-06)

> Cited by `core/agent.py:73,104` and `docs/algorithms/metacognition.html`. The original doc was
> lost (no local VCS) and reconstructed 2026-07-07 from the surviving toggle comments + the memory
> ledger. Percentages are historical; the **decisions + re-enable criteria** are load-bearing.

**The finding.** A paired, time-matched ablation (`scripts/ablation_paired.py`) killed the
shared-upstream contention confound and showed the full in-session cognitive stack did **not** beat
a stripped baseline: trivial suite both 100% (the earlier "45%" was a contention artifact); hard
suite **full 78% vs thin 80%, McNemar p=1.0, full ~1.8× latency** (agent RSS grew ~270MB→~2GB over
2.3h, OOM at repeat 5). A 6-agent review diagnosed every layer as one of: **(a) advisory not
load-bearing** (MCTS "strong hint, not a mandate"; recalled skills are prose never executed;
selfhood narrative; confidence logged-only; RRF computed a ranking then discarded it via fixed
per-source budgets); **(b) ungrounded signal** (MCTS scored self-prediction of un-executed actions;
the dual-solver arbiter sampled 2 completions, threw both away, dispatched the original — dominant
latency, 0 answer changes; the grounded `hypothesis.py` loop was dead code); **(c) open loop**
(postmortem defects read only by an operator tool; router+PRM shipped untrained→escalate-all;
nothing graduated). **Worst single bug:** the temperature policy sampled AWAY from correctness —
graded factual Q&A classified "conversational" → temp 1.0 + presence_penalty 1.5.

**Applied (toggles are module constants in `core/agent.py`):**

| Change | Site | Status |
|--------|------|--------|
| Greedy sampling for graded turns | `_is_factual_query` + `FACTUAL_SAMPLING_PARAMS` | live |
| RRF emits by fused score under ONE global budget (12k→4k) | `bus.py _format_markdown` | live |
| Relevance-gate every tier (dropped `"user"` graph seed; vector distance-gate; episodic threshold) | `bus.py` fetchers | live |
| Metacog dual-solver arbiter OFF | `_METACOG_ARBITER_ENABLED = False` | **OFF** |
| Grounded hypothesis test→evaluate→survive loop wired | `_HYPOTHESIS_GROUNDING_ENABLED = True` | live (needs `--deep-reason`) |
| MCTS turn-start hint OFF | `_MCTS_TURNSTART_ENABLED = False` | **OFF** |
| Normalized graduation + discriminative credit + mints a `proposed` macro | `skills_auto` | live |
| Router trains/loads at startup (not escalate-all) | `router/trainer.bootstrap_router` | live |
| Selfhood wake-up prefix OFF; workspace prefix gated on active project | `_SELFHOOD_PREFIX_ENABLED = False` | **OFF** |

**Re-enable criteria (why each OFF toggle is parked, not deleted):**
- `_MCTS_TURNSTART_ENABLED` — only with an **execution-grounded** value fn (not self-prediction).
  Its intended grounded replacement (verifier-judged best-of-N that SUBSTITUTES the winner) landed
  2026-07-07 as the async-critic bounded repair (§5 #18).
- `_SELFHOOD_PREFIX_ENABLED` — the prefix injects no facts/tools/constraints (pure token cost). The
  load-bearing selfhood path is the cross-session memory substrate (Track B), which IS proven.
- `_METACOG_ARBITER_ENABLED` — net-negative as built; superseded by #18.

**Do NOT re-enable an OFF layer** without meeting its criterion or a fresh paired-ablation win.
The default-OFF state is the measured-neutral configuration, not an accident.

**Track B / B3.** Cross-session MEMORY tiers are PROVEN (Track B: 98% recall treatment vs 0%
control). The pure-idle loops (dream/self-play, reflection critique, skills-auto graduation, PRM)
were unadjudicated until B3 — see §4/§6: B3's first live pass (2026-07-07) proved the self-play
loop productive; the deeper "does idle output improve outcomes" question is still open.

---

## 4. WHAT REMAINS TO DO

Everything below is open. Start here. Grouped: (A) improvement-review partials/blocked,
(B) static-hunt deferred findings, (C) functional-hunt deferred findings.

### 4A. Improvement-review items not fully closed (from the 6-agent review)

- **#5 — agent.py guard-seam refactor (PARTIAL, high value).** The seam is established
  (`core/stream_guards.py` holds the pure streaming guards, re-exported). Remaining: the four big
  `handle_chat` extractions (XML tool-call parser `~7142-7900` → finalization chain `~9420-10361` →
  final-generation streamer closure `~6145-6632` → tool guard/dispatch/result pipeline `~8200-9200`)
  and a `TurnState` dataclass replacing the shared loop locals + a `StreamGuard/TurnHook` pipeline
  interface. Each needs live-driven hot-path validation (the unit suite alone can't catch a
  parse/dispatch regression). #26 (test builders) de-risks it. **This is the single biggest
  maintainability lever left** — agent.py is 11k+ lines and every hardening session appends to it.
- **#6 — flip `GHOST_PIN_TOOL_SCHEMAS` default (CODE-READY).** The blocker (#7c byte-stable tool
  set) is fixed and the pin is **validated holding in prod**. Remaining: watch llama-server `n_past`
  with the pin on for a session to quantify the prefill saving, then set it as the launcher default
  (it's currently set in the live env; make it durable). Operator session.
- **#7a/b — tool-schema diet (PARTIAL).** #7c (byte-stable advertised set) is done. Remaining: cut
  the mega tool descriptions ~⅓ (manage_projects 10.6KB — ~4KB is workflow doctrine that belongs in
  the system prompt / error messages; browser 5.7KB; file_system 4KB) and consolidate the ~12
  selfhood/meta tools behind one `self_report(area=…)` or semantic routing. Needs **live tool-use
  validation** (a description trim can regress tool selection; can't be verified headless).
- **#27b — PRM keep/delete verdict (PARTIAL, blocked on more B3).** B3 showed the self-play loop
  PRM feeds IS live-productive → "delete PRM (~2k lines)" is **not** triggered. Remaining: the
  PRM-specific sub-arm — self-play with frontier-weighted (`--frontier-selfplay`) vs uniform seeding,
  compared on verified-lesson yield — for a full keep/wire decision. Pre-register the criterion.
- **#4 — B3 deeper run (first pass done).** Remaining: more repeats + a **richer seed set** so the
  dream loop fires (it needs >3 seed auto-memories for entropy; the 10-task seed was too thin) and
  the **probe-outcome McNemar** (does idle-produced learning *improve* task success, not just exist?).
  Harness is ready (`scripts/ablation_trackb3.py`); the report-builder bug is fixed. Operator session.
- **#1 — version control:** SKIPPED per operator (repo versioned on another server). No action.

Also deferred within done items: #3 launchd supervisor plist in-repo (out-of-repo launcher);
#9 warm-runner browser tier (long-lived in-sandbox Playwright process); #27c persistent token→node
graph inverted index (the forgetting pass + node-cache landed).

### 4B. Static bug-hunt deferred findings (source review — still open unless marked RESOLVED)

Severity in parens. Many are latent (no prod caller), multi-process (single-tenant today), or
model-behavior edges.

- **`current_project_id` cross-conversation race (projects + api)** — **RESOLVED 2026-07-07.** #22
  (turn serialization) closes the chat-turn-vs-chat-turn window; the residual was the stateless
  `/api/upload` + `/api/download` endpoints, which carry no conversation context and read the racy
  global (a concurrent switch/reconcile could land an upload in another conversation's sandbox).
  Fix: `project_scoped_sandbox(..., explicit_project_id=)` + a `?project_id=<id>` query param on
  both endpoints → a client scopes race-free; the global stays the fallback when absent. Tests:
  test_upload_project_scope.py (6). Docs: file_system.html, api/routes.html. (Full per-conversation
  threading of every `record_*` call remains the deeper option, but the exploitable API surface is
  now closed.)
- **workspace `current_project_id` event-stamping race** (high/likely, but see #22) — events recorded
  mid-turn get stamped with whichever conversation last assembled a prompt. #22's serialization
  should close the concurrent window; confirm, else thread the active project id through each
  `record_*` call.
- **memory projects metadata split-lock + skills cross-process lock** (med/multi-process) —
  `projects.py` append_ledger/set_config_value do a read-modify-write across two lock acquisitions
  (lost update); `skills.py` writes a fixed `.tmp` with only an in-process RLock (torn tmp under
  multi-process). Mirror `frontier.py`'s fcntl advisory lock.
- **graph.execute_graph_compression resurrects expired facts** (med/latent, UNWIRED) — node-merge
  SELECTs without a `valid_until` filter and re-INSERTs without setting it, so superseded facts come
  back as current; also loses self-loops. Fix before wiring.
- **vector smart_update template over-match + correct_fragment id-collision** (low-med) —
  smart_update deletes the nearest neighbor at L2²<0.50 (can erase a distinct fact sharing the
  "User <key> is <value>" template); correct_fragment deletes old id then add()s new — if the new
  text hashes to an existing id, add() no-ops and the fragment is lost.
- **prm binary-floor gates continuous training + train↔serve feature skew** (med/low) — class-balance
  floor computed on BINARY labels while the model fits CONTINUOUS labels; several step features
  (steps_so_far, failures_so_far, tool_used/failed_this_turn) are always 0 at the live scoring site
  (turn start) but vary at train → weaker deployed discrimination than train_accuracy implies. Needs
  a training-signal redesign. (Relevant to #27b's PRM verdict.)
- **reflection selection oldest-first** (low) — `Reflector.run` is oldest-first within a tick (a
  recency window reaches fresh failures faster); `_truncate` head-keeps `tc.error`/`failure_reason`
  (drops the tail exception of a traceback); a diagnosis with "plan:" is truncated at that word.
  (The non-persistent-dedup half was RESOLVED 2026-07-04.)
- **distill outcome-heuristic false positives** (low-med) — the tool-error heuristic's bare
  substrings ("exception"/"traceback") match benign read content; `[ATTEMPT_ABORTED_*]` regex is
  searched in the user-facing final_response. Bounded today by the 3-repeat/two-signal gates.
  (Structured `ToolCall.error` on the chat path landed 2026-07-07 as #27d — improves this.)
- **agent.py correction-lookup fingerprint mismatch on prepended turns** (med) — calib-negative +
  trajectory stashes key on the pre-prepend response text; the next-turn lookup fingerprints the
  returned (banner/clarifying/digest-prepended) text → cache miss drops the "confidently wrong"
  signal on hedged turns. Fix touches the two-mode finalize+return hot path.
- **streamed-turn calibration gap** (low) — streamed final generations bypass the finalize tail →
  write no calibration JSONL pair, log a competence-only `below=`; verdict deferred to a next-turn
  banner. Async-verification design gap, not a finalize skip.
- **agent.py trajectory tool-result pairing on id-less duplicates** (low-med/contingent) — two
  same-named tool calls in one turn with empty ids collide on the `name` key in `pending_calls` →
  one result dropped (blank ToolCall on disk), Signal-3 error-repeat promotion undercounted.
  Contingent on local models streaming id-less tool calls. Needs an index-fallback key.
- **acquired_skills / composed / qwen_bridge** (low/med) — AcquiredSkillManager re-instantiated
  per call, each its own RLock (lost failure_count increment under concurrency); composed macro can't
  thread step N output into step N+1 ($var only resolves against initial params — feature gap);
  qwen_bridge `_run_coro_blocking` runs each native coroutine on a fresh loop (cross-loop error if a
  native tool caches a loop-bound client — agent_qwen.py variant only).
- **file_system replace bad-byte write-back corruption** — **RESOLVED 2026-07-07.** The replace
  read + guarded write + streaming path now use `errors="surrogateescape"`, so untouched non-UTF-8
  bytes round-trip to their exact originals instead of being persisted as U+FFFD; the
  syntax-regression guard fails open when `ast.parse` raises `UnicodeEncodeError` on a lone
  surrogate. Tests: test_replace_bad_byte_roundtrip.py (5). Docs: tools/file_system.html.
- **file_system + darkweb SSRF-on-redirect / body-cap** — **RESOLVED 2026-07-07.**
  `tool_download_file` disables auto-redirect and follows hops manually, re-validating each
  `Location` with the SSRF guard (`_download_redirect_target`, bounded at `_MAX_DOWNLOAD_REDIRECTS`).
  The onion fetch (`_fetch_raw_html`) now STREAMS the body (`iter_content`/`iter_bytes`) and stops at
  `_MAX_ONION_BODY_BYTES` instead of materializing `r.text` whole. Tests: test_download_redirect_ssrf.py
  (9). Docs: file_system.html, darkweb_search.html. Browser SSRF CORE was RESOLVED 2026-07-04
  (in-sandbox route interceptor); residual (still open): `file://` container-read outside the sandbox
  subtree when not project-scoped, and DNS-rebind of a subresource hostname in non-Tor mode.
- **execute nits** (low) — `_inline_py` `-c` body detector false-block on a chained command reusing
  the delimiter quote (annoyance, never wrong execution); `stateful=True, args=[…]` drops argv[2:];
  file-not-found retry re-runs side-effecting commands (substring heuristic); single global stateful
  kernel (documented tradeoff).
- **tool_failure loose FATAL patterns** (low) — `invalid.?(arg|param|schema)` / `tool.*not found`
  checked before DIAGNOSTIC, so `ValueError: invalid argument…` / `FileNotFoundError: 'tool.py' not
  found` get marked FATAL instead of self-correctable. Also: dead retry helpers
  (`get_retry_delay`/`should_retry`/`MAX_RETRIES`, off-by-one) — wire-or-delete.
- **sandbox/docker.py** (med/rare + high-on-Linux) — remove-while-exec race (in-flight command dies,
  self-heals; do NOT auto-retry — commands aren't idempotent); no client-side deadline on probes/exec
  (a wedged daemon hangs a worker); **Linux exec-user vs root-provisioned env** (Playwright/Chromium
  in /root mode 700, sudo refuses unknown uids — masked on macOS which execs as root; forces a full
  re-provision to fix — do when a Linux deployment matters).
- **router serve-only / scaling features** (low) — context_turn_coupling computed at serve but never
  trained (column all-zeros, inert); code_fence_count/coding_language raw not log1p; multi_step
  uses unanchored substrings. Needs a schema bump + retrain.
- **optim GEPA pipeline** (med, offline/opt-in) — not dspy-compatible: `tuner.compile` handed plain
  `TrainExample`s never converted to `dspy.Example` (real dspy run crashes, masked by the test mock);
  signature input-field name mismatch; the A/B `_ab_runner` feeds the instruction as a bare system
  prompt (not how the agent embeds it at inference → OOD gate). Needs a dspy env. (Adoption-ordering
  half RESOLVED 2026-07-04.)
- **scripts measurement robustness** (low) — `ablation_eval` globs a fixed report dir and folds
  stale/foreign-model JSONs into the verdict + treats (task,repeat) as independent in Wilson CIs
  (over-narrow); `selfhood_functional_test` Section C/G non-atomic RMW on the live store + Section D
  hardcodes :8088; `load_tokens` O(n²) filler; timestamp collisions. (**Note:** `ablation_trackb3.py`
  had a report-builder bug — FIXED 2026-07-07.)
- **interface external GPU servers** (med, home-lab threat model) — TTS/STT + image-gen HTTP servers
  bind 0.0.0.0 with NO auth on expensive endpoints (LAN resource exhaustion) + unbounded reads +
  blocking model calls on the async loop; needs a shared-key check coordinated with the clients. The
  clockwork desktop client parses SSE with `aiter_text` on non-line-aligned chunks (token misframing),
  ships a placeholder key, leaks camera/QTimer on close.
- **entrypoint / misc lifecycle nits** (low) — `--no-memory` leaks a `/tmp/ghost_no_memory_*` dir per
  boot; `_host_signal_to_bus` logs hardcoded 85/90 thresholds not the configured values; unbounded
  numeric CLI args; scratchpad connections not closed; episode float-epoch timestamp (degraded
  ranking); `--prm-online-update` holdout slice contains the just-trained sample (leak);
  `_reflected_trajectory_ids` unbounded (RESOLVED 2026-07-04 — now persisted+bounded);
  utils/telemetry `stop()` can swallow caller cancellation.

### 4C. Functional bug-hunt deferred findings (live behavior — still open)

- **[affordance] `workspace` tool has no `search` action** (low, self-heals) — the model repeatedly
  guesses `workspace{action:"search"}` → strike → recovers. Add a `search`/`recall` alias or clearer
  description.
- **[behavior] "project X" recall-routing variance** (low, model-dependent) — "when does project
  Kestrel ship?" made the model check the *projects* tool and give up, while "search your memory for
  Kestrel" recalled it (0.76). Storage correct; nudge via tool description or a recall-fallback on
  empty project lookups.
- **[search] Yandex fails over Tor** (low, known) — per-exit-node reachability; circuit rotation per
  attempt already implemented; search succeeds via other backends. MEASURE across exit nodes before
  changing the backend set (see `tor-search-reachability` memory). Possible cheap win: shorter
  per-backend Tor timeout for known-flaky backends (needs measurement).
- **[response] long skill-list truncated** (low) — "list all skills" tabulated 24 built-in tools and
  ran out of budget before acquired/composed; verifier LATE-REFUTED. Check whether truncated-answer
  auto-continuation fires here, or nudge to summarize not tabulate.
- **[infra] smart-memory upstream 503 not retried** (low) — single occurrence (llama busy). Check
  whether the background smart-memory task retries/degrades vs silently drops the injection.
- **[infra] torch leaked-semaphore at shutdown** (low, cosmetic) — `resource_tracker: 1 leaked
  semaphore` on every restart (the local embedder). Check the embedder subprocess/pool teardown.
- **[coding] huge-reasoning no-file-spec** (reviewed, no fix) — model emits only prose reasoning, no
  spec; salvage logic already present. Genuine model-behavior edge.

---

## 5. Completed ledgers

### 5A. 6-agent improvement review — 27-item board (2026-07-07)

DONE (24): #2 (600s self-stall), #3 (RSS watchdog+health), #8 (docker probe TTL), #9 (browser text
preview), #10 (one truncation policy + spill), #11 (line-ranged read), #12 (de-quadratic streaming),
#13 (episodic semantic recall), #14 (per-item RRF), #15 (vector tier scoring), #16 (retrieval-credit
inflation), #17 (bound autobiographical), #18 (async-critic bounded repair), #19 (reflection→bg
priority), #20 (`spawn_bg`), #21 (/api/health + config dump), #22 (serialize turns), #23 (workspace
save off-loop), #24 (dream heuristic keying), #25 (redesign doc + docs truth), #26 (test builders),
#27a (wire context_manager), #27c (graph forgetting + node-cache), #27d (ToolCall.error on chat path).
PARTIAL/CODE-READY/BLOCKED: #4, #5, #6, #7, #27b (see §4A). SKIPPED: #1 (git).
Full detail of each in §6 and in git history on the other server.

### 5B. Static bug hunt — 28 units, all CLEAR (2026-07-03/04)

utils · sandbox · workspace · tools-infra · tools-fs-exec · tools-web · tools-knowledge ·
tools-skills · tools-projects · memory · router · api · core-llm · core-planning · core-verify ·
core-projects · core-dream · core-agent(×2) · prm · reflection · selfhood · distill · eval · optim ·
skills_auto · entrypoint · interface · scripts. Each: every file reviewed for concrete-failure bugs;
confirmed bugs fixed with regression tests + HTML docs; full suite green on the session date.
Residuals in §4B. (Regression tests: `tests/test_bughunt_unit*.py`.)

### 5C. Functional bug hunt — 14 subsystems, all CLEAR (2026-07-04)

Core chat turn · Introspection & selfhood · Memory & profile · Code execution/sandbox ·
Filesystem/workspace · Web search & research · Tasks & scheduling · Skills · Projects & autonomy ·
Metacognition & verification · Deep-reason/planning · Vision & image-gen · DBA/database ·
API/interface. Real fixes shipped in units 1,3,5,7,11 (+ the post-sweep chess post-mortem and the
skills_auto graduation wiring). Residuals in §4C.

---

## 6. Session history (newest first)

### 2026-07-07 — Live validations after the improvement review
- **KV pin (#6) validated in prod.** Across the 2 turns of a real coding request, the
  `stable-prefix h=a449281b len=26116` log line was byte-identical → the pin holds and #7c
  (byte-stable tool set) works; no re-prefill between turns. `GHOST_PIN_TOOL_SCHEMAS=1` set by
  operator.
- **B3 first live pass (#4).** Operator stopped prod (disabled the launchd job) to free RAM;
  isolated treatment/control agents booted on :8046 against the shared llama :8088. Treatment
  (`--bio-time-scale 30 --bio-deterministic`) fired reflection + synthetic self-play within minutes
  of accelerated idle and produced **1 `self_play` lesson** ("alternative idiom for sql tasks",
  challenge SOLVED, score +1.900); control (scale 1) produced **0**. First live proof a pure-idle
  loop is productive. Dream didn't fire (too few seed memories for entropy); reflection had no failed
  turns. Harness completed both arms; a report-builder bug (reused trackb2's `_build_report` with
  wrong meta keys) crashed the final formatting — data recovered from the temp GHOST_HOMEs, and
  `_b3_report` written to fix it. Prod restored via the venv-python restart (the bare homebrew python
  lacks uvicorn).

### 2026-07-07 — 6-agent improvement review + implementation (24/27 done)
A 6-agent parallel review (core loop / tools / memory / learning loops / service layer / cross-cut)
produced 27 high-impact items. Implemented 24 headless with tests + docs; the rest need live sessions
(§4A). Full suite 6398→6508 (+110 tests). Highlights: the 600s inline-`is_background` self-stall
(#2/#19, same root cause as the fixed critic deadlock); RSS watchdog + `/api/health` + resolved-config
dump (#3/#21); the MemoryBus read-path rebuild (#13/#14/#15/#16 — wired episodic semantic recall,
per-item RRF, tier-scoring, deferred retrieval credit); de-quadratic streaming (#12); one truncation
policy + spill-to-file (#10); turn serialization (#22); `spawn_bg` unification (#20); context_manager
wiring (#27a); graph forgetting (#27c); reconstructed the lost redesign doc (#25); the guard-module
seam (#5, partial). Per-item detail in the Log table (preserved in git history).

### 2026-07-05 — Post-sweep live regressions (from the operator's real sessions)
- **skills_auto graduation producer wiring** (BUGHUNT unit-25 deferred, closed). The pipeline was
  structurally unreachable in prod: async-critic records the trajectory UNKNOWN before the verdict
  lands, and nothing backfilled the corpus (2058 UNKNOWN, 0 extractor-eligible). Fixed:
  `_record_late_verdict → _backfill_trajectory_outcome` (CONFIRMED≥0.7→PASSED guarded UNKNOWN-only;
  REFUTED≥0.7→FAILED + Reflector/PRM negative; corrections sidecar wins). ROOT CAUSE of "no late
  verdict ever": `--no-verifier` had been in the launcher (unintended) — a SIGUSR2 task-dump proved
  the verdict task was never spawned; permanently REMOVED (postmortem-off stays deliberate).
  LIVE-VERIFIED end-to-end.
- **Deferred closures:** router matmul warning noise (Apple Accelerate FPE on a provably-finite
  matmul outside `errstate` — wrapped + `_vectorize` choke-point sanitisation); WEB-EXEC fail-open
  (an inconclusive probe on a web-artifact write now caps CONFIRMED at 0.6 + fires the "actually RUN
  it" repair); correction-classifier affirmation false-positive (praise opening "actually" no longer
  promotes-to-FAILED — affirmation veto).
- **The chess post-mortem (5 fixes).** "play chess against each other, with YOU" failed totally: the
  LLM wrote 5 crash bugs + 1 comprehension inversion, and the agent had every signal to stop and used
  none. Fixed: async-critic mode shipped untested writes (the in-loop repair was gated behind
  `not async` — the pure unverified-mutation predicate now fires inline; this is the same gap #18
  later fully closed); project constraints never reached the autoadvance coding executor; participant-
  mode architecture steer + `/api/game/move`; poisoned memories + no surgical delete (added
  `VectorMemory.delete_fragment` + `POST /api/memory/delete`); `update_profile` couldn't delete +
  idempotency hash poisoned by failures (commit at success, not dispatch).

### 2026-07-04 — Functional bug hunt (14 units) + post-hunt strategy
- All 14 live subsystems driven with real requests → CLEAR. Real fixes: turn-loop `<think>`-strip cut
  at a quoted `<tool_call>` mention (unit 1); **insert_fact hung the turn + dropped the fact** — the
  bus path awaited graph-triplet extraction INLINE with `is_background=True` → self-deadlock (the
  original of the #2 class; fixed by publish-then-extract-in-background, unit 3); native tool-call
  corruption repair (units 5/7 — 3 variants: the upstream server leaks following calls' XML into the
  first arg under `--native-tools`); MCTS value fn returned flat 0.50 because the sim call's 256-token
  budget was consumed by `<think>` → disabled thinking on the sim call (unit 11).
- **Post-hunt strategy (make success measurable):** built an offline invariant GATE
  (`eval_baseline gate`), consolidated the ~5 disagreeing "was this turn good?" signals into one
  grounded `resolve_turn_outcome` (verifier verdict now folds into the trajectory corpus, not just
  calibration/selfhood), persisted the reflection dedup set (loop now progresses through the backlog),
  and built an execution-grounded **behavioral** eval (`eval/behavioral.py` — drives the live agent +
  inspects its sandbox/DB/trajectory; `mean_tool_calls=2.4` vs capability's 0.0). Also a
  sandbox-verdict runner + discriminating behavioral eval (closed a recurring gap).

### 2026-07-03/04 — Static bug hunt (28 units)
- Every subsystem reviewed one-lens-per-unit (often several parallel review agents), findings
  re-verified in source before fixing. ~28 unit sessions, hundreds of findings, ~150 confirmed fixes
  + regression tests + HTML docs. Notable HIGH fixes: browser SSRF-on-redirect (in-sandbox route
  interceptor covering redirects + subresources + `.last_url` re-nav + Tor-bypass); destructive
  file_system ops resolving to the sandbox ROOT (allow_root=False guard); manage_projects delete
  hard-deleting the ACTIVE project (auto-fill yields to explicit title); project_advancer missing the
  EXIT-CODE banner (failed build marked DONE); dream leaking synthetic trajectories to the prod log;
  the competence-detector EXIT-CODE substring gap (codes 3-9/127/130 scored SUCCESS). The `**d`
  dataclass-from-dict schema-drift silent-drop/wipe pattern recurred across selfhood/distill/eval and
  was hardened everywhere. Residuals → §4B. Regression tests: `tests/test_bughunt_unit*.py`.
