# Ghost Agent — FUNCTIONAL Bug Hunt (live, port 8000)

A resumable, system-wide **functional** bug hunt. Unlike `BUGHUNT.md` (static
source review, unit by unit), this one drives the **live agent** with real
`/api/chat` requests and hunts for bugs & inefficiencies **in its runtime
behaviour and the log stream**. Fix → test → document → mark CLEAR → next
subsystem. Do NOT reuse `BUGHUNT.md` for state; update it only if a finding
also needs a static-side note.

Started 2026-07-04.

---

## How to resume (instructions to Claude)

1. Read the **Subsystems** table. Take the **first non-CLEAR** unit (top-down).
2. Confirm the agent is up and the harness works (see **Harness** below).
3. Drive that subsystem with a spread of real requests. Watch the pretty-stream
   log for: errors / warnings / tracebacks, execution strikes, tool failures,
   retries & loops, mis-routing, wrong/empty answers, latency cliffs, wasted
   LLM calls, cache misses, guard mis-fires, and behaviour that contradicts the
   request. Cross-check the `.err` log and `$GHOST_HOME/system/*.log` when a
   line is cryptic.
4. For each candidate: **reproduce**, then confirm it's a REAL bug against the
   source (concrete failure scenario) before fixing. Log speculative ones under
   **Deferred findings**.
5. Fix confirmed bugs in `src/`. Add regression tests under `tests/`
   (unit-level; functional repro described in the test docstring). Update the
   HTML docs under `docs/` (see `docs/audit_fixes.html`).
6. **Restart** the agent (picks up code changes — the live process is the OLD
   code until restarted), re-drive the same requests to confirm the fix live.
7. Run the full suite green:
   `GHOST_API_KEY=test-key PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python -m pytest tests/ -q`
8. Mark the unit CLEAR with a Session-log entry (findings, fixes, test files).
   Move to the next.

**Ground rules:** never wipe memory / `$GHOST_HOME`. Prefer closing loops over
new modules. A "bug" needs a concrete failure scenario; an "inefficiency" needs
a measured before (log evidence: wasted turns, redundant calls, latency).

---

## Harness (live-agent test rig)

- **Agent**: `python -m src.ghost_agent.main --port 8000` under launchd
  (`/Library/LaunchDaemons/com.local.ghost-agent.plist`, **KeepAlive=true**).
  Upstream LLM = Eva `http://127.0.0.1:8088` (qwen-3.6-35b-a3). Auth OFF
  (`--api-key ""`). Flags as of 2026-07-05 (final): `--verbose --deep-reason
  --smart-memory 0.9 --max-context 240000 --enable-metacog --metacog-mem-high 98
  --metacog-mem-floor-mb 300 --mandatory-tor --autoadvance-idle`. Verifier ENABLED
  (the Jul-4 `--no-verifier` was unintended and is permanently removed); the absence
  of `--postmortem --postmortem-propose-patch` IS a deliberate operator cost/value
  decision (do not re-add). The exec line in `bin/start-ghost-agent.sh` is the only
  source of truth, not this list.
- **Send a request** (model name is validated — must be `qwen-3.6-35b-a3`):
  ```bash
  curl -s -m 180 -X POST http://127.0.0.1:8000/api/chat \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen-3.6-35b-a3","messages":[{"role":"user","content":"..."}],"stream":false}'
  ```
  Response: `choices[0].message.content` (also mirrored at `message.content`).
- **Watch the log** (launchd stdout, ANSI-colored, APPENDS across restarts):
  `/Users/vasilis/Data/AI/Logs/ghost-agent.log`  (errors: `…/ghost-agent.err`).
  Record `wc -l` before a request; `tail -n +N | sed $'s/\x1b\\[[0-9;]*m//g'`
  after. App-logger file (WARN/ERROR, non-pretty): `$GHOST_HOME/system/ghost-agent.log`.
- **Convenience probe** (this session): `scratchpad/fprobe.sh "prompt"`.
- **Restart** (after a code change): `ps auxw | grep python | grep 8000 | awk '{print $2}' | xargs kill`
  → launchd waits for llama-server then relaunches (~a few s + Tor probe).
  Confirm back up: `curl -s http://127.0.0.1:8000/health` → `{"status":"ok"}`.
- `GHOST_HOME=/Users/vasilis/Data/AI/Data/`, cwd `/Users/vasilis/Data/AI/Agent`.

---

## Subsystems (resumable — first non-CLEAR, top-down)

| # | Subsystem | Drives / observes | Status |
|---|-----------|-------------------|--------|
| 1 | Core chat turn | greeting, identity, short Q&A; persona, streaming, no-tool path | CLEAR |
| 2 | Introspection & selfhood | "tell me about yourself", recall, narrative; introspect/self_state | CLEAR |
| 3 | Memory & profile | "remember that…", "what do you know about me", knowledge_base, update_profile | CLEAR |
| 4 | Code execution / sandbox | "run python to…", calculate/count; execute tool + Docker | CLEAR |
| 5 | Filesystem / workspace | write/read/list files; file_system tool, workspace continuity | CLEAR |
| 6 | Web search & research | "search the web…", "research…"; search/deep_research over Tor | CLEAR |
| 7 | Tasks & scheduling | "schedule…/remind me…"; manage_tasks, cron | CLEAR |
| 8 | Skills | "create/list/compose a skill"; manage_skills, composed skills | CLEAR |
| 9 | Projects & autonomy | "start a project to…", advance; projects, autoadvance-idle | CLEAR |
| 10 | Metacognition & verification | low-confidence turns, verifier, critic nodes, arbiter, replan | CLEAR |
| 11 | Deep-reason / planning | hard multi-step problems; planner, MCTS, hypothesis | CLEAR |
| 12 | Vision & image-gen | "generate an image of…", "describe image"; image_generation (Jetson), vision_analysis | CLEAR |
| 13 | DBA / database | postgres connect + queries; database tool | CLEAR |
| 14 | API / interface layer | endpoints, SSE streaming, resume/cancel, upload/download, error paths | CLEAR |

Status values: PENDING · IN-PROGRESS · CLEAR.

---

## Deferred findings (not yet fixed — revisit)

- **[router] `router/model.py:178` matmul non-finite — RESOLVED 2026-07-05.** Diagnosis
  was wrong: inputs at that line are provably finite (design matrix nan_to_num'd at fit
  entry; per-epoch divergence guard raises before line 178 is reachable), so routing was
  NOT garbage. The warnings are spurious FPE flags from Apple's Accelerate BLAS (this
  venv's numpy backend) on matmul with finite data — and that one matmul sat OUTSIDE the
  fit loop's `errstate` suppression (reconfirmed live at the 17:17 boot). Wrapped it;
  also moved feature sanitisation to the `_vectorize` choke point so predict_proba /
  partial_fit / bce_loss no longer take raw vectors (one NaN feature = blanket 0.5
  predict / silently-dropped online step / NaN holdout loss). tests/test_router_nan_guard.py (+4).
- **[coding] huge-reasoning, no file spec** (unit 4, REVIEWED — no fix) — `coding_executor: no
  file spec parsed (content=0 reasoning=67454)`. Reviewed: coding_executor.py already salvages by
  re-parsing reasoning+content combined (`extract_json_from_text(f"{reasoning}\n{content}")`),
  logs a diagnostic, and returns `(spec, was_empty)` for graceful handling. This case failed
  because the model emitted only PROSE reasoning (no spec anywhere) — genuine model-behavior
  edge, not a code defect. The other occurrences are `content=0 reasoning=0` EMPTY upstream
  (llama contention). Left as-is; salvage already present.
- **[infra] smart-memory upstream 503 not retried** (unit 3/infra) — `Smart memory task
  failed: Server error '503 Service Unavailable' for .../v1/chat/completions`. Single
  occurrence (llama-server busy). Check whether the smart-memory background task retries /
  degrades vs silently drops the injection. Low priority unless recurring.
- **[affordance] `workspace` tool has no `search` action** (unit 5) — the model repeatedly
  guesses `workspace{action:"search"}` (a natural expectation) → strike → recovers. Args are
  clean JSON (not corruption). Consider a `search`/`recall` alias or clearer tool description.
  Low; self-heals. Take under unit 5 (workspace).
- **[behavior] "project X" recall variance** (unit 2/9) — asking "when does project
  Kestrel ship?" made the model check the *projects* tool and give up, while "search your
  memory for Kestrel" recalled it fine (match 0.76). Storage is correct; the model just
  doesn't always route a "project"-worded query to recall. Model-dependent, self-corrects
  with phrasing. Low. Could nudge via tool description or a recall-fallback on empty project
  lookups.
- **[search] Yandex backend fails over Tor** (unit 6) — `search error … yandex.com/search/site …
  error sending request` (2×/query, ~20s each). Known per-exit-node reachability (see the
  `tor-search-reachability` memory). Circuit rotation per attempt IS already implemented
  (search.py:387); Yandex is just hard to reach over Tor. Search still succeeds via other
  backends + cites a source. Memory says MEASURE across exit nodes before changing the backend
  set — do NOT pin/remove blindly. Left as-is. Possible cheap win: shorter per-backend Tor
  timeout for known-flaky backends (needs measurement).
- **[response] long skill-list truncated** (unit 8) — "list all skills" made the model render a
  24-row built-in-tools table and run out of output budget before the acquired/composed skills;
  verifier LATE-REFUTED ("Output truncated"). Skills themselves work (list + acquired execution
  fine). Check whether truncated-answer auto-continuation fires here, or nudge the model to
  summarize rather than tabulate everything. Low; only very long responses.
- **[infra] torch leaked-semaphore at shutdown** (`.err`) — `resource_tracker: There appear
  to be 1 leaked semaphore objects to clean up at shutdown` on every restart (the local
  embedder). Cosmetic; check the embedder subprocess/pool teardown. Low.

---

## Session log

(newest first)

### 2026-07-05c — skills_auto graduation producer wiring (BUGHUNT.md unit-25 deferred, closed)
The graduation pipeline was structurally unreachable in production: corpus had 2058
UNKNOWN chat turns (1116 with ≥2 tool calls — the latent pool) and ZERO extractor-eligible
PASSED-with-tools trajectories, because async-critic mode records the trajectory before
the verdict lands and nothing backfilled the corpus. Fixes:
1. **[core] late verdict → corpus backfill** — `_record_late_verdict` now routes
   high-conf verdicts through new `_backfill_trajectory_outcome`: CONFIRMED ≥0.7 →
   UNKNOWN→PASSED (cache-verified UNKNOWN only, never upgrades FAILED), REFUTED ≥0.7 →
   FAILED (Reflector/PRM negative). Sidecar overlay, last-write-wins → user corrections
   still override. tests/test_critic_async.py (+5), test_graduation_producer_wiring.py (3).
2. **[skills_auto] latent (a)(b)(c)** — consolidator single-member cluster-specific
   signature (dual store keys), support over-count on same-batch samples, signature-hash
   delimiter injection. All fixed + regression-pinned (+6). Store empty in prod → hash
   change migrates nothing.
3. **[config, ROOT CAUSE of "no late verdict ever"] `--no-verifier` in the production
   launcher** — live verification kept failing (verdict task invisible: no socket, no
   log, absent from asyncio.all_tasks); a new SIGUSR2 task-dump handler (main.py;
   `kill -USR2 <pid>` → every live task + await stack in the log) proved the task was
   never spawned. `bin/start-ghost-agent.sh` (edited Jul 4 19:10) passed
   `--no-verifier`, so the whole verdict subsystem (verdicts, WEB-EXEC, verdict-driven
   lesson retraction, correction banners, corpus labels) was OFF — while the log
   printed "verdict deferred — verifying asynchronously" every turn, hiding the
   ablated state even from the operator's stream (misleading-message FIXED: an ablated
   verifier now logs "verifier is ABLATED (--no-verifier); nothing will land late" —
   live-verified). FINAL RESOLUTION after an operator round-trip (the flag was
   unintended; a miscommunication briefly re-added it): `--no-verifier` permanently
   REMOVED — verifier runs in production; the postmortem flags dropped in the same
   Jul-4 edit stay off (that half WAS deliberate). Backfill chain re-verified after
   the final restart: trajectory 0961d276 → passed, LATE CONFIRMED (100%),
   verifier_late sidecar line.
4. **[observability] async-verdict done-callback swallowed exceptions bare** — a dying
   verdict task vanished without a trace; now logs "async verdict task died: …" +
   "LATE verdict was empty" for the None-result path.
Docs: docs/algorithms/skill_acquisition.html ("Producer wiring" section),
docs/self_improvement.md, docs/audit_fixes.html (Round 10). Full suite: 6366 passed /
11 skipped. LIVE-VERIFIED end-to-end after the config fix: sum7 request →
"late verdict backfilled into the corpus: trajectory f91d7efa → passed" +
"LATE CONFIRMED (100%)" in the log, verifier_late line in corrections.jsonl, and
iter_trajectories yields outcome=passed with seq ['file_system','execute'] —
extractor-eligible. NOTE the Harness section flags above are stale vs the launcher
(no --postmortem since Jul 4; --metacog-mem-high 98 --metacog-mem-floor-mb 300 added).

### 2026-07-05b — deferred-findings closure: router matmul noise, WEB-EXEC fail-open, correction-classifier FP (3 fixes)
1. **[router] matmul warning spam** — RESOLVED (see Deferred findings above: Accelerate
   FPE noise on a provably-finite matmul left outside `errstate`, + `_vectorize`
   choke-point sanitisation for the predict/online paths). tests/test_router_nan_guard.py (+4).
2. **[verifier] WEB-EXEC fail-open** (req-70 residue, observed live 2026-06-20:
   "WEB-EXEC check skipped" → CONFIRMED 100% on vision alone) — an inconclusive probe
   (skipped or crashed) on a turn that WROTE web artifacts now caps a CONFIRMED at 0.6
   (below every ≥0.7 consumption gate); REFUTED never weakened; sync in-loop gate now
   fires the "actually RUN it" repair on a sub-0.7 CONFIRMED over an untested write
   (mirrors the async pure-predicate path). tests/test_verifier_web_exec.py (+4),
   tests/test_verifier_auto_repair.py (+1 end-to-end cap→repair).
3. **[distill] correction-classifier affirmation FP** (BUGHUNT.md unit-22 deferred) —
   praise that opens with "actually" and echoes the request no longer promotes the
   prior turn to FAILED / retracts its lesson: affirmation veto with negative-marker
   blocking. tests/test_user_correction.py (+5).
Docs: docs/core/verifier.html, docs/self_improvement.md, docs/audit_fixes.html (Round 10, 3 rows).
Full suite: 6352 passed / 11 skipped. Agent restarted; router warning absence + health
verified live post-restart.

### 2026-07-05 — POST-SWEEP live regression: the chess post-mortem (5 fixes shipped)
User-reported total failure of "play chess against each other. with YOU, not a generated
chess AI … turn based from the terminal" (2026-07-04 21:59–23:22). Post-mortem verdict:
LLM wrote 5 distinct crash bugs + 1 comprehension inversion, but the AGENT had every
signal to stop the bleeding and used none. Fixes (all tested, docs updated, live-verified):
1. **[unit 10] Async-critic mode shipped untested writes** — the in-loop VERIFIER-GATE
   AUTO-REPAIR was gated behind `not _critic_async_enabled()`; production runs
   GHOST_CRITIC_ASYNC=1, so six consecutive turns finalised on never-run
   `terminal_chess.py` writes ("flagged INCOMPLETE" ×6, shipped anyway). Fix: the
   unverified-mutation check is a pure predicate (no LLM) — async mode now runs it inline
   and forces the bounded "actually RUN it" re-entry; verdict itself stays deferred.
   LIVE-VERIFIED: write-only probe → `verifier gate UNVERIFIED → auto-repair round 1/1`
   → model executed the file → "Confirmed working" grounded in real output.
   `tests/test_async_unverified_repair.py` (5).
2. **[unit 9] Project constraints never reached the autoadvance coding executor** — the
   captured "with YOU - Ghost plays directly, not a generated chess AI" constraint was
   stored on the project record but `_generate_build_spec`'s prompt never saw it; the
   first engine violation was written by that exact path. Fix: `project_advancer` passes
   `metadata.constraints` → `build_coding_task` → spec prompt block.
3. **[unit 9] Participant-mode architecture steer** — new `has_participant_constraint()` +
   `PARTICIPANT_STEER` (`utils/constraints.py`): player-role constraints now append a
   binding directive (state-file + per-chat-turn reasoning, or thin client POSTing to
   `/api/game/move`) to the post-write steer AND the executor spec prompt. Prompt's
   PARTICIPANT MODE section now states a USER-run terminal client CAN reach
   127.0.0.1:8000 (only the sandbox can't — the model had over-generalised the loopback
   fact and dismissed the correct design). `/api/game/move` LIVE-VERIFIED: e2e4 → LLM
   replied e5 + comment, attempt 1. `tests/test_participant_constraint_steer.py` (15).
4. **[unit 3] Poisoned memories + no surgical delete** — the misread "i'm not playing
   against you…" complaint got consolidated as "user explicitly prefers a random AI move
   selection", then dream-fused with Unit-3 test probes (Marlin/teal/number-7) into a
   fabricated project memory. No safe delete existed (`delete_by_query` = semantic top-1).
   Fix: `VectorMemory.delete_fragment()` + `POST /api/memory/delete` (exact-id first,
   unique-substring fallback, refuses ambiguity, in-process). Deleted live: both Marlin
   fragments + the tabs-over-spaces probe residue. `tests/test_game_move_api.py` (+9).
5. **[unit 3+core] update_profile couldn't delete + idempotency guard poisoned by failures**
   (caught LIVE during verification): `ProfileMemory.delete()` existed but was unreachable
   — empty-value call hard-errored; the corrected retry was blocked "args already applied"
   (hash recorded at DISPATCH, not on success); model finalised on a false "Done — removed".
   Fix: empty value now routes to delete (+ scrubs the derived vector fact); durable
   idempotency hash commits at result time on SUCCESS only, batch-local pending set keeps
   intra-response dedup. LIVE-VERIFIED: retry actually removed the key from disk.
   `tests/test_profile_delete_and_idempotency.py` (11).
Full suite: 6241 passed / 11 skipped. NOTE: `tests/test_thinking_loop_guards.py` has 2
env-sensitive tests that fail if `FORCE_COLOR` is set in the shell — run with
`env -u FORCE_COLOR` (pre-existing, not a regression).
Remaining from the post-mortem, NOT fixed (model-behavior, not agent): qwen-3.6 wrote the
5 chess bugs and misread the complaint; the repair gate now contains the blast radius.

### 2026-07-04 — unit 14 (API / interface layer) — CLEAR (no code change) — **SWEEP COMPLETE (14/14)**
- SSE streaming (stream:true) works: OpenAI-compatible `data:` chunks (delta.role/content/finish),
  keep-alive comment line. Malformed bodies → clean 400 (`"just a string"` → 400; missing messages
  → `InvalidRequestShape`), not crashes. /api/download path-traversal (`../../../etc/passwd`) → 404
  (secure — os.path.basename hardening holds); nonexistent → 404. /health ok.
- All 14 functional subsystems CLEAR. 5 real fixes shipped (units 1,3,5,7,11); 9 verified healthy.

### 2026-07-04 — unit 13 (DBA / database) — CLEAR (no code change)
- postgres_admin tool connects to the default DB (postgresql://ghost@127.0.0.1:5432/agent, live
  PG 18.4) and runs queries: `SELECT version()` → real result; a GROUP BY aggregate over pg_tables
  → correctly-formatted rows (information_schema 4, pg_catalog 64, public 4). Connect + query +
  result-set handling all healthy. SQL-domain metacog active (competence tracked).

### 2026-07-04 — unit 12 (Vision & image-gen) — CLEAR (no code change)
- image_generation → Jetson: hit 3× transient 500s + circuit breaker, model retried, second call
  SUCCEEDED — produced a REAL 534KB gen_6f491e53.png (downloadable http=200). Honest response
  ("temporary error, let me try again" + valid link — NOT hallucinated). Graceful error recovery.
- vision_analysis → Eva: accurately described the generated image (red carbon-fiber bike, "TREK"
  branding, off-white wall). End-to-end vision works. Both nodes reachable (Jetson + Eva up).

### 2026-07-04 — unit 11 (Deep-reason / planning) — CLEAR (real fix shipped)
- MCTS fires on classifier-"hard" tasks (app-log `MCTS[sim]: selected … over N alternatives`);
  simple puzzles (jug puzzle) correctly answered directly without MCTS. Gating works.
- **BUG FOUND + FIXED (inefficiency): MCTS value function returned flat 0.50 → no ranking.**
  Every MCTS selection logged score=0.50. `_simulate_parallel` asks the model for JSON
  {progress,cost,risk}, but on this REASONING model max_tokens=256 was fully consumed by
  `<think>` → empty content (confirmed vs Eva: even 1024 tokens → 3500+ reasoning chars, 0
  content). `_parse_json("")` → {} → all default 0.5 → flat 0.50 for every candidate. MCTS paid
  for N sims but ranked nothing. Fix: disable thinking on the sim call
  (`chat_template_kwargs={"enable_thinking":False}` + `/no_think`) — verified vs Eva it now emits
  JSON with varied scores. Test: test_mcts.py::test_simulate_disables_thinking_and_scores. Docs
  updated. Suite green (6187). Restarted (couldn't force MCTS to fire live — classifier-gated —
  but the fix is confirmed by direct Eva measurement + unit test).

### 2026-07-04 — unit 10 (Metacognition & verification) — CLEAR (no code change)
- Functional at every observable level: per-turn composite confidence (gates async verification
  at threshold 0.97); verifier LATE CONFIRMED (good answers) + LATE REFUTED (e.g. the truncated
  skill-list); critic routes to Eva and degrades gracefully (ReadTimeout→fallback) under load.
  Reasoning-trap probe (bat & ball) answered correctly ($0.05, not the $0.10 trap).
- Boot stats explain non-events: `arbiter=on gated_domains=shell,sql` (arbiter is DOMAIN-GATED by
  design — my "other"-domain turns correctly didn't trigger it); `replans_tried=0` every boot
  because replan is host-resource-pressure-driven (triggers.py:370) and the host never crossed the
  (deliberately high, mem=98) thresholds during a task — inert by design, not a bug.
- Outcome-consolidation (verifier-refute → FAILED trajectory) was already unit-tested + gated
  (probe:outcome_consolidation) in the static hunt. No new bug.

### 2026-07-04 — unit 9 (Projects & autonomy) — CLEAR (no code change)
- manage_projects list + create work and persist (projects.db registry + sandbox/projects/<id>/
  workspace dir created; digest updated). Create captured the "don't start yet" constraint and
  respected the human gate. Healthy.
- autoadvance-idle not directly exercised (fires only on 30-min idle cooldown; can't trigger
  on-demand via a request). Left the test project "Hunt Demo CLI" (7599058e59b9) on file — benign.

### 2026-07-04 — unit 8 (Skills) — CLEAR (no code change)
- `manage_skills(list)` + `manage_composed_skills(list)` both work; acquired skill EXECUTION
  works (generate_password ran in the sandbox → produced a 20-char password). Skill dispatch +
  sandbox execution healthy.
- Verifier LATE-REFUTED the skill-LIST response for truncation (model tabulated all 24 built-in
  tools and ran out of budget before acquired/composed). Response-length edge, not a skills bug;
  logged as deferred.

### 2026-07-04 — unit 7 (Tasks & scheduling) — CLEAR (real fix shipped)
- manage_tasks create / list / stop / stop_all all work + persist (scheduler-backed).
- **BUG FOUND + FIXED (native tool-call repair, 3rd variant):** live `manage_tasks` stop got
  `{"action": "stop</parameter>\n<parameter=task_identifier>\ntask_…"}` — the SAME call's second
  PARAMETER leaked into the first (close-then-sibling-`<parameter>`, not a new `<function>`), which
  my repair didn't catch → "Unknown action" strike, recovered only by model retry. Generalized
  `_repair_one_native_tool_call` from "close-then-new-call" to "close-then-continuation": leaked
  sibling `<parameter>`s now fold back into the same call's args. Tests: +2 in
  test_native_toolcall_repair.py (19 total). Docs updated. Suite green (6186). Restarted →
  verified create+stop_all clean, no strikes.
- Note (not a bug): the model scheduled a plain "to-do" as an hourly recurring job
  (`interval:3600`) — its own judgment; the tool requires an explicit cron_expression (no default).

### 2026-07-04 — unit 6 (Web search & research) — CLEAR (no code change)
- Drove a web search ("most recent F1 champion") over Tor: DDG backend returned results, model
  synthesized the answer (Lando Norris, 2025) with a source URL. 80s. Functionally healthy.
- Yandex backend failed over Tor (2×, ~20s each) — known per-exit-node reachability; circuit
  rotation per attempt is already implemented (search.py:387); search succeeds via other
  backends. Deferred (see tor-search-reachability memory: measure before changing the set).
- Did NOT trigger a full deep_research (heavy Eva fan-out; would saturate the single upstream).

### 2026-07-04 — unit 5 (Filesystem / workspace) — CLEAR (real fix shipped)
- file_system write/read/list all work (write hunt_notes.txt → read back → list workspace).
- **BUG FOUND + FIXED (native tool-call repair gap):** a live file_system read fired my native
  tool-call repair, but when the leaked value STARTS with framing (`m.start()==0`, no clean
  prefix) the old code left the tag-soup as the primary arg and dispatched it → garbage
  `file_system` call + wasted strike. Fixed `_repair_one_native_tool_call` to DROP the phantom
  primary and promote the first recovered call; `_repair_native_tool_calls` now flags repaired
  when the primary is rewritten (not just when extras exist). Tests: +1 in
  test_native_toolcall_repair.py (17 total). Docs updated. Suite green (6184). Restarted →
  re-verified: list+read clean, no strike.
- **IMPORTANT operational note:** rapid-fire probing SATURATES the single Eva upstream — every
  turn also fires an async verifier (critic node = Eva) + background tasks, so a burst backs up
  the queue (saw foreground requests take 189–315s, critic ReadTimeouts). A direct 5-token Eva
  call hit 20s under load but 0.2s once drained. NOT a bug — the single-server constraint. **Pace
  probes** (one at a time, let each fully finish) for the rest of the hunt.

### 2026-07-04 — unit 4 (Code execution / sandbox) — CLEAR (no code change)
- Drove: sum-of-squares (→338350, exit 0, 7s), `print(1/0)` (correctly diagnosed
  ZeroDivisionError/exit 1), write+run primes.py (correct, exit 0). All healthy — sandbox exec,
  exit-code handling, error surfacing, multi-turn write→execute all correct.
- Reviewed the `coding_executor: no file spec` warnings: salvage logic already present
  (re-parses reasoning+content); residual cases are empty-upstream (llama flakiness) and
  prose-only model output — not code defects. No fix needed.

### 2026-07-04 — unit 3 (Memory & profile) — CLEAR (real fix shipped)
- update_profile: healthy — "prefer tabs over spaces" wrote `preferences.python_indentation`
  (persisted to `user_profile.json`); profile recall accurate.
- **BUG FOUND + FIXED (HIGH): knowledge_base `insert_fact` hung the turn and dropped the fact.**
  "Add … Bluefin …" ran to the client timeout (>150s), and a follow-up couldn't find Bluefin.
  Root cause: `tool_remember`'s bus path `await`ed the graph-triplet extraction LLM call INLINE,
  BEFORE `publish_fact`, with `is_background=True` — which parks on `_wait_for_foreground_clear`
  until `foreground_requests==0`, i.e. it waited for THIS turn to finish → self-deadlock (600s
  ceiling). The turn hung AND (hang before publish) the fact was never stored. Fix: publish the
  fact immediately (empty triplets), extract triplets in a fire-and-forget background task
  (bounded `asyncio.wait_for` 20s) that adds them to the graph — off the critical path, where
  `is_background=True` is finally correct. Tests: `tests/test_insert_fact_hang.py` (4) +
  updated `test_memory_bus_integration.py`. Docs: audit_fixes.html. Suite green (6183). Restarted
  → verified live: "store … Kestrel …" returned in 7s (was >150s), recall quoted the fact back
  (match 0.76).
- Deferred (logged): workspace-no-`search`-action strike; "project X" recall-routing variance.

### 2026-07-04 — unit 2 (Introspection & selfhood) — CLEAR (no code change)
- Drove: "tell me about yourself" (introspect summary), stats ("1,472 experiences" + topic
  clusters), recall ("what do you remember about postgres" → introspect recall + memory
  recall), and a self_state WRITE ("note an open question…").
- All healthy: `introspect` summary/stats/recall render correctly; the original native
  tool-call `introspect` corruption is CONFIRMED FIXED live (action='summary' dispatched
  clean, no "must be one of" error). self_state authoring **persisted to disk** — verified
  `state.json` now holds `"text": "Should we refactor the complexity router?"` (write loop
  closed, will surface next session). `/api/chat` JSON is well-formed (newlines properly
  escaped — ruled out a suspected malformed-JSON bug).
- Latency note (not a bug): a cross-store recall took ~71s (introspect recall + memory
  recall + knowledge_base over 3 turns). Worth a perf look if it recurs; logged, not fixed.

### 2026-07-04 — unit 1 (Core chat turn) — CLEAR
- Harness established: live agent :8000, `qwen-3.6-35b-a3` upstream Eva, auth off, launchd
  KeepAlive restart. Ledger + memory pointer created.
- Drove greeting / factual / arithmetic / instruction-following / multi-turn context /
  capability probes. Conversational path is CLEAN: correct answers, trivial fast-path for
  greetings (1.6s bypass), sensible temp routing (0.20 factual / 1.00 open-ended), no
  wasteful tool calls, metacog not over-firing, multi-turn context threaded correctly
  (7×"teal"→28, honoured the profile's strict-output pref).
- **BUG FOUND + FIXED (turn-loop parsing):** recurring `system_parse_error` on coding tasks
  — the `<think>` stripper's lazy `.*?(?:</think>|(?=<tool_call|function)|$)` cut the block
  at a *quoted* `<tool_call>` mention inside the reasoning (model quoting "emit one
  `<tool_call>` per turn"), leaving `<tool_call>\` per turn…</think>\n<tool_call>…` as the
  parse target → malformed-call strike + wasted turn. Replaced all 4 sites with
  `_strip_think_blocks` (prefers the real `</think>`; only strips an *unclosed* think up to a
  REAL tool-call opening). Tests: `tests/test_think_strip_toolcall_mention.py` (12). Docs:
  audit_fixes.html. Full suite green (6179 passed, 11 skipped). Restarted → verified live: a
  write+execute coding task ran clean (file_system → execute → correct Fibonacci output),
  zero parse errors / strikes.
- Deferred (logged above): router matmul non-finite, coding huge-reasoning no-file-spec,
  smart-memory 503, torch semaphore leak.
