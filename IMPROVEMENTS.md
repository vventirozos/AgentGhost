# High-Impact Improvements — Resumable Ledger

Source: 6-agent parallel code review, 2026-07-07 (core loop / tools / memory / learning
loops / service layer / cross-cutting). Ran AFTER the system-wide bug hunt (all 28 units
CLEAR, see BUGHUNT.md) and the 2026-06-28 cognitive-layer redesign — so these are
architectural / performance / loop-closing improvements, not demonstrable bugs.
All file:line cites verified on 2026-07-07; code moves — re-verify before acting.

## How to resume (instructions to Claude)

1. Read this file. Finish any item marked `in-progress` first; otherwise pick the first
   `OPEN` item top-to-bottom, unless the user names one. **Check the Dependencies
   section** — some items gate or compound with others.
2. Set status to `in-progress`. Re-verify the finding's claims against current code
   before changing anything.
3. Implement. Per repo convention: add unit tests in `tests/`, update the matching HTML
   docs in `docs/`.
4. Run the full suite:
   `GHOST_API_KEY=test-key PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python -m pytest tests/ -q`
   (must be green).
5. Record in the Log at the bottom (item, date, what was done, anything deferred), set
   status to `DONE` with date. Flag/env changes require manually relaunching the live
   agent (GHOST_HOME exported; no supervisor — see run-and-test-setup conventions).

## Status board

| #   | Item                                                        | Area        | Impact   | Effort | Status |
|-----|-------------------------------------------------------------|-------------|----------|--------|--------|
| 1   | Put codebase under git + lockfile                           | cross-cut   | critical | S      | SKIPPED (repo already versioned on another server per operator 2026-07-07) |
| 2   | Fix 600s self-stall: inline `is_background=True` on request path | core   | critical | S      | DONE 2026-07-07 |
| 3   | RSS watchdog + controlled restart + supervision seam        | service     | critical | S      | DONE 2026-07-07 (watchdog+execv+health; launchd plist deferred) |
| 4   | Run B3 idle-loop ablation (`--bio-time-scale`, `--bio-deterministic`) | learning | critical | M | HARNESS-READY 2026-07-07 (flags + scripts/ablation_trackb3.py built; LIVE run needs operator session) |
| 5   | agent.py guard-pipeline seam + 4 handle_chat extractions    | core        | critical | L      | PARTIAL 2026-07-07 (guard-module seam established: core/stream_guards.py + first guards migrated; the 4 big handle_chat extractions + TurnState remain — need live-driven hot-path validation) |
| 6   | Flip `GHOST_PIN_TOOL_SCHEMAS=1` after A/B                   | core        | high     | S      | CODE-READY 2026-07-07 (blocker #7c fixed; live A/B + flip needs operator session) |
| 7   | Tool-schema diet + byte-stable advertised tool set          | tools       | high     | M      | PARTIAL 2026-07-07 (byte-stable skills DONE; description-diet + meta-cluster consolidation need live tool-use validation) |
| 8   | Docker readiness probe: collapse to 1 exec + success TTL    | tools       | high     | S      | DONE 2026-07-07 |
| 9   | Browser: `navigate`/`click` return page text (later: warm runner) | tools | high     | S/L    | DONE 2026-07-07 (tier a; warm-runner L deferred) |
| 10  | One truncation policy for execute output + spill-to-file    | tools       | high     | M      | DONE 2026-07-07 |
| 11  | Line-ranged `read` (start_line/end_line) in file_system     | tools       | high     | S      | DONE 2026-07-07 |
| 12  | De-quadratic streaming loops (probe gating, incremental scrub) | core     | high     | M      | DONE 2026-07-07 |
| 13  | Wire episodic semantic recall (`vector_memory=` kwarg)      | memory      | high     | S      | DONE 2026-07-07 |
| 14  | Per-item RRF for skill/episodic tiers (no monolithic blobs) | memory      | high     | S      | DONE 2026-07-07 |
| 15  | Vector scoring: shrink tier multiplier; handle identity/synthesis types | memory | high | M   | DONE 2026-07-07 |
| 16  | Fix 4x retrieval-credit inflation + per-turn playbook rewrites | memory   | high     | M      | DONE 2026-07-07 |
| 17  | Bound autobiographical.jsonl (compaction + tail reads)      | memory      | high     | M      | DONE 2026-07-07 |
| 18  | Async-critic: bounded verdict await at loop exit (closes deferred #4) | learning | high | M  | DONE 2026-07-07 |
| 19  | Reflection/postmortem LLM calls → background priority       | learning    | high     | S      | DONE 2026-07-07 |
| 20  | One `spawn_bg` helper for all fire-and-forget tasks         | cross-cut   | high     | S      | DONE 2026-07-07 |
| 21  | `/api/health` + boot-time resolved-config dump              | service     | high     | S      | DONE 2026-07-07 |
| 22  | Serialize turns (semaphore→1) or contextvar per-turn globals | service    | high     | S      | DONE 2026-07-07 |
| 23  | `/api/workspace/save` off the event loop + size ceiling     | service     | high     | S-M    | DONE 2026-07-07 |
| 24  | Fix dream heuristic single-slot collapse (task= keying)     | learning    | high     | S      | DONE 2026-07-07 |
| 25  | Reconstruct COGNITIVE_LAYER_REDESIGN.md + docs/README truth | cross-cut   | high     | S      | DONE 2026-07-07 |
| 26  | Canonical test builders (`make_context`/`make_agent`)       | cross-cut   | high     | M      | DONE 2026-07-07 (builders landed; opportunistic migration ongoing) |
| 27a | Wire or delete `core/context_manager.py` (dead L0-L4 compression) | core  | medium   | S/M    | DONE 2026-07-07 (wired) |
| 27b | PRM: adjudicate via B3 sub-arm, then delete or wire (~2k lines) | learning | medium  | S      | BLOCKED-ON-#4 (B3 harness ready; PRM verdict = a B3 sub-arm; delete/keep decision needs the live run) |
| 27c | Graph memory: forgetting pass + token→node inverted index   | memory      | medium   | M      | DONE 2026-07-07 (forgetting + node-cache; full inverted index deferred) |
| 27d | Close deferred #5 as landed; populate `ToolCall.error` on chat path | learning | medium | S    | DONE 2026-07-07 |

## Dependencies / ordering

- **#1 (git) first.** Everything else is safer with rollback. One session: `git init`,
  `.gitignore` (`__pycache__/`, `.pytest_cache/`, venvs, runtime-written files),
  initial commit, `pip freeze > requirements.lock` from the live venv. GHOST_HOME data
  lives outside the tree. Adopt commit-per-hardening-session.
- **#2 and #19 share a root cause** (llm.py priority routing has no "inline on the
  request path" concept) — fix as one design pass if possible.
- **#24 BEFORE #4**: B3 will falsely kill the dream loop if its lesson output still
  collapses into one playbook slot. **#27b is a sub-arm of #4** — pre-register the
  criterion, don't invest in PRM before the verdict.
- **#6 + #7 compound**: the KV pin only pays off if the advertised tool set is
  byte-stable across turns (the per-query acquired-skill RAG filter breaks it).
  Together they are likely the largest single latency win available.
- **#26 (test builders) de-risks #5** (handle_chat extraction) — the suite is currently
  welded to the monolith's internal shape.
- **#3 and #21 pair**: the supervisor wants a health probe target; the health endpoint
  wants the RSS number the watchdog computes.

## Items (detail)

### 1. Put the codebase under version control — critical, S
71,834 lines, 536 test files, daily hardening, no `.git` (verified). Cost already
realized: `COGNITIVE_LAYER_REDESIGN.md` — cited by `core/agent.py:73` and
`docs/algorithms/metacognition.html:50` — no longer exists anywhere on disk; a past
sweep wipe needed trajectory-JSONL replay to recover source. requirements.txt is all
floating `>=`, no lockfile → env not reconstructible.

### 2. Fix the 600s self-stall on the request path — critical, S
`_wait_for_foreground_clear` (`core/llm.py:891-914`) has no early-exit while
`request_active`; the API layer holds `foreground_requests > 0` for the life of
`handle_chat` (`api/routes.py:355/387/396`). Any `chat_completion(..., is_background=True)`
awaited INLINE on the request path parks against its own request for the full 600s.
Reachable in live config: Context Shield (`agent.py:8938`, fires on tool output >
~84KB), context condenser (`agent.py:2011`), Perfect-It (`agent.py:10918` via 9585).
Same class as the fixed critic-node deadlock — these sites were missed. Latent: with
worker nodes, `LLMClient.route()` (llm.py:349-354) puts the stall inside query
expansion (agent.py:5158) and RAG-fusion decompose (bus.py:252).
Fix: these calls ARE foreground work — `is_background=False`, or a `from_request`
param skipping the wait; defensively cap the wait ~30s even when request_active.

### 3. RSS watchdog + supervision seam — critical, S
No process-RSS self-monitoring anywhere (zero psutil.Process/memory_info hits in src);
known 270MB→2GB growth on a ~94%-RAM box. (a) RSS check at top of `_biological_tick`
(`agent.py:2185` — ABOVE the `memory_system is None` early-return at :2189), when
quiescent (`foreground_tasks==0 and foreground_requests==0`, :2201-2204) and RSS >
GHOST_MAX_RSS_MB (~1.5GB): one WARNING, then controlled restart via
`os.execv(sys.executable, [sys.executable, "-m", "ghost_agent.main", *sys.argv[1:]])`.
(b) Supervision: launchd plist in-repo (`KeepAlive=true`, `ThrottleInterval` generous,
distinct exit code for boot-refusal so a down Tor doesn't crash-loop) OR `--supervise`
respawn wrapper in main(). Note `main.py:1430` already assumes a launchd restart that
doesn't exist; `bin/start-ghost-agent.sh` is NOT in this repo — bring the launcher into
the tree. (c) Attribution: add `proc_rss_mb` to HostSnapshot (`utils/telemetry.py:368`
is host-percent only); on USR2 or threshold, log tracemalloc top-20 diff vs boot
baseline (behind a flag). Known transient-spike candidates: full trajectory-corpus
materialization `list(collector.iter_trajectories())[-50:]` (`agent.py:10435`, also
:2474 — give the collector a `tail(n)`); async-critic verdict tasks pinning full
`list(messages)` snapshots (`agent.py:9701-9709`, snapshot only what the verifier
reads); `malloc_trim` (`agent.py:1759`) is Linux-only = no-op on Darwin.

### 4. Run B3 — idle-loop adjudication is one flag away — critical, M
The paired harness exists (`scripts/ablation_trackb2.py`, scripts/ABLATION.md); only
hard-coded windows/dice block it: windows at `agent.py:2276` (600<idle<=3600) and
:3020-3022 (idle>3600 + `random()<0.2`), dice at :2292 (`random()<0.5`), cooldowns
:2166-2183. Precedent for per-phase overrides exists (:2424, :2597, :2665). Add
`--bio-time-scale N` (divide all window bounds + cooldowns; hoist inline literals to
the class-constant block first) and `--bio-deterministic` (skip dice). Then a ~150-line
`scripts/ablation_trackb3.py` cloned from trackb2: seed K tasks both arms, sleep
through M accelerated idle epochs, diff playbook lessons by `source` +
GraduatedSkillStore count + proposed macros, run the trackb probe suite. Control arm =
scale 1 (loops never fire in-window). DO #24 FIRST. Adjudicates: dream/self-play,
reflection critique, skills_auto graduation, postmortem, PRM (#27b).

### 5. agent.py guard seam + handle_chat extractions — critical, L (4 independent M steps)
`handle_chat` = `agent.py:4590-10369` (5,783 lines, 51% of file); per-turn state as
bare locals, incl. `locals().get(...)` introspection (:5697, :5772-5781, :5929, :9867);
48 module-level toggle constants; 169 `except Exception` in this file (60 bare-pass).
`core/strikes.py` already proved the extraction pattern. Moves: (i) `TurnState`
dataclass replacing shared locals (StrikeLedger becomes a field); (ii) guard pipeline
interface — `StreamGuard.inspect(chunk, buf, state) -> Verdict(CONTINUE|ABORT|INJECT)`
+ `TurnHook.after_tool(result, state)`; migrate first the near-pure guards: thinking-
loop probe (`_detect_thinking_loop` already module-level ~:1027), tool-call-collapse
probe, cross-turn repetition guard, truncation-continuation. Extraction order by
value/risk: (b) XML tool-call parser :7142-7900 (pure fn of content+tool names, highest
test value) → (d) finalization chain :9420-10361 → (a) final-generation streamer
closure :6145-6632 → (c) tool guard/dispatch/result pipeline :8200-9200 (needs a small
outcome struct instead of ~10 loose locals). Do #26 first or alongside.

### 6. Flip the KV pin — high, S (zero code)
`GHOST_PIN_TOOL_SCHEMAS` defaults "0" (`agent.py:6024-6032`); live env has no override.
With pin off, the stable injection (`_compose_injection`, :4100-4177 — tool scaffolding
~850 tok, persona, playbook, hydrated memory 4-6KB) rides the moving last user message
→ ~1.5-4k tokens re-prefilled per turn × 10-40 turns/request. The pin path (anchor to
first user message, turn-1 byte-stability handled :4162-4176) is built; stable-prefix
sha1 already logged (:6013-6022). A/B: confirm hash constant across a request in live
log; watch llama-server n_past with `GHOST_PIN_TOOL_SCHEMAS=1` for a session; flip
default / set in launcher env.

### 7. Tool-schema diet + byte-stable tool set — high, M
35 static tools = 53,460 chars ≈ 14k tokens per request (`registry.py:51-462`);
manage_projects 10,648 chars (`tools/projects.py:1997+` — ~4KB is workflow doctrine,
not schema), browser 5,705, file_system 3,494; 12+ of 35 slots are selfhood/meta tools
diluting a 35B's tool choice. (a) Cut mega-descriptions ~1/3: move policy prose to the
system prompt or into the error messages where incidents happen. (b) Consolidate the
meta cluster behind one `self_report(area=…)` tool or semantic routing. (c) Pin
acquired-skill selection per conversation, not per query (`registry.py:584-597`
byte-changes the system prompt each message → llama-server prompt-prefix KV cache
invalidated → full re-prefill). (c) is prerequisite for #6's full payoff.

### 8. Docker readiness probe tax — high, S
`execute()` → `ensure_running()` → unconditional `_is_container_ready()`
(`sandbox/docker.py:595-606, :146-166, :105-144`): container.reload() + host
touch/unlink + 2 exec_runs ≈ 100-400ms per command, under `self._lock` (serializes
parallel tool calls); execute.py's root-fallback retry (:408-419) pays twice. Fix:
collapse probe to ONE exec (`sh -c 'stat <syncfile> && echo OK'`; reload redundant) +
5-10s success-TTL, invalidated on any exec failure/126/127/OCI error (preserves
deleted-inode detection).

### 9. Browser: return text from navigate/click; warm runner later — high, S then L
One-shot subprocess per op: Playwright start + `launch_persistent_context` (1-3s) +
full Tor page load (5-20s) (`tools/browser.py:378-423`); `navigate` returns only
status/title (:452-470) so the universal next step `extract_text` re-launches and
re-navigates the SAME page (:479-484); profile lock (:52-63) makes it strictly serial.
Tier (a) S: include capped innerText excerpt (4-8KB) + title in `navigate` result and
post-click text in `click`; say so in the schema ("you usually do NOT need
extract_text"). Tier (b) L: long-lived runner process inside the sandbox holding the
persistent context (detached exec + JSON over FIFO/socket — same pattern as the
Jupyter kernel, `execute.py:586-641`).

### 10. Execute output truncation: one policy + spill-to-file — high, M
Live config `--max-context 240000` → agent trunc_limit 336k chars (`agent.py:8923-8947`),
so docker's static 256KB head+tail (`docker.py:640-656`) is the only real gate: one
noisy command can inject ~70k tokens that persist in history. execute.py's 512KB layer
(:830-850) is unreachable dead code; bash branch has no tool-level trim (:476-479);
file_search 40KB, browser 64KB — four divergent policies. Fix: execute returns
12-24KB head+tail by default; on truncation spill FULL output to a sandbox file and
return the path ("full output saved to .out/last_run.log — use file_system
search/read_chunked"); delete dead layer; one shared `truncate_head_tail(text, budget,
label)` in utils used by docker.py, execute.py, file_search, agent loop.

### 11. Line-ranged read — high, S
After a failed replace, recovery = re-read WHOLE file (read_byte_budget allows 336KB ≈
90k tokens at live config; `file_system.py:478-561`) or improvise sed via execute
(+probe +turn). `read_chunked` (:1910-2051) pages by BYTES — never aligns with the
line numbers that failure messages (:933-971) and `rg --line-number` output hand out.
Fix: `start_line`/`end_line` params on `operation='read'` (line-number-prefixed slice,
like the failure snippet), advertised in schema + pointed at from the replace-failure
message and the too-large-file error (:511).

### 12. De-quadratic the streaming loops — high, M
Per token chunk: two full-buffer `.lower()` (`agent.py:6736, 6749`); ungated
`_detect_tool_call_loop(full_content)` regex over whole buffer per content chunk
(:6771 — `TOOL_CALL_LOOP_PROBE_EVERY` :1024 defined but never used); once thinking >
32K chars, `_detect_thinking_loop(guard_buf)` runs per-token over 32-200K chars (:6806-
6812 — the `next_loop_probe` gating at :6819 doesn't cover this branch); full-content
`_stream_scrub_pattern.sub` per chunk on final-generation streams (:6292). Fix:
incremental lowercase tail (test last ~64 chars); gate :6771 and :6806 on the existing
probe cadence; cursor + small holdback window for the scrub. Zero behavior change;
removes seconds of CPU per long turn + major allocation churn (RSS pressure; see #3c).

### 13. Wire episodic semantic recall — high, S (one line)
Write path embeds every episode trigger explicitly for semantic recall
(`agent.py:10882-10891`); `EpisodicMemory._vector_search` has full relevance-gated
mapping (`memory/episodes.py:222-241, 266-329`). But `MemoryBus._fetch_episodic` calls
`search_similar(query, 5)` WITHOUT `vector_memory=` (`core/bus.py:360-366`) → always
substring fallback over 100 most recent (episodes.py:244-264). 117 episode embeddings
(45% of live collection) have a dead retrieval path. Pass the kwarg; audit
`search_recoveries` call sites (System 3 pivots) for the same omission.

### 14. Per-item RRF for skill/episodic tiers — high, S
`_fetch_skill` returns the whole rendered playbook as ONE item (`bus.py:342-358`),
`_fetch_episodic` one blob (:360-380): rank 1 constant regardless of relevance;
per-item floor and `_PER_SOURCE_CAP` (:439) are no-ops; under sub-query fan-out,
near-identical multi-KB blobs each eat the global budget (exact-text dedup :221-231
doesn't collapse them). Both sources have per-item lists internally
(skills.py:1069-1088; episodes list pre-format). Emit `[{"source":"skill","text":
render_lesson_for_prompt(l)} for l in chosen]` — dedup and RRF then work per lesson.

### 15. Vector scoring: tiers annihilate ranking; orphaned types — high, M
`combined_score = p_score*10 + dist + time_penalty` (`memory/vector.py:647`): tiers ±10
apart vs dist+penalty ~2.3 → cross-tier relevance can never win (name-memory at dist
1.4 beats document at 0.1). `type="identity"` (53 embeddings, writer
tools/memory.py:862) and `type="synthesis"` (25, dream consolidation, dream.py:1091)
have NO case in `search()` (:553-599) → generic else, lowest priority; also not in
`_PRUNABLE_TYPES` (:135) → grow forever. Bus consumes search() as formatted string and
re-splits (bus.py:322) → computed scores discarded; RRF fuses on category order, not
relevance. Fix: score type from metadata (identity ≈ name-memory tier, synthesis ≈
document_summary), shrink multiplier (e.g. p_score*0.3), add identity/synthesis to
prunable or cap them, add `search_scored()` returning (text, score) for the bus.

### 16. Retrieval-credit inflation + write amplification — high, M
`hydrate_context` fans out per sub-query (up to 4; `bus.py:186-196, 244`); each
`vector.search()` bumps retrieval_count/last_accessed for up to 12 memories INCLUDING
gated-out candidates (vector.py:267-290, 662-678 — spaced-repetition half-life :638-640
and prune ranking :404-419 get corrupted credit); each `get_playbook_context` calls
`record_retrieval` per lesson = full 100KB skills_playbook.json rewrite per lesson
(skills.py:818-849 via 709-724) — up to ~20 sync rewrites (~2MB) per turn. `retrievals`
inflates 4x while `helpful_retrievals` credits once per window (:940-946) → hit_rate
crushed → stale penalty (:318-319, r>=5 & hit<0.35 → score*0.5) flags genuinely useful
lessons → `prune_low_utility` can evict them. Fix: `record_retrievals=False` from bus
fetchers; single deduped bump (one Chroma update, one playbook rewrite) after
`_format_markdown` decides what enters the prompt.

### 17. Bound autobiographical.jsonl — high, M
No cap (877KB / 1,702 lines at ~6 weeks; contrast workspace/activity.py:32-33 compacts
at 2MB/2000 lines). Per turn: `recent(3)` full-file parse; `search_my_past` cache
invalidated by same turn's own `capture_turn` append → full re-index per turn
(`selfhood/autobiographical.py:491-527`); `note_referenced_experiences` → `recent(50)`
full parse; `update_outcome` (verdict backfill) reads all lines + rewrites entire file
(:529-581). 3-4 O(n) passes + one O(n) rewrite per turn, monotonic growth = quadratic
over lifetime. Fix: WorkspaceActivity compaction pattern (byte cap + keep-newest-N +
roll old entries into a summary record), tail-read `recent()` (seek from end),
`update_outcome` via line-offset index or migrate log to SQLite like episodes.py.

### 18. Async-critic bounded verdict await — closes deferred #4 — high, M
Sync path already implements #4's intent: verifier-judged repair with answer
substitution, bounded by `_MAX_VERIFIER_REPAIRS` (`agent.py:8022-8140`, substitution
:8131). Parallel best-of-N would be strictly worse on one llama slot (N× decode every
turn vs regenerate-on-refute). Gap: `GHOST_CRITIC_ASYNC=1` (production) skips the
loop-exit verdict entirely (:8050-8064; gate :4179-4188) → refuted answers ship with
only a next-turn note (:9735-9744), though the critic runs on a SECOND model costing
the main slot nothing. Fix: in the async branch at :8050, spawn critic verdict at
loop-exit, await with 20-30s budget (reuse `_critic_gate_timeout` machinery) when the
last tool is substantive; on timeout fall back to defer. Then mark ledger #4 CLOSED
("landed: sequential repair, critic-judged") — do NOT build parallel best-of-N.

### 19. Reflection/postmortem at background priority — high, S
`_critique_fn`/`_verify_plan_fn` (`main.py:717, :755`) and postmortem `_analyze_fn`/
`_patch_fn` (:891, :912) call the shared 35B with NO `is_background=True` (contrast
dream.py:1042, :1371 which do it right). Consequences: post-turn `reflect_one`
(agent.py:10733-10756) contends head-on with the user's next turn (main.py:779-781
documents the contention and raises timeout to 120s instead of fixing priority); the
calls increment foreground_tasks so `_wait_for_foreground_clear` (llm.py:907-914) and
self-play's user-live check (tools/memory.py:1291) misread idle reflection as an
active user. Fix: `is_background=True` for watchdog-path closures; route post-turn
reflect_one via `use_critic=True` (off-host model). Same root cause family as #2 —
consider one priority-routing design pass.

### 20. One `spawn_bg` helper — high, S
Four coexisting fire-and-forget conventions across 23 create_task sites:
`spawn_task` (utils/logging.py:21 — contextvars, but no strong ref/exception handler);
`_pending_background_tasks` strong-ref set (agent.py:2023-2033, 9616-9622 — the
correct pattern, documented at :2019); module-level `_GRAPH_EXTRACT_TASKS`
(tools/memory.py:145-147); bare unstored create_task VIOLATING the repo's own GC rule
at agent.py:10695 (lesson retraction) and :10724 (PRM online update). Failure = silent
(60 bare-pass handlers in agent.py alone). Fix: `spawn_bg(coro, *, name)` composing
contextvars + strong-ref registry drained at shutdown + done-callback logging
non-CancelledError via logger.warning (auto-renders in live stream). Migrate 23 sites,
delete the 3 ad-hoc variants, add a grep-based test forbidding bare `create_task(`.

### 21. /api/health + resolved-config dump — high, S
No health/status route exists (verified across routes.py/app.py/game_routes.py/
projects_routes.py). Blind spots: failed VectorMemory init logs once (main.py:380-381)
then `_biological_tick` returns forever (agent.py:2189) — ALL biological phases
silently dead while HTTP answers; RSS/tasks/queues invisible without ssh. Config truth
is split 5 ways: 57 argparse flags (main.py:107-209), 28 GHOST_* env vars (some read
at import in _env.py:53-69), 4 module toggles (agent.py:86-106), interface server's
own env (interface/server.py:33-53), out-of-repo launcher. Fix: `GET /api/health`
(register ABOVE the `/{path:path}` catch-all at routes.py:740; behind verify_api_key):
rss_mb, uptime, asyncio task count, foreground counters, biological_watchdog_alive,
memory_system_loaded, scheduler_jobs, last_turn latency, config digest. At boot: one
structured "Resolved Config" pretty_log block (vars(args) redacted + consumed GHOST_*
+ module toggles) also written to `$GHOST_HOME/system/last_config.json` for post-crash
forensics. NetMon (ghost:8080) can scrape it; #3's supervisor probes it.

### 22. Serialize turns or contextvar the per-turn globals — high, S
`agent_semaphore = asyncio.Semaphore(10)` (`agent.py:1742`) admits 10 concurrent turns,
but per-turn state is on the singleton context: `last_user_content` (written :4615,
:4663; read mid-turn by tools/projects.py:270,326 + tools/memory.py:1103) and
`current_project_id` (25 refs; drives project-scoped sandbox writes + routes.py:700,
731). APScheduler cron jobs (main.py:421-456) fire mid-user-turn → a scheduled
project switch lands user writes in the wrong project's sandbox. Fix (cheapest
correct): semaphore→1 (single llama slot anyway) + `_run_proactive_task` waits for
foreground clear; if real concurrency ever wanted, move both fields to contextvars.

### 23. Workspace save off the event loop — high, S-M
`/api/workspace/save` (`api/routes.py:464-491`) does sync os.walk + zipfile.write
inline in the coroutine (no to_thread) → blocks ALL requests/SSE for the duration;
`zip_buffer.getvalue()` (:489) holds the whole archive in RAM, no ceiling (load side
caps 500MB at :571; save side none). Fix: asyncio.to_thread → spool file
(SpooledTemporaryFile), byte ceiling mirroring load (413 on exceed), FileResponse
deleting spool on completion. Interface proxy (interface/server.py:507-541) unchanged.

### 24. Dream heuristic single-slot collapse — high, S (one line) — DO BEFORE #4
Every REM heuristic written with constant `task="[System] Dream Heuristic"`
(`core/dream.py:1103`); skills dedup keys on normalized trigger (skills.py:527-531)
and keeps a new solution only if LONGER (:586-604) → N heuristics × M dreams collapse
into ONE churning playbook slot; frequency inflates toward bogus graduation (freq>=5,
dream.py:1329); trigger matches no user query (BM25-keyed). The dream loop's primary
lesson product structurally cannot accumulate influence — would silently bias B3
toward "kill" for the wrong reason. Fix: `task=h[:80]` (or extracted trigger clause) +
`source="dream"` so existing dedup/cap/utility machinery treats heuristics as
first-class lessons and B3 can count them by source.

### 25. Reconstruct redesign rationale + docs/README truth — high, S
`COGNITIVE_LAYER_REDESIGN.md` is GONE (find + mdfind across /Users/vasilis/Data) yet
cited by agent.py:73,:104 and docs/algorithms/metacognition.html:50. docs/core/
arbiter.html + mcts.html predate the redesign and present toggled-off layers
(_METACOG_ARBITER_ENABLED=False :106 gate :8553; _MCTS_TURNSTART_ENABLED=False :86
gate :4976; _SELFHOOD_PREFIX_ENABLED=False :90 gate :4865) as live; README.md:11
headlines MCTS. ~610 lines of tests (test_arbiter.py, test_mcts*.py) pin dead-path
behavior unmarked. Fix: (a) reconstruct the redesign doc from the agent.py:73-106
comment blocks NOW while recoverable; (b) status banner in the 3 HTML docs naming the
gating constant, value, re-enable criterion; (c) fix README; (d) mark ablated-layer
test files (docstring or `ablated` pytest marker).

### 26. Canonical test builders — high, M
Measured duplication: 22 separate `def agent()` fixtures; 77 files constructing
GhostAgent directly + 17 via `__new__`; 233 SimpleNamespace fakes across 65 files; 17
copies of `class FakeBgTasks`; 60 session-named files (conftest.py is only 158 lines).
Constructor/context shape changes fan out across ~90+ files — actively deters #5.
Fix: `make_context(**overrides)` / `make_agent(llm=…, tools=…)` in conftest or
tests/helpers.py wrapping the existing patterns; new tests must use them; subject-named
files going forward; merge session files opportunistically when touched.

### 27a. Wire or delete core/context_manager.py — medium, S/M
Complete tested L0-L4 progressive-compression subsystem, ZERO production callers
(`context_manager.py:23-252`; only tests/test_context_manager.py references it). Live
path degrades in one cliff via `_prune_context` (agent.py:1897-2042), which pays an
LLM summarization through #2's stall path and evicts the KV prefix. Either wire
`compress_if_needed` in front of _prune_context at agent.py:5690 (L1-L3 are
deterministic/LLM-free; verify pairing rules vs tool-call/result pairs, cf.
process_rolling_window agent.py:1867-1887) keeping _prune_context as L4 emergency —
or delete module + test. Wiring is the loop-closing-aligned choice.

### 27b. PRM: adjudicate then delete or wire — medium, S (after #4)
MCTS (PRM's intended consumer) default-off → live value chain is: idle retrain phase
2.7 (agent.py:2565-2657, every 3h idle) → hot-swap → weight self-play cluster picks
(dream.py:1826-1858) — terminates inside the unadjudicated self-play loop. prm/
(1,615 lines) + frontier_selection.py + phase 2.7 + MCTS branch maintained for an
unmeasured effect. Fix: B3 sub-arm — self-play frontier-weighted vs brittle-pool
seeding (`--frontier-selfplay` already toggles, dream.py:1823), compare verified-
lesson yield; pre-register criterion; if self-play fails B3, delete the ~2k lines +
phase in one sweep.

### 27c. Graph memory forgetting + inverted index — medium, M (high at months horizon)
Only uncapped tier (3,297 triplets; vector 5000-cap, episodes 500, skills 50).
Temporal supersession covers only 10 functional predicates (graph.py:105-173);
everything else accumulates forever, weight-1 stale edges rank alongside reinforced
facts. Per turn: `_map_words_to_seeds` materializes all nodes, substring + difflib per
query word = O(words × nodes) (:342-371); 3-hop spreading activation O(degree³) around
hubs (:381-440); full in-RAM mirror (:61-75). Fix: dream-cycle prune (drop weight==1
edges older than N days — weight column already stored, unused for forgetting);
persistent token→node inverted index updated in _upsert_edge/_remove_edge; cap
per-node expansion top-K by weight.

### 27d. Close deferred #5 as landed; populate ToolCall.error — medium, S
The feared naive auto-FAILED labeling is already solved conservatively:
distill/outcome_heuristics.py:207-224 (>=3 identical tool errors → FAILED), :260-296
(verifier REFUTED + structural fold-in), wired at record time agent.py:11072-11102,
with thresholds + never-demote flood control. Residual: signal depends on regex-
sniffing result TEXT because the chat recorder leaves ToolCall.error empty (docstring
:101-106; only self-play/batch set it) — atypical error shapes (e.g. native-tools
corruption incidents) are missed. Fix: populate ToolCall.error where the chat turn
records results (feeding agent.py:11040); `_looks_like_tool_error` checks the flag
first, text sniff as fallback. Mark deferred #5 closed-as-landed in ledger/docs.

## Log

| Date | Item | Result |
|------|------|--------|
| 2026-07-07 | — | Ledger created from 6-agent review. All items OPEN. |
| 2026-07-07 | #2 | Inline request-path calls (Context Shield agent.py, `_prune_context` condenser, --perfect-it via new `foreground=` param) flipped to foreground; `chat_completion` skips the foreground wait for off-main-pool background calls (also fixes `route()`'s latent stall); 120s park now logs one `BG Queue Wait` line. Tests: test_foreground_yield.py (priority-routing section). Docs: docs/core/llm.html. |
| 2026-07-07 | #19 | `_critique_fn`/`_verify_plan_fn`/`_analyze_fn`/`_patch_fn` in main.py now `is_background=True` — reflection/postmortem yield to live users and stop inflating foreground_tasks. Same tests/docs as #2. |
| 2026-07-07 | #13/#14/#16 | MemoryBus read path rebuilt: `_fetch_episodic` passes vector store (semantic recall live); skill/episodic tiers emit one RRF item per lesson/episode (`SkillMemory.get_playbook_items`, `EpisodicMemory.format_episode`); retrieval credit deferred to post-fusion survivors via `_credit_surfaced` → `VectorMemory.bump_retrievals` + `SkillMemory.record_retrievals_bulk` (was ~4× inflation + ~20 playbook rewrites/turn). `VectorMemory.search` split into `_search_selection` + `search_items`(per-item, no side effects) + string renderer. Tests: test_bus_per_item_fusion.py (new, 22) + updated test_memory_bus*/episodic/async_memory/contextual/arch_perf/recent/audit. Docs: bus.html, vector.html, episodes.html, skill_acquisition.html. |
| 2026-07-07 | #15 | Vector `_TIER_WEIGHT` 10→0.3 (category prior no longer absolute — adjacent tiers can be crossed by a decisively closer match); `identity`/`synthesis` types scored from metadata at their curated tiers (were orphaned into the auto else-branch); `synthesis` added to `_PRUNABLE_TYPES`, `identity` deliberately kept exempt. Tests in test_bus_per_item_fusion.py. Docs: vector.html. |
| 2026-07-07 | #8 | `sandbox/docker.py`: `_probe_container_ready` collapsed reload+stat+echo → one `sh -c 'stat && echo OK'`; success-TTL (`_READY_TTL_S`=8s, `mark_ready`/`invalidate_ready`/`_ready_is_fresh`) skips the probe after a confirmed-good command; exit 126/127/128 + exec exceptions invalidate. Tests: test_sandbox_readiness_ttl.py (11). Docs: sandbox/docker.html. |
| 2026-07-07 | #11 | `file_system` `operation='read'` gains `start_line`/`end_line` (aliases start/from_line, end/to_line): streamed, line-number-prefixed slice, exempt from the whole-file size cap, ≤2000-line span. Too-large error + large-file replace-failure guidance now point at it. `_read_line_range` helper; registry schema updated. Tests: test_read_line_range.py (10). Docs: tools/file_system.html. |
| 2026-07-07 | #24 | `core/dream.py`: REM-cycle heuristics keyed on their own content (`task=h[:80]`, `trigger=`, `source="dream"`) instead of the constant `[System] Dream Heuristic` that collapsed all heuristics into one churning playbook slot. Unblocks honest B3 measurement of the dream loop. Tests: test_dream_consolidation_metrics.py (new class). Docs: core/dream.html. |
| 2026-07-07 | #3 | `core/agent.py`: `_rss_watchdog_check` (opt-in `GHOST_MAX_RSS_MB`, default off) runs at top of `biological_watchdog` before the tick guard; on over-limit + idle → clean sandbox stop + `os.execv` in-place restart (preserves argv/GHOST_HOME). `_current_rss_mb` helper. launchd plist deferred (out-of-repo launcher). Tests: test_health_and_config.py. Docs: core/agent.html. |
| 2026-07-07 | #21 | `GET /api/health` (routes.py, above the catch-all, API-key gated): rss/uptime/task-count/foreground counters/`biological_watchdog_alive`/`memory_system_loaded`/scheduler jobs/resolved config. `_build_resolved_config` (main.py) flattens 5 config sources (redacts key, captures module toggles) → boot log + `$GHOST_HOME/system/last_config.json`. Tests: test_health_and_config.py (9). Docs: core/agent.html. |
| 2026-07-07 | suite | Full suite green after first batch: 6429 passed / 11 skipped / 0 failed. Fixed a __init__-bypass robustness issue in docker TTL helpers (getattr defaults, matching `_get_lock` pattern) surfaced by test_bughunt_unit2_sandbox + test_sandbox_log_quiet. |
| 2026-07-07 | #20 | `utils/logging.py`: `spawn_bg(coro, *, name)` composes contextvars + strong-ref registry (`_BG_TASKS`) + failure-logging done-callback; `drain_background_tasks` wired into lifespan shutdown. Migrated the 2 GC-violating bare create_tasks (lesson retract, PRM update) + consolidated `_GRAPH_EXTRACT_TASKS` (memory.py). Grep-guard test forbids new bare create_task outside an audited allow-list. Tests: test_spawn_bg.py (6), test_insert_fact_hang.py updated. Docs: logging.html. |
| 2026-07-07 | #23 | `/api/workspace/save` builds the zip in a worker thread (`asyncio.to_thread`) to a temp spool file with a 500MB `_MAX_WORKSPACE_SAVE_BYTES` ceiling (413 on exceed), streamed via `FileResponse` + background cleanup — was inline sync walk/deflate freezing the event loop + unbounded in-RAM archive. Tests: test_workspace_save_offloaded.py (5) + updated test_open_audit_fixes. Docs: api/routes.html. |
| 2026-07-07 | #22 | Turns serialized: `agent_semaphore` 10→1 (per-turn `last_user_content`/`current_project_id` on singleton context → concurrent turns clobber project scope; cron-mid-turn hazard). `tools/tasks.should_defer_scheduled_task` makes scheduled jobs skip when a user is live. Tests: test_turn_serialization.py (5) + test_agent semaphore assert updated. Docs: core/agent.html. |
| 2026-07-07 | #12 | Streaming loop de-quadratic'd: stop-marker scan → bounded tail (`_tail_has_stop_marker`); tool-call probe gated on `next_tool_probe`/TOOL_CALL_LOOP_PROBE_EVERY (was 2 full-buffer regex/chunk); n-gram detector runs on `next_loop_probe` cadence at all sizes (was per-token past 32K); final-gen scrub skipped until a `<` appears (`_scrub_seen_lt`). Detection behavior unchanged. Tests: test_streaming_dequadratic.py (8) + existing loop-guard suite green. Docs: core/agent.html. |
| 2026-07-07 | #10 | One truncation policy: `utils/text_truncate.truncate_head_tail`. docker `execute(spill_large_output=)` — execute tool returns ~24KB view + writes FULL output to `.ghost_runs/run_N.log` (model reads via file_system); bash path also spilled; rg/find/browser keep legacy 256KB no-spill. Dead 512KB execute.py layer removed. Tests: test_execute_output_spill.py (7). Docs: tools/execute.html. |
| 2026-07-07 | #9 | Browser `op_navigate`/`op_click` include a capped ~8KB `text` preview (`_body_excerpt`, `nav_text_chars`), collapsing navigate→extract_text→read into one op; schema + docstring tell the model it usually doesn't need extract_text. Warm-runner L tier deferred. Tests: test_browser_navigate_text.py (6). Docs: tools/browser.html. |
| 2026-07-07 | #27d | `_record_turn_trajectory` populates `ToolCall.error` on the chat path (was self-play/batch only); `outcome_heuristics._tool_call_failed` checks the flag first, text sniff fallback. Closes deferred #5 as landed. Tests: test_trajectory_failure_heuristic.py (+4). Docs: self_improvement.md. |
| 2026-07-07 | #17 | `autobiographical.py`: compaction (`_maybe_compact_locked`, 2MB/2000-line cap + `[Consolidated]` summary record, clears search cache) mirroring workspace/activity; `recent()` tail-reads via deque instead of full-file parse. Bounds the quadratic per-turn growth. Tests: test_autobiographical_bounds.py (5). Docs: algorithms/selfhood.html. |
| 2026-07-07 | #18 | Async-critic (`GHOST_CRITIC_ASYNC=1`) now awaits the off-host verdict at loop-exit for a bounded budget (`_critic_repair_await_budget`, default 25s, `GHOST_CRITIC_REPAIR_BUDGET` override, 0 disables) and REPAIRS a REFUTED answer in-loop instead of shipping with only a next-turn note; verdict cached for the post-loop gate; timeout falls back to defer. Closes ledger #4 as landed (no parallel best-of-N). Tests: test_critic_async.py (repair + disabled-defer). Docs: core/verifier.html. |
| 2026-07-07 | #27a | Wired `core/context_manager.py` (was zero callers): `handle_chat` runs `compress_if_needed(max_level=3)` — deterministic L1-L3 content compression, all messages preserved — BEFORE the LLM `_prune_context` (now L4 emergency), so the summarization LLM call fires less. `_get_context_manager` lazy accessor + `max_level` cap. Tests: test_context_manager.py (+3). Docs: core/context_manager.html. |
| 2026-07-07 | #25 | Reconstructed `COGNITIVE_LAYER_REDESIGN.md` at repo root (from agent.py toggle comments + memory ledger); status banners added to docs/core/arbiter.html + mcts.html; README MCTS/swarm claims fixed; ablated-layer test files marked (test_arbiter/test_mcts docstrings). Guard test test_cognitive_redesign_doc.py pins the doc exists + toggles match. |
| 2026-07-07 | #27c | `graph.py`: `prune_stale_edges` (weight-1 edges older than N days dropped from DB+mirror; reinforced edges survive — weight is the decay signal), wired into dream cycle; cached node-list snapshot (`_nodes_snapshot`/`_invalidate_node_cache`) replaces per-query `list(nodes())`. Full inverted index deferred. Tests: test_graph_forgetting.py (6). Docs: memory/graph.html. |
| 2026-07-07 | #7/#6 | `_RequestState.stable_tool_query()` pins acquired-skill routing to the request's first substantive query → advertised tool set byte-stable across turns (the blocker that made the `GHOST_PIN_TOOL_SCHEMAS` KV pin ineffective). Tests: test_stable_tool_query.py (4). Docs: core/agent.html. Description-diet + meta-cluster consolidation (#7a/b) and the pin A/B+flip (#6) need a live operator session — not flipped blind per the ledger's own guidance. |
| 2026-07-07 | #4 | Added `--bio-time-scale N` (`_bio_scaled`/`_bio_cooldown` divide all idle windows + cooldowns) and `--bio-deterministic` (`_bio_roll` fires idle phases every tick); hoisted the ~15 inline window literals to scaled calls; defaults preserve prod timing. Built `scripts/ablation_trackb3.py` (isolates pure-idle loops: treatment accelerated+deterministic vs control at scale 1; compares probe McNemar + learning artifacts by source). Tests: test_bio_time_scale.py (7). Docs: scripts/ABLATION.md. LIVE run needs operator. |
| 2026-07-07 | #26 | `tests/helpers.py`: `make_context(**overrides)` / `make_agent(**overrides)` + shared `FakeBgTasks`, exposed as `agent_context`/`built_agent` conftest fixtures — one seam replacing the 22 fixtures / ~94 hand-built contexts / 233 SimpleNamespace fakes that welded the suite to the monolith's shape (which deterred #5). Tests: test_canonical_builders.py (6, incl. a real handle_chat smoke). Docs: architecture.html. |
| 2026-07-07 | #5 | Guard-module seam established: `core/stream_guards.py` houses the pure streaming guards (`_detect_thinking_loop`/`_detect_tool_call_loop`/`_tail_has_stop_marker` + constants), re-exported into agent.py (behavior-preserving, re-export identity pinned). New stream guards land there, not inline. The 4 big handle_chat extractions + TurnState deferred (need live hot-path validation). Tests: test_stream_guards_module.py (5). Docs: core/agent.html. |
| 2026-07-07 | ALL | Session complete. Full suite: 6508 passed / 11 skipped / 0 failed (was 6398 at session start; +110 new tests). Every item completable in this environment is DONE or has its apparatus built. Remaining needs a LIVE operator session: #6 (KV-pin A/B), #4 (B3 multi-hour run) + #27b (PRM verdict = B3 sub-arm), #5-big / #7-descriptions (live hot-path / tool-use validation). #1 SKIPPED (git on another server). |
