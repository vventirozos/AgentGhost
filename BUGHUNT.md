# System-Wide Bug Hunt — Resumable Ledger

Goal: code-review every subsystem for bugs, fix what's confirmed, test, update docs,
and mark the unit CLEAR. One or a few units per session; this file is the state.

## How to resume (instructions to Claude)

1. Read this file. Pick the **first unit whose status is not CLEAR** (top to bottom),
   or finish any unit marked `in-progress` first.
2. Set status to `in-progress`, then review every file in the unit's scope for real,
   demonstrable bugs (concrete failure scenario required — not style).
3. For each confirmed bug: fix it, add a regression test in `tests/`, update the
   matching HTML docs in `docs/`.
4. Run the full suite: `GHOST_API_KEY=test-key PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python -m pytest tests/ -q`
   (must be green — ~4300 tests / ~3 min).
5. Record findings in the log at the bottom (unit, date, bugs found/fixed/deferred),
   set status to `CLEAR` with the date. Deferred/uncertain findings go to the
   **Deferred findings** section so they aren't lost.

Definition of CLEAR: every file in scope reviewed; confirmed bugs fixed with
regression tests; docs updated; full suite green on that session's date.

## Units

| # | Unit | Scope | ~Lines | Status |
|---|------|-------|--------|--------|
| 1 | utils | src/ghost_agent/utils/ (helpers, logging, sanitizer, egress_guard, telemetry, token_counter, stylometry, constraints) | 2637 | CLEAR 2026-07-03 |
| 2 | sandbox | src/ghost_agent/sandbox/docker.py | 590 | CLEAR 2026-07-03 |
| 3 | workspace | src/ghost_agent/workspace/ | 1514 | CLEAR 2026-07-03 |
| 4 | tools-infra | tools/: registry, validators, fallback_chains, tool_failure, tasks, self_state, introspect, uncertainty_tool | 2300 | CLEAR 2026-07-03 |
| 5 | tools-fs-exec | tools/: file_system, execute, workspace, workspace_track, system | 3612 | CLEAR 2026-07-03 |
| 6 | tools-web | tools/: browser, search, darkweb_search, vision, image_gen | 3123 | CLEAR 2026-07-03 |
| 7 | tools-knowledge | tools/: memory, database, report_pdf, postmortem_review | 2387 | pending |
| 8 | tools-skills | tools/: composed_skills, acquired_skills, swarm, qwen_bridge | 2055 | pending |
| 9 | tools-projects | tools/projects.py | 2062 | pending |
| 10 | memory | src/ghost_agent/memory/ | 5891 | pending |
| 11 | router | src/ghost_agent/router/ | 1056 | pending |
| 12 | api | src/ghost_agent/api/ | 1223 | pending |
| 13 | core-llm | core/: llm, prompts, context_manager, bus | 2711 | pending |
| 14 | core-planning | core/: planning, mcts, frontier_selection, hypothesis, entropy | 2244 | pending |
| 15 | core-verify | core/: verifier, calibration, uncertainty, confidence, metacog, metacog_log, strikes, arbiter | 2892 | pending |
| 16 | core-projects | core/: project_advancer, project_research, project_concepts, project_safety, project_digest, coding_executor, workspace_cleanup, triggers, rrf_weights | 3921 | pending |
| 17 | core-dream | core/: dream, challenge_templates, journal_challenges, adversarial_generator, self_play_scoring, solution_novelty | 6697 | pending |
| 18a | core-agent (1/2) | core/agent.py lines ~1–5300 (init, loop, tool dispatch) | 5300 | pending |
| 18b | core-agent (2/2) | core/agent.py lines ~5300–end + agent_qwen.py | 5245 | pending |
| 19 | prm | src/ghost_agent/prm/ | 1599 | pending |
| 20 | reflection | src/ghost_agent/reflection/ | 1711 | pending |
| 21 | selfhood | src/ghost_agent/selfhood/ | 2278 | pending |
| 22 | distill | src/ghost_agent/distill/ | 1564 | pending |
| 23 | eval | src/ghost_agent/eval/ | 1062 | pending |
| 24 | optim | src/ghost_agent/optim/ | 792 | pending |
| 25 | skills_auto | src/ghost_agent/skills_auto/ | 621 | pending |
| 26 | entrypoint | src/ghost_agent/main.py, _env.py, __init__.py | ~400 | pending |
| 27 | interface | interface/server.py, slack_project_commands.py | ~? | pending |
| 28 | scripts | scripts/*.py (best-effort; eval/ablation tooling) | ~? | pending |

## Deferred findings (not yet fixed — revisit)

- **browser redirect / loopback SSRF + Tor bypass (HIGH — top priority)** — Chromium
  bypasses the proxy for loopback and does NOT re-check redirects, so a page navigated
  to a public host that returns `302 → http://127.0.0.1:9051` (Tor control),
  `169.254.169.254` (cloud metadata), or a LAN host reaches host-local services under
  host networking (agent API, postgres, Tor control), and the host-side guard only vetted
  the initial URL. DNS-rebind (resolve public at check, internal at nav) is the same class.
  Related: the `.last_url` sidecar re-navigation (runner-side) skips the SSRF check
  entirely; and `file://` can read any container-readable file (not restricted to the
  sandbox subtree). Fix candidates: (a) runner-side Playwright `page.route` interceptor
  that aborts requests to loopback/private/link-local/metadata when a proxy is set;
  (b) `--proxy-bypass-list="<-loopback>"` to force loopback through the SOCKS proxy in
  Tor mode. BOTH need a live Chromium (Docker) to verify they block the vector without
  breaking self-play fixtures — deferred rather than shipped untested. (high / likely)
- **darkweb onion body-cap after full download** — `_fetch_onion_text` reads the whole
  response (`.get()`, non-streaming) before `_cap_body`; an untrusted onion engine sending
  a chunked multi-GB body with no Content-Length OOMs the host. Same streaming-cap shape
  as the (fixed) vision URL fetch — apply it to the onion fetch. (med / likely)
- **vision cross-project read fallback** — a missing project-scoped image is retried
  against the sandbox ROOT, letting one project's agent read another project's artifacts
  via `vision`. Noted by-design (read-only tool); revisit if project isolation tightens.
  (low)

- **file_system replace bad-byte write-back corruption** — `tool_replace_text` reads
  with `errors="replace"` then writes the whole file back, so a mostly-text file with
  a few invalid-UTF-8 bytes has every bad byte persisted as U+FFFD (corrupting regions
  the edit didn't touch). NOT fixed by strict-refuse: tolerating such files is a
  deliberate feature (test_high_roi_fixes::test_replace_text_handles_text_with_bad_bytes).
  Correct fix: decode/encode via `errors="surrogateescape"` (round-trips bad bytes
  losslessly) through the shared `_write_replace_guarded` write path + the streaming
  path. Deferred because it touches the shared write path used by all replace
  strategies. The streaming .tmp-leak part of this finding WAS fixed. (med / likely)
- **file_system download SSRF via redirect** — `tool_download_file` validates the
  original URL with `url_ssrf_reason` but both clients follow redirects, so on the
  non-Tor WEB path a 302 → `169.254.169.254` / `127.0.0.1:<port>` / internal host
  bypasses the guard. Only reachable outside mandatory-Tor (under Tor the request
  routes through the SOCKS proxy, which can't reach link-local/metadata). Fix:
  disable auto-redirect and manually follow ≤N hops re-validating each Location —
  deferred because it restructures the streaming/proxy download and risks breaking
  the working path. (med / possible; non-Tor only)
- **execute `_inline_py` `-c` body detector false-block** — a chained command whose
  trailing segment reuses the delimiter quote (`python3 -c "print(1)" && echo "a; b"`)
  is mis-parsed by the `$`-anchored regex (the body absorbs the `&&` segment), tripping
  the "quote-escape corruption" BLOCK and forcing a re-emit. A false-block, never a
  wrong execution. A non-greedy tweak does NOT fix it (verified — the `$` anchor forces
  expansion). Proper fix: extract the real `-c` arg with shlex, which risks the existing
  corruption-detection/auto-convert logic. (med / likely; annoyance only)
- **execute stateful args dropped** — `execute(stateful=True, args=[...])`: the jupyter
  runner reads only `sys.argv[1]` (the wrapper) and never forwards argv[2:] into the
  kernel namespace, so a script reading sys.argv silently gets the kernel's argv.
  (low / likely)
- **execute file-not-found retry re-runs side-effecting commands** — the root/remap
  retry assumes "file-not-found ⇒ nothing executed", but `_looks_like_file_not_found`
  is a substring heuristic that can match output from a command that DID work before
  failing (e.g. "config.py not found, aborting" after real writes), duplicating side
  effects on retry. (low / possible)
- **execute single global stateful kernel** — shared `.jupyter_runner.py` + one kernel
  at `/workspace/.kernel.json`: concurrent stateful calls race the runner file and share
  one kernel namespace, and namespace state persists across projects/conversations
  (documented tradeoff — kernel pinned to /workspace). (low)
- **workspace_track mark_seen false-positive** — a non-empty but un-normalizable URL
  (e.g. `#frag` → normalizes to "") makes `mark_url_seen` return False, which the tool
  reports as "already seen" though it was never recorded. Implausible input. (low)

- **tool_failure loose FATAL patterns pre-empt diagnostics** — `invalid.?(arg|param|schema)`
  and `tool.*not found` / `not found.*tool` are checked before the DIAGNOSTIC bucket, so
  `ValueError: invalid argument …` and `FileNotFoundError: 'tool.py' not found` get marked
  FATAL ("do not retry") instead of self-correctable DIAGNOSTIC. Not fixed because tightening
  the fatal patterns risks the reverse regression (a genuinely-fatal arg/tool error getting
  retried). Needs care — maybe check explicit Python-exception DIAGNOSTIC types before the
  loose text FATAL patterns. (low / possible)
- **tool_failure dead retry helpers** — `get_retry_delay`/`should_retry`/`MAX_RETRIES` have no
  callers in src/ (agent.py drives retries via its own counters). `should_retry` has a latent
  off-by-one (4 tries for MAX_RETRIES=3). Harmless while unwired; wire-or-delete when the retry
  path is next touched. (low / certain-unwired)
- **fallback_chains ping-pong** — deep_research↔web_search list each other as first fallback.
  Advisory only (LLM decides), so by-design; noted as a smell, not fixed.
- **tool_failure swarm hint needles** — `"not configured"` never appears in swarm's output
  (dead hint) and `"0 of"` is a fragile bare substring. Fix against swarm's real error strings
  when reviewing swarm (unit 8). (low)
- **registry acquired-skill runner not project-scoped** — the acquired-skill runner executes
  with `sandbox_dir=context.sandbox_dir` (root), bypassing `_proj_ws()` scoping the other file
  tools use, so with a project active it can't see project files. Deferred: changing this shifts
  acquired-skill file paths (projects/<id>/acquired_skills/) and could break skill telemetry;
  weigh when reviewing acquired_skills (unit 8). (low / possible)

- **workspace current_project_id event-stamping race** — `WorkspaceModel.current_project_id`
  is process-global, reassigned per turn during prompt assembly (agent.py) while up to
  10 chats run concurrently (Semaphore(10)). Events recorded mid-turn (research/command/
  task outcomes) get stamped with whichever conversation most recently assembled a prompt,
  so they can be filed under the wrong project. Same documented race class as
  `context.current_project_id` (tools/projects.py, tools/file_system.py). Proper fix:
  thread the active project id through each record_* call instead of reading the global.
  (high / likely; deferred because the fix touches every capture call site across tools.)
- **workspace narrative.regenerate staleness** — read events → await critique_fn → persist
  has no re-entrancy guard, so two overlapping idle-phase consolidations can persist an
  older narrative over a newer one (atomic replace = no corruption, just staleness until
  next cycle). Fix: an asyncio.Lock around regenerate. (low / possible)
- **workspace _PROJECTS_PATH_RE URL false-match** — `derive_event_project_id` runs the
  `projects/<hex>` regex over the event URL too, so a free-chat pull of e.g.
  `gitlab.com/x/projects/deadbeef1/…` is misattributed to a "project". Documented as
  intentional legacy-recovery, but it can drop an agnostic event from a project view.
  Fix candidate: restrict the path-derivation haystack to filesystem fields. (low / possible)

- **sandbox/docker.py remove-while-exec race** — ensure_running's recreate path can
  force-remove the container while another thread is mid-exec_run (the lock only
  guards provisioning, by design). The in-flight command dies with "Container
  Execution Error" and the system self-heals next call. A full fix needs container
  generation counting / in-flight refcounts. (med / rare; do NOT auto-retry the
  command — commands are not idempotent.)
- **sandbox/docker.py no client-side deadline on probes/exec** — a wedged docker
  daemon socket can hang a worker thread forever (probes and command exec have no
  HTTP-level timeout). Naive fix (global docker client timeout) would kill legitimate
  long-running execs; needs a probe-only client or per-call timeout plumbing.
  (low-med / rare)
- **sandbox/docker.py Linux exec-user vs root-provisioned env** — on Linux, execute()
  runs as host uid:gid but everything is provisioned as root: Playwright's Chromium
  lives in /root/.cache/ms-playwright (mode 700) so the in-sandbox browser can't find
  it; sudo refuses unknown uids; runtime pip install unwritable. Masked in prod
  (macOS → exec as root). Fix sketch: set PLAYWRIGHT_BROWSERS_PATH=/ms-playwright at
  container creation, chmod -R a+rX after install, HOME=/tmp in exec env, marker bump.
  Do this when a Linux deployment actually matters — it forces a one-time full
  re-provision. (high on Linux / zero on current prod)
- **utils/telemetry.py `stop()`** — `except (asyncio.CancelledError, Exception): pass`
  around `await self._task` can swallow a cancellation aimed at the *caller* of
  `stop()` (not just the expected one from the task we cancelled), absorbing shutdown
  cancellation. Rare; fixing needs care with shutdown semantics. (low / possible)
- **utils/helpers.py `is_removal_or_negation_text`** — substring matching without word
  boundaries can false-positive ("...cannot have a spare..." contains "not have a").
  Reviewed and left as-is: fail-safe direction, module documents false positives as
  cheap by design. No fix intended unless it shows up in practice.

## Session log

(newest first)

### 2026-07-03 — unit 6: tools-web — CLEAR
- Reviewed all 5 files via 3 lenses (browser; search+darkweb; vision+image_gen)
  + own pass over browser anonymity/injection. ~25 findings, 17 fixed, ~8 deferred.
  Runner injection (JSON+shlex) and per-attempt circuit isolation verified clean.
- HIGH: browser SSRF guard did host getaddrinfo even in Tor mode (DNS leak) →
  resolve=False when anonymous; + scheme allowlist + fail-closed.
- Also: WebRTC flag fix (removed local-IP-exposing --disable-features);
  vision URL byte-cap + non-image refusal + content-null coerce + leading-prefix
  strip; image_gen empty/data-URI b64 reject + width/height coerce + prompt-
  fallback exclusions; search/darkweb distill fallback scrubbed; deep_research
  fetch honours proxy + suppresses NEWNYM (helper parameterized) + max_context
  sizing; browser timeout_ms/max_chars safe-coerce + clamp; darkweb don't-cache-
  all-errors; fact_check None-guard.
- Cross-unit: added proxy_override/renew_identity params to utils/helpers
  helper_fetch_url_content (unit 1, backward-compatible defaults).
- Tests: tests/test_bughunt_unit6_tools_web.py (22). One existing test updated
  (map-reduce 40k→context-aware sizing).
- Docs: tools/browser.html (Tor/DNS hardening), tools/search.html (deep_research),
  audit_fixes.html (Round 9 unit 6).

### 2026-07-03 — unit 5: tools-fs-exec — CLEAR
- Reviewed all 5 files via 3 lenses (file_system; execute; system/workspace/track)
  + own read of the path-scoping core. ~20 findings, 13 fixed (1 HIGH), 6 deferred.
  Core traversal/symlink containment verified sound (resolve→is_relative_to).
- HIGH: destructive ops (delete/rename/copy) could resolve to the sandbox/project
  ROOT (/workspace, ., '', projects/<active-id>) and rmtree the whole workspace →
  _get_safe_path allow_root=False guard.
- 12 fixed (see below); 7 deferred incl. the SSRF-redirect and the replace bad-byte
  write-back (strict-refuse reverted — it broke the deliberate bad-byte tolerance
  feature; the correct surrogateescape fix is deferred).
- Also: writes force encoding=utf-8 (locale mojibake + truncate-on-error);
  streaming replace temp cleanup (leaked .tmp on no-match/error); empty-content
  guard narrowed (allows "none"); flexible-replace
  count=1; inspect encoding; <3.9 containment fallback hardened; system action
  normalize; execute non-UTF8 read guard + both-args warning + grep-no-match
  telemetry; workspace/track str-coerce action (never-raises).
- Reverted a non-greedy _inline_py regex tweak that did NOT fix the false-block
  (verified) — deferred instead.
- Tests: tests/test_bughunt_unit5_fs_exec.py (24). Existing 261 fs/exec tests green.
- Also hardened a PRE-EXISTING flaky timing test unrelated to unit 5:
  test_memory_bus.py::test_publish_fact_concurrent_fanout asserted elapsed<100ms for
  4 concurrent 30ms ops — flaked under full-suite CPU contention. Widened to 50ms
  sleep / 150ms bound (preserves the concurrent-vs-sequential distinction). Not a
  unit-5 bug; belongs to core/bus (unit 13) but fixed now to keep the green-gate
  reliable for the remaining units.
- Docs: tools/file_system.html (path-safety, write/replace rows), audit_fixes.html
  (Round 9 unit 5).

### 2026-07-03 — unit 4: tools-infra — CLEAR
- Reviewed all 8 files via 3 lenses (registry; tool_failure/validators/fallback;
  tasks/self_state/introspect/uncertainty) + own read. ~20 findings, 12 fixed,
  ~7 deferred, several non-issues cleared (registry schema↔dispatch symmetry verified).
- Fixes: validators schema-qualified/quoted DELETE/UPDATE table names; validators
  string-literal-aware paren balance + WHERE detection (fixes both a valid-SELECT
  false-reject AND a where-in-string/subquery bypass); tool_failure 401/403 word-
  boundary + http/status context; tasks action normalise; tasks malformed-interval
  reject (was silent-60s-masquerade); tasks memory-write isolation; registry browser
  single _proj_ws() call (_run_browser helper); registry acquired-skill RAG failure
  degrades to advertise-all; **kwargs on manage_skills/self_play_loop/list_lessons;
  composed-skills known_tools from live dispatch table; self_state resolve/close
  require target; uncertainty dead-assignment removal.
- Tests: tests/test_bughunt_unit4_tools_infra.py (31). Two existing tests updated
  for intended behavior changes (interval reject; browser async dispatch).
- Docs: tools/tasks.html, tools/tool_failure.html, tools/registry.html,
  audit_fixes.html (Round 9 unit 4).

### 2026-07-03 — unit 3: workspace — CLEAR
- Reviewed all 7 files via 2 lenses (model/state; aux modules). 18 findings,
  15 confirmed+fixed, 3 deferred. Both reviewers independently flagged the
  activity.jsonl unbounded-growth bug.
- Fixes: reactions utf-8-sig/replace decode (BOM false-positive + invalid-byte
  false-negative); activity self-compaction + iter_events lock-across-yield +
  recent() deque; state track/untrack path normalisation; touch_session prior
  timestamp; corrupt state.json sidecar + tolerant from_dict; research dedup
  mark-after-append; nav counter recency eviction; TaskOutcome fired/finished
  timestamps; changelog scoping (legacy+agnostic) + undated sort; strip find
  offset; cross-project prefix tolerance.
- Tests: tests/test_bughunt_unit3_workspace.py (23). Existing 111 workspace tests
  still green.
- Docs: algorithms/workspace.html (activity, state, reactions rows),
  audit_fixes.html (Round 9 unit 3).

### 2026-07-03 — unit 2: sandbox — CLEAR
- Reviewed docker.py via 2 lenses (concurrency/lifecycle, logic/security); 15 findings,
  11 confirmed+fixed, 3 deferred, 1 not-a-bug (host networking is documented design).
- Fixes: always-commit image cache after provision (stale-cache reinstall loop);
  self.image no longer mutated to cached tag (rmi bricked fallback pull); 5-min
  provision-failure backoff (reinstall storm); in-container `timeout` on all installs
  (they hold the provision lock); readiness probe retries once on transient errors
  (false negative force-removed healthy containers); never-ready container now raises;
  marker/chromium checks + tor spawn once per container generation (per-command
  overhead + "Environment Ready" log flood); duplicate readiness probe removed from
  execute(); host workspace mkdir before create (root-owned bind source loop);
  CPU_QUOTA<=0 = uncapped (0 bricked creation); get_stats handle snapshot.
- Tests: tests/test_bughunt_unit2_sandbox.py (17). Two existing tests updated for
  intended behavior changes (timeout prefix; MagicMock workspace → real tmp dir).
- Docs: sandbox/docker.html (lifecycle, tor), audit_fixes.html (Round 9 unit 2).

### 2026-07-03 — unit 1: utils — CLEAR
- Reviewed all 8 files (2 parallel review agents, findings re-verified in source).
- 10 findings → 8 confirmed and fixed, 2 deferred (above). Fixes:
  1. sanitizer: loose fence regex still quadratic (measured 16K spaces ≈ 1.2s; regression
     of the earlier ReDoS fix) → linear prefix-match + str.find scan, equivalence-tested.
  2. sanitizer: `fix_python_syntax` ?-stutter strip corrupted *valid* string literals
     ("Ready?Set?Go?Now" → "Ready"), corruption parsed & was committed silently → valid
     code now passes through untouched.
  3. egress_guard: multicast blocked (CPython `is_global=True` for 224/4, ff00::/8)
     despite documented-allowed → `is_multicast` checked before `is_global`.
  4. stylometry: `scrub_query` dropped leading content keywords (bare "search"/"lookup"
     prefixes; gratitude tokens anywhere) → framed forms only; gratitude trailing-only.
  5. token_counter: unlocked `_SHORT_TOKEN_CACHE` eviction race → threading.Lock.
  6. helpers: binary-URL short-circuit used raw URL not path (missed "?dl=1") → urlparse path.
  7. logging: `setup_logging` crashed on bare filename (makedirs("")) → guarded.
  8. telemetry: `HostSnapshot.healthy` ignored disk + hardcoded thresholds → mirrors
     HostTelemetry defaults at call time, disk axis added.
- Tests: tests/test_bughunt_unit1_utils.py (29 tests). Full suite: 5874 passed, 11 skipped.
- Docs: audit_fixes.html (Round 9), tools/search.html (scrub contract),
  algorithms/mandatory_tor.html (multicast note).
