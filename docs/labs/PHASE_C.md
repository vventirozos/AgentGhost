# Phase C — the `run` module, built ALONGSIDE services/jobs (design, 2026-11-04)

Source of truth: UNIFIED_DESIGN.md §3.2, §4, §12 (code facts), §13 finding 6, §14 row C. This note pins the build.
Acceptance (§14): `run` starts/stops `server` runs in the workspace and `job`/`app` runs on labs; old services/jobs
untouched; both registries agree in a reconcile test. Rollback: revert; the old tools never changed.

## C1. What is reused, not rebuilt (inventory 11-04)
There is no shared workspace launcher today: `sandbox/services.py` `ServiceSupervisor` (2,869 lines: port leasing,
pid file + `$!` fallback, generation stamps, adopt, loopback rules, token) and `sandbox/jobs.py` `JobSupervisor`
(nonce'd exit sentinel, `timeout -k`, TTL 45 min, reap) each build their own `setsid nohup`. Phase C does **not** write
a third launcher. Workspace runs are **adapters** over the two supervisors:
- `server` @ workspace → `ServiceSupervisor.start/stop/restart/status/logs` (same project key, same lease, same token).
- `job` @ workspace → `JobSupervisor` detached start (the `execute` promotion path is NOT changed in C — finding 6:
  promotion moves to `job` runs in the same deploy as the tool switch, end of D).
- `app` @ workspace → refused (`app` lives on kept machines, §5).
So a workspace run started through `run` IS a services/jobs row; the run registry holds a pointer, and "both registries
agree" is true by construction and checked by `reconcile`.

## C2. Lab runs (new code, small)
- `job` @ `lab_x/member` → `POST /v1/labs/{id}/exec {detach:true}` → `job_id`; status/logs/collect via
  `GET /v1/labs/{id}/jobs/{job_id}`; stop via `DELETE` (process-group kill). States from the host: running|exited|lost.
- `server` @ `lab_x/member` → same detached job + a declared `port`; probed by `POST /v1/labs/{id}/http` to that
  member:port. No tailnet URL (disposable labs are not exposed); `url` answers "reach it with `lab http`".
- `app` @ `lab_x/member` → only on a **kept** lab (`/keep` is Phase D's release path; in C, `run start lifetime=app`
  on a disposable lab is refused with `not_kept` wording). On a kept lab: `POST /v1/labs/{id}/app/{start|stop|restart}`;
  state and URL from the lab detail `app` block. Owner origin only (host returns `owner_only`; the agent refuses first).

## C3. The registry (host-side, fixes P6)
`<memory_dir>/runs.db` (sqlite, never on the bind mount). Row: `run_id (run_<8hex>), name, where, lifetime, command,
port, owner, origin, backend (services|jobs|labs_job|labs_app), backend_ref (entry key | job id | lab_id/job_id |
lab_id), started_at, state, last_probe_at, ended_at`. `(owner, name)` unique while live. State is **probed, not
remembered**: every `status` asks the backend and writes the probe back. `reconcile()` = for each live row probe the
backend; a backend row that vanished → `lost`; a services/jobs row with no run row is NOT adopted in C (they are the old
tools' rows) — reported only.

## C4. The tool (`run`)
Actions `start (name, command, where, lifetime, port?)`, `status [name|all]`, `logs`, `stop`, `restart`, `collect`,
`url`. `where` required (`workspace` | `lab_<8hex>/<member>`), no default. Results are ToolOutcomes with status +
world_changed like `lab`; a start returns the run's "still running, not evidence" banner (`is_promoted_result`
semantics: a run start is never proof of a result). Origins: member/replay refused (REPLAY_FORBIDDEN); autopilot may
start/stop `job`/`server` runs, never `app` (D5); a probe stops only its own runs. Owner key as `lab`.
**Exposure while alongside:** `GHOST_RUN_TOOL = off | probe | on`, default **probe** at the C deploy — advertised only
to probe-origin requests (so live acceptance runs, the owner's tool list and KV head are unchanged). `on` comes with
the Phase D switch (aliases, `manage_services`/`jobs` removed, briefing RUNNING block, promotion → job runs).

## C5. Wiring (re-grep every `"execute"`/`"manage_services"` frozenset, as for `lab`)
REPLAY_FORBIDDEN, egress (command text is code: CODE_TOOLS; `url_literals`), strikes (run logs = program output),
evidence (a start is not evidence; `collect` of an exited job is), outcome heuristics, agent work/mutating/batch-wall
sets, batch order (`_batch_call_depends`: run after execute in the workspace; run on lab_x after lab calls on lab_x),
status-question refusal, isolation args. Not in C: briefing, aliases, promotion, boot reconcile (only a boot line).

## C6. Out of scope (Phase D)
Projects on the inventory; archive/delete/release over runs; kept machines via `/keep`; `run` replacing
`manage_services`/`jobs`; token check move; deleting the old modules; chess-coach/jiu-jitsu rows deleted.

## C7. Design review r1 (11-04, independent reader) — 3 HIGH, 4 MED, 2 LOW, all folded in; these OVERRIDE C1–C5
- **H1** `JobSupervisor` has no detached start (`run()` blocks the budget and kills a quiet job at 124). → ONE new method
  `start_detached(command, workdir, owner, label)` on the supervisor reusing `_write_script`/`_launch`/the `_promote` row/
  `_record_launch` and the concurrency cap; every existing path unchanged.
- **H2** a run-started job would be adopted by the `jobs` tool and woken with "use jobs collect". → rows carry
  `started_via="run"`; `_sync_sandbox_jobs` skips them; their wake text names `run(action='collect', name=…)`; the
  reconcile test asserts each run job is reported once.
- **H3** services keys are `<project>:<name>`; a conversation owner would become a bare `name` and the §4SK replace path
  would kill another conversation's (or a legacy) service. → conversation owners map to a safe key half
  `c-<sha1(conversation key)[:12]>`; `run start` refuses when the key exists and is not its own pointer.
- **M1** `app` cannot pass in C (`/app/*` is owner-origin only; no `/keep` until D; the app command is fixed at keep).
  → **`app` runs move to Phase D** with `/keep`. C acceptance = `server`@workspace, `job`@workspace, `job`/`server`@lab.
- **M2** a probe-only tool list re-prefills the head (~166 s per probe turn, §4LE). → `run` is **not advertised** in C
  (callable registered, dispatch refuses it from the model); live acceptance is a script calling `tool_run` in a
  probe-origin scope. The owner's tool list changes once, at the D switch. `GHOST_RUN_TOOL` dropped.
- **M3** service state is prose. → after a start read the row from `list_entries()` (key, pid, port, token); `backend_ref`
  keeps the token; on probe a token mismatch = `replaced` (a `manage_services` start reused the name), not `running`.
- **M4** lab jobs never end and every read `touch`es the lab (keeps a disposable lab alive). → lab job commands are
  wrapped agent-side in `timeout --foreground -k 30 2700` (45 min, = workspace TTL; `--foreground` so the host's group kill on stop reaches it — live 11-04); nothing polls labs in the background — only
  an explicit `run status/logs/collect` reads them (the model is using the lab, so the touch is right); the boot line
  and reconcile never touch labs.
- **L1** a run on a destroyed lab → `lost (lab destroyed)`; a host kill (`exited, exit_code null, killed`) = stopped.
- **L2** lab job fields `{job_id, state, exit_code, output_tail, started_at, ended_at}`; `/http` `{member, port, path,
  method}` → `status|null, error`. A `where=lab_x/…` on another owner's lab needs the explicit id and the result names
  its owner (§12 finding 13), as `lab` does.
