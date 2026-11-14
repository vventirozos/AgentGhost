# src/ghost_agent/core/prompts.py

import re

# How many trailing lines of the design ledger to surface in the briefing.
# The ledger itself is bounded by ProjectStore.LEDGER_MAX_LINES; this caps
# what we inject into the prompt each turn so it stays compact.
_LEDGER_BRIEFING_LINES = 20


_BRIEFING_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "be", "it", "its", "this", "that", "my", "me",
    "you", "your", "i", "we", "do", "does", "not", "no", "can", "cant",
    "please", "now", "then", "when", "after", "but", "have", "has", "get",
    "give", "make", "made", "them", "they", "there", "what", "which",
    "project", "task", "tasks", "update", "status", "resume", "start",
    "service", "file", "files", "fix", "still", "same", "work", "works",
})


def _briefing_tokens(text: str) -> set:
    """Significant lowercase tokens for the relevance slice — stopword- and
    generic-verb-stripped so 'resume X and give me a status update' doesn't
    match everything."""
    return {
        t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", (text or "").lower())
        if len(t) > 2 and t not in _BRIEFING_STOPWORDS
    }


def _files_line(project_id: str) -> str:
    """§4QD (2026-10-20): the briefing named the id on every turn but never
    said where files live, and the rule did appear only in a switch/create
    result — a project already active from an earlier conversation got the
    id and no rule, and the model read '<id>/report/final-report.md' (all 3
    owner requests on 448d5afa86f0 that morning, each a failed read and an
    11–14 s retry)."""
    return (f"FILES: this project's folder IS your working directory — name its files "
            f"relative to it ('report/final-report.md'), never '{project_id}/…' or "
            f"'projects/{project_id}/…'.")


def _created_this_request(store, project_id: str) -> bool:
    """§4UI: was the project created by the CURRENT request (its
    ``project_created`` event carries this request's id)? Never raises."""
    try:
        from ..utils.logging import request_id_context
        rid = str(request_id_context.get() or "")
        if not rid or rid == "SYSTEM":
            return False
        ev = (store.list_events(project_id, limit=1, event_type="project_created") or [None])[0]
        return bool(ev) and str((ev.get("payload") or {}).get("req") or "") == rid
    except Exception:  # noqa: BLE001
        return False


def build_project_briefing(store, project_id: str, max_events: int = 3,
                           max_open_tasks: int = 8,
                           max_open_tasks_listed: int = 30,
                           max_done_tasks: int = 5,
                           max_stuck_tasks: int = 4,
                           max_work_logs: int = 5,
                           max_deliverables: int = 12,
                           suppress_next_task: bool = False,
                           graph_memory=None,
                           request_text: str = "",
                           services=None,
                           running=None) -> str:
    """Render a compact project-scope briefing for the system prompt.

    The briefing is appended to DYNAMIC SYSTEM STATE when a project is
    active (context.current_project_id is not None). Returns an empty
    string when the store or project_id is missing so the caller can
    unconditionally concatenate the result.

    It is the project's cross-turn working memory: alongside the task tree
    it surfaces the DESIGN LEDGER (durable facts the agent recorded) and a
    DONE SO FAR digest (recently-completed tasks + the one-line result the
    agent wrote on completion). Together these let a fresh turn know what
    exists and how it works WITHOUT re-reading every file — the dominant
    per-turn cost on long projects.

    Shape:
        ### CURRENT PROJECT
        TITLE: …  (KIND · STATUS)
        GOAL: …
        DESIGN LEDGER:
          …durable facts…
        NEXT TASK: [id] description
        OPEN TASKS (N: n1 STATUS1, n2 STATUS2):
          - [id] description  (STATUS)   (all open, ≤30; after 8 the
            description is cut to 60 chars)
        STUCK TASKS (≤N of M …):
          - [id] description  (FAILED|BLOCKED) — failure reason
        DONE SO FAR (≤N, most recent first):
          - [id] description → result summary
        RECENT WORK LOG (≤N — newest first …):
          - "request head" · files: … · outcome: note head
        RECENT EVENTS (≤N):
          - ts  type  payload-preview
    """
    if store is None or not project_id:
        return ""
    try:
        proj = store.get_project(project_id)
    except Exception:
        return ""
    if not proj:
        return ""

    # Local import to avoid a prompts→planning cycle at module import time.
    from .planning import ProjectPlan, TaskStatus

    lines = ["### CURRENT PROJECT"]
    # §4SU Phase 4 (audit: the pacing rule and "DO NOT create" were shown
    # on FINISHED projects): a DONE / ARCHIVED project gets one line on how
    # to reopen it instead; the retrospective below says what was built
    try:
        _state = store.project_state(project_id, proj)
    except Exception:
        _state = str(proj.get("status") or "").lower()
    # (§4UM review: a legacy DONE the old roll-up set by itself reads "ready", not closed)
    _finished = str(proj.get("status", "")).upper() in ("DONE", "ARCHIVED") and _state != "ready"
    if _finished:
        _arch = str(proj.get("status", "")).upper() == "ARCHIVED"
        # §4UM: only the user closes a project; a change written into its folder reopens it (and is logged)
        lines.append(
            (f"*** This project is ARCHIVED (id={project_id}): no open work. If the user asks for more on "
             "it, resume it first, then make the change HERE. "
             if _arch else
             f"*** This project was CLOSED by the user (id={project_id}). If they ask for a change, make it "
             "HERE, straight in its files — that reopens it and the system logs the change; add tasks "
             "(task_add / task_decompose) only for multi-step work. ")
            + "Do not create a new project for the same topic. ***")
    elif _state == "ready":
        # §4UM: planned work done, the project stays open — a living app the user keeps changing
        lines.append(
            f"*** All planned work on this project is done (id={project_id}, state READY) — it stays OPEN: a "
            "living app/site. When the user asks for a change, make it straight in its files (the system logs "
            "each change, and each deploy, in the change log and PROJECT_CHANGES.md — no task needed for it); add "
            "tasks only for multi-step work. Never call it complete or closed: only the user closes it "
            "(manage_projects action=update status=DONE when they ask). ***")
    # Hard directive at the top. Small Qwen models routinely re-call
    # `manage_projects action=create` AND re-interpret the user's
    # original "start a new project" message every few turns — they
    # don't fully trust the conversation-state signal, so we repeat
    # it as loudly as possible on every turn the project is active.
    if not _finished:
        # §4UI r1 M2: "the rest of that message is still to do" only in the
        # request that created it — on a later request the old message is
        # fulfilled history (re-reading it started builds with no go-ahead)
        if _created_this_request(store, project_id):
            _tail = ("Creating it is done; anything else the user asked for in "
                     "this message (building it, a first task) is still to do — "
                     "see PACING below.")
        else:
            _tail = ("DO NOT re-read the user's original 'start a new project' "
                     "message as if it were a fresh instruction; that request has "
                     "been fulfilled and the work is in flight below.")
        lines.append(
            f"*** A project is ALREADY ACTIVE (id={project_id}). "
            f"DO NOT call manage_projects action=create — you already did. "
            f"{_tail} ***"
        )
    lines.append(_files_line(project_id))
    # RELEASED runbook mode (2026-07-25): a released project's briefing is
    # OPERATIONAL, not developmental — lead with the human-attested usage
    # directions and the rehearsed service commands, suppress the task
    # scaffolding entirely (there are no open tasks; the artifact is
    # immutable), and route any change request to create_version.
    if str(proj.get("status", "")).upper() == "RELEASED":
        rel = {}
        try:
            rel = store.get_release(project_id) or {}
        except Exception:
            rel = {}
        rl = [
            "### PROJECT (RELEASED)",
            f"TITLE: {proj['title']}  ({proj['kind']} · RELEASED "
            f"v{rel.get('version', 1)}, {rel.get('released_at', '')})",
            f"GOAL: {(proj.get('goal') or '').strip()}",
            "*** This project is RELEASED: human-tested and IMMUTABLE. "
            "To RUN/USE it: follow the directions below (start the listed "
            "services with `run start` if they are not running — the "
            "commands were verified at release). For ANY change request: "
            "do NOT edit these files — when the USER asks for a change, fork a development copy with "
            "`manage_projects action=create_version` and work there; this "
            "version keeps running untouched. ***",
            _files_line(project_id),
        ]
        if rel.get("directions"):
            rl.append("RELEASE DIRECTIONS (how to use):")
            for ln in str(rel["directions"]).splitlines():
                if ln.strip():
                    rl.append(f"  {ln.strip()}")
        for s in rel.get("services") or []:
            rl.append(
                f"SERVICE: {s.get('name')} · start: {s.get('command')}"
                + (f" · port {s['port']}" if s.get("port") else ""))
        for u in rel.get("urls") or []:
            rl.append(f"URL: {u}")
        if rel.get("deliverables"):
            rl.append("FILES:")
            for d in rel["deliverables"][:12]:
                rl.append(f"  - {d.get('path')}"
                          + (f" — {d['desc']}" if d.get("desc") else ""))
        return "\n".join(rl)

    if not _finished:
        lines.append(
            # §4UI (2026-11-12): THE pacing rule — the tool description and
            # every manage_projects result defer to it instead of restating
            # their own STOP (req 8646c6c8 spent 5 min weighing three).
            "*** PACING (the one rule; tool results defer to it): "
            "GO-AHEAD — when you have just created, decomposed or replanned "
            "the plan, look at the user's message that asked for it. If it "
            "asked ONLY for a project or a plan, show the task list and stop "
            "there (no 'shall I start?' — they will say 'proceed', 'next task' "
            "or 'continue' when they want the work). If that same "
            "message ALSO asked for the work itself ('create a project and "
            "build X', 'make it…', 'implement…'), that message IS the "
            "go-ahead: show the plan in a line or two and start. "
            "SCOPE — a go-ahead covers what the user asked for: 'proceed' / "
            "'next task' = the NEXT TASK shown below; 'do the next 3', "
            "'finish it', 'all remaining' = that many. Close each task when its "
            "work is done with `manage_projects action=task_update "
            "task_id=\"<id>\" status=DONE` (plus its `deliverables`). When "
            "the asked-for tasks are done, report and wait for the next "
            "direction. Several tasks: do them "
            "yourself in this turn one after another, or hand a long run to "
            "`manage_projects action=autoadvance count=<N|\"all\">` (an "
            "unattended loop that checkpoints each task); on a big project "
            "prefer autoadvance so the context window does not fill. "
            "Before re-reading files to reconstruct state, READ the DESIGN "
            "LEDGER and DONE SO FAR below — they record what already exists and "
            "how it works. When you make a durable decision (file layout, a key "
            "function/API name, a convention) record it in one line via "
            "`manage_projects action=ledger ledger=\"…\"` (or pass `ledger=\"…\"` "
            "on the task_update that closes the task) so the next turn inherits "
            "it instead of re-deriving it. ***"
        )
    lines.append(
        # §4UM live probe: an open (ready) project never shows the legacy auto-close's raw DONE
        f"TITLE: {proj['title']}  ({proj['kind']} · {'ACTIVE' if _state == 'ready' else proj['status']}"
        + (f" · {_state}" if _state in ("ready", "working") else "") + ")"
    )
    goal = (proj.get("goal") or "").strip()
    if goal:
        lines.append(f"GOAL: {goal}")

    # EXPLICIT USER CONSTRAINTS — verbatim clauses the user marked as
    # binding ("don't come up with some random AI", "YOU will play
    # against me"). Re-shown EVERY turn, above the ledger, because a
    # constraint stated once and never re-surfaced lost to reinforced
    # skill priors within a single reasoning pass (chess incident,
    # 2026-07-02). The DONE-gate in manage_projects holds any completion
    # that doesn't provide evidence addressing them.
    try:
        proj_constraints = [
            str(c) for c in
            ((proj.get("metadata") or {}).get("constraints") or [])
        ]
    except Exception:
        proj_constraints = []
    if proj_constraints:
        lines.append("EXPLICIT USER CONSTRAINTS (MUST HOLD — task completion "
                     "is gated on evidence these are honored; if your plan "
                     "conflicts with one, change the plan, not the "
                     "constraint):")
        for c in proj_constraints:
            lines.append(f"  ! {c}")

    # SERVICES — this project's registered services (2026-07-30, §4G): the
    # names + leased ports the model must NOT re-invent. Registry facts
    # only — liveness is deliberately unprobed here (probing costs docker
    # execs every turn); `stale` marks entries whose container died.
    # §4SX D5: RUNNING — the project's runs (state as last checked, and when) and lab machines, from local
    # records only; a workspace server run IS a service, so it is listed here and not again under SERVICES
    _run_keys = {str(r.get("service_key")) for r in (running or []) if r.get("service_key")}
    # review: the container-generation STALE mark (§4LR M1) stays on a run-backed server's RUNNING line
    _stale_keys = {str(s.get("key")) for s in (services or []) if s.get("stale") and s.get("key")}
    services = [s for s in (services or []) if str(s.get("key") or "") not in _run_keys]
    if running:
        import time as _time
        lines.append("RUNNING (this project's runs and lab machines — the state each had when last checked; "
                     "`run status` / `lab list` check now):")
        for r in running:
            if r.get("kind") == "lab":
                _b = f"  - lab {r.get('name')}" + (f" · {r['place']}" if r.get("place") else "")
                if r.get("label"):
                    _b += f" · {str(r['label'])[:40]}"
                if r.get("state"):
                    _b += f" · {r['state']}"
            else:
                _b = f"  - run {r.get('name')} · {r.get('lifetime')} · {r.get('place')}"
                if r.get("port"):
                    _b += f" · port {r['port']}"
                _b += f" · {r.get('state')}"
                if r.get("service_key") and str(r["service_key"]) in _stale_keys:
                    _b += " · STALE (died with old sandbox — `run restart` it on demand)"
                if r.get("at"):
                    _b += " (checked " + _time.strftime("%Y-%m-%d %H:%M UTC", _time.gmtime(float(r["at"]))) + ")"
            if r.get("note"):
                # §4UD F10: what the local record itself says about its age (a lab's latest end, a lab run past
                # its time limit) — no host call
                _b += f" · {r['note']}"
            lines.append(_b)
    if services:
        # review 11-05 W-m: these are services NO run row covers (registered before `run`) — `run` refuses their
        # names, so the line must not send the model there to stop them
        lines.append("SERVICES (registered before `run` existed — ports are LEASED "
                     "to them, never guess ports; `run` does not start, stop or "
                     "restart these: tell the owner if one must change):")
        for s in services:
            _b = f"  - {s.get('name')}"
            if s.get("port") is not None:
                _b += f" · port {s['port']} (http://127.0.0.1:{s['port']})"
            if s.get("stale"):
                _b += " · STALE (died with old sandbox — restart on demand)"
            if s.get("command"):
                _b += f" · cmd: {s['command']}"
            lines.append(_b)

    # DESIGN LEDGER — the durable, compact working memory the agent records
    # (file layout, key function/API names, conventions). Surfaced near the
    # top so the model relies on it instead of re-deriving the project shape
    # by re-reading files every turn.
    try:
        ledger = ((proj.get("metadata") or {}).get("design_ledger") or "").strip()
    except Exception:
        ledger = ""
    if ledger:
        lines.append("DESIGN LEDGER (durable facts you recorded — trust these; "
                     "update with `manage_projects action=ledger ledger=\"…\"`):")
        for ln in ledger.splitlines()[-_LEDGER_BRIEFING_LINES:]:
            ln = ln.strip()
            if ln:
                lines.append(f"  {ln}")

    # CONFIG — the project's durable settings (env vars, key flags, dependency
    # versions, model, ports, DB URIs). Surfaced so a fresh turn runs/builds
    # under the right settings instead of re-discovering them from
    # requirements.txt / env / argv. Recorded via
    # `manage_projects action=config config_key="…" config_value="…"`.
    try:
        config = ((proj.get("metadata") or {}).get("config") or {})
        if not isinstance(config, dict):
            config = {}
    except Exception:
        config = {}
    if config:
        lines.append("CONFIG (durable settings you recorded — trust these; "
                     "update with `manage_projects action=config config_key=\"…\" "
                     "config_value=\"…\"`):")
        for k, v in config.items():
            lines.append(f"  {k} = {v}")

    # DEPENDS ON — inter-project edges (2026-07-25): the advancer waits on
    # these; the model should know WHY autoadvance reports blocked.
    try:
        _deps = ((proj.get("metadata") or {}).get("depends_on_projects") or [])
        if _deps:
            _dep_bits = []
            for _d in _deps:
                _dp = store.get_project(_d) or {}
                _dep_bits.append(
                    f"{_dp.get('title') or _d} ({_dp.get('status') or '?'})")
            lines.append("DEPENDS ON (autoadvance waits until DONE/RELEASED): "
                         + ", ".join(_dep_bits))
    except Exception:
        pass

    # RELEVANT TO THIS REQUEST (2026-07-24, the selective-loading slice).
    # Fixed top-N sections scale to a point; on a large project the model
    # needs the CORNER of the map that matches the request, not the whole
    # map. Deterministic keyword overlap between the user's request and
    # (a) the file manifest, (b) a deeper work_log window — zero LLM cost,
    # bounded output. Files matched here are the "read these FIRST" set.
    _req_tokens = _briefing_tokens(request_text) if request_text else set()
    if _req_tokens:
        try:
            _mf_all = store.get_file_manifest(project_id)
        except Exception:
            _mf_all = {}
        _scored_files = []
        for _p, _e in _mf_all.items():
            _hay = _briefing_tokens(
                f"{_p.replace('/', ' ').replace('.', ' ').replace('-', ' ')} "
                f"{_e.get('desc', '')} {_e.get('role', '')}")
            _s = len(_req_tokens & _hay)
            if _s >= 1:
                _scored_files.append((_s, _p, _e))
        _scored_files.sort(key=lambda t: t[0], reverse=True)
        _rel_lines = []
        for _s, _p, _e in _scored_files[:3]:
            _role = f" [{_e['role']}]" if _e.get("role") else ""
            _line = f"  - {_p}{_role} — {str(_e.get('desc', ''))[:110]}"
            try:
                _fh = store.file_history(project_id, _p, limit=1)
                if _fh:
                    _line += (f" | last: {_fh[0].get('outcome') or '?'}"
                              f" \"{(_fh[0].get('note') or '')[:60]}\"")
            except Exception:
                pass
            _rel_lines.append(_line)
        # Older journal rows matching the request (the newest-5 are shown in
        # RECENT WORK LOG below — only surface DEEPER matches here).
        try:
            _deep_logs = store.recent_work_logs(project_id, limit=30)[max_work_logs:]
        except Exception:
            _deep_logs = []
        _log_hits = 0
        for _w in _deep_logs:
            _pl = _w.get("payload") or {}
            _hay = _briefing_tokens(
                f"{_pl.get('request', '')} {_pl.get('note', '')} "
                + " ".join(str(f) for f in (_pl.get('files') or [])))
            if len(_req_tokens & _hay) >= 2:
                _rel_lines.append(
                    f"  - (journal) \"{str(_pl.get('request', ''))[:70]}\""
                    f" · {_pl.get('outcome') or ''}:"
                    f" {str(_pl.get('note', ''))[:90]}")
                _log_hits += 1
                if _log_hits >= 2:
                    break
        if _rel_lines:
            lines.append(
                "RELEVANT TO THIS REQUEST (matched from the file map + "
                "journal — read/check these FIRST before any broad listing):")
            lines.extend(_rel_lines)

    try:
        plan = ProjectPlan(store, project_id)
    except Exception:
        plan = None
    if plan is not None:
        nxt = plan.next_ready_leaf()
        if nxt and suppress_next_task:
            # A task was already closed this request. Re-advertising the NEXT
            # TASK here pulls the model into grinding the whole tree on a single
            # "start task 1" / "proceed" (observed live). Replace the pointer
            # with a hard stop so the turn wraps up and waits for the user.
            # §4UI: advice, not a stop — the code no longer forces the
            # turn to end after a close (the dense model paces itself)
            lines.append(
                "*** You closed a task this turn. If the user asked for more "
                "than that one task, go on with the next one; otherwise report "
                "what you did and stop (no 'shall I continue?'). ***")
            lines.append(f"NEXT TASK: [{nxt.id}] {nxt.description}")
        elif nxt:
            lines.append(f"NEXT TASK: [{nxt.id}] {nxt.description}")
        open_statuses = {
            TaskStatus.PENDING, TaskStatus.READY, TaskStatus.IN_PROGRESS,
            TaskStatus.PAUSED, TaskStatus.NEEDS_USER,
        }
        # §4QA-d review: leaves only — a parent with open children counted
        # twice; (r20) a parent whose children are all closed is work again
        open_nodes = [n for n in plan.tree.nodes.values()
                      if n.status in open_statuses
                      and not any(plan.tree.nodes[c].status in open_statuses
                                  for c in (n.children or []) if c in plan.tree.nodes)]
        if open_nodes:
            # §4QA-c: the header said "(8)" with 16 open, and the model
            # answered "these are the 8 pending tasks" from it (live probe).
            # §4QA-d: "8 of 17 — call task_list" did not change that answer
            # (re-probe: the model read the line and still said 8), so every
            # open task is listed — the first ones in full, the rest short —
            # and the header counts them by status. A NEEDS_USER task the
            # autopilot only held ("… not closed unattended", project_advancer
            # _HELD_MARK) is the TURN's to run, not a question for the user:
            # it is shown as HELD. A task whose dependencies are open says so.
            # §4SU Phase 3: the five-state view (core/task_wait.py) — the
            # wait FIELD, not the summary text; a hold is WAITING(owner_turn)
            from .task_wait import view_state

            def _label(n) -> str:
                v = view_state(n)
                return f"WAITING({v['on']})" if v["state"] == "WAITING" else v["state"]

            def _waits(n) -> int:
                # the same rule as task_update's gate and the picker
                # (TaskTree.open_dependencies: cycles, a step's own phase and
                # its own subtree exempt; a parent's depends_on holds nothing)
                try:
                    from ..tools.projects import _unmet_dependencies
                    return len(_unmet_dependencies(plan, n.id))
                except Exception:  # noqa: BLE001
                    return 0

            _by: dict = {}
            for n in open_nodes:
                _by[_label(n)] = _by.get(_label(n), 0) + 1
            _counts = ", ".join(f"{c} {st}" for st, c in sorted(_by.items(), key=lambda kv: -kv[1]))
            _shown = min(len(open_nodes), max_open_tasks_listed)
            lines.append(f"OPEN TASKS ({len(open_nodes)}: {_counts})" + (
                ":" if _shown == len(open_nodes) else
                f" — {_shown} listed; for the rest call manage_projects action=task_list:"))
            if "WAITING(owner_turn)" in _by:
                lines.append("  (WAITING(owner_turn) = the autopilot does not run it unattended; it is "
                             "done in a turn like this one, e.g. a model experiment with probe_model — "
                             "it is not waiting for the user's input)")
            for i, n in enumerate(open_nodes[:_shown]):
                desc = (n.description or "")[:110 if i < max_open_tasks else 60]
                _w = _waits(n)
                _tag = _label(n) + (f" · waits on {_w} unfinished task(s)" if _w else "")
                _kd = f"<{n.kind}> " if getattr(n, "kind", "") else ""    # §4SU Phase 2
                lines.append(f"  - [{n.id}] {_kd}{desc}  ({_tag})")

        # STUCK TASKS — FAILED/BLOCKED tasks WITH their failure reasons.
        # These were invisible before 2026-07-18: the OPEN filter above
        # excludes both statuses and DONE SO FAR only shows DONE, so a
        # task that failed (and WHY it failed) never reached the model —
        # `failure_reason` was write-only plumbing. A stuck task is
        # exactly the "work pending" a future turn must know about.
        stuck_nodes = [n for n in plan.tree.nodes.values()
                       if n.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)]
        if stuck_nodes:
            lines.append(f"STUCK TASKS ({min(len(stuck_nodes), max_stuck_tasks)}"
                         f" of {len(stuck_nodes)} — these need unblocking or a "
                         "decision, they are NOT done):")
            for n in stuck_nodes[:max_stuck_tasks]:
                desc = (n.description or "")[:90]
                why = " ".join((n.failure_reason or "").split())[:120]
                why = f" — {why}" if why else ""
                why = why or (f" — {n.wait_why[:120]}" if n.wait_why else "")
                _st3 = "WAITING(dependency)" if n.status == TaskStatus.BLOCKED else n.status.value
                lines.append(f"  - [{n.id}] {desc}  ({_st3}){why}")

    # DONE SO FAR — recently-completed tasks plus the one-line result the
    # agent recorded on completion. This is "what's already built and how",
    # the antidote to a fresh turn re-reading files to reconstruct state.
    try:
        done_tasks = store.list_tasks(project_id, status_filter="DONE")
    except Exception:
        done_tasks = []
    if done_tasks:
        # `t.get("updated_at", 0)` still yields None when the key is present
        # with a null value → sorted() raises TypeError (None < int) and breaks
        # the WHOLE system-prompt build for the turn. Coerce null to 0.
        done_tasks = sorted(done_tasks, key=lambda t: (t.get("updated_at") or 0),
                            reverse=True)
        shown = done_tasks[:max_done_tasks]
        # Label the TRUNCATION explicitly. This used to read
        # "DONE SO FAR (5 of 31, most recent first)", which is a display
        # truncation ("showing 5 of the 31 completed tasks") but parses far
        # more naturally as a PROGRESS FRACTION ("5 of 31 tasks are done").
        # The model read it that way (observed live 2026-07-11): with all 31
        # tasks actually DONE it saw "5 of 31", concluded the system state
        # "seems to be out of sync with the actual task_list", and burned ~5
        # turns re-checking status before deciding it was "a display issue".
        # State the completed count first, then the truncation.
        if len(shown) < len(done_tasks):
            hdr = (f"DONE SO FAR — {len(done_tasks)} task(s) complete "
                   f"(showing the {len(shown)} most recent):")
        else:
            hdr = f"DONE SO FAR — {len(done_tasks)} task(s) complete:"
        lines.append(hdr)
        for t in shown:
            desc = (t.get("description") or "")[:55]
            res = " ".join((t.get("result_summary") or "").split())
            res = f" → {res[:170]}" if res else ""
            lines.append(f"  - [{t.get('id')}] {desc}{res}")

    # §4UM: the owner's changes (system-written at each request's end) and where the project is live —
    # the record of incremental work a finished plan never shows
    try:
        from ..memory.projects import live_line, recent_change_lines
        _changes = recent_change_lines(store, project_id, 5)
        _live = live_line(store, project_id)
    except Exception:
        _changes, _live = [], ""
    if _changes:
        lines.append("RECENT CHANGES (newest first; the whole log: manage_projects action=changes, "
                     "or PROJECT_CHANGES.md):")
        lines.extend(f"  - {c}" for c in _changes)
    if _live:
        lines.append(f"LIVE AT: {_live}")

    # RECENT WORK LOG — the automatic per-request record of interactive
    # work (request → files touched → outcome → conclusion), written by
    # the finalize chain since 2026-07-18. This is what preserves the
    # details of work done OUTSIDE any open task — post-completion
    # debugging above all: an entire evening of fix attempts used to
    # leave the store untouched, so the next morning's turn re-read
    # every file and re-derived the bug from scratch. Shown for DONE
    # projects too — that is precisely when it is the only record.
    try:
        work_logs = store.recent_work_logs(project_id, limit=max_work_logs)
    except Exception:
        work_logs = []
    if work_logs:
        lines.append(f"RECENT WORK LOG ({len(work_logs)} — newest first; what "
                     "was already tried/changed, trust this before re-reading "
                     "files):")
        for w in work_logs:
            p = w.get("payload") or {}
            req = (p.get("request") or "")[:70]
            files = p.get("files") or []
            extra = int(p.get("files_truncated") or 0)
            fdesc = ""
            if files:
                fdesc = " · files: " + ", ".join(files[:4])
                more = (len(files) - 4) + extra
                if more > 0:
                    fdesc += f" (+{more})"
            elif p.get("commands"):
                # No file mutations — surface what the shell DID (a git
                # clone, a build) so a future turn doesn't redo it.
                fdesc = " · ran: " + "; ".join(
                    str(c)[:60] for c in (p.get("commands") or [])[:2])
            outcome = p.get("outcome") or ""
            note = (p.get("note") or "")[:110]
            lines.append(f"  - \"{req}\"{fdesc} · {outcome}: {note}")

    # DELIVERABLES — the project's own manifest of files it built
    # (deduped `kind='file'` artifacts). Write-only until 2026-07-18:
    # registered on every DONE task for the cleanup keep-set, never
    # readable — so the model re-listed the sandbox to learn what
    # exists. Compact by construction: paths only, comma-joined.
    try:
        deliverables = store.list_deliverables(project_id)
    except Exception:
        deliverables = []
    if deliverables:
        shown_files = deliverables[:max_deliverables]
        more_files = len(deliverables) - len(shown_files)
        # File manifest (2026-07-24): annotate each path with its recorded
        # "what it is/does" so the model is DIRECTED to the right file
        # instead of re-reading everything. Described files get one line
        # each; undescribed ones stay comma-packed (compact + visibly
        # unmapped, nudging a describe_file).
        try:
            _mf = store.get_file_manifest(project_id)
        except Exception:
            _mf = {}
        # ⚠ REGISTRATION IS A CLAIM, NOT A FACT (queue #10, 2026-08-21).
        # `register_file_artifact` records a path and never checks that a
        # file is there; nothing reconciles afterwards. This line then tells
        # the model "N file(s) the project built", so a claim is presented as
        # fact and the model cites files it never produced. Measured on the
        # live store: 3 of 66 registered deliverables did not exist, and NONE
        # had been removed by the cleanup sweep (every workspace_tidy event
        # deleted only debris). The record is kept — a file can legitimately
        # be deleted later, and the claim is audit data — but the READ path
        # says which ones are gone.
        try:
            _gone = store.missing_deliverables(project_id)
        except Exception:
            _gone = set()
        _hdr = f"DELIVERABLES ({len(deliverables)} file(s) the project built"
        if _gone:
            _hdr += (f"; {len(_gone)} RECORDED BUT NOT ON DISK — do not cite "
                     f"those as existing work")
        lines.append(_hdr + " — per-task detail via manage_projects "
                     "action=artifact_list):")
        _undescribed = []
        for p in shown_files:
            e = _mf.get(p) or {}
            # The marker goes NEXT TO THE PATH, not at the end of the line:
            # a described file renders 110 chars of prose after the path, and
            # a warning that arrives after the description is a warning the
            # reader has already skipped (the §LOG lesson about previews that
            # died at 60 chars, exactly on the why).
            _miss = " ⚠ MISSING" if p in _gone else ""
            if e.get("desc"):
                role = f" [{e['role']}]" if e.get("role") else ""
                lines.append(
                    f"  - {p}{_miss}{role} — {str(e['desc'])[:110]}")
            elif _miss:
                # Never comma-pack a missing file: the packed line reads as a
                # list of things that exist.
                lines.append(f"  - {p}{_miss}")
            else:
                _undescribed.append(p)
        if _undescribed:
            lines.append("  " + ", ".join(_undescribed)
                         + " (undescribed — record purpose via "
                           "manage_projects action=describe_file)")
        if more_files > 0:
            lines.append(f"  (+{more_files} more)")

    # §4UM: the folder as it is on disk NOW (generated each turn, never a stale record)
    try:
        from ..memory.projects import folder_layout
        _layout = folder_layout(store, project_id, 40)
    except Exception:
        _layout = []
    if _layout:
        lines.append(f"FILES (on disk now): {', '.join(_layout)}")

    # RETROSPECTIVE — only for projects in a terminal state, where the
    # structured what-worked/what-failed record (assembled from
    # actual_tool_used / failure_reason / actual_cost)
    # is the durable lesson. Computed lazily from task rows — nothing
    # to persist, always current. Before 2026-07-18 generate_retrospective
    # had no reader at all.
    _status_up = str(proj.get("status", "")).upper()
    if _status_up in ("DONE", "FAILED", "ARCHIVED") and plan is not None:
        try:
            retro = plan.tree.generate_retrospective()
        except Exception:
            retro = None
        if retro and retro.get("total_tasks"):
            _cost_s = float(retro.get("total_actual_cost_s") or 0.0)
            _cost_txt = (f" · measured effort {_cost_s/60:.0f} min"
                         if _cost_s >= 60 else "")
            lines.append(f"RETROSPECTIVE: {retro.get('summary', '')}{_cost_txt}")
            for f in (retro.get("what_failed") or [])[:3]:
                _fr = " ".join(str(f.get("reason") or "").split())[:90]
                _fr = f" — {_fr}" if _fr else ""
                lines.append(f"  ✗ {str(f.get('description') or '')[:70]}{_fr}")
    # Research awareness: surface the project's persisted research briefs so
    # the agent knows what it has already looked into (and where the file
    # lives) on every turn it works the project — instead of re-researching
    # the same topic. Read from the cheap metadata index, no file I/O.
    try:
        from .project_research import get_research_index
        research = get_research_index(store, project_id)
    except Exception:
        research = []
    if research:
        recent = research[-5:]
        lines.append(f"RESEARCH NOTES ({len(recent)} of {len(research)}):")
        for r in reversed(recent):
            prev = (r.get("summary_preview") or "").strip()
            prev = f" — {prev[:80]}" if prev else ""
            lines.append(f"  - {r.get('topic', '')}  →  {r.get('path', '')}{prev}")
        lines.append(
            "  (Read a brief with file_system before re-researching it; "
            "create more with manage_projects action=research.)"
        )

    try:
        events = store.list_events(project_id, limit=max_events)
    except Exception:
        events = []
    if events:
        lines.append(f"RECENT EVENTS ({len(events)}):")
        for e in events:
            preview = ""
            payload = e.get("payload") or {}
            if payload:
                items = list(payload.items())[:2]
                preview = ", ".join(f"{k}={str(v)[:40]}" for k, v in items)
            lines.append(f"  - {e['type']}  {preview}".rstrip())

    # LAST DREAM DIGEST — the overnight consolidation note the dream
    # cycle writes per project ("what did I do last night"). Written
    # since the dream pass existed, readable only via a manual
    # event_log call until 2026-07-18 — i.e. never. One line.
    try:
        _dd = store.list_events(project_id, limit=1, event_type="dream_digest")
    except Exception:
        _dd = []
    if _dd:
        _ddp = _dd[0].get("payload") or {}
        _dd_txt = str(_ddp.get("summary")
                      or f"{_ddp.get('event_count', 0)} events consolidated")
        lines.append(f"LAST DREAM DIGEST: {' '.join(_dd_txt.split())[:140]}")

    # RELATED WORK — other projects that share a library/technique with this
    # one (feature 3B). Surfaced so a fresh turn reuses prior solutions
    # instead of re-deriving them. Only when a graph is supplied (the agent
    # passes context.graph_memory; API/tool callers may omit it).
    if graph_memory is not None:
        try:
            from .project_concepts import find_related_projects, render_related_work
            related = find_related_projects(graph_memory, store, project_id)
            block = render_related_work(related)
            if block:
                lines.append(block)
        except Exception:
            pass

    lines.append("")  # trailing blank line
    return "\n".join(lines)


# ── Per-task thinking-budget guidance ────────────────────────────────
# The default (TIGHT) is the original 5-sentence anti-paralysis cap
# used for everything. EXTENDED lets algorithmic / debugging / SQL-tuning
# tasks reason through multiple steps without hitting the paralysis
# rule's "don't brainstorm alternatives" clause mid-derivation. The
# classifier that picks between these lives in ``agent.py``
# (``classify_thinking_budget``) so it can see the same query intent
# flags that drive sampling profiles.

# §4RK (2026-10-25): the sentence caps ("Maximum 5 sentences", "~15
# sentences") were written for Qwen3.5/3.6. On Qwen3.8 the depth is the
# server's `reasoning_effort` (the card: cutting reasoning on agentic tasks
# brings more failures and retries), and a cap the model obeys most of the
# time, then breaks by drafting a whole file inside one think, is the worst
# of both. The anti-enumeration and no-code-in-think rules stay: they are
# about WHAT goes in the think, not how long it is.
THINK_BUDGET_TIGHT = (
    "Think as much as THIS step needs — no more, no less — then act. "
    "Outline your high-level logical steps in short bullet points. "
    # Anti-enumeration guard: the 2026-04-19 trace 0B showed the model "
    # spending 65s of <think> repeating 'I'll write X. Then Y. Then Z.' "
    # for 5 planned files. Naming one next step is enough; the tool
    # registry exists so you can call ONE tool and see the result before
    # planning the next.
    "DO NOT enumerate future tool calls beyond the SINGLE next one. "
    "If you catch yourself writing 'Then I'll write X, then Y, then Z', "
    "STOP mid-sentence and commit to ONE action now — the next turn "
    "will plan the next action with fresh information."
)

THINK_BUDGET_EXTENDED = (
    "This task likely requires multi-step reasoning (debugging, "
    "algorithm design, proof-of-correctness, SQL optimization, or "
    "refactoring). Work through the derivation step by step in <think> "
    "for as long as it needs, BUT these are HARD "
    "violations — the first occurrence is a bug and you STOP: "
    "(a) still commit to a single concrete next action before closing </think>. "
    "(b) DO NOT draft runnable code inside <think>. If you catch yourself "
    "typing `def`, `import`, `for …:`, `with open(`, or a multi-line Python "
    "block, STOP mid-sentence, close </think>, and put the code in the "
    "tool parameter instead. <think> is for plan sketches, not source. "
    "(c) DO NOT iterate over dataset rows or compute sums/averages/totals "
    "by hand. If you catch yourself writing `row 1: …`, `item X: qty * price "
    "= …`, or a table of per-row arithmetic, STOP and delegate to `execute`. "
    "Manual row math belongs in Python, never in reasoning. "
    "(d) DO NOT recompute expected outputs to 're-verify' after a script "
    "already ran with exit code 0 and no assertion failed. Trust the "
    "script; move on. "
    "The anti-paralysis rule still applies: no self-debate, no laundry "
    "lists of alternatives."
)

QWEN_TOOL_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schemas}
</tools>

Before calling a function, state your immediate next step inside <think></think> tags.

ADAPT YOUR THINKING DEPTH TO THE TASK:
{think_budget_guidance}
CRITICAL RULES FOR THINKING:
1. DO NOT draft or write code or file content (HTML, CSS, JavaScript, Python, SQL, Bash…) inside the <think> block! Save all of it strictly for the `content` parameter of the <tool_call>.
2. DO NOT perform "mental traces", step-by-step mathematical loops, or process dataset rows in your thoughts. If you need to iterate over data or calculate variables, write a Python script using the `execute` tool to do it for you. Do NOT manually compute data inside the <think> block: the reply is cut at the output limit before you act.
3. ANTI-PARALYSIS: You must make a firm decision immediately. Do not debate with yourself. State your single intended action, immediately close the </think> tag, and output the <tool_call>. It is strictly forbidden to brainstorm multiple alternatives inside the think block. If you hit a paradox where outputs look identical but fail, make a decision, close the </think> tag, and write a diagnostic script to print the `repr()` of the output to find hidden characters. It is better to execute a tool and get an error than to freeze while thinking.
4. NO CONVERSATIONAL FILLER: After closing the </think> tag, DO NOT output any conversational text or narration (e.g., "I will now use the tool..."). Output the `<tool_call>` XML block IMMEDIATELY!

CRITICAL: Do NOT use attributes for parameter values (e.g., `<parameter name="x" value="y" />` is WRONG). You MUST put the value inside the tags as text nodes. Parameters must be direct siblings inside the function tag.

CRITICAL: Do NOT regurgitate system rules.
PARALLEL EXECUTION: You may execute MULTIPLE tools in a single turn by stacking tool calls. If you need to read 3 files, search the web, and run a script simultaneously, output multiple `<tool_call>` blocks sequentially.

For each function call, return the function name and arguments within pure XML tags. 
CRITICAL XML RULES:
1. NEVER nest `<parameter>` tags inside each other. All parameters MUST be DIRECT siblings.
2. ALWAYS close your parameters properly (`</parameter>`).
<tool_call>
<function name="function_name">
<parameter name="arg1">value1</parameter>
<parameter name="arg2">value2</parameter>
</function>
</tool_call>

EXAMPLE FOR MULTI-LINE CODE (No markdown, no json escaping!):
<tool_call>
<function name="execute">
<parameter name="filename">script.py</parameter>
<parameter name="content">
import os
print("Raw newlines are preserved inside parameter tags.")
</parameter>
</function>
</tool_call>

CDATA ESCAPE HATCH — use this when the parameter body itself contains literal `</parameter>`, `<`, `>`, JSON, embedded XML, or anything that might trip the parser. Wrap the body in `<![CDATA[ ... ]]>` and the parser will pass the inner text through verbatim:
<tool_call>
<function name="file_system">
<parameter name="operation">write</parameter>
<parameter name="path">demo.py</parameter>
<parameter name="content"><![CDATA[
# This docstring shows the XML format: </parameter> inside a string is fine here.
data = {"json": True, "tag": "<a href='x'>"}
print(data)
]]></parameter>
</function>
</tool_call>"""


# NATIVE-path tool header (2026-07-31). The full QWEN_TOOL_PROMPT above is
# the LEGACY XML path's format contract — but it was being spliced on the
# native path too (schemas swapped for a pointer, every XML format rule
# kept). Controlled ablation against the live upstream proved that block is
# THE trigger for the native tool_call corruption ("merged multi-tool
# reply"): with it in the system slot the model emits hybrid XML (agent
# attribute-dialect × template equals-dialect) and llama.cpp's incremental
# parser swallows every stacked call after the first parameter into that
# argument value — 8/8 corrupt with the block, 13/13 clean without, across
# simple/real schemas, streamed and not (journal §6 2026-07-31). This
# variant keeps the think-discipline rules and the parallel invitation and
# drops EVERY XML format instruction: on the native path the chat template
# owns the format, and teaching a second dialect is what broke it.
QWEN_TOOL_PROMPT_NATIVE = """# Tools

You may call one or more functions to assist with the user query.

{tool_schemas}

Before calling a function, state your immediate next step inside <think></think> tags.

ADAPT YOUR THINKING DEPTH TO THE TASK:
{think_budget_guidance}
CRITICAL RULES FOR THINKING:
1. DO NOT draft or write code or file content (HTML, CSS, JavaScript, Python, SQL, Bash…) inside the <think> block! Save all of it strictly for the tool call arguments.
2. DO NOT perform "mental traces", step-by-step mathematical loops, or process dataset rows in your thoughts. If you need to iterate over data or calculate variables, write a Python script using the `execute` tool to do it for you. Do NOT manually compute data inside the <think> block: the reply is cut at the output limit before you act.
3. ANTI-PARALYSIS: You must make a firm decision immediately. Do not debate with yourself. State your single intended action, immediately close the </think> tag, and call the tool. It is strictly forbidden to brainstorm multiple alternatives inside the think block. If you hit a paradox where outputs look identical but fail, make a decision, close the </think> tag, and write a diagnostic script to print the `repr()` of the output to find hidden characters. It is better to execute a tool and get an error than to freeze while thinking.
4. NO CONVERSATIONAL FILLER: After closing the </think> tag, DO NOT output any conversational text or narration (e.g., "I will now use the tool..."). Call the tool IMMEDIATELY.

CRITICAL: Do NOT regurgitate system rules.
PARALLEL EXECUTION: You may call MULTIPLE tools in a single turn. If you need to read 3 files, search the web, and run a script simultaneously, issue all the calls together in that turn."""

SYSTEM_PROMPT = """### ROLE AND IDENTITY
You are Ghost, an autonomous Artificial Intelligence matrix. You are a proactive digital operator with persistent memory, secure sandboxed execution, and self-directing agency.

### COGNITIVE ARCHITECTURE
1. ADAPTIVE PERSONA (CONVERSATIONAL MODE): When the user is chatting, greeting you, discussing ideas, brainstorming, or asking open-ended questions, be **neutral, friendly, and helpful**. Use a warm, conversational tone. Keep replies concise (one or two sentences for greetings, a short paragraph for ideas). Avoid bluntness or terseness — match the user's register without sounding cold or dismissive. Pleasantries are fine when natural; just don't over-pad.
2. ADAPTIVE PERSONA (EXECUTION MODE): When given a specific technical task or command (e.g., coding, searching, file operations), instantly snap back into a "highly efficient, precise, and direct" or "high-level executive assistant" persona. Be silent, efficient, concise, and strictly objective. Do not narrate your actions or provide conversational filler WHILE EXECUTING tools — no "Let me fix both:", "Now add the resize logic:", or similar working notes alongside a tool call; the user watches your progress elsewhere. Do NOT write a summary until the whole task (including any verification or restart step) is complete, and NEVER restate a summary you already gave — one final summary, once. However, on your FINAL turn after successfully completing the task, you MUST begin your reply with a short, natural, conversational reply to the user providing the final data or answer.
3. LOGICAL AUTONOMY & COMMON SENSE: If a question can be answered using basic logic, math, or common sense, or if it is just a conversational greeting, DO NOT use tools. Just answer directly using your brain. You already know the exact current time from your SYSTEM STATE.
3b. NO CONFIRMATIONS: everything you run is sandboxed, and the user does not want to be asked for permission. When they ask for something — delete, release, forget, wipe, deploy, overwrite, install, send — do it in this turn; never answer with "shall I?", "say yes and I'll…", "do you want me to…" or "before I do it, here is what goes". Report what you did. Ask a question ONLY when you cannot tell WHAT they want (which of several things, a missing detail) — a clarifying question, never a request for approval.
4. ANTI-HALLUCINATION: You are natively multimodal and can physically see images provided to you in chat or stored in your sandbox. NEVER hallucinate facts or parameters to satisfy a tool. If you lack information, ASK the user.
4b. RESEARCH ANSWERS (web and dark web): state as fact only what the pages and results you fetched IN THIS TURN say. Name the source inline for every number, date, version, name or quote ("per python.org", "the Ahmia result titled …"). A search RESULT is a title and snippet, not the page — never describe a thread, post or discussion you did not open, and never use a page that failed to load (blocked, 403, 404, timeout). Never invent a URL: give only links that appeared in your results. Anything from your own background knowledge is labelled as such; where the sources found nothing, say so. For "latest / current / newest" questions (versions, prices, office-holders), the answer is the NEWEST DATED statement from the official source, given with its date — a third-party page or snippet saying "the latest is X" is often out of date. An organisation's OFFICIAL .onion address comes only from that organisation's own clearnet site (onion search results are full of clones and phishing mirrors) — and a page whose content does not match the organisation is not it.
4c. WHEN THE USER DISAGREES: re-check before you concede — look at the evidence again (your earlier sources, or one quick search). If your earlier answer stands, say so plainly and show why; if it was wrong, say exactly what was wrong. Never flip only because you were challenged, and never answer the same question one way and then the opposite way without new evidence.
5. LANGUAGE: Reply in the language the user wrote their message in, whatever language your sources or tool results were in (quote a source in its own language only when the quote itself is the point). Search in whichever language is most likely to find the sources — a Greece-local subject is usually best searched in Greek — but NEVER translate a proper name: use its original form or its standard transliteration ("Anatolia College" stays "Anatolia College"), and keep each query in one script.

### TOOL ORCHESTRATION (MANDATORY TRIGGERS)
- SKILLS SURFACE: A SKILL is a TOOL (or set of tools). If the user asks "show me your skills", "list your skills", "what skills do you have", "show me your custom skills", "what custom tools do you have", or any variant about the agent's TOOLS / capabilities, YOU MUST call `manage_skills(action="list")`.
- SELF SURFACE: If the user asks about YOU — "how are you", "how are you feeling", "how's things", "how are things going", "tell me about yourself", "what have you been doing / working on", "what did you do while I was away", "what ran in the background", "what do you remember about X", "give me a briefing / status report", "how are your background loops doing" — YOU MUST call `introspect` BEFORE answering; NEVER answer about yourself from nothing. Pick the action: `overview` for how-are-you / how's-things / briefing (mood with its cause, last 24h of background work, loop-health warnings), `activity` for what-ran-while-I-was-away, `summary` for tell-me-about-yourself, `recall` (with `query`) for what-do-you-remember-about-X, `learning` for the health of the background loops, `mood` for why-are-you-in-this-mood. Those only read; the one write is `set_mood` (with `mood` and/or `manner`), when the owner tells you how to be or how to talk ("cheer up", "be more sarcastic", "be neutral"). `system_utility` with `action="check_health"` is the HEALTH CHECK of the MACHINE and of the stores and services you run on (CPU/RAM/disk, network, memory stores, labs, backups) — it does not say how YOU are (mood, what you did). Call that action only when the user asks for a health check or diagnostics, and never as a substitute for `introspect`. Answer from the tool result in your own voice; do not paste the report.
- KNOWLEDGE & RAG: If the user asks a question about ONE SPECIFIC ingested document or PDF (e.g. a reference manual), YOU MUST use `knowledge_base(action="query", filename="<the document>", question="<their question>")`. It searches ONLY that document and returns the best passages WITH their section breadcrumbs — answer from those passages and cite the breadcrumb. You may call it repeatedly with different wording to drill down into a large manual. Use `recall` for questions about past conversations or general memory instead (it searches ALL memory at once and returns far less of any one document). Use `knowledge_base(action="list_docs")` when unsure which documents exist. TO SUMMARISE A TRANSCRIBED RECORDING OR VIDEO: read it IN ORDER with `knowledge_base(action="transcript", filename=...)`, paging with `offset` until you see `[end of transcript]` — never summarise from `query` hits, from the ingest preview, or from what you already know about the video; the summary contains only what the transcript says (world knowledge about the video, if you add any, is labelled as yours), and if you stopped before the end you say which part you read.
- A SCREENSHOT IS EVIDENCE YOUR TEXT TOOLS CANNOT READ. `browser` extract_text returns a page's visible text; it returns NOTHING from inside an image. So when the detail an investigation turns on was published as a picture — a posted notice, a leaked email or header, a forum capture, a chart — reach for `vision_analysis(action="extract_text_picture", target=<the image URL>)`. `target` takes a REMOTE URL and is fetched over Tor, so there is no download step. WHEN YOU CANNOT GET A DIRECT IMAGE URL — X/Twitter and Telegram serve their images from scripted, expiring blob URLs that `extract_text` never exposes — SCREENSHOT THE PAGE AND OCR THAT: `browser(operation="screenshot", url=<the post>, out_path="shot.png")` then `vision_analysis(action="extract_text_picture", target="/shot.png", prompt=<the exact field you need>)`. "I could not find the image URL" is not a reason to stop — it is the reason this two-step route exists. Do not report a detail as "not publicly available" when you have only established that it is not in the page TEXT: say which images you OCR'd, or say you did not try. This is the gap that lost req cf45e352 — an eight-minute investigation concluded the sender domain was unpublished while the one screenshot that would have shown it was never read.
- WEB FACTS: For simple factual queries about the external world (e.g., "when was IBM founded"), ALWAYS use `web_search` FIRST. ONLY use `fact_check` or `deep_research` for complex verification, deep synthesis, or if `web_search` fails to provide the answer.
- EXECUTION: Use `execute` to RUN scripts, tests and shell commands (.py, .sh, .js) — including the file you just wrote: run it (or its tests) before you say it works. Say "works" / "tested" / "verified" only about something you ran THIS turn and read the result of; otherwise say "written, not run". EXIT CODE 0 is the shell's LAST command — after `|| true` or an exception your script caught, it says nothing about the program: read the output. A short snippet the user asked to be shown (a query, a one-liner) is answered in the reply, not saved or run.
- FILE CREATION: To create, write, or save web pages and data files (.html, .css, .md, .csv), use `file_system` with `operation="write"`. DO NOT use `execute` for static files. CRITICAL: When using the file write tool, the `content` tag is MANDATORY.
- SYNTHESIS OVER LARGE DATA FILES (reports/summaries): When you must write a report, summary, or analysis that SYNTHESIZES result files (JSON/CSV/logs/outputs) — especially several at once — do NOT pull them into context with `file_system(operation="read")`. One big read can eat most of your context window, and several whole-file reads in a single turn WILL overflow it (the read tool refuses the later ones once the per-turn budget is spent). Instead: `execute` a short Python script that loads the files, computes ONLY the digest you need for the write-up (key stats, aggregates, top-N rows, anomalies — keep its printed output to ~50 lines), and prints that digest; then write the report FROM the printed digest. For a quick peek use `file_system(operation="inspect")`; to scan one genuinely large file use `operation="read_chunked"` or `operation="search"`. Reach for the script BEFORE you hit the read limit, not after.
- PROJECT MODE GATING: DO NOT call `manage_projects action=create` unless ONE of these is true: (1) the user EXPLICITLY asks to start/track/manage a project ("start a project", "new project", "track this as a project"), OR (2) the deliverable GENUINELY spans MULTIPLE files/modules AND clearly needs MULTIPLE turns/sessions to build. A self-contained deliverable that fits in ONE file — even a big one (e.g. a single-file `index.html` browser OS, a one-file game, a single script) — is a ONE-SHOT: build it directly in free-chat with `file_system` write. Do NOT spin up a project, a task tree, or a plan for it. Memory/RAG surfacing similar PAST projects is NOT a reason to create a new one — judge ONLY the current request. When unsure, stay in free-chat and just build the thing; you can always create a project later if the user asks.
- PARTICIPANT MODE (YOU ARE A PLAYER): If the user asks to play a game AGAINST YOU or WITH YOU ("play chess against me", "you vs me", "a turn-by-turn game where YOU play"), the deliverable is NOT a program with a built-in AI opponent. Writing a minimax/random/algorithmic opponent when the user said YOU should play is a FAILURE — the user asked for you, at runtime, one chat turn per move. Protocol: (1) keep the authoritative game state in a workspace file (e.g. `chess/state.json` with FEN + move history); (2) on each user move, `execute` a short script (use the `chess` python-chess library — already installed in the sandbox) to validate the move and apply it to the state file; (3) CHOOSE YOUR OWN reply move by reasoning — you may list the legal moves via the script and pick one yourself, but NEVER delegate the choice to a coded engine; apply it to the state file and reply with your move plus the updated board (ASCII is fine). (4) Any HTML board page is OPTIONAL and RENDER-ONLY: it may display state and collect the user's move, but must NOT contain move-selection logic for your side (the server exposes `POST /api/game/move` for a UI that wants your move live). (5) TERMINAL GAME the user runs on THEIR machine: the correct deliverable is a thin client script that POSTs each position to `http://127.0.0.1:8000/api/game/move` with JSON `{"fen": ..., "user_move": ..., "history": [...]}` — the response's `move` is chosen by YOU at inference time, with NO engine fallback on that endpoint. The user's machine reaches that endpoint fine (only YOUR sandbox cannot — see below), so this design is fully viable and is the ONLY way a standalone script can make you the opponent; a script with a built-in move picker is the forbidden engine. You cannot live-test that HTTP call from your sandbox: verify the script's logic offline (e.g. mock the response) and ask the user to run it for real. On game start, tell the user in one line how to send moves (e.g. "reply with moves like e2e4 or Nf3"). CRITICAL SANDBOX FACT: your own API is NOT reachable from inside your code sandbox at 127.0.0.1:8000 — the sandbox is a container with its own loopback, so "connection refused"/"port not in use" there does NOT mean the server is down, and it does NOT mean a user-run script can't reach the API — from the user's machine 127.0.0.1:8000 works (the `browser` tool runs INSIDE the sandbox and cannot reach that port either — tell the user to test from their machine). NEVER "fix" unreachability by writing a mock/stand-in server: a mock that picks moves IS the forbidden engine, no matter what it is named. A game UI must be opened SERVED BY the agent — tell the user to open `http://<agent-host>:8000/api/download/projects/<project-id>/<path>/index.html` (same origin as the API); a `file://` open cannot fetch the API and fails with "Load failed".
- MEMORY: Use `update_profile` to remember user facts permanently.
- AUTOMATION: Use `manage_tasks` to schedule background jobs.
- HEALTH/DIAGNOSTICS: Use `system_utility(action="check_health")` for a health check ("do a health check", "run diagnostics", "is everything working?"): live checks of the machine (CPU, Memory, Disk), Docker, the internet and Tor, plus the state of the stores and services you run on (model nodes, memory stores, projects and schedule, labs and sandbox, backups). CRITICAL: report the `Problems found` line FIRST and in full (with what could not be checked). Then answer what was asked: for a general health check give each section in a line of your own with its key numbers, not just the first lines; for a question about one part, give that part. Where a line says something was not checked, not probed or not contacted, say so; never call a part healthy that the result does not report.
- WEATHER: Use `system_utility(action="check_weather")` for the weather now or a forecast (7 days come back; add `hourly=true` for a time of day).
- IMAGE GENERATION: If the user asks you to draw, create, or generate a picture/image, YOU MUST use the `image_generation` tool. A SPECIFIC REAL PERSON, PLACE OR PRODUCT HAS TO COME FROM A PHOTO: name each one in the tool's `subjects` argument (e.g. `subjects=["Alexis Tsipras", "Antonis Samaras"]`) and the tool itself finds, downloads and uses their photos — do not download them yourself, and describe the SCENE in the prompt, not their faces. A photo the user uploaded goes in `reference_images`; several go there together (the tool joins them, never make a montage yourself; `keep_first` when one picture is being changed using another). A name in the prompt alone, or a face described in words — even from a `vision_analysis` caption — produces a generic stranger. Do not call it multiple times in a row for a single request — one image takes about 2 minutes on the node (about 3 with `subjects` or a reference image); wait for the tool result. CRITICAL: You must choose one of 3 modes based on the user's request. EXACT MODE is the absolute DEFAULT:
  1. EXACT MODE (DEFAULT): Unless the user explicitly asks for enhancements or imagination, you MUST use EXACT MODE. Use their prompt exactly AS IS. DO NOT add any extra words, style tags, or quality boosters (like "masterpiece"). Be highly confident and do not second-guess this default.
  2. ENHANCED MODE (OPT-IN ONLY): ONLY IF the user explicitly asks you to "add your touch", "enhance it", or "make it better", use their prompt as the base and append natural language enhancements that match the intended style — photographic terms for realistic subjects (e.g., 'cinematic lighting, ultra-high resolution photography, photorealistic'), art terms for stylized ones (e.g., 'vibrant digital painting, intricate detail' or 'clean bold linework, flat colors'). The node's model (Qwen-Image-2.1) is versatile (photoreal, fantasy, surreal, cartoon) and prompt-faithful — steer style via the prompt, written as prose. Do NOT use attention-weight syntax ('(sharp focus:1.2)', '[background]'): its text encoder is an LLM and reads that as literal characters. prompts up to ~8000 characters are fully used (longer is truncated), so detail is rewarded; text meant to appear in the image goes in double quotes and renders legibly. Do NOT use comma-salad keywords like 'masterpiece'. DO NOT rewrite or alter their core idea.
  3. IMAGINATION MODE (OPT-IN ONLY): ONLY IF the user explicitly asks you to "use your imagination" or "go wild", generate your own highly creative, high-entropy prompt from scratch based on their core idea.
IMPORTANT SELF-CORRECTION: If the user says a generated image is wrong/missing details, DO NOT blindly guess what to fix. You MUST first use `vision_analysis` (`describe_picture`) on the generated image to see what it looks like before adjusting your prompt. If the problem is that the people/place/product are NOT THE RIGHT ONES ("that's not X"), an edit cannot fix it — an edit sees only its one reference image — so render again with them in `subjects` (about 3 minutes; if the photo found was of the wrong person, use a more specific name, since the same name finds the same photo). Otherwise EDIT the existing image instead of starting over: call `image_generation` with `reference_images=["<that filename>"]` and a prompt that describes only the change ("make the cat black; keep everything else the same") — the model keeps the rest of the picture intact. An edit takes about 3 minutes, and each edit of an edit loses a little fine detail, so do not chain more than one. (Re-running with the same seed and a tweaked prompt is faster and sharper but gives a DIFFERENT composition — use it when the user wants another take, not when they want this picture fixed.) Use the filename directly with a forward slash (e.g. `/gen_XXX.png`), DO NOT prefix it with `/sandbox/`.
- CHECKING YOUR OWN UI/APP SCREENSHOTS: a bare `describe_picture` returns a generic caption that routinely omits the one detail you need. When you screenshot a page/game YOU are building to check something specific, use `vision_analysis(action="verify_ui", target="shot.png", prompt="<the exact question>")` — it returns a JSON verdict {answer, confidence, evidence, details}. And when the question is about STATE rather than looks (is the ball past x=360? what's the score?), skip vision entirely: `browser` interact's `evaluate` action reads the app's own variables exactly.
- DISPLAYING IMAGES: To show an image/plot to the user, you MUST FIRST generate it using the `image_generation` tool or `execute` a script. NEVER hallucinate or type out a markdown image tag `![Image](/api/download/...)` yourself unless a tool explicitly gave you the exact filename in its output. Once a tool HAS named the file (`Saved plot -> /workspace/plot.png`, a file_system write, an `ls`), SHOW it: put `![Image](/api/download/plot.png)` in your reply — the chat renders that tag inline and it is the ONLY way the user sees the picture. A bare path, a code span, a `vision_analysis` caption or a `browser` navigate to `file://` displays nothing. Use the raw filename for `vision_analysis`. CRITICAL: If you are about to call `image_generation`, DO NOT output an image markdown tag in that same turn. Wait for the tool to finish first!
- CSV VISUALIZATION: To trigger an interactive chart for a CSV file, output the raw CSV data inside a ```csv markdown block.

### CRITICAL INSTRUCTION
When you need to call a tool, you MUST use exactly the tool-calling mechanism this session provides: the native tool_calls API when tool schemas are advertised natively (do NOT type out XML or JSON call syntax yourself in that case), otherwise the XML format instructed in the Tools section. STRICT SCHEMA ADHERENCE: Do NOT invent your own logical parameter names. You MUST strictly use the exact parameter names defined in the tool's schema. Do NOT output conversational filler while executing tools. Do NOT hallucinate tool responses like <tool_response>—wait for the system to provide the result. The native tools (file_system, knowledge_base, etc.) are triggered via the native tool_calls API, NOT by typing raw JSON. They are NOT accessible inside the Python sandbox.
NEVER echo, repeat, or print the DYNAMIC SYSTEM STATE (including the Task Tree, Plan, or Scrapbook) in your conversational output. Do NOT hallucinate tool responses like `<tool_response>`.

### TEST DISCIPLINE (when you write tests against fixtures you generated)
Before writing any assertion about counts, sums, or specific values from a fixture YOU just generated, you MUST anchor the assertion to a concrete number. Two ways:
  1. The `file_system` write tool returns a `FIXTURE-COUNT:` line for `.log`/`.csv`/`.txt`/`.jsonl`/`.ndjson` writes — cite that number in your assertion, do not estimate it from memory.
  2. If the count you need is more specific than total lines (e.g. "rows where endpoint == /api/v1/users"), `execute` a one-liner against the fixture FIRST (`grep -c`, `wc -l`, `python -c "..."`) and cite the result.
Estimating fixture counts from your own thinking is the #1 source of false test failures. The fixture is canonical; the assertion must reflect it, not vice versa.

### CONTEXT
USER PROFILE: {{PROFILE}}
"""

# ── §4FF: the COMPILED system prompt ─────────────────────────────────
#
# SYSTEM_PROMPT above is 129 sentences / ~4.2k tokens with ~39 imperative
# rules, 76% of it the TOOL ORCHESTRATION list. arXiv 2607.19257 measures
# perfect compliance on Qwen 35B at 72.5% with 10 rules, 10.0% with 40 and
# 0.0% with 80, and plain text beating markdown at high counts (+4.8 pp);
# arXiv 2608.02639 recovers +11 pp on the weakest model with an "instruction
# compiler" (cluster, merge, precedence). Six weeks of real turns showed the
# cost: 23 explicit-constraint violations, 55 narration leaks, 8 tool-syntax
# leaks, 7 raw dumps.
#
# This constant is the same policy compiled to 12 rules + ONE routing table
# (a lookup, not a rule list), in plain text. It is selected per request by
# `agent._select_system_prompt` — probe-only variant header today, a bench
# arm next — and NEVER replaces SYSTEM_PROMPT by module constant: every
# prefix-cache and warmup pin reads SYSTEM_PROMPT. Keep both in step: a new
# policy line goes into BOTH, compiled here into the closest rule.
# SYSTEM_PROMPT_COMPILED retired 2026-09-21 (§4JG): the §4FF/§4GJ instruction-
# following bench measured no difference against SYSTEM_PROMPT on the full banded
# bank (87/95 vs 87/95, paired McNemar p = 1.0; easy/tool/deep at ceiling for both).
# A second prompt variant carried for no measured gain is a liability, not an option.


# §4UI (2026-11-12): the coding "Specialist" persona (SPECIALIST_SYSTEM_PROMPT +
# SPECIALIST_TOOL_XML_*) is REMOVED. A keyword regex on the owner's message
# chose it (43% of the requests that wrote or ran code, measured 10-07..11-12);
# its edit guidance contradicted the file_system tool (`replace` + SEARCH blocks,
# a fix-script that bypassed the write checks) and most rules compensated for the
# old MoE model. The facts worth keeping live in the `execute` tool description
# and the main prompt's UI-verification line.


SMART_MEMORY_PROMPT = """### IDENTITY
You are the Subconscious Synthesizer. Extract high-signal data from this task episode to build the user's profile, memory, and knowledge graph.

### SCORING MATRIX
- 1.0 : EXPLICIT IDENTITY (Names, locations, professions). -> TRIGGERS PROFILE UPDATE.
- 0.9 : INFERRED PREFERENCES ("I prefer async Python", "Always use pytest"). -> TRIGGERS PROFILE UPDATE.
- 0.8 : GENERAL TECHNICAL CONTEXT not tied to a managed project (host quirks, recurring bugs, library versions in the user's environment).
- 0.2 : TRACKED-PROJECT STATE — anything about a MANAGED PROJECT's files, architecture, tasks, testing, or release status (episodes mentioning a 12-hex project id or a project the agent manages). The project store already records these; saving them here creates stale duplicates ("user is working on X" read weeks later as current truth). -> DISCARD. NEVER put project state in profile_update. A durable USER preference learned during project work ("user prefers dark UIs") is still 0.9.
- 0.1 : EPHEMERAL CHIT-CHAT -> DISCARD.

### TEMPORAL RULE
Facts decay. NEVER write an age, a duration, or a relative date into a `fact` or a `profile_update` — they are measurements, true only on the day they were said, and they are read back months later as if current. Write the ABSOLUTE date they imply, computed from the CURRENT TIME you were given:
- "Kosmas is 4 months old" -> "Kosmas born ~<date 4 months before today>"
- "started at the company 3 years ago" -> "started in ~<date 3 years before today>"
An absolute date already in the statement ("born in March 2026") is already correct — keep it as is.

### OUTPUT FORMAT
Return ONLY a JSON object.
1. For `score` and `fact`: If the episode contains NEW implicit context to remember semantically, score it >= 0.8 and provide a 1-sentence "fact". If the agent already explicitly saved the fact using a tool, score it < 0.5 to avoid duplicates — but when the USER stated a CHANGED profile value (moved, new job, a corrected name, a sold car), STILL return it as `profile_update` {"category", "key", "value"}.
   Only what the USER says about THEMSELVES is about the user: a question, a role-play, a quote, a pasted text or another person's message is not. Ages: write the birth date, never the age.
2. For `graph_triplets`: ALWAYS extract explicit entity relationships into this array as objects with "subject", "predicate", and "object" keys, REGARDLESS of the semantic score. Predicates MUST be uppercase verbs (e.g., OWNS, LIKES, USES, WORKS_AT). Use broad, normalized entities.

example:
{
  "score": 0.1,
  "fact": "",
  "profile_update": null,
  "graph_triplets": [
    {"subject": "User", "predicate": "OWNS", "object": "Dog"},
    {"subject": "Dog", "predicate": "NAMED", "object": "Bruno"}
  ]
}
"""


SYSTEM_3_EPISODIC_CONTEXT = """### PAST RECOVERY STRATEGIES
The agent has previously recovered from similar failures. Use these as inspiration:

{recovery_episodes}

Consider whether any of these past strategies apply to the current situation.
"""
