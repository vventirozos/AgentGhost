#!/usr/bin/env python3
r"""§4UY battery — projects subsystem fixes (2026-11-18, findings P1–P23 of §4UX-3): the one who-may-act gate,
the model's notes, the legacy first plan, task-turn progress, plan-only pacing, what a check is, the declared
file's age, every writer through the hooks, the change log and work log, the ready briefing, task kinds, ordered
closes, unclosed turns, request claims, stopped batches, the trash, a finished batch's status.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uy_projects.py [-jN] [all|rerun M1 M2 …]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = "src/ghost_agent/tools/projects.py"; MP = "src/ghost_agent/memory/projects.py"
PT = "src/ghost_agent/core/project_turns.py"; PA = "src/ghost_agent/core/project_advancer.py"
LG = "src/ghost_agent/utils/logging.py"; TD = "src/ghost_agent/core/task_deliverable.py"
TC = "src/ghost_agent/core/task_close.py"; PR = "src/ghost_agent/core/prompts.py"
RG = "src/ghost_agent/tools/registry.py"; PV = "src/ghost_agent/utils/provenance.py"
LB = "src/ghost_agent/tools/labs.py"; OC = "src/ghost_agent/tools/outcome.py"
SE = "src/ghost_agent/selfhood/sentiment.py"
MUT = [
    ("CONTROL no-op", TP, r'''def action_class(act: str, kwargs: Dict[str, Any]) -> str:''',
     r'''def action_class(act: str, kwargs: Dict[str, Any]) -> str:  # no-op'''),
    ("CONTROL known-bad", TP, r'''    if action_class(act, kwargs) != "owner":
        return None''', r'''    return None'''),
    # G — P1/P9: the one gate
    ("G1 owner-check-off", TP, r'''    if not _owner_turn_now() or origin not in ("", "owner"):''', r'''    if origin not in ("", "owner"):'''),
    ("G2 member-scope-ok", TP, r'''    if not _owner_turn_now() or origin not in ("", "owner"):''', r'''    if not _owner_turn_now():'''),
    ("G3 untrusted-off", TP, r'''    if _src and not _reduce_only(context, store, act, kwargs):
        return _err(''', r'''    if False:
        return _err('''),
    ("G4 probe-any-project", TP, r'''        if (proj.get("metadata") or {}).get("probe_created"):''', r'''        if True:'''),
    ("G5 budget-always-read", TP, r'''        return "owner" if isinstance(md, dict) and md else "read"''', r'''        return "read"'''),
    ("G6 autopilot-always-read", TP, r'''        return "read" if en is None or str(en).strip() == "" else "owner"''', r'''        return "read"'''),
    ("G7 entry-skips-gate", TP, r'''        _wall = _who_may_act(context, store, act, kwargs)''', r'''        _wall = None'''),
    ("G8 notes-not-namespaced", TP, r'''    return {MODEL_NOTES_KEY: {**prev, **md}}''', r'''    return dict(md)'''),
    # A — P4 pacing
    ("A4 task-turn-records", TP, r'''    if not root or root != _current_request_id():
        return                                 # (a task's turn''', r'''    if not root:
        return                                 # (a task's turn'''),
    # K — P13 / P14 / P16 / P17 / P22
    ("K1 kind-dropped", TP, r'''            task_kind, kind = kind, ""''', r'''            kind = ""'''),
    ("K2 close-no-settle", TP, r'''            if not (deps & others) and time.monotonic() - t0 >= CLOSE_SETTLE_S:''', r'''            if not (deps & others):'''),
    ("K3 claim-not-noted", TP, r'''                    note_request_claim(store, _current_request_id(), tid)''', r'''                    pass'''),
    ("K4 claim-not-released", MP, r'''        release_request_claims(request_id)''', r'''        pass'''),
    ("K5 stop-not-recorded", TP, r'''            with _pin_evt(project_id), _record_batch_stop(store, project_id, _batch_t0):''', r'''            with _pin_evt(project_id):'''),
    ("K6 done-batch-partial", TP, r'''                if _ran and not (batch.stop_reason in ("failed", "repeated_failures", "completed_with_failures")''', r'''                if False and not (batch.stop_reason in ("failed", "repeated_failures", "completed_with_failures")'''),
    ("K7 ready-says-active", TP, r'''            "status": (READY_VIEW_STATUS if _state == "ready" else proj["status"]),''', r'''            "status": proj["status"],'''),
    # L — P2 legacy first plan
    ("L1 stamp-current-request", MP, r'''                if _legacy is not None:''', r'''                if False:'''),
    ("L2 stamp-not-read", MP, r'''        if legacy_ready_at(store, project_id, before=first_at - 1.0) is not None:''', r'''        if False:'''),
    ("L3 failed-rollup-counts", MP, r'''        if st in _FINISHED_ROLLUPS and (before is None or ts < before):''', r'''        if (before is None or ts < before):'''),
    # C — P8/P10/P11/P18/P19/P20 change log
    ("C1 job-end-not-kept", MP, r'''    _note_end_state(e, now)''', r'''    pass'''),
    ("C2 job-writes-foreign", MP, r'''                e["wrote_t"] = float(st["t"])   # the job's writes since the request ended are this entry's''', r'''                pass'''),
    ("C3 wake-not-seeded", TP, r'''            seed_from_starter(_starter, request_id)''', r'''            pass'''),
    ("C4 writer-not-watched", TP, r'''            watch_tool_folder(context, folder_of())''', r'''            pass'''),
    ("C5 deploys-in-briefing", MP, r'''        return [change_line(e) for e in store.list_changes(project_id, limit=n, kind="change")]''', r'''        return [change_line(e) for e in store.list_changes(project_id, limit=n)]'''),
    ("C6 added-not-in-manifest", MP, r'''            _manifest_added(store, pid, entry)''', r'''            pass'''),
    ("C7 ellipsis-cut", MP, r'''    return cut.rstrip() + "…"''', r'''    return out[:CHANGE_SUMMARY_MAX]'''),
    ("C8 foreign-never-rendered", MP, r'''                if first != CHANGES_FILE_HEADER:
                    # §4UY P10''', r'''                if first != CHANGES_FILE_HEADER:
                    return None
                    # §4UY P10'''),
    ("C9 worklog-raw-paths", MP, r'''                return self._normalize_rel_path(_pid, f) or f''', r'''                return f'''),
    # T — P3/P6/P15 task turns
    ("T1 progress-no-parent", PT, r'''            set_progress_parent(rid, outer)''', r'''            pass'''),
    ("T2 progress-no-fallback", LG, r'''        while s is None and _up and _hops < 4:''', r'''        while False:'''),
    ("T3 progress-not-propagated", LG, r'''                ps["progress_t"] = s["progress_t"]''', r'''                pass'''),
    ("T4 batch-reads-deadline-only", PA, r'''        if request_recent_progress(rid):''', r'''        if False:'''),
    ("T5 failed-run-counts", PT, r'''        ran = (_ran(res) and _succeeded(res) and self._touches_project(name, kwargs, op)) if "c" in evs else False''', r'''        ran = (_ran(res) and self._touches_project(name, kwargs, op)) if "c" in evs else False'''),
    ("T6 any-run-counts", PT, r'''        ran = (_ran(res) and _succeeded(res) and self._touches_project(name, kwargs, op)) if "c" in evs else False''', r'''        ran = (_ran(res) and _succeeded(res)) if "c" in evs else False'''),
    ("T7 syntax-check-runs", PT, r'''        return []                                        # §4UY P6: a syntax check reads the code, never runs it''', r'''        return ["c"]'''),
    ("T8 linter-runs", PT, r'''        return ["w"] if any(a in ("--fix", "--write") or a.startswith("--fix=") for a in args) else []''', r'''        return ["w"] if any(a in ("--fix", "--write") or a.startswith("--fix=") for a in args) else ["c"]'''),
    ("T9 turn-files-unknown", PT, r'''        if rel.rstrip("/") in known or any(k.startswith(rel.rstrip("/") + "/") for k in known):''', r'''        if False:'''),
    ("T10 unclosed-fails", PT, r'''        if status == "IN_PROGRESS" and not unattended:''', r'''        if False:'''),
    ("T11 unclosed-not-a-stop", PA, r'''        if cls == "blocked" and res.task_id and res.summary.startswith(_task_turn_unclosed_prefix()):''', r'''        if False:'''),
    # D — P7 declared file
    ("D1 existing-file-closes", TD, r'''    if mt >= float(since) - DELIVERABLE_SINCE_SLACK_S:''', r'''    if True:'''),
    ("D3 since-not-passed", TC, r'''r = gate_stale_deliverable(ws, judged_desc, since=_task_since(store, task_id),''', r'''r = gate_stale_deliverable(ws, judged_desc, since=None,'''),
    # R — review round (2026-11-18): P16 claims, P9 reduce-only, P4 split / later go-ahead, P6 real runs, P7 evidence
    ("R1 claim-any-request", TP, r'''    if running_before or _owner_turn_now():''', r'''    if False:'''),
    ("R2 task-turn-claimed", TP, r'''    if _current_request_id().startswith(TASK_TURN_RID_PREFIX):''', r'''    if False:'''),
    ("R3 owner-kind-released", TP, r'''        return str((store.get_task(tid) or {}).get("kind") or "").lower() != "owner"''', r'''        return True'''),
    ("R4 reduce-only-walled", TP, r'''    if _src and not _reduce_only(context, store, act, kwargs):
        return _err(''', r'''    if _src:
        return _err('''),
    ("R5 budget-raise-exempt", TP, r'''                elif v > cur[k]:
                    return False''', r'''                elif False:
                    return False'''),
    ("R6 checkpoint-off-exempt", TP, r'''                    if v <= 0 or (cur[k] > 0 and v > cur[k]):''', r'''                    if cur[k] > 0 and v > cur[k]:'''),
    ("R7 pause-with-rename", TP, r'''            return fields == {"status"} and str(kwargs.get("status") or "").strip().upper() == "PAUSED"''', r'''            return str(kwargs.get("status") or "").strip().upper() == "PAUSED"'''),
    ("R8 split-latches", TP, r'''    if split_of and go_ahead is None:
        return''', r'''    if False:
        return'''),
    ("R10 dash-m-not-a-head", PT, r'''                if _PY_RE.fullmatch(os.path.basename(w[0])):     # review: `python3 -m pytest` runs pytest''', r'''                if False:'''),
    ("R11 syntax-flag-anywhere", PT, r'''any(a in _SYNTAX_ONLY_FLAGS[head] for a in _leading_flags(args)):''', r'''any(a in _SYNTAX_ONLY_FLAGS[head] for a in args):'''),
    ("R12 unittest-not-a-runner", PT, r'''"docker", "podman", "unittest",''', r'''"docker", "podman",'''),
    ("R13 touched-ignored", TD, r'''        if str(target.resolve()) in set(touched or ()):''', r'''        if False:'''),
    ("R14 failed-call-is-evidence", TD, r'''        if not _call_succeeded(name, res):
            return''', r'''        pass'''),
    ("R15 close-without-evidence", TC, r'''since=_task_since(store, task_id), touched=_touched_now())''', r'''since=_task_since(store, task_id), touched=())'''),
    ("R16 wrapper-records-nothing", TP, r'''                note_tool_evidence(_current_request_id(), name, kwargs, res, folder)''', r'''                pass'''),
    # S — review 2 (2026-11-18): P4 refusal latch / explicit-false split, P6 `-m mod.sub`, P7 modes + readers
    ("S2 false-split-ignored", TP, r'''    if split_of and go_ahead is None:''', r'''    if split_of:'''),
    ("S4 dash-m-module-ignored", PT, r'''        if _runs_project_module(cmd, root, known):
            return True''', r'''        if False:
            return True'''),
    ("S5 module-full-path-only", PT, r'''cands = ["/".join(parts) + ".py", "/".join(parts), parts[0] + ".py"]''', r'''cands = ["/".join(parts) + ".py"]'''),
    ("S7 module-known-ignored", PT, r'''            if c in known or any(k.startswith(c + "/") for k in known):
                return True''', r'''            if False:
                return True'''),
    ("S8 owner-kind-checked", TC, r'''    if kind.strip().lower() == "owner":
        return "off"''', r'''    if False:
        return "off"'''),
    ("S10 background-only-noted", TC, r'''    return "refuse"
''', r'''    return "note"
'''),
    ("S11 note-not-surfaced", TP, r'''                    close_notes.extend(f"{tid}: {_n.reason}" for _n in _verdict.notes)''', r'''                    pass'''),
    ("S12 evidence-cd-ignored", TD, r'''            if cmd:
                base = leading_cd(cmd, root)''', r'''            if False:
                base = leading_cd(cmd, root)'''),
    ("S13 evidence-not-normalised", TD, r'''                rel = project_rel(w, root)
                if not rel:''', r'''                rel = w
                if not rel:'''),
    ("S14 long-word-drops-call", TD, r'''                continue                             # ENAMETOOLONG and the like: this word only''', r'''                raise'''),
    ("S15 fs-success-by-content", TD, r'''    if name == "execute":
        from .project_turns import _succeeded''', r'''    if True:
        from .project_turns import _succeeded'''),
    # P — review 3 (2026-11-18): the `_Pacing` table (one mutant per row), P7 owner batch, P6 inline, P20 note
    ("P1 plan-only-never", TP, r'''    if rec is None or rec.state != "plan_only":
        return ""''', r'''    return ""'''),
    ("P4 autoadvance-skips-check", TP, r'''            _po = _plan_only_request(go_ahead)''', r'''            _po = ""'''),
    ("P12 ran-not-go", TP, r'''            _note_autoadvance_ran()                 # §4UY P4: an autoadvance that ran is the decision: go''', r'''            pass'''),
    ("P13 request-start-not-dropped", TP, r'''    _on_new_request(_drop_pacing)''', r'''    pass'''),
    ("P14 hook-not-called", PV, r'''    for fn in list(_ON_NEW_REQUEST):''', r'''    for fn in []:'''),
    ("P15 prose-old-latch", PR, r'''"NEW message from the user; that message re-opens the decision "''', r'''"NEW message from the user; that message changes it "'''),
    ("P16 owner-turn-refused", TC, r'''            if _owner_turn_now() or _owner_task_turn():
                return "note"''', r'''            if False:
                return "note"'''),
    ("P17 owner-task-turn-refused", TC, r'''            if _owner_turn_now() or _owner_task_turn():''', r'''            if _owner_turn_now():'''),
    ("P18 advancer-owner-refused", TC, r'''    if actor == "advancer_owner":
        return "note"''', r'''    if False:
        return "note"'''),
    ("P19 import-after-quote", PT, r'''_IMPORT_RE = re.compile(r"(?:^|[\s;'\"(])''', r'''_IMPORT_RE = re.compile(r"(?:^|[\s;])'''),
    ("P20 open-arg-counts", PT, r'''_OPEN_ARG_RE = re.compile(r"\bopen\(\s*(['\"])[^'\"]*\1")''', r'''_OPEN_ARG_RE = re.compile(r"(?!x)x")'''),
    ("P22 delete-rmtree", MP, r'''                    if self._trash_workspace(project_id, ws_p):''', r'''                    if False:'''),
    ("P23 removed-not-recorded", MP, r'''                        _note_trash(self, project_id, "removed")''', r'''                        pass'''),
    ("P24 note-always-trash", TP, r'''                _folder = store.deleted_folder(rid)''', r'''                _folder = "trash"'''),
    # Q — review 4 (2026-11-18): strict lifting (a new user message), create's explicit go_ahead, exec() runs
    ("Q1 message-not-needed", TP, r'''    return rec.state != "plan_only" or _added_count(root) > rec.n''', r'''    return True'''),
    ("Q2 message-never-lifts", TP, r'''    return rec.state != "plan_only" or _added_count(root) > rec.n''', r'''    return rec.state != "plan_only"'''),
    ("Q3 n-not-set", TP, r'''    rec.state, rec.n = "plan_only", _added_count(root)''', r'''    rec.state = "plan_only"'''),
    ("Q4 plan-go-ignored", TP, r'''        if _go_lifts(rec, root):
            rec.state = "go"''', r'''        if False:
            rec.state = "go"'''),
    ("Q5 omitted-overrides-go", TP, r'''    elif _go_ahead_false(go_ahead) or rec.state == "undecided":''', r'''    else:'''),
    ("Q6 false-ignored", TP, r'''    elif _go_ahead_false(go_ahead) or rec.state == "undecided":''', r'''    elif rec.state == "undecided":'''),
    ("Q7 omitted-resets-n", TP, r'''    elif _go_ahead_false(go_ahead) or rec.state == "undecided":''', r'''    elif _go_ahead_false(go_ahead) or rec.state != "go":'''),
    ("Q8 adv-go-ahead-ignored", TP, r'''    if _go_ahead_given(go_ahead) and _go_lifts(rec, root):
        return ""''', r'''    if False:
        return ""'''),
    ("Q9 adv-go-ahead-always-lifts", TP, r'''    if _go_ahead_given(go_ahead) and _go_lifts(rec, root):
        return ""''', r'''    if _go_ahead_given(go_ahead):
        return ""'''),
    ("Q10 create-go-ahead-ignored", TP, r'''            if created_task_ids or go_ahead is not None:''', r'''            if created_task_ids:'''),
    ("Q11 create-always-records", TP, r'''            if created_task_ids or go_ahead is not None:''', r'''            if True:'''),
    ("Q12 exec-open-stripped", PT, r'''    return text if _EXEC_RE.search(text) else _OPEN_ARG_RE.sub("open(", text)''', r'''    return _OPEN_ARG_RE.sub("open(", text)'''),
    ("Q13 open-never-stripped", PT, r'''    return text if _EXEC_RE.search(text) else _OPEN_ARG_RE.sub("open(", text)''', r'''    return text'''),
    ("Q14 content-open-counts", PT, r'''    return _names_project_file(_strip_open_reads(str(code or "")), root, known)''', r'''    return _names_project_file(str(code or ""), root, known)'''),
    # B — P12 briefing
    ("B1 ready-gets-pacing", PR, r'''    _in_flight = not _finished and _state != "ready"''', r'''    _in_flight = not _finished'''),
    ("B2 files-repeat-deliverables", PR, r'''    _layout = [f for f in _layout if f not in _shown]''', r'''    pass'''),
]
#: module-level code / a constant: coverage maps no test to it
# P5 — the question wall (operator decision after §4UY: a code rule, not prose): `lab` publish actions are refused
# in the owner's turn of a request that only asked. W = the wall (tools/labs.py), Q = the predicate (provenance)
_P5 = "tests/test_4uy_projects_p5.py::"
_P5_TABLE_OLD = r'''PUBLISH_ACTIONS = frozenset({"put", "deploy", "switch", "rollback", "restore", "keep"})
NOT_PUBLISH_ACTIONS = frozenset({"templates",'''
MUT += [
    ("PW1 entry-skips-wall", LB, r'''    _qwall = await _question_wall(context, action, origin, lab_id)       # §4UY P5: before any argument error''', r'''    _qwall = None'''),
    ("PW2 wall-in-every-request", LB, r'''    if action not in PUBLISH_ACTIONS or origin != "owner" or not _asked_only_a_question(context):''', r'''    if action not in PUBLISH_ACTIONS or origin != "owner":'''),
    ("PW3 origin-not-read", LB, r'''    if action not in PUBLISH_ACTIONS or origin != "owner" or not _asked_only_a_question(context):''', r'''    if action not in PUBLISH_ACTIONS or not _asked_only_a_question(context):'''),
    ("PW4 any-turn-is-the-owners", LB, r'''        if not _owner_turn_now():
            return False
        from ..utils.logging import request_id_context
        from ..utils.provenance import question_only_request''', r'''        from ..utils.logging import request_id_context
        from ..utils.provenance import question_only_request'''),
    ("PW5 put-walled-on-disposable", LB, r'''            if st == 404 or (st == 200 and isinstance(det, dict) and str(det.get("lifetime") or "") != "kept"):''', r'''            if st == 404:'''),
    ("PW6 put-open-when-unsure", LB, r'''            if st == 404 or (st == 200 and isinstance(det, dict) and str(det.get("lifetime") or "") != "kept"):''', r'''            if st != 200 or not isinstance(det, dict) or str(det.get("lifetime") or "") != "kept":'''),
    ("PW7 kept-cache-not-read", LB, r'''        if lab_id not in _KEPT_LAB_SEEN:
            try:
                st, det = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin, timeout=10)''', r'''        if True:
            try:
                st, det = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin, timeout=10)'''),
    ("PW8 put-never-asks-kept", LB, r'''_PUBLISH_IF_KEPT = frozenset({"put"})''', r'''_PUBLISH_IF_KEPT = frozenset()'''),
    ("PW9 put-without-lab-walled", LB, r'''            return None                          # no lab named: the argument error follows, nothing is sent''', r'''            pass'''),
    ("PW10 code-not-a-wall", OC, r'''    "question_not_instruction",                      # §4UY P5: a deploy to the live site in a request that only asked
''', r''''''),
    ("PW11 wall-after-argument-errors", LB, r'''    _qwall = await _question_wall(context, action, origin, lab_id)       # §4UY P5: before any argument error
    if _qwall is not None:
        return _qwall
    if action not in ("templates", "create", "list") and not _LAB_ID_RE.match(lab_id):''', r'''    _qwall = await _question_wall(context, action, origin, lab_id) if _LAB_ID_RE.match(lab_id) else None
    if _qwall is not None:
        return _qwall
    if action not in ("templates", "create", "list") and not _LAB_ID_RE.match(lab_id):'''),
    ("PQ1 semicolon-asks-in-any-text", PV, r'''        if not ("?" in mark or "\u037e" in mark or (";" in mark and _semicolon_asks(frag))):''', r'''        if not ("?" in mark or "\u037e" in mark or ";" in mark):'''),
    ("PQ2 greek-semicolon-not-read", PV, r'''        if not ("?" in mark or "\u037e" in mark or (";" in mark and _semicolon_asks(frag))):''', r'''        if not ("?" in mark or "\u037e" in mark):'''),
    ("PQ3 greek-mark-not-read", PV, r'''        if not ("?" in mark or "\u037e" in mark or (";" in mark and _semicolon_asks(frag))):''', r'''        if not ("?" in mark or ((";" in mark or "\u037e" in mark) and _semicolon_asks(frag))):'''),
    ("PQ4 one-question-suffices", PV, r'''        if not ("?" in mark or "\u037e" in mark or (";" in mark and _semicolon_asks(frag))):
            return False''', r'''        if not ("?" in mark or "\u037e" in mark or (";" in mark and _semicolon_asks(frag))):
            continue'''),
    ("PQ5 trailing-words-ignored", PV, r'''    return seen and not _has_letter(t[pos:])''', r'''    return seen'''),
    ("PQ6 spans-read", PV, r'''    t = "\n".join(_mask_line(ln) for ln in t.split("\n"))''', r'''    pass'''),
    ("PQ7 fences-read", PV, r'''    t = _Q_FENCE_RE.sub(" " + _Q_MASK + " ", text.replace(_Q_MASK, " "))''', r'''    t = text.replace(_Q_MASK, " ")'''),
    ("PQ8 ellipsis-ends-a-sentence", PV, r'''            continue                              # a bare ellipsis: the sentence goes on''', r'''            pass'''),
    ("PQ9 numbers-are-sentences", PV, r'''        if not _has_letter(frag):
            continue''', r'''        if not frag.strip():
            continue'''),
    ("PQ10 no-sentence-is-a-question", PV, r'''    return seen and not _has_letter(t[pos:])''', r'''    return not _has_letter(t[pos:])'''),
    ("PQ11 any-message-asks", PV, r'''    return bool(texts) and all(is_pure_question(t) for t in texts)''', r'''    return bool(texts) and any(is_pure_question(t) for t in texts)'''),
    ("PQ12 latest-message-only", PV, r'''    return bool(texts) and all(is_pure_question(t) for t in texts)''', r'''    return bool(texts) and is_pure_question(texts[-1])'''),
    ("PQ13 added-messages-not-read", PV, r'''        if rid in _ASKED:                                # §4UY P5: never an entry without its own message
            _ASKED[rid].append(t)''', r'''        if rid in _ASKED:                                # §4UY P5: never an entry without its own message
            pass'''),
    ("PQ14 own-id-not-root", PV, r'''        root = _lineage(_rid(req_id))[-1]
        entry = _ASKED.get(root)''', r'''        root = _rid(req_id)
        entry = _ASKED.get(root)'''),
    ("PQ15 non-string-read", PV, r'''    if not isinstance(text, str):
        return False
    t = _Q_FENCE_RE''', r'''    text = str(text or "")
    t = _Q_FENCE_RE'''),
    # review r1: A-1 quote / bracket spans and the `;` word, A-2 one entry, A-3 linear scan, A-4 client notes
    ("PQ16 apostrophe-opens-anywhere", PV, r'''        elif ch in _Q_APOS and (i == 0 or line[i - 1].isspace() or line[i - 1] in _Q_OPENS_AFTER) \
                and last_at(_Q_APOS[ch]) > i:''', r'''        elif ch in _Q_APOS and last_at(_Q_APOS[ch]) > i:'''),
    ("PQ17 apostrophe-closes-anywhere", PV, r'''    return k + 1 >= len(line) or line[k + 1].isspace() or line[k + 1] in _Q_CLOSES_BEFORE''', r'''    return True'''),
    ("PQ18 single-quotes-not-masked", PV, r'''        elif ch in _Q_APOS and (i == 0''', r'''        elif False and (i == 0'''),
    ("PQ19 latin-last-word-never-asks", PV, r'''    return last.isalpha() and greek_before        # a Latin word inside a Greek sentence ("είναι live;")''', r'''    return False'''),
    ("PQ20 lone-span-never-asks", PV, r'''        return greek_before                       # a span standing alone: the word it follows''', r'''        return False'''),
    ("PQ29 latin-word-needs-no-greek-before", PV, r'''    return last.isalpha() and greek_before        # a Latin word inside a Greek sentence ("είναι live;")''', r'''    return last.isalpha()'''),
    ("PQ30 any-last-token-after-greek", PV, r'''    return last.isalpha() and greek_before        # a Latin word inside a Greek sentence ("είναι live;")''', r'''    return greek_before'''),
    ("PQ31 glued-span-reads-the-sentence", PV, r'''        if word:                                  # a span glued to a word (`foo()`, `print("hi")`): that word
            return bool(_Q_GREEK_RE.search(word))''', r'''        if word:                                  # a span glued to a word (`foo()`, `print("hi")`): that word
            return bool(_Q_GREEK_RE.search(frag))'''),
    ("PQ32 lone-span-reads-the-sentence", PV, r'''        return greek_before                       # a span standing alone: the word it follows''', r'''        return bool(_Q_GREEK_RE.search(frag))'''),
    ("PQ33 assignment-word-counts", PV, r'''    greek_before = bool(_Q_GREEK_RE.search(before)) and not before.endswith(("=", ":"))''', r'''    greek_before = bool(_Q_GREEK_RE.search(before))'''),
    ("PQ34 spans-skipped-not", PV, r'''    before = next((w for w in reversed(words[:-1]) if w.strip(_Q_MASK)), "")''', r'''    before = words[-2] if len(words) > 1 else ""'''),
    ("PQ35 note-span-unbounded", SE, r'''a file named .{0,700}?(?:"''', r'''a file named .*?(?:"'''),
    ("PQ36 mention-span-unbounded", SE, r'''(?:\|[^>\n]{0,200})?>"),''', r'''(?:\|[^>]*)?>"),'''),
    ("PQ38 mark-before-span-no-end", PV, r'''(?=\\s|\\Z|\ue000)|\n+")''', r'''(?=\\s|\\Z)|\n+")'''),
    ("PQ21 added-rebuilds-entry", PV, r'''        if rid in _ASKED:                                # §4UY P5: never an entry without its own message
            _ASKED[rid].append(t)''', r'''        if True:
            _ASKED.setdefault(rid, []).append(t)'''),
    ("PQ22 read-does-not-refresh", PV, r'''        _ASKED.move_to_end(root)
        texts = list(entry)''', r'''        texts = list(entry)'''),
    ("PQ23 new-request-keeps-entry", PV, r'''        _ASKED[rid] = [str(text or "")]                  # §4UY P5: a new request under this id starts over''', r'''        _ASKED.setdefault(rid, [str(text or "")])'''),
    ("PQ24 entry-never-evicted", PV, r'''        while len(_ASKED) > _CAP:
            _ASKED.popitem(last=False)''', r'''        pass'''),
    ("PQ25 client-notes-read", PV, r'''        return message_text(text)
    except Exception:  # noqa: BLE001 — unreadable: judged as it is''', r'''        return str(text or "")
    except Exception:  # noqa: BLE001 — unreadable: judged as it is'''),
    ("PQ26 failed-ingest-note-unknown", SE, r'''found inside it\.\]|"
               r"as if you had read it\.\])", re.DOTALL),''', r'''found inside it\.\])", re.DOTALL),'''),
    ("PQ27 no-closer-precheck", PV, r'''            ahead = [line.find(c, i + 1) for c in _Q_PAIRS[ch] if last_at(c) > i]''', r'''            ahead = [k for k in (line.find(c, i + 1) for c in _Q_PAIRS[ch]) if k >= 0]'''),
    ("PQ28 brackets-not-masked", PV, r'''"\u00ab": "\u00bb", "(": ")",
            "[": "]"}''', r'''"\u00ab": "\u00bb"}'''),
    ("PW12 unsure-says-no-answer", LB, r'''could not be checked against the labs host, and it would''', r'''could not be checked against the labs host (no answer), and it would'''),
]
# one mutant per publish action: moved to the other side of the table (the import check still passes)
MUT += [(f"PT{i} {a}-not-publish", LB, _P5_TABLE_OLD,
         "PUBLISH_ACTIONS = frozenset({" + ", ".join(f'"{x}"' for x in
                                                      ("put", "deploy", "switch", "rollback", "restore", "keep")
                                                      if x != a)
         + "})\nNOT_PUBLISH_ACTIONS = frozenset({" + f'"{a}", "templates",')
        for i, a in enumerate(("put", "deploy", "switch", "rollback", "restore", "keep"))]

FORCE = {"G5 budget-always-read": ["test_no_owner_action_runs_outside_the_owners_clean_turn"],
         "G6 autopilot-always-read": ["test_no_owner_action_runs_outside_the_owners_clean_turn"],
         # module-level lines (a table, a set): no test "covers" them at call time
         "PW8 put-never-asks-kept": [_P5 + "test_put_into_a_disposable_lab_runs_in_a_question_request"],
         "PQ26 failed-ingest-note-unknown": [_P5 + "test_a_slack_file_note_is_not_the_users_sentence"],
         "PQ28 brackets-not-masked": [_P5 + "test_anything_else_is_not_a_pure_question"],
         **{k: [_P5 + "test_reading_the_client_notes_is_linear",
                _P5 + "test_the_bounded_note_patterns_still_take_out_whole_notes"]
            for k in ("PQ35 note-span-unbounded", "PQ36 mention-span-unbounded")},
         "PQ38 mark-before-span-no-end": [_P5 + "test_a_message_whose_every_sentence_ends_with_a_question_mark_is_a_pure_question"],
         "PW10 code-not-a-wall": [_P5 + "test_no_publish_action_reaches_the_host_in_a_request_that_only_asked",
                                 _P5 + "test_every_lab_action_is_publish_or_not_and_the_code_is_declared"],
         **{f"PT{i} {a}-not-publish": [_P5 + "test_no_publish_action_reaches_the_host_in_a_request_that_only_asked"]
            for i, a in enumerate(("put", "deploy", "switch", "rollback", "restore", "keep"))}}
PINS = ["tests/test_4uy_projects.py", "tests/test_4uy_projects_p5.py", "tests/test_4sx_b_lab_tool.py",
        "tests/test_4sx_d3_kept.py", "tests/test_4sx_d4_autopilot_labs.py", "tests/test_4ud_labs.py",
        "tests/test_4ul_labs.py", "tests/test_4ul_labs_review.py", "tests/test_4ul_labs_rereview.py",
        "tests/test_lab_deploy_errors_6043a546.py",
        "tests/test_4ur_who_may_act_wall.py", "tests/test_4um_living_projects.py",
        "tests/test_4um_review.py", "tests/test_4ul_review2.py", "tests/test_4ul_autopilot_turns.py",
        "tests/test_4ul_review_fixes.py", "tests/test_4lp_projects.py", "tests/test_4sx_d2_inventory.py",
        "tests/test_4mq_long_horizon.py", "tests/test_4pf_open_items.py", "tests/test_4qa_project_task_integrity.py",
        "tests/test_4sr_gate_words.py", "tests/test_advance_pacing.py", "tests/test_4su_phase4.py",
        "tests/test_4uo_wake_changes_and_lookup_notes.py", "tests/test_projects_tool.py",
        "tests/test_4qa_probe_quotes.py", "tests/test_4uy_code_spec.py"]


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    pins = [p for p in PINS if os.path.exists(os.path.join(ROOT, p))]
    cov = BC.build_map(srcs, pins, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        if label in FORCE:
            ids = sorted(set(ids or []) | {t if "::" in t else f"tests/test_4uy_projects.py::{t}"
                                           for t in FORCE[label]})
        if ids:
            return ids, f"cov {len(ids)} tests"
        return list(pins), f"fallback {len(pins)} files"
    print(f"{len(pins)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
    res = BC.run_parallel(batch, originals, select, workers=workers, log=lambda m: print(m, flush=True))
    by = {}
    for _, st, _, _ in res:
        by[st] = by.get(st, 0) + 1
    print("SUMMARY " + " ".join(f"{k.lower()}={v}" for k, v in sorted(by.items())), flush=True)
    for lab, st, tier, tail in res:
        if st != "KILLED":
            print(f"  {st:8} {lab} — [{tier}] {tail}", flush=True)


if __name__ == "__main__":
    main()
