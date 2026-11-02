#!/usr/bin/env python3
"""§4SU Phases 3+4 battery — wait FIELDS (core/task_wait.py; store, tree, advancer, briefing,
tool, web) and the model-facing surface (replan, list display, go-ahead marker, finished
briefing, status aliases, one `refused` list).
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4su3.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TW = 'src/ghost_agent/core/task_wait.py'
TP = 'src/ghost_agent/tools/projects.py'
PL = 'src/ghost_agent/core/planning.py'
RT = 'src/ghost_agent/api/projects_routes.py'
PA = 'src/ghost_agent/core/project_advancer.py'
TC = 'src/ghost_agent/core/task_close.py'
MS = 'src/ghost_agent/memory/projects.py'
PR = 'src/ghost_agent/core/prompts.py'
AG = 'src/ghost_agent/core/agent.py'
MUT = [
    ('CONTROL known-bad', TW, 'FOLDED = {"READY": "PENDING", "PAUSED": "NEEDS_USER"}', 'FOLDED = {}'),
    ('CONTROL no-op', TW, 'def view_label(task) -> str:', 'def view_label(task) -> str:  # no-op'),
    # ── task_wait.py
    ('W1 hold-tail-unread', TW, '    if low.endswith(_HELD_TAIL):', '    if False:'),
    ('W2 gate-mark-unread', TW, '    if low.startswith("human gate:"):', '    if False:'),
    ('W3 rollup-gate-lost', TW, '        return "owner", (f"rollup_{gate}" if gate else "rollup"), rest[:300]', '        return "owner", "rollup", rest[:300]'),
    ('W4 blocked-is-owner', TW, '    return ("dependency", "blocked") if fold_status(status) == "BLOCKED" else ("owner", "asked")', '    return ("owner", "asked")'),
    ('W5 default-is-cascade', TW, '    return ("dependency", "blocked") if fold_status(status) == "BLOCKED" else ("owner", "asked")', '    return ("dependency", "child_failed") if fold_status(status) == "BLOCKED" else ("owner", "asked")'),
    ('W6 hold-ignores-status', TW, '    return _get(task, "wait_on") == "owner_turn" and fold_status(_get(task, "status")) == "NEEDS_USER"', '    return _get(task, "wait_on") == "owner_turn"'),
    ('W7 running-not-renamed', TW, '    return {"state": {"IN_PROGRESS": "RUNNING"}.get(st, st)}', '    return {"state": st}'),
    ('W8 enum-repr', TW, '    v = getattr(v, "value", v)          # a TaskStatus enum → its value', '    pass'),
    ('W9 paused-not-owner', TW, '    if raw == "PAUSED":\n        return PAUSED_WAIT[:2]', '    if False:\n        return PAUSED_WAIT[:2]'),
    # ── store
    ('S1 no-migration', MS, '                ("wait_on", "TEXT NOT NULL DEFAULT \'\'"),', ''),
    ('S2 migration-skips-marks', MS, '                on, code, why = wait_from_marks(r["status"], r["result_summary"])', '                on, code, why = "owner", "asked", ""'),
    ('S3 leave-keeps-wait', MS, '            if _st not in WAITING_STATUSES:\n                fields.update(wait_on="", wait_code="", wait_why="")', '            if False:\n                fields.update(wait_on="", wait_code="", wait_why="")'),
    ('S4 enter-keeps-old', MS, '            elif not fields.get("wait_on"):\n                # r1 MAJOR', '            elif False:\n                # r1 MAJOR'),
    ('S5 wait-on-unchecked', MS, '            if fields["wait_on"] not in WAIT_ON:', '            if False:'),
    ('S6 add-no-fold', MS, '        status = fold_status(status) or "PENDING"    # PAUSED → NEEDS_USER (owner, paused)', '        status = str(status or "PENDING")'),
    ('S7 no-cols-guard', MS, '            fields = {k: v for k, v in fields.items() if not k.startswith("wait_")}', '            pass'),
    ('S8 update-paused-no-wait', MS, '            if _raw_st == "PAUSED" and not fields.get("wait_on"):', '            if False:'),
    # ── tree
    ('P1 status-keeps-wait', PL, '            node.set_wait(*(wait if wait and status in (TaskStatus.NEEDS_USER, TaskStatus.BLOCKED)\n                            else ("", "", "")))', '            if wait:\n                node.set_wait(*wait)'),
    ('P2 settle-no-default', PL, '                if not n.wait_on:\n                    n.set_wait(*default_wait(n.status.value), n.result_summary)', '                pass'),
    ('P3 settle-no-clear', PL, '            elif n.wait_on or n.wait_code or n.wait_why:\n                n.set_wait("", "", "")', '            elif False:\n                n.set_wait("", "", "")'),
    ('P4 gate-park-code', PL, '        if parent.status != TaskStatus.NEEDS_USER or parent.wait_code != "human_gate":', '        if parent.status != TaskStatus.NEEDS_USER:'),
    ('P5 rollup-code-lost', PL, '            parent.set_wait("owner", f"rollup_{_gate}" if _gate else "rollup", why)', '            parent.set_wait("owner", "asked", why)'),
    ('P6 not-hydrated', PL, '                wait_on=str(row.get("wait_on") or ""),', ''),
    ('P7 not-persisted', PL, '                    wait_on=node.wait_on, wait_code=node.wait_code, wait_why=node.wait_why,', ''),
    ('P8 cascade-owner', PL, '                parent.set_wait("dependency", "child_failed", "a subtask failed")   # §4SU Phase 3 r1\n                self._check_parent_failure(parent.parent_id, visited)', '                parent.set_wait("owner", "asked", "a subtask failed")\n                self._check_parent_failure(parent.parent_id, visited)'),
    # ── advancer
    ('A1 take-back-any', PA, '                and _t.get("wait_on") == "owner_turn"        # §4SU Phase 3: the field, not the text', '                and True'),
    ('A2 pick-any-needs-user', PA, '                  and n.wait_on == "owner_turn"]                # §4SU Phase 3', '                  ]'),
    ('A3 hold-wait-is-owner', PA, '                       wait=("owner_turn", "gate_form", _why))', '                       wait=("owner", "gate_form", _why))'),
    ('A4 review-no-wait', PA, '                           wait=("owner", "review", task_topic(nxt.description)[:200]))', '                           )'),
    ('A5 batch-hold-by-text', PA, '        _is_hold = (_row.get("wait_on") == "owner_turn" if _row', '        _is_hold = (str(_row.get("result_summary") or "").endswith(_HELD_MARK) if _row'),
    ('A6 exp-hold-owner', PA, '                               wait=("owner_turn", "model_experiment", _why))', '                               wait=("owner", "model_experiment", _why))'),
    # ── briefing / tool / web
    ('B1 briefing-raw-status', PR, '                return f"WAITING({v[\'on\']})" if v["state"] == "WAITING" else v["state"]', '                return n.status.value'),
    ('B2 finished-shows-active', PR, '    if not _finished:\n        lines.append(\n            f"*** A project is ALREADY ACTIVE', '    if True:\n        lines.append(\n            f"*** A project is ALREADY ACTIVE'),
    ('B3 finished-no-line', PR, '    if _finished:\n        _arch', '    if False:\n        _arch'),
    ('T1 waiting-not-alias', TP, '                    _su = {"WAITING": "NEEDS_USER", "RUNNING": "IN_PROGRESS"}.get(', '                    _su = {}.get('),
    ('T2 list-full-rows', TP, '            _out = {} if not bool((metadata or {}).get("verbose")) else {"projects": projs}', '            _out = {"projects": projs}'),
    ('T3 list-filter-view', TP, '            if _vf in ("WAITING", "RUNNING"):', '            if False:'),
    ('T4 slim-raw-status', TP, '                     "state": view_label(t)}', '                     "state": t.get("status")}'),
    ('T5 replan-drops-done', TP, '            _drop = [t for t in _all if str(t.get("status") or "").upper() != "DONE" and not _has_done(t["id"])]', '            _drop = [t for t in _all if str(t.get("status") or "").upper() != "DONE"]'),
    ('T6 replan-ignores-running', TP, '            if _running:\n                return _err(f"NOT replanned', '            if False:\n                return _err(f"NOT replanned'),
    ('T7 replan-drops-gate', TP, '            if _gated:\n                for _t in _gated:', '            if False:\n                for _t in _gated:'),
    ('T8 replan-cohesive-late', TP, '            if len(_new) > 1 and _is_cohesive_single_file(_rp.get("title"), _rp.get("goal")):', '            if False:'),
    ('T9 replan-no-ledger', TP, '                    store.append_ledger(project_id, f"REPLANNED: {len(_replan[\'drop_ids\'])} open task(s) "', '                    (lambda *a: None)(project_id, f"REPLANNED: {len(_replan[\'drop_ids\'])} open task(s) "'),
    ('T10 no-go-ahead-key', TP, '            return _ok({"awaits_go_ahead": True,      # §4SU Phase 4: a plan for the user — the turn ends\n                        "created": ids,', '            return _ok({"awaits_go_ahead": False,\n                        "created": ids,'),
    ('T11 sniffer-ignores-refused', TP, '        _held = any(payload.get(k) for k in ("refused", "missing"))', '        _held = any(payload.get(k) for k in ("missing",))'),
    ('T12 gate-not-by-field', TP, '                                           "waiting": plan.tree.nodes[tid].wait_code})        # §4SU Phase 3', '                                           "waiting": ""})'),
    ('T13 autoadv-holds-by-status', TP, '                _holds = [_row(t) for t in _rows if t.get("wait_on") == "owner_turn"][:10]', '                _holds = []'),
    ('C1 no-retry-clause', TC, '"When a fix is yours to do, do it, then retry with status=done AND the result in the same call; "', '""'),
    ('G1 guard-any-row', AG, "    return raw.startswith(('{\"confirmation_needed\": true', '{\"awaits_go_ahead\": true'))", "    return '\"awaits_go_ahead\": true' in raw or '\"confirmation_needed\": true' in raw"),
    ('G2 guard-any-tool', AG, '        if str(row.get("name") or "") != "manage_projects":\n            return False\n        content = row.get("content")', '        content = row.get("content")'),
    ('R1 web-click-by-text', RT, '            if existing.get("wait_code") == "rollup_constraints":', '            if str(existing.get("result_summary") or "").startswith("not closed: constraints"):'),
    # ── review r1/r2 fixes
    ('F1 release-no-restore', MS, '        if hold:\n            _summary = fields.pop("result_summary", None)', '        if False:\n            _summary = fields.pop("result_summary", None)'),
    ('F2 reaper-no-restore', MS, '            if not hold:\n                continue\n            keep_ids.add(r["id"])', '            if True:\n                continue\n            keep_ids.add(r["id"])'),
    ('F3 claim-not-logged', PA, '            if _nxt_held_wait is not None:\n                # §4SU Phase 3 r2', '            if False:\n                # §4SU Phase 3 r2'),
    ('F6 stop-writes-pending', PA, '                store.release_claim(tid, failure_reason="stopped mid-step")', '                store.update_task(tid, status="PENDING", failure_reason="stopped mid-step")'),
    ('F7 crash-writes-pending', PA, '                store.release_claim(nxt.id,\n                                    failure_reason=f"coding executor crashed', '                store.update_task(nxt.id, status="PENDING",\n                                    failure_reason=f"coding executor crashed'),
    ('F8 cascade-over-owner', PL, '        # must block (it was approvable with a FAILED child under it)\n        if self._owner_own_wait(parent):', '        # must block (it was approvable with a FAILED child under it)\n        if False:'),
    ('F9 completion-closes-owner-wait', PL, '        # (a gated one goes on: its roll-up parks it as the gate, r1)\n        if self._owner_own_wait(parent):', '        # (a gated one goes on: its roll-up parks it as the gate, r1)\n        if False:'),
    ('F10 gated-wait-not-reparked', PL, '                and _human_gate_reason(parent.postconditions) is None)\n\n', '                )\n\n'),
    ('F31 gate-park-not-blocked', PL, '        return (parent.status == TaskStatus.NEEDS_USER and parent.wait_code != "human_gate"', '        return (parent.status == TaskStatus.NEEDS_USER and True'),
    ('F32 done-no-unblock', PL, '                self._check_parent_unblock(node.parent_id)\n                self._check_parent_completion(node.parent_id)', '                self._check_parent_completion(node.parent_id)'),
    ('F11 unblock-any-code', PL, '        if parent.status != TaskStatus.BLOCKED or parent.wait_code != "child_failed":', '        if parent.status != TaskStatus.BLOCKED:'),
    ('F12 migrate-gate-as-hold', MS, '                    if _g:\n                        on, code, why = "owner", "human_gate", str(_g)[:300]', '                    if False:\n                        on, code, why = "owner", "human_gate", str(_g)[:300]'),
    ('F13 replan-dedupes', TP, '                if _replan is not None:\n                    # replan: the new plan IS the plan', '                if False:\n                    # replan: the new plan IS the plan'),
    ('F14 replan-no-recheck', TP, '                if _replan["kept_parents"]:', '                if False:'),
    ('F15 replan-archived', TP, '            if str((store.get_project(project_id) or {}).get("status") or "").upper() == "ARCHIVED":', '            if False:'),
    ('F16 replan-stale-running', TP, '                        and _now_ts - float(t.get("updated_at") or 0) < 900]', '                        ]'),
    ('F17 replan-deletes-nothing', TP, '                for _rid in _replan["drop_roots"]:', '                for _rid in []:'),
    ('F18 replan-empty-ok', TP, '            elif _replan is not None:\n                return _err("NOT replanned: no new task was created. Nothing changed.")', '            elif False:\n                return _err("NOT replanned: no new task was created. Nothing changed.")'),
    ('F19 claim-end-unlogged', MS, '                if self.claimed_hold(row["project_id"], task_id) is not None:\n                    self.log_event(row["project_id"], task_id, "hold_claim_ended", {})', '                if False:\n                    self.log_event(row["project_id"], task_id, "hold_claim_ended", {})'),
    ('F20 released-parent-unjudged', PL, '                    self._check_parent_completion(node.id)\n                    if node.status == status:', '                    pass\n                    if node.status == status:'),
    ('F27 reopened-done-parent-rejudged', PL, '                if node.children and prior_status in (TaskStatus.NEEDS_USER, TaskStatus.PAUSED):', '                if node.children:'),
    ('F28 claim-ends-only-leaving', MS, '        if (updated and row and new_status is not None and new_status != "IN_PROGRESS"):', '        if (updated and row and prev_task_status == "IN_PROGRESS" and new_status not in (None, "IN_PROGRESS")):'),
    ('F29 dep-failed-parent-pending', TC, '            leaves = _failed_leaves(tree, d)', '            leaves = [d]'),
    ('F33 dep-hold-as-owner', TC, '        elif st == "NEEDS_USER" and on == "owner_turn":', '        elif False:'),
    ('F34 dep-blocked-is-failed', TC, '        failed_like = st == "FAILED" or (st == "BLOCKED" and code == "child_failed")', '        failed_like = st in ("FAILED", "BLOCKED")'),
    ('F35 done-keeps-refusal-text', PL, '                        or str(parent.result_summary or "").startswith("not closed:")):', '                        ):'),
    ('F30 dep-running-close-it', TC, '            lines.append(f"{d} is running now — wait for it")', '            lines.append(f"{d} is open — do it first")'),
    ('F21 released-parent-no-failure', PL, '                    if node.status == status:\n                        self._check_parent_failure(node.id)', '                    if False:\n                        self._check_parent_failure(node.id)'),
    ('F22 replan-ignores-live-step', TP, '                if unattended_step_live(store, project_id):\n                    _running', '                if False:\n                    _running'),
    ('F23 replan-takes-task-id', TP, '            if task_id and task_id != project_id:\n                # r3', '            if False:\n                # r3'),
    ('F24 replan-partial-kept', TP, '                    for _nid in ids:\n                        try:\n                            store.delete_task(_nid)', '                    for _nid in []:\n                        try:\n                            store.delete_task(_nid)'),
    ('F25 dep-owner-close-it', TC, '            lines.append(f"{d} waits for the owner — tell the owner what it waits on and stop")', '            lines.append(f"{d} is open — do it first")'),
    ('F26 reaper-restore-to-pending', MS, '                logger.warning("orphaned hold %s not restored — left IN_PROGRESS", r["id"], exc_info=True)', '                keep_ids.discard(r["id"])'),
    ('F36 cascade-recodes-model-block', PL, '        if parent.status == TaskStatus.BLOCKED and parent.wait_code != "child_failed":\n            return\n', ''),
    ('F37 migrate-blocked-as-cascade', MS, '                    if not _kid_failed or _own:\n                        code = "blocked"', '                    if False:\n                        code = "blocked"'),
    ('F38 dep-parent-pending', TC, '                lines.append(f"{d} failed — replan it (its subtasks give no single task to redo)")', '                lines.append(f"{d} failed — redo it (task_update status=PENDING) or replan")'),
    ('F39 dep-blocked-leaf-as-failed', TC, '        return [tid] if n.status.value in ("FAILED", "BLOCKED") else []', '        return [tid] if n.status.value == "FAILED" else []'),
    ('F40 direct-done-keeps-refusal', PL, '                                                or str(node.result_summary or "").startswith("not closed:")):', '                                                ):'),
    ('F41 rollup-close-no-unblock', PL, '                self._check_parent_unblock(parent.parent_id)   # r7: a roll-up close lifts the block above too\n                self._check_parent_completion(parent.parent_id, visited)\n\n        elif parent.dependency_type == DependencyType.ANY:', '                self._check_parent_completion(parent.parent_id, visited)\n\n        elif parent.dependency_type == DependencyType.ANY:'),
    ('F42 done-subtree-redone', TC, '    if kids and n.status.value == "DONE":\n        return []', '    if False:\n        return []'),
    ('F44 best-close-no-unblock', PL, '                    parent.failure_reason = "All BEST-dependency children failed"\n                self._check_parent_unblock(parent.parent_id)', '                    parent.failure_reason = "All BEST-dependency children failed"\n                pass'),
]


PINS = ["tests/test_4su_phase3.py", "tests/test_4su_phase4.py", "tests/test_4su_phase0.py", "tests/test_4su_phase1.py"]


def _tests():
    pat = r"wait_on\|wait_code\|NEEDS_USER\|manage_projects\|build_project_briefing\|advance_once\|_awaits\|task_close\|replan"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    batch = [m for m in MUT if m[0].split(" ")[0] in set(args[1:]) or m[0].startswith("CONTROL")] if args[0] == "rerun" else MUT
    full = _tests()
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in batch if originals[f].count(old) != 1]
    if bad:
        print("ANCHOR NOT UNIQUE/FOUND:", bad); sys.exit(2)
    cov = BC.build_map(srcs, full, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        return (sorted(set(ids) | set(PINS)), f"cov {len(ids)} tests + pins") if ids else (PINS, "fallback pins")
    print(f"{len(full)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
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
