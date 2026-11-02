#!/usr/bin/env python3
"""§4SU Phase 0 battery — the confirmed project-system bugs (human gate, system-text evidence, refused
closes reported, pause kept, parents unblock, stuck autopilot, HTTP advance, sweep → trash).
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4su0.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUT = [
    ('CONTROL known-bad', 'src/ghost_agent/core/planning.py', '                if (gate_reason is not None and not human_approved\n                        and prior_status != TaskStatus.DONE):', '                if False:'),
    ('CONTROL no-op', 'src/ghost_agent/core/workspace_cleanup.py', 'def project_trash_dir(store) -> Optional[Path]:', 'def project_trash_dir(store) -> Optional[Path]:  # no-op'),
    ('H1 parked-is-approval', 'src/ghost_agent/core/planning.py', '                        and prior_status != TaskStatus.DONE):', '                        and prior_status not in (TaskStatus.NEEDS_USER, TaskStatus.DONE)):'),
    ('H7 park-not-logged', 'src/ghost_agent/tools/projects.py', '                            store.log_event(project_id, tid, "human_gate_parked",', '                            store.log_event(project_id, tid, "human_gate_parked_x",'),
    ('H9 approval-not-passed', 'src/ghost_agent/tools/projects.py', '                                       human_approved=_approved)', '                                       human_approved=False)'),
    ('H10 web-not-approval', 'src/ghost_agent/api/projects_routes.py', '            human_approved=True,\n        )', '            human_approved=False,\n        )'),
    ('N1 not-closed-off', 'src/ghost_agent/tools/projects.py', '                        not_closed.append({"task_id": tid, "status": _after.value,', '                        (lambda *_: None)({"task_id": tid, "status": _after.value,'),
    ('N2 payload-off', 'src/ghost_agent/tools/projects.py', '                payload["not_closed"] = not_closed', '                pass'),
    ('N3 unregister-off', 'src/ghost_agent/tools/projects.py', '                                    store.unregister_file(project_id, _rel)', '                                    pass'),
    ('N4 unregister-all', 'src/ghost_agent/tools/projects.py', '                                for _rel in set(store.list_deliverables(project_id)) - _reg_before:', '                                for _rel in set(store.list_deliverables(project_id)):'),
    ('N5 ledger-always', 'src/ghost_agent/tools/projects.py', '                    elif st_enum == TaskStatus.DONE and (ledger or "").strip() and project_id:', '                    if st_enum == TaskStatus.DONE and (ledger or "").strip() and project_id:'),
    ('N6 ledger-off', 'src/ghost_agent/tools/projects.py', '                            store.append_ledger(project_id, ledger.strip())\n                        except Exception:\n                            logger.debug("ledger append skipped", exc_info=True)\n                    if _waive_note', '                            pass\n                        except Exception:\n                            logger.debug("ledger append skipped", exc_info=True)\n                    if _waive_note'),
    ('E1 held-is-evidence', 'src/ghost_agent/tools/projects.py', '    if (_HELD_MARK in low or low.startswith("human gate:") or low == "flagged for human review"', '    if (low.startswith("human gate:") or low == "flagged for human review"'),
    ('E2 gate-is-evidence', 'src/ghost_agent/tools/projects.py', '    if (_HELD_MARK in low or low.startswith("human gate:") or low == "flagged for human review"', '    if (_HELD_MARK in low or low == "flagged for human review"'),
    ('E3 flag-is-evidence', 'src/ghost_agent/tools/projects.py', '    if (_HELD_MARK in low or low.startswith("human gate:") or low == "flagged for human review"', '    if (_HELD_MARK in low or low.startswith("human gate:")'),
    ('E4 experiment-is-evidence', 'src/ghost_agent/tools/projects.py', '            or low.startswith(MODEL_EXPERIMENT_PREFIX.lower())):', '            ):'),
    ('E5 nothing-is-evidence', 'src/ghost_agent/tools/projects.py', '        return ""\n    return text\n', '        return ""\n    return ""\n'),
    ('G1 gate-rewritable', 'src/ghost_agent/tools/projects.py', '                    extras["postconditions"] = list(postconditions) + _kept_gates', '                    extras["postconditions"] = list(postconditions)'),
    ('R1 rollup-gate-off', 'src/ghost_agent/core/planning.py', '        reason = _human_gate_reason(parent.postconditions)\n        if reason is None:\n            return False', '        reason = None\n        if reason is None:\n            return False'),
    ('R2 rollup-any-off', 'src/ghost_agent/core/planning.py', '            if any(s == TaskStatus.DONE for s in child_statuses):\n                if self._park_gated(parent):\n                    return', '            if any(s == TaskStatus.DONE for s in child_statuses):\n                if False:\n                    return'),
    ('R3 unblock-model-block', 'src/ghost_agent/core/planning.py', '        if (parent.failure_reason or "") not in ("", "All ANY-dependency children failed"):\n            return', '        if False:\n            return'),
    ('P1 park-unregisters', 'src/ghost_agent/tools/projects.py', '                        if _reg_before is not None and _after == TaskStatus.FAILED:', '                        if _reg_before is not None:'),
    ('P2 park-why-result', 'src/ghost_agent/tools/projects.py', '                        _why_not = (f"human gate: {_gr}" if _after == TaskStatus.NEEDS_USER and _gr', '                        _why_not = (f"human gate: {_gr}" if False'),
    ('S2 stuck-any-failed', 'src/ghost_agent/core/project_advancer.py', '        if not culprits:\n            return', '        if not failed_ids:\n            return'),
    ('T1 trash-off', 'src/ghost_agent/core/workspace_cleanup.py', '                if not fpath.is_symlink() and _to_project_trash(store, project_id, root, fpath, rel, _trash_stamp):', '                if False:'),
    ('T9 moved-is-deleted', 'src/ghost_agent/core/workspace_cleanup.py', '            _gone = [r for r in deleted if r not in set(_moved)]', '            _gone = []'),
    ('D1 claim-release-unpauses', 'src/ghost_agent/memory/projects.py', '                    and prev_task_status != "IN_PROGRESS"):', '                    ):'),
    ('C1 unblock-off', 'src/ghost_agent/core/planning.py', '                self._check_parent_unblock(node.parent_id)\n', '                pass\n'),
    ('C2 unblock-ignores-failed', 'src/ghost_agent/core/planning.py', '            still = any(st in failed for st in child_statuses)', '            still = False'),
    ('C3 unblock-no-climb', 'src/ghost_agent/core/planning.py', '        self._check_parent_unblock(parent.parent_id, visited)', '        pass'),
    ('C4 unblock-any-type', 'src/ghost_agent/core/planning.py', '            still = bool(child_statuses) and all(st in failed for st in child_statuses)', '            still = any(st in failed for st in child_statuses)'),
    ('S1 stuck-off', 'src/ghost_agent/core/project_advancer.py', '        _pause_if_stuck(context, store, project_id)\n', '        pass\n'),
    ('S3 stuck-while-running', 'src/ghost_agent/core/project_advancer.py', '        if any(st(t) == "IN_PROGRESS" for t in tasks):\n            return', '        if False:\n            return'),
    ('V1 http-live-off', 'src/ghost_agent/api/projects_routes.py', '    if _live:\n        raise HTTPException(409,', '    if False:\n        raise HTTPException(409,'),
    ('T2 event-off', 'src/ghost_agent/core/workspace_cleanup.py', '            store.log_event(project_id, None, "workspace_sweep", {', '            (lambda *a: None)(project_id, None, "workspace_sweep", {'),
    ('T3 purge-off', 'src/ghost_agent/core/workspace_cleanup.py', '            purge_project_trash(store)\n', '            pass\n'),
    ('T4 purge-keeps-old', 'src/ghost_agent/core/workspace_cleanup.py', '        if now - day > (PROJECT_TRASH_KEEP_DAYS + 1) * 86400:', '        if False:'),
    ('T5 purge-everything', 'src/ghost_agent/core/workspace_cleanup.py', '        if now - day > (PROJECT_TRASH_KEEP_DAYS + 1) * 86400:', '        if True:'),
    ('T6 trash-in-sandbox', 'src/ghost_agent/core/workspace_cleanup.py', '    return (Path(md).parent / PROJECT_TRASH_DIRNAME) if md else None', '    return None'),
    ('T7 log-named-off', 'src/ghost_agent/core/workspace_cleanup.py', '            if _is_debris(rel) and not rel.lower().endswith(".log"):', '            if _is_debris(rel):'),
    ('T8 trash-overwrites', 'src/ghost_agent/core/workspace_cleanup.py', '        while dest.exists():\n            n += 1', '        while False:\n            n += 1'),
    ('H4 not-parked-approves', 'src/ghost_agent/tools/projects.py', '    return status_now == TaskStatus.NEEDS_USER and _owner_released(', '    return _owner_released('),
    ('H2 same-request-approves', 'src/ghost_agent/tools/projects.py', '        return str((ev.get("payload") or {}).get("req") or "") != _current_request_id()', '        return True'),
    ('H5 no-record-approves', 'src/ghost_agent/tools/projects.py', '        if ev is None:\n            return False\n        return str(', '        if ev is None:\n            return True\n        return str('),
    ('H6 advancer-event-ignored', 'src/ghost_agent/tools/projects.py', '("human_gate_parked", "human_gate_triggered"))\n        if ev is None', '("human_gate_parked",))\n        if ev is None'),
    ('G3 delete-own-only', 'src/ghost_agent/tools/projects.py', '                    if _c.get("parent_id") == _cur and _c["id"] not in _sub:', '                    if False:'),
    ('G4 delete-no-park-event', 'src/ghost_agent/tools/projects.py', '                for _c in _held:\n                    store.log_event(_pid_t, _c["id"], "human_gate_parked",', '                for _c in []:\n                    store.log_event(_pid_t, _c["id"], "human_gate_parked",'),
    ('G5 delete-done-gate-holds', 'src/ghost_agent/tools/projects.py', '                      and str(_c.get("status") or "").upper() != "DONE"\n', '\n'),
    ('M1 rollup-no-event', 'src/ghost_agent/core/planning.py', '        for _pid_parked in list(getattr(self.tree, "rollup_parked", []) or []):', '        for _pid_parked in []:'),
    ('M2 rollup-not-reported', 'src/ghost_agent/tools/projects.py', '                    for _rp in list(getattr(plan.tree, "rollup_parked", []) or []):', '                    for _rp in []:'),
    ('L1 latest-oldest', 'src/ghost_agent/memory/projects.py', '+ ",".join("?" * len(types)) + ") ORDER BY id DESC LIMIT 1",', '+ ",".join("?" * len(types)) + ") ORDER BY id ASC LIMIT 1",'),
    ('S4 stuck-despite-runnable', 'src/ghost_agent/core/project_advancer.py', '            if not h:\n                return', '            if False:\n                return'),
    ('S5 stuck-no-block-walk', 'src/ghost_agent/core/project_advancer.py', '                    out |= {i for i in failed_ids if _under(i, p)}', '                    pass'),
    ('S6 stuck-no-dep-chain', 'src/ghost_agent/core/project_advancer.py', '                    out |= _held(by_id[d], seen)', '                    pass'),
    ('G2 delete-gated-free', 'src/ghost_agent/tools/projects.py', '            _held = [_c for _c in _gated if not _owner_released(\n', '            _held = [_c for _c in [] if not _owner_released(\n'),
    ('H3 any-turn-approves', 'src/ghost_agent/tools/projects.py', '        if not _owner_turn_now():\n            return False\n        task = store.get_task(tid)', '        if False:\n            return False\n        task = store.get_task(tid)'),
    ('P3 project-done-past-gate', 'src/ghost_agent/tools/projects.py', '                    if _open_gated:\n                        return _err("NOT updated: the project', '                    if False:\n                        return _err("NOT updated: the project'),
    ('P4 project-done-gate-on-done', 'src/ghost_agent/tools/projects.py', '                    _open_gated = [t for t in (store.list_tasks(project_id) or [])\n                                   if str(t.get("status") or "").upper() != "DONE"\n', '                    _open_gated = [t for t in (store.list_tasks(project_id) or [])\n                                   if True\n'),
    ('M3 rollup-copies-summary', 'src/ghost_agent/core/planning.py', '                    if self._park_gated(parent):\n                        # r3: park BEFORE copying', '                    parent.result_summary = best.result_summary\n                    if self._park_gated(parent):\n                        # r3: park BEFORE copying'),
    ('H16 queued-ignored', 'src/ghost_agent/tools/projects.py', '            if any(_HOLD_BACK_RE.search(_gate_text(later)) for later in (added_user_messages() or [])):', '            if False:'),
    ('H18 id-anywhere', 'src/ghost_agent/tools/projects.py', '_TASK_REF_RE = re.compile(r"(?<![0-9a-z])[0-9a-f]{6,32}(?![0-9a-z])")', '_TASK_REF_RE = re.compile(r"[0-9a-f]{6,32}")'),
    ('H19 short-prefix', 'src/ghost_agent/tools/projects.py', '[0-9a-f]{6,32}(?![0-9a-z])")', '[0-9a-f]{1,32}(?![0-9a-z])")'),
    ('H22 curly-apostrophe', 'src/ghost_agent/tools/projects.py', '.replace("\\u2019", "\'")', ''),
    ('G6 project-delete-free', 'src/ghost_agent/tools/projects.py', '            if _held_g:\n                for t in _held_g:', '            if False:\n                for t in _held_g:'),
    ('G7 project-delete-approve-word', 'src/ghost_agent/tools/projects.py', 'and not _owner_released(store, rid, t["id"], context, verbs=(_DELETE_WORD_RE,))]', 'and not _owner_released(store, rid, t["id"], context, verbs=(_APPROVE_WORD_RE, _DELETE_WORD_RE))]'),
    ('H20 delete-approve-word', 'src/ghost_agent/tools/projects.py', 'store, _pid_t, _c["id"], context, verbs=(_DELETE_WORD_RE,))]', 'store, _pid_t, _c["id"], context, verbs=(_APPROVE_WORD_RE, _DELETE_WORD_RE))]'),
    ('H21 instruction-plain-yes', 'src/ghost_agent/tools/projects.py', '"message: ask them now to reply \\"approve <task id>\\" with the id shown here, and stop. "', '"message: ask them now and stop. "'),
    ('H8 park-no-req', 'src/ghost_agent/tools/projects.py', '    return {"reason": str(reason)[:200], "req": _current_request_id()}', '    return {"reason": str(reason)[:200]}'),
    ('H11 name-ignored', 'src/ghost_agent/tools/projects.py', '        if not _message_names_task(ask, task, rivals, verbs):\n            return False', '        if False:\n            return False'),
    ('H12 verb-not-bound', 'src/ghost_agent/tools/projects.py', '            if mm and tid.startswith(mm.group(1)):', '            if True:'),
    ('H25 others-named-ok', 'src/ghost_agent/tools/projects.py', '    if named != {tid}:\n        return False', '    if tid not in named:\n        return False'),
    ('H27 stale-context', 'src/ghost_agent/tools/projects.py', '        ask = _gate_text(_owner_message_now(context))\n', '        ask = _gate_text(getattr(context, "last_user_content", ""))\n'),
    ('H15 hold-back-ignored', 'src/ghost_agent/tools/projects.py', '        if not ask.strip() or _HOLD_BACK_RE.search(ask):', '        if not ask.strip():'),
    ('H23 question-approves', 'src/ghost_agent/tools/projects.py', '[?;;？¿]|', ''),
    ('H24 nt-approves', 'src/ghost_agent/tools/projects.py', '"would|could|must|need|ai)n\'?t(?!\\w)|', '"would|could|must|need|ai)n\'t(?!\\w)|'),
    ('H28 conditions-approve', 'src/ghost_agent/tools/projects.py', 'r"until|till|before|only|provided|assuming|"', 'r""'),
    ('H29 nt-too-wide', 'src/ghost_agent/tools/projects.py', 'r"(?:(?<!\\w)(?:do|does|did|ca|can|wo|is|are|was|were|have|has|had|should|"', 'r"(?:(?<!\\w)(?:\\w+|"'),
]







PINS = ["tests/test_4su_phase0.py"]


def _tests():
    pat = (r"HUMAN_GATE\|human_gate\|human_approved\|postcondition\|task_update\|PAUSED\|BLOCKED\|"
           r"advance_unattended\|advance_project\|sweep_project_workspace\|workspace_sweep\|project_trash")
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
