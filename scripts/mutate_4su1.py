#!/usr/bin/env python3
"""§4SU Phase 1 battery — one close check (core/task_close.check_close) on the model, owner, advancer and roll-up paths.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4su1.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = 'src/ghost_agent/core/task_close.py'
TP = 'src/ghost_agent/tools/projects.py'
PL = 'src/ghost_agent/core/planning.py'
RT = 'src/ghost_agent/api/projects_routes.py'
PA = 'src/ghost_agent/core/project_advancer.py'
MUT = [
    ('CONTROL known-bad', TC, '        v.refused.append(r)\n        v.ok = False\n        return v ', '        continue\n        v.ok = False\n        return v '),
    ('CONTROL no-op', TC, 'def hold_kind(gate: str) -> str:', 'def hold_kind(gate: str) -> str:  # no-op'),
    ('K1 order-swapped', TC, 'GATE_ORDER = ("dependencies", "deliverable",', 'GATE_ORDER = ("deliverable", "dependencies",'),
    ('K2 owner-refused', TC, '        if actor == "owner":\n', '        if False:\n'),
    ('K3 any-actor-waives', TC, '        if actor == "model" and gate in MODEL_WAIVABLE and gate in over:', '        if gate in MODEL_WAIVABLE and gate in over:'),
    ('K4 deliverable-waivable', TC, 'MODEL_WAIVABLE = frozenset({"model_run", "postconditions"})', 'MODEL_WAIVABLE = frozenset({"model_run", "postconditions", "deliverable"})'),
    ('K5 short-waiver-ok', TC, 'if gate == "model_run" and (len(over[gate]) < MIN_WAIVER_CHARS or not single_task):', 'if gate == "model_run" and (not single_task):'),
    ('K6 bulk-waiver-ok', TC, 'if gate == "model_run" and (len(over[gate]) < MIN_WAIVER_CHARS or not single_task):', 'if gate == "model_run" and (len(over[gate]) < MIN_WAIVER_CHARS):'),
    ('K7 probe-old-result', TC, '    if str(result_text or "").strip():\n        probe.result_summary = str(result_text)', '    if False:\n        probe.result_summary = str(result_text)'),
    ('K8 hold-kind-generic', TC, '    return {"deliverable": "undelivered", "model_run": "unrun_experiment"}.get(gate, f"refused_{gate}")', '    return f"refused_{gate}"'),
    ('K9 evidence-off', TC, '    if str(result or "").strip():\n        return None\n    stored', '    if True:\n        return None\n    stored'),
    ('K10 model-run-judges-rewrite', TC, '"model_run": lambda: gate_model_run(store, project_id, stored_desc,', '"model_run": lambda: gate_model_run(store, project_id, judged_desc,'),
    ('K11 rewrite-not-judged', TC, '    judged_desc = description if description else stored_desc', '    judged_desc = stored_desc'),
    ('K12 crash-refuses', TC, '            logger.warning("close gate %s crashed (passes)", gate, exc_info=True)\n            r = None', '            logger.warning("close gate %s crashed (passes)", gate, exc_info=True)\n            r = Refusal(gate, "crashed")'),
    ('T1 undelivered-key-off', TP, '                            undelivered.append(tid)\n                            undelivered_reasons.append(f"{tid}: {_r.reason}")', '                            pass'),
    ('T2 refused-not-listed', TP, '                            continue\n                        refused.append(_rd)', '                            continue\n                        pass'),
    ('T3 not-model-data-dropped', TP, '                        _over.setdefault("model_run", str(not_model_data))', '                        pass'),
    ('T4 model-run-always-skipped', TP, '                        skip=({"model_run"} if _done_before else set()),', '                        skip={"model_run"},'),
    ('T5 waiver-not-returned', TP, '                            overrides_used.append({"task_id": tid, "gate": _g, "reason": _why_w})', '                            pass'),
    ('T6 waiver-still-demoted', TP, '                                       postconditions_waived=bool(_post_waiver))', '                                       postconditions_waived=False)'),
    ('T7 bulk-is-single', TP, '                        single_task=len(set(target_ids)) == 1)', '                        single_task=True)'),
    ('T8 override-not-ledgered', TP, '                                store.append_ledger(project_id, f"OVERRIDE {_g} on task {tid} (by the agent): {_why_w}")', '                                pass'),
    ('T9 postcondition-not-reported', TP, '                            not_closed.append({"task_id": tid, "status": plan.tree.nodes[tid].status.value,\n                                               "why": _r.reason[:300]})', '                            pass'),
    ('T10 waiting-key-off', TP, '                            waiting.append(tid)\n                            waiting_reasons.append(_r.reason)', '                            pass'),
    ('H1 owner-not-recorded', RT, '            _over = list(_v.overridden)', '            _over = []'),
    ('H2 fields-not-first', RT, '        if leftover:\n            try:\n                store.update_task(tid, **leftover)', '        if False:\n            try:\n                store.update_task(tid, **leftover)'),
    ('H3 owner-postcondition-demoted', RT, '_was_done or any(r.gate == "postconditions" for r in _over),', '_was_done,'),
    ('H4 tid-not-canonical', RT, '    tid = _canon_id(tid)          # §4SU Phase 1', '    tid = tid          # §4SU Phase 1'),
    ('R1 checker-not-installed', PL, '        self.tree.close_checker = self._rollup_close_check', '        pass'),
    ('R2 refusal-ignored', PL, '        if not why:\n            return False', '        if True:\n            return False'),
    ('R3 refusal-logged-as-gate', PL, '                    self.store.log_event(self.project_id, _pid_parked, "rollup_refused",', '                    self.store.log_event(self.project_id, _pid_parked, "human_gate_triggered",'),
    ('R4 rollup-skips-deliverable', PL, 'result=result_text, skip={"dependencies", "evidence"})', 'result=result_text, skip={"dependencies", "evidence", "deliverable"})'),
    ('R5 rollup-skips-model-run', PL, 'result=result_text, skip={"dependencies", "evidence"})', 'result=result_text, skip={"dependencies", "evidence", "model_run"})'),
    ('R6 all-not-held', PL, '            if child_statuses and all(s == TaskStatus.DONE for s in child_statuses):\n                if self._rollup_held(parent, self._done_children_text(parent)):', '            if child_statuses and all(s == TaskStatus.DONE for s in child_statuses):\n                if False:'),
    ('R7 not-recorded-parked', PL, '        new_summary = f"not closed: {why}"[:2000]', '        new_summary = ""'),
    ('A1 build-refusal-ignored', PA, '        if not _v.ok:\n            _refusal = _v.first', '        if False:\n            _refusal = _v.first'),
    ('A2 build-hold-kind-lost', PA, '                                hold_kind(_refusal.gate))', '                                _hold)'),
    ('A3 research-skips-file', PA, '                skip={"dependencies", "evidence"})\n        except', '                skip={"dependencies", "evidence", "deliverable", "model_run"})\n        except'),
    ('A4 research-hold-kind-lost', PA, '_st, _why, _hold = TaskStatus.NEEDS_USER, f"{_r.reason} — {_HELD_MARK}", hold_kind(_r.gate)', '_st, _why, _hold = TaskStatus.NEEDS_USER, f"{_r.reason} — {_HELD_MARK}", ""'),
    ('F1 not-closed-is-evidence', TP, ' or low.startswith("not closed:")\n', '\n'),
    ('F2 postcondition-echo', TC, '        probe.result_summary = _stored_evidence(getattr(node, "result_summary", ""))', '        pass'),
    ('F3 done-parent-rejudged', PL, '        if parent.status == TaskStatus.DONE:\n            # r1 M2', '        if False:\n            # r1 M2'),
    ('F4 digest-misses-refusal', 'src/ghost_agent/core/project_digest.py', '"human_gate_triggered", "rollup_refused")', '"human_gate_triggered")'),
    ('F5 refusal-text-kept', PL, '                if str(parent.result_summary or "").startswith("not closed:"):', '                if False:'),
    ('F6 re-click-records', RT, '        if st == TaskStatus.DONE and not _was_done:', '        if st == TaskStatus.DONE:'),
    ('F7 owner-no-constraints', RT, '                                         result=_result, constraints=_cons)', '                                         result=_result)'),
    ('F8 fileless-override-recorded', TP, '                                    and _files_for_task(store, project_id, tid, deliverables))', '                                    )'),
    ('F9 refused-labelled-postcondition', PA, '    elif not _step_ok and _refused_gate:', '    elif False:'),
    ('G1 reclick-demotes', RT, 'postconditions_waived=_was_done or any(', 'postconditions_waived=any('),
    ('G2 refusal-text-kept-on-done', PL, '            elif status == TaskStatus.DONE and str(node.result_summary or "").startswith("not closed:"):', '            elif False:'),
    ('G3 children-system-text', PL, '_stored_evidence(self.nodes[c].result_summary) for c in parent.children', 'str(self.nodes[c].result_summary or "") for c in parent.children'),
    ('G4 parked-not-reported', RT, '        _parked = [p for p in', '        _parked = [] and [p for p in'),
    ('I1 best-raw-text', PL, 'self._rollup_held(parent, self._evidence_of(best.result_summary))', 'self._rollup_held(parent, str(best.result_summary or ""))'),
    ('I2 restate-off', PL, '                or (str(parent.result_summary or "").startswith("not closed:")\n                    and parent.result_summary != new_summary)):', '                or False):'),
    ('I3 restate-any-park', PL, '                or (str(parent.result_summary or "").startswith("not closed:")\n                    and parent.result_summary != new_summary)):', '                or (parent.result_summary != new_summary)):'),
    ('J1 best-by-raw-length', PL, "best = max(done_children, key=lambda c: (len(self._evidence_of(c.result_summary)),", "best = max(done_children, key=lambda c: (len(c.result_summary or ''),"),
    ('J2 gate-instruction-by-status', TP, '_parked = [n["task_id"] for n in not_closed if str(n["why"]).startswith("human gate:")]', '_parked = [n["task_id"] for n in not_closed if n["status"] == "NEEDS_USER"]'),
    ('J3 no-dedupe', TP, '                not_closed = list({n["task_id"]: n for n in not_closed}.values())', '                not_closed = list(not_closed)'),
    ('J4 digest-no-dedupe', 'src/ghost_agent/core/project_digest.py', '                if ev.get("task_id") and _tk in _nu_seen:', '                if False:'),
    ('Q1 audit-misses-declared', TP, '                    if _decl_a and _decl_a not in _audit_rels:\n                        _audit_rels.append(_decl_a)', '                    pass'),
    ('Q2 owner-audit-not-recorded', RT, '            if str(existing.get("result_summary") or "").startswith("not closed: constraints:"):', '            if False:'),
    ('P1 constrained-parent-closes', PL, '                if _files_for_task(self.store, self.project_id, parent.id, [decl] if decl else []):', '                if False:'),
    ('P2 project-constraints-ignored', PL, '            cons += [str(c) for c in (meta.get("constraints") or [])]', '            pass'),
    ('P3 fileless-held', PL, '                if _files_for_task(self.store, self.project_id, parent.id, [decl] if decl else []):', '                if True:'),
    ('O1 audit-reclose-held', TP, '                            and not _task_audit[0] and _done_before):', '                            and not _task_audit[0] and False):'),
    ('O2 instruction-not-closed-always', TC, '    if _stay == len(refused):', '    if False:'),
    ('O3 digest-touched-early', 'src/ghost_agent/core/project_digest.py', '            else:  # needs-user\n', '            else:  # needs-user\n                touched.add(pid)\n'),
    ('N1 reclose-note-off', TP, '                        if _done_before:\n                            # r5/r7: a refused RE-close', '                        if False:\n                            # r5/r7: a refused RE-close'),
    ('N2 reclose-legacy-keys', TP, '                            refused.append(_rd)\n                            continue\n', '                            refused.append(_rd)\n'),
    ('N3 stale-entry-kept', TP, '                        not_closed[:] = [n for n in not_closed if n["task_id"] != tid]', '                        pass'),
    ('N4 digest-shows-done', 'src/ghost_agent/core/project_digest.py', '                    if _t_now.get("status") == "DONE":\n                        continue', '                    if False:\n                        continue'),
]


PINS = ["tests/test_4su_phase1.py", "tests/test_4su_phase0.py", "tests/test_4st_experiment_gate.py", "tests/test_4pg_autopilot_review.py"]


def _tests():
    pat = r"task_update\|update_status\|check_close\|advance_once\|update_task\|postcondition\|summarize_since"
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
