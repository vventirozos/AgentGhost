#!/usr/bin/env python3
"""§4SU Phase 5 battery — autopilot opt-in, held tasks re-checked, the batch goes past a model experiment.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4su5.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUT = [
    ('CONTROL known-bad', 'src/ghost_agent/core/project_advancer.py', '        if kind not in HOLD_KINDS_RECHECKED:\n            return None', '        if False:\n            return None'),
    ('CONTROL no-op', 'src/ghost_agent/core/project_advancer.py', 'async def _recheck_held(context, store, plan, project_id, nxt, held_summary):', 'async def _recheck_held(context, store, plan, project_id, nxt, held_summary):  # no-op'),
    ('A1 autoadvance-turns-on', 'src/ghost_agent/tools/projects.py', '                    if _prev.get("autopilot"):\n                        from ..core.project_advancer import resume_autopilot_metadata', '                    if True:\n                        from ..core.project_advancer import resume_autopilot_metadata'),
    ('A3 off-silent', 'src/ghost_agent/tools/projects.py', '                        _autopilot_note = ("autopilot stays OFF: only this batch runs.', '                        _autopilot_note = ("" and "autopilot stays OFF: only this batch runs.'),
    ('B1 recheck-off', 'src/ghost_agent/core/project_advancer.py', '    if _nxt_held_summary:\n        _rc = await', '    if False:\n        _rc = await'),
    ('B2 any-kind', 'src/ghost_agent/core/project_advancer.py', 'HOLD_KINDS_RECHECKED = frozenset({"unverified_build", "undelivered", "unrun_experiment"})', 'HOLD_KINDS_RECHECKED = frozenset({"unverified_build", "undelivered", "unrun_experiment", "verify_task", "research_missed"})'),
    ('B3 missing-file-ok', 'src/ghost_agent/core/project_advancer.py', '            if not p.is_file() or p.is_symlink() or p.stat().st_size == 0:\n                return None', '            if False:\n                return None'),
    ('B4 empty-file-ok', 'src/ghost_agent/core/project_advancer.py', '            if not p.is_file() or p.is_symlink() or p.stat().st_size == 0:', '            if not p.is_file() or p.is_symlink():'),
    ('B5 deliverable-ignored', 'src/ghost_agent/core/project_advancer.py', '        if deliverable_problem(str(ws), nxt.description):\n            return None', '        if False:\n            return None'),
    ('B6 experiment-ignored', 'src/ghost_agent/core/project_advancer.py', '                                   files=files, result_text=result):\n            return None', '                                   files=files, result_text=result) and False:\n            return None'),
    ('B7 hold-text-kept', 'src/ghost_agent/core/project_advancer.py', '        result = held_summary[: -len(" — " + reason)] if reason and held_summary.endswith(" — " + reason) else held_summary', '        result = held_summary'),
    ('B8 kind-not-recorded', 'src/ghost_agent/core/project_advancer.py', '"files": list(cres.files or [])[:8], "hold": _hold_kind_of(_why)})', '"files": list(cres.files or [])[:8]})'),
    ('B9 verify-kind', 'src/ghost_agent/core/project_advancer.py', '    if w.startswith("needs a real check"):\n        return "verify_task"', '    if w.startswith("needs a real check"):\n        return "unverified_build"'),
    ('B10 research-kind-lost', 'src/ghost_agent/core/project_advancer.py', '        step_payload["hold"] = _hold or _hold_kind_of(_why)', '        step_payload["hold"] = _hold_kind_of(_why)'),
    ('B11 unverified-no-files-ok', 'src/ghost_agent/core/project_advancer.py', '        if kind == "unverified_build" and not files:\n            return None', '        if False:\n            return None'),
    ('C1 experiment-stops', 'src/ghost_agent/core/project_advancer.py', '            if owner_requested and len(_experiments) <= hard_cap:\n                continue', '            if False:\n                continue'),
    ('C2 no-exclude', 'src/ghost_agent/core/project_advancer.py', '                                     exclude_ids=set(_experiments), **_kw)', '                                     exclude_ids=set(), **_kw)'),
    ('C3 exclude-ignored', 'src/ghost_agent/core/project_advancer.py', '        _ex = set(exclude_ids or ())', '        _ex = set()'),
    ('C4 final-stop-lost', 'src/ghost_agent/core/project_advancer.py', '        stop_reason = "model_experiment"\n    return AdvanceManyResult(', '        pass\n    return AdvanceManyResult('),
    ('C5 instruction-names-last', 'src/ghost_agent/tools/projects.py', '        _exp = next((a for a in batch.advanced if a.get("task_id")\n                     and str(a.get("summary") or "").startswith("model experiment")),', '        _exp = next((a for a in reversed(batch.advanced) if a.get("task_id")\n                     and str(a.get("summary") or "").startswith("")),'),
    ('C6 experiments-not-returned', 'src/ghost_agent/core/project_advancer.py', 'list(_gated), list(_held), sorted(_not_run), list(_experiments))', 'list(_gated), list(_held), sorted(_not_run), [])'),
    ('C7 unattended-continues', 'src/ghost_agent/core/project_advancer.py', '            if owner_requested and len(_experiments) <= hard_cap:', '            if len(_experiments) <= hard_cap:'),
    ('R1 desc-changed-ok', 'src/ghost_agent/core/project_advancer.py', '        if str(pl.get("description") or "") != str(nxt.description or ""):\n            return None', '        if False:\n            return None'),
    ('R2 gate-ignored', 'src/ghost_agent/core/project_advancer.py', '        if enforce_human_gate(store.get_task(nxt.id) or {}):\n            return None', '        if False:\n            return None'),
    ('R3 postcondition-unchecked', 'src/ghost_agent/core/project_advancer.py', '            if plan.tree._check_postconditions(_probe):\n                return None', '            if False:\n                return None'),
    ('R4 outside-ok', 'src/ghost_agent/core/project_advancer.py', '                p.resolve().relative_to(ws)   # r1: confined to the workspace', '                pass'),
    ('R5 not-registered', 'src/ghost_agent/core/project_advancer.py', '                store.register_file_artifact(nxt.id, rel, description=_short_summary(result)[:180])', '                pass'),
    ('R6 deadline-converted', 'src/ghost_agent/core/project_advancer.py', '    if _experiments and stop_reason in ("count_reached", "hard_cap", "project_done", "open_but_waiting"):', '    if _experiments and stop_reason in ("count_reached", "hard_cap", "project_done", "open_but_waiting", "deadline"):'),
    ('R7 race-reset', 'src/ghost_agent/tools/projects.py', '                            if m.get("autopilot"):\n                                m.update(resume_autopilot_metadata())', '                            if True:\n                                m.update(resume_autopilot_metadata())'),
    ('A2 on-not-reset', 'src/ghost_agent/tools/projects.py', '                        _bump_meta(store, project_id, _reset_if_on)', '                        pass'),
]









PINS = ["tests/test_4su_phase5.py", "tests/test_4sr_gate_words.py", "tests/test_4lp_projects.py", "tests/test_4mq_long_horizon.py"]


def _tests():
    pat = r"advance_once\|advance_many\|autoadvance\|autopilot\|_HELD_MARK\|model_experiment\|held"
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
