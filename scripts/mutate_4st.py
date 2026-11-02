#!/usr/bin/env python3
"""§4ST battery — an experiment task closes DONE only on a kept probe_model transcript (judged on what RAN).
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4st.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUT = [
    ('CONTROL known-bad', 'src/ghost_agent/core/experiment_gate.py', '        if not (by_words or shaped):\n            return None, []', '        if True:\n            return None, []'),
    ('CONTROL no-op', 'src/ghost_agent/core/experiment_gate.py', 'def _named_text_files(ws: Path, texts: Iterable[str]) -> List[str]:', 'def _named_text_files(ws: Path, texts: Iterable[str]) -> List[str]:  # no-op'),
    ('G1 by-words-off', 'src/ghost_agent/core/experiment_gate.py', '        if not (by_words or shaped):', '        if not shaped:'),
    ('G2 shape-off', 'src/ghost_agent/core/experiment_gate.py', '        if not (by_words or shaped):', '        if not by_words:'),
    ('G3 kept-is-delivered-off', 'src/ghost_agent/core/experiment_gate.py', '        if any(bodies.get(rel) == text for rel, text in rec):', '        if False:'),
    ('G3b kept-by-path', 'src/ghost_agent/core/experiment_gate.py', '        if any(bodies.get(rel) == text for rel, text in rec):', '        if any(rel in bodies for rel, text in rec):'),
    ('G16 prose-filter-off', 'src/ghost_agent/core/experiment_gate.py', '                        if rel.lower().endswith(_PROSE_SUFFIXES)\n', '                        if True\n'),
    ('G16b no-md', 'src/ghost_agent/core/experiment_gate.py', '_PROSE_SUFFIXES = (".md", ".txt", ".rst", ".html", ".htm", ".tex")', '_PROSE_SUFFIXES = (".txt", ".rst", ".html", ".htm", ".tex")'),
    ('G17 model-label-off', 'src/ghost_agent/core/experiment_gate.py', '                        and (_MODEL_LABEL_RE.search(t)\n', '                        and (False\n'),
    ('G18 needs-model-off', 'src/ghost_agent/core/experiment_gate.py', '                             or (_OUTPUT_SHAPE_RE.search(t) and _names_a_model(t))))', '                             or _OUTPUT_SHAPE_RE.search(t)))'),
    ('G19 own-off', 'src/ghost_agent/core/experiment_gate.py', '"close\'s result"), sorted(rel for rel in shaped if rel in own\n', '"close\'s result"), sorted(rel for rel in shaped if True\n'),
    ('G20 crash-debug', 'src/ghost_agent/core/experiment_gate.py', '        logger.warning("model run gate skipped (the close stands)", exc_info=True)', '        logger.debug("model run gate skipped (the close stands)", exc_info=True)'),
    ('W14 bulk-waives', 'src/ghost_agent/tools/projects.py', '                            if len(_waive) < MIN_WAIVER_CHARS or len(set(target_ids)) > 1:', '                            if len(_waive) < MIN_WAIVER_CHARS:'),
    ('S1 sim-is-build-input', 'src/ghost_agent/core/project_advancer.py', '        if parts[0] == _QUARANTINE_DIR:\n            return False', '        if False:\n            return False'),
    ('S2 sim-is-brief', 'src/ghost_agent/core/project_advancer.py', '            if "research" not in _segs or _segs[0] == _QUARANTINE_DIR:     # §4ST r2', '            if "research" not in _segs:     # §4ST r2'),
    ('S3 claim-by-heading', 'src/ghost_agent/core/experiment_gate.py', '    r"(?im)^[ \\t>]*[*_]{0,2}[ \\t]*transcripts?[ \\t]*[*_]{0,2}[ \\t]*:")\n#: r2', '    r"(?im)^[ \\t>]*(?:#{1,6}[^\\n]*\\btranscripts?\\b|[*_]{0,2}[ \\t]*transcripts?[ \\t]*[*_]{0,2}[ \\t]*:)")\n#: r2'),
    ('S4 no-dot-slash', 'src/ghost_agent/core/experiment_gate.py', '            while r.startswith("./"):\n                r = r[2:]', '            while False:\n                r = r[2:]'),
    ('S5 build-crash-debug', 'src/ghost_agent/core/project_advancer.py', '        logger.warning("model run gate (build) skipped', '        logger.debug("model run gate (build) skipped'),
    ('S6 quarantine-keeps-record', 'src/ghost_agent/core/experiment_gate.py', '                store.unregister_file(project_id, rel)\n', '                pass\n'),
    ('S7 recovery-promotes-sim', 'src/ghost_agent/core/workspace_cleanup.py', '            for rel in sorted(keep):\n                if rel.split("/", 1)[0] == _QUARANTINE_DIR:', '            for rel in sorted(keep):\n                if False:'),
    ('S8 partial-recovery-promotes-sim', 'src/ghost_agent/core/workspace_cleanup.py', '            for rel in sorted(found):\n                if rel.split("/", 1)[0] == _QUARANTINE_DIR:', '            for rel in sorted(found):\n                if False:'),
    ('Q5 folder-renamed', 'src/ghost_agent/core/experiment_gate.py', '                dest_rel = f"{QUARANTINE_DIR}/{head}/{nb}" if head else f"{QUARANTINE_DIR}/{nb}"', '                dest_rel = f"{QUARANTINE_DIR}/{nb}"'),
    ('G4 result-cite-off', 'src/ghost_agent/core/experiment_gate.py', '            if _cites(str(result_text or "")) or any(_cites(t) for t in bodies.values()):', '            if any(_cites(t) for t in bodies.values()):'),
    ('G5 body-cite-off', 'src/ghost_agent/core/experiment_gate.py', '            if _cites(str(result_text or "")) or any(_cites(t) for t in bodies.values()):', '            if _cites(str(result_text or "")):'),
    ('G6 hop-off', 'src/ghost_agent/core/experiment_gate.py', '                if t and _cites(t):\n                    return None, []', '                if False:\n                    return None, []'),
    ('G6b hop-skips-bodies-off', 'src/ghost_agent/core/experiment_gate.py', '                if rel in bodies:\n                    continue\n                t = _read(ws, rel)', '                t = _read(ws, rel)'),
    ('G7 desc-names-are-outputs', 'src/ghost_agent/core/experiment_gate.py', '        for rel in _named_text_files(ws, [str(result_text or "")]):', '        for rel in _named_text_files(ws, [str(result_text or ""), desc]):'),
    ('G8 result-named-off', 'src/ghost_agent/core/experiment_gate.py', '        for rel in _named_text_files(ws, [str(result_text or "")]):', '        for rel in []:'),
    ('G9 declared-off', 'src/ghost_agent/core/experiment_gate.py', '        for rel in list(files or ()) + [declared_deliverable(desc) or ""]:', '        for rel in list(files or ()):'),
    ('G9b files-off', 'src/ghost_agent/core/experiment_gate.py', '        for rel in list(files or ()) + [declared_deliverable(desc) or ""]:', '        for rel in [declared_deliverable(desc) or ""]:'),
    ('G10 shape-heading-off', 'src/ghost_agent/core/experiment_gate.py', '    r"(?im)^[ \\t>]*(?:#{1,6}[^\\n]*\\btranscripts?\\b"                       # a heading naming transcripts', '    r"(?im)^[ \\t>]*(?:#{1,6}[^\\n]*\\bNEVERMATCHX\\b"                       # a heading naming transcripts'),
    ('G11 shape-label-off', 'src/ghost_agent/core/experiment_gate.py', '    r"|[*_]{0,2}[ \\t]*(?:transcript|model[ \\t]+(?:response|answer|output)s?)"', '    r"|[*_]{0,2}[ \\t]*(?:NEVERMATCHX)"'),
    ('G11b shape-model-label-off', 'src/ghost_agent/core/experiment_gate.py', '    r"|[*_]{0,2}[ \\t]*(?:transcript|model[ \\t]+(?:response|answer|output)s?)"', '    r"|[*_]{0,2}[ \\t]*(?:transcript)"'),
    ('G11c shape-case', 'src/ghost_agent/core/experiment_gate.py', '    r"(?im)^[ \\t>]*(?:#{1,6}[^\\n]*\\btranscripts?\\b"                       # a heading naming transcripts', '    r"(?m)^[ \\t>]*(?:#{1,6}[^\\n]*\\btranscripts?\\b"                       # a heading naming transcripts'),
    ('G13 realpath-off', 'src/ghost_agent/core/experiment_gate.py', '        if not os.path.realpath(target).startswith(real_ws + os.sep):\n            return None\n        with open', '        if False:\n            return None\n        with open'),
    ('G13b links-refused', 'src/ghost_agent/core/experiment_gate.py', '        real_ws = os.path.realpath(ws)\n        if not os.path.realpath(target)', '        real_ws = os.path.realpath(ws)\n        if os.path.islink(target):\n            return None\n        if not os.path.realpath(target)'),
    ('G13c traversal-allowed', 'src/ghost_agent/core/experiment_gate.py', '        if not rel or ".." in rel.split("/"):', '        if not rel:'),
    ('G14 quarantine-dir-walked', 'src/ghost_agent/core/experiment_gate.py', '                _dirs[:] = [d for d in _dirs if d != QUARANTINE_DIR]     # a quarantined run is no data', '                pass'),
    ('G15 problem-mentions', 'src/ghost_agent/core/experiment_gate.py', 'from probe_model: run the prompts with probe_model(save_to=', 'from the model: run the prompts with the tool(save_to='),
    ('Q1 quarantine-off', 'src/ghost_agent/core/experiment_gate.py', '            os.replace(src, dest)\n            moved.append(dest_rel)', '            moved.append(dest_rel)'),
    ('Q2 quarantine-overwrites', 'src/ghost_agent/core/experiment_gate.py', '            while (ws / dest_rel).exists():', '            while False:'),
    ('G12 claim-filter-off', 'src/ghost_agent/core/experiment_gate.py', '                                         and _TRANSCRIPT_CLAIM_RE.search(bodies[rel]))', '                                         )'),
    ('Q3 quarantine-follows-links', 'src/ghost_agent/core/experiment_gate.py', '            if os.path.islink(src) or not src.is_file():', '            if not src.is_file():'),
    ('Q4 quarantine-requarantines', 'src/ghost_agent/core/experiment_gate.py', '            if not rel or ".." in rel.split("/") or rel.split("/")[0] == QUARANTINE_DIR:', '            if not rel or ".." in rel.split("/"):'),
    ('W1 waiver-min-zero', 'src/ghost_agent/tools/projects.py', '                            if len(_waive) < MIN_WAIVER_CHARS or len(set(target_ids)) > 1:', '                            if len(_waive) < 1 or len(set(target_ids)) > 1:'),
    ('W2 waiver-ignored', 'src/ghost_agent/tools/projects.py', '                            if len(_waive) < MIN_WAIVER_CHARS or len(set(target_ids)) > 1:', '                            if True:'),
    ('W3 waiver-note-off', 'src/ghost_agent/tools/projects.py', '                    if _waive_note:\n                        _res_text = ', '                    if False:\n                        _res_text = '),
    ('W4 waived-even-if-demoted', 'src/ghost_agent/tools/projects.py', '                    if _waive_note and plan.tree.nodes[tid].status == TaskStatus.DONE:', '                    if _waive_note:'),
    ('W5 waiver-log-off', 'src/ghost_agent/tools/projects.py', '                                               outcome="experiment gate waived",', '                                               outcome="done",'),
    ('W6 gate-on-reclose', 'src/ghost_agent/tools/projects.py', '                    if not _done_before:\n                        from ..core.experiment_gate', '                    if True:\n                        from ..core.experiment_gate'),
    ('W7 gate-off', 'src/ghost_agent/tools/projects.py', '                        if _unrun:\n                            _waive = ', '                        if False:\n                            _waive = '),
    ('W8 rewrite-file-off', 'src/ghost_agent/tools/projects.py', '                            _gate_files.append(declared_deliverable(_desc_for_gate) or "")', '                            pass'),
    ('W9 rewrite-desc-judged', 'src/ghost_agent/tools/projects.py', '                            plan.tree.nodes[tid].description,\n                            files=_gate_files,', '                            _desc_for_gate,\n                            files=_gate_files,'),
    ('W10 no-deliverables', 'src/ghost_agent/tools/projects.py', '                        _gate_files = list(deliverables or [])', '                        _gate_files = []'),
    ('W11 rejected-key-dropped', 'src/ghost_agent/tools/projects.py', '"waiting_on_dependencies", "unrun_experiments"))', '"waiting_on_dependencies"))'),
    ('W12 payload-key-off', 'src/ghost_agent/tools/projects.py', '                payload["unrun_experiments"] = unrun', '                pass'),
    ('W13 waived-payload-off', 'src/ghost_agent/tools/projects.py', '                payload["experiment_gate_waived"] = waived', '                pass'),
    ('B1 build-gate-off', 'src/ghost_agent/core/project_advancer.py', '            cres = await asyncio.to_thread(_experiment_gate_build, store, project_id, nxt, cres)', '            pass'),
    ('B2 build-quarantine-off', 'src/ghost_agent/core/project_advancer.py', '        _moved = quarantine(store, project_id, _shaped)\n        pretty_log(', '        _moved = []\n        pretty_log('),
    ('B3 build-keeps-files', 'src/ghost_agent/core/project_advancer.py', '        return _dc.replace(cres, ok=False, files=[f for f in (cres.files or []) if f not in _shaped],', '        return _dc.replace(cres, ok=False, files=list(cres.files or []),'),
    ('B4 build-ok-stays', 'src/ghost_agent/core/project_advancer.py', '        return _dc.replace(cres, ok=False, files=[f for f in (cres.files or []) if f not in _shaped],', '        return _dc.replace(cres, ok=True, files=[f for f in (cres.files or []) if f not in _shaped],'),
    ('R1 research-gate-off', 'src/ghost_agent/core/project_advancer.py', '        if _unrun:\n            # nothing is moved here', '        if False:\n            # nothing is moved here'),
    ('R2 research-owner-held', 'src/ghost_agent/core/project_advancer.py', '            if owner_requested:\n                _st, _fail_reason = TaskStatus.FAILED, _unrun', '            if False:\n                _st, _fail_reason = TaskStatus.FAILED, _unrun'),
    ('R3 research-no-file', 'src/ghost_agent/core/project_advancer.py', '                files=[research_path] if research_path else [], result_text=str(result_summary or ""))', '                files=[], result_text=str(result_summary or ""))'),
]


PINS = ["tests/test_4st_experiment_gate.py"]


def _tests():
    pat = r"experiment_gate\|unrun_experiment\|not_model_data\|_finalize_coding\|advance_once\|probe_transcripts"
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
