#!/usr/bin/env python3
"""Whole-file mutation battery for the 2026-09-13 §4GJ fixes (lint gate, pin ratchet, memory reconciler, escalation ordering, §R R2).

WHY THIS LIVES IN THE REPO: the numbers quoted in PROJECT_JOURNAL §4GJ
(see PROJECT_JOURNAL §4GJ for the score) must be re-runnable, not a claim from a scratch directory that is
deleted with the session (the §4DG lesson behind scripts/mutate_fs_ledger.py).

Runs against a COPY of the repo — never the live tree:

    rsync -a --exclude __pycache__ src tests scripts docs pytest.ini /tmp/mtree/
    mkdir -p /tmp/pristine && cp -R /tmp/mtree/src /tmp/pristine/   # sibling of the tree
    PYTHONPATH=src python scripts/mutate_4gj_2026_09_13.py --tree /tmp/mtree [--slice a:b] [--list]

One mutant per run, py_compile before trusting a verdict, __pycache__ purged,
signals trapped, the tree restored from the pristine copy and HASH-checked
after every mutant. The harness never names the expected killer: every
mutant runs the same full test set. The "pre-fix tree" control needs the
pre-fix copies of the three files in ../orig next to the tree; it is
reported SKIPPED when they are absent (the §4GG run had them: every new pin
failed on that tree — 24/25, 7/8, 10/12 and an ImportError).
"""
import hashlib, os, re, shutil, signal, subprocess, sys, time

ARGV = sys.argv
TREE = ARGV[ARGV.index("--tree") + 1]
PRISTINE = os.path.join(os.path.dirname(TREE.rstrip("/")), "pristine")
ORIG = (ARGV[ARGV.index("--orig") + 1] if "--orig" in ARGV
        else os.path.join(os.path.dirname(TREE.rstrip("/")), "..", "orig"))   # pre-fix files
PY = sys.executable
# Every file the section touched: restored from the pristine copy after every
# mutant. Forks own disjoint slices; see the journal entry for who owned what.
FILES = [
    "src/ghost_agent/core/agent.py",
    "src/ghost_agent/sandbox/docker.py",
    "src/ghost_agent/sandbox/egress_gate.py",
    "src/ghost_agent/memory/vector.py",
    "src/ghost_agent/memory/frontier.py",
    "src/ghost_agent/memory/episodes.py",
    "src/ghost_agent/tools/memory.py",
    "src/ghost_agent/core/dream.py",
    "src/ghost_agent/tools/file_system.py",
    "scripts/if_bench.py",
    "scripts/if_bench_combine.py",
    "tests/test_pin_quality_ratchet.py",
    "scripts/lint.py",
    "src/ghost_agent/tools/registry.py",
    "src/ghost_agent/memory/temporal.py",
    "tests/test_lint_gate.py",
    "src/ghost_agent/tools/browser.py",
    "src/ghost_agent/tools/search.py",
    "src/ghost_agent/tools/darkweb_search.py",
    "src/ghost_agent/sandbox/jobs.py",
    "tests/test_4gi_review_round2.py",
    "tests/test_egress_failclosed_4gi.py",
    "src/ghost_agent/api/routes.py",
    "tests/test_4gi_api_routes.py",
    "src/ghost_agent/tools/projects.py",
    "src/ghost_agent/core/project_advancer.py",
    "src/ghost_agent/core/project_research.py",
    "src/ghost_agent/core/coding_loop.py",
    "src/ghost_agent/core/isolation.py",
    "src/ghost_agent/core/workspace_cleanup.py",
    "tests/test_4gi_symlink_class.py",
]
AGENT, DOCKER, GATE, VECTOR, FRONTIER, EPISODES, MEMTOOL, DREAM, FS, IFBENCH, IFCOMB, RATCHET, LINT, REGISTRY, TEMPORAL, LINTTEST, BROWSER, SEARCH, DARKWEB, JOBS, RR2TEST, EFCTEST, ROUTES, ROUTESTEST, PROJECTS, ADVANCER, RESEARCH, CODINGLOOP, ISOLATION, CLEANUP, SYMTEST = FILES

_VERDICT = ["tests/test_mechanical_refute_beats_escalation.py",
            "tests/test_forced_final_no_answer.py", "tests/test_4fn_judge_fixes.py",
            "tests/test_escalation_discipline.py", "tests/test_verdict_fact_recording.py",
            "tests/test_evidence_gate_real_rows.py"]
_SYM = ["tests/test_4gi_symlink_class.py", "tests/test_workspace_zip_symlinks_4gj.py",
        "tests/test_4gj_r3_call_sites.py",
        "tests/test_file_system_replace.py", "tests/test_model_writable_root_ops.py",
        "tests/test_tool_layer_hardening_4dx.py", "tests/test_4gi_api_routes.py"]
_SANDBOX = ["tests/test_egress_gate_consumer_4gi.py", "tests/test_egress_failclosed_4gi.py",
            "tests/test_sandbox_tor_egress.py", "tests/test_4gi_review_round2.py",
            "tests/test_registry_guard_4gi.py", "tests/test_4gi_symlink_class.py"]
_MEMORY = ["tests/test_memory_store_reconcile_4gj.py", "tests/test_memory_audit_fixes_2026_07_26.py",
           "tests/test_episodes_recall_fixes.py", "tests/test_rag_document_qa.py"]
_API = ["tests/test_4gi_api_routes.py", "tests/test_workspace_zip_symlinks_4gj.py", "tests/test_4gi_review_round2.py",
        "tests/test_verdict_fact_recording.py", "tests/test_calibration_audit.py"]
_TOOLS3 = ["tests/test_browser_wallclock_ceiling.py", "tests/test_research_outcome_status.py",
           "tests/test_fact_check.py", "tests/test_egress_gate_consumer_4gi.py"]
_BENCH = ["tests/test_if_bench_bank.py", "tests/test_if_bench_combine_bands.py"]
_LINT = ["tests/test_lint_gate.py"]
_PINS = ["tests/test_pin_quality_ratchet.py", "tests/test_outcome_consumers_r7.py",
         "tests/test_outcome_consumers_r6.py", "tests/test_outcome_consumers_r5.py",
         "tests/test_outcome_consumers_r4.py", "tests/test_outcome_consumers_r3.py"]
TESTS = _VERDICT + _SANDBOX
TEST_SETS = {AGENT: _VERDICT, DOCKER: _SANDBOX, GATE: _SANDBOX,
             VECTOR: _MEMORY, FRONTIER: _MEMORY, EPISODES: _MEMORY,
             MEMTOOL: _MEMORY, DREAM: _MEMORY + _SYM, FS: _MEMORY, IFBENCH: _BENCH, IFCOMB: _BENCH, RATCHET: _PINS, LINT: _LINT, REGISTRY: _LINT, TEMPORAL: _LINT + _MEMORY,
             FS: _MEMORY + _LINT, MEMTOOL: _MEMORY + _LINT, LINTTEST: _LINT, BROWSER: _TOOLS3, SEARCH: _TOOLS3, DARKWEB: _TOOLS3, JOBS: _SANDBOX, RR2TEST: _SANDBOX, EFCTEST: _SANDBOX, ROUTES: _API, ROUTESTEST: _API, PROJECTS: _SYM, ADVANCER: _SYM, RESEARCH: _SYM,
             CODINGLOOP: _SYM, ISOLATION: _SYM, CLEANUP: _SYM, SYMTEST: _SYM,
             FS: _MEMORY + _LINT + _SYM, ROUTES: _API + _SYM}

_B = [
    # Re-anchored §4GK round 4: the `(\\.0+)?` branch was unreachable because
    # `_strip` deleted the decimal point, so a correct "12.0" scored as a
    # violation. The fixed predicate is the new anchor.
    ("IFB always-true number checker", IFBENCH, '        return bool(re.fullmatch(r"[-+]?\\d+(?:\\.\\d+)?", t)) and abs(float(t) - n) < 1e-9', "        return True"),
    ("IFB deep item mislabelled easy", IFBENCH, 'ck_number_equals(12), True, "deep"),', 'ck_number_equals(12), True, "easy"),'),
    ("IFB constraint dropped from the prompt", IFBENCH, '"Answer on exactly one line, nothing else. In the sandbox: create the three files "', '"In the sandbox: create the three files "'),
    ("IFB dodge floor removed", IFBENCH, "    return lambda r: len(_strip(r).split()) >= n", "    return lambda r: True"),
    ("IFB json values not checked, keys only", IFBENCH, "        if not isinstance(obj, dict) or set(obj) != set(pairs):", "        if not isinstance(obj, dict) or not set(pairs) <= set(obj):"),
    ("IFB band filter ignored", IFBENCH, "        items = [it for it in items if it[4] in want_b]", "        items = list(items)"),
    ("IFB no-tools filter ignored", IFBENCH, "    items = [it for it in items if not (no_tools and it[3])]", "    items = list(items)"),
    ("IFC combined run reports no bands", IFCOMB, '           "by_band": by_band,\n', ""),
    ("IFC unbanded rows dropped", IFCOMB, 'bands = sorted({str(p[a].get("band") or "unbanded") for p in pairs})',
     'bands = sorted({str(p[a]["band"]) for p in pairs if p[a].get("band")})'),
    ("IFC per-band mcnemar uses the pooled counts", IFCOMB,
     '            "mcnemar": {f"{a}_only": sb, f"{b_}_only": sc, "p": mcnemar_exact(sb, sc)},',
     '            "mcnemar": {f"{a}_only": bb, f"{b_}_only": cc, "p": mcnemar_exact(bb, cc)},'),
]

_M = [
    ("M1 reset_all skips the outline sidecar", MEMTOOL, '                    for attr, empty in (("library_file", "[]"),\n                                        ("outlines_file", "{}")):', '                    for attr, empty in (("library_file", "[]"),):'),
    ("M1 reset_all resets catalogues even on a failed batch", MEMTOOL, '                if not failed_batches:\n                    # Both sidecars', '                if True:\n                    # Both sidecars'),
    ("M3 delete_document_by_name catalogue outside the lock", VECTOR, '            self._update_library_index(filename, "remove")\n            # \u2026a forgotten document', '            pass\n            # \u2026a forgotten document'),
    ("M3 delete_document_by_name leaves the outline behind", VECTOR, '            self.drop_document_outline(filename)\n        return True, "Deleted"', '            pass\n        return True, "Deleted"'),
    ("M2 frontier: no fsync before the rename", FRONTIER, '                fh.flush()\n                os.fsync(fh.fileno())', '                fh.flush()'),
    ("M2 frontier: fixed temp name again", FRONTIER, 'tmp = self.file_path.with_suffix(f".{os.getpid()}.tmp")', 'tmp = self.file_path.with_suffix(".tmp")'),
    ("M4 forget_episode reports success on failure", VECTOR, '                "orphan until the next reconcile: %s", episode_id, e)\n            return False', '                "orphan until the next reconcile: %s", episode_id, e)\n            return True'),
    ("M4 episodes ignore the False return", EPISODES, '                    if forget(vid) is False:\n                        _orphaned.append(vid)', '                    forget(vid)'),
    ("M4 episodes drop the orphan warning", EPISODES, '            if _orphaned:\n                logger.warning(', '            if False:\n                logger.warning('),
    ("M4 live_episode_ids returns empty instead of None", EPISODES, '                           "reconcile will be skipped, not run blind", e)\n            return None', '                           "reconcile will be skipped, not run blind", e)\n            return set()'),
    ("M5 reconcile: episode arm runs on a None id set", VECTOR, '        if live_episode_ids is None:\n            report["skipped"].append(', '        if False:\n            report["skipped"].append('),
    ("M5 reconcile: a failed document scan repairs blind", VECTOR, '            report["skipped"].append(f"document scan failed: {e}")\n            return', '            report["skipped"].append(f"document scan failed: {e}")\n            live_sources = set()'),
    ("M5 reconcile: unbounded", VECTOR, '            if len(report["catalogue_dropped"]) >= cap:\n                report["bounded"] = True\n                break', '            if False:\n                report["bounded"] = True\n                break'),
    ("M5 reconcile: drops catalogue entries that DO have rows", VECTOR, '            if str(name) not in live_sources:', '            if True:'),
    ("M5 reconcile: never adopts unlisted documents", VECTOR, '        for name in sorted(live_sources - set(map(str, catalogue))):', '        for name in []:'),
    ("M5 reconcile: reaps rows with an unusable episode_id", VECTOR, '            try:\n                ep = int(ep)\n            except (TypeError, ValueError):', '            try:\n                ep = int(ep) if ep is not None else -1\n            except (ValueError,):'),
    ("M5 reconcile: outline arm judged against the STALE catalogue", VECTOR, '            listed = set(map(str, self.get_library()))', '            listed = set(map(str, catalogue))'),
    ("M5 reconcile: raises instead of reporting", VECTOR, '        except Exception as e:  # noqa: BLE001 \u2014 housekeeping never fails a cycle\n            logger.warning("memory reconcile aborted: %s", e)', '        except ValueError as e:\n            logger.warning("memory reconcile aborted: %s", e)'),
    ("M5 dream: reconciler never scheduled", DREAM, '                _rec = await self._reconcile_memory_stores()', '                _rec = ""'),
    ("M5 dream: passes no live ids (would reap every episode vector)", DREAM, '        report = await asyncio.to_thread(recon, live_ids)', '        report = await asyncio.to_thread(recon, set())'),
]

_P = [
    ("M1 parse-reach always true", RATCHET, "    cur = node\n    for _ in range(12):", "    return True\n    cur = node\n    for _ in range(12):"),
    ("M2 per-file growth ignored", RATCHET, "        if count > per_file.get(name, 0)", "        if False"),
    ("M3 missing baseline passes", RATCHET, "    if not path.exists():", "    if False:"),
    ("M4 fixture taint disabled", RATCHET, "    roots = set(_FIXTURE_SEEDS) | _fixture_names(tree)", "    roots = set()"),
    ("M5 fail-open on unknown uses", RATCHET, "        textual = not uses or not all(_reaches_parse(u, parents) for u in uses)", "        textual = not uses or not any(_reaches_parse(u, parents) for u in uses)"),
    ("M7 slack rule neutered", RATCHET, '    return int(baseline["total_textual"]) - sum(current.values())', "    return 0"),
]

_L = [
    ("L gate: zero-tolerance arm disabled", LINT, '    zero_tol = [m for m in messages if m["symbol"] in ZERO_TOLERANCE]', "    zero_tol = []"),
    ("L gate: new fingerprints tolerated", LINT, "    new_fps = {fp: n for fp, n in counts.items() if fp not in entries}", "    new_fps = {}"),
    ("L gate: ratchet ignores growth", LINT, "    grown = {fp: (n, entries[fp]) for fp, n in counts.items()\n             if fp in entries and n > entries[fp]}", "    grown = {}"),
    ("L gate: fingerprint keyed on line number", LINT, '    return f"{msg[\'symbol\']}|{path}|{msg[\'message\'].strip()}"', '    return f"{msg[\'symbol\']}|{path}|{msg[\'line\']}"'),
    ("L gate: missing pylint reads as green", LINT, '        print(f"LINT UNAVAILABLE: {exc}", file=sys.stderr)\n        return 2', '        print(f"LINT UNAVAILABLE: {exc}", file=sys.stderr)\n        return 0'),
    ("L gate: seed launders growth in a tracked family", LINT, "        if refused:\n            print(f\"REFUSING to seed already-tracked symbol(s): \"", "        if False:\n            print(f\"REFUSING to seed already-tracked symbol(s): \""),
    ("L gate: unused families dropped from the run", LINT, '"--disable=all", "--enable=E,W0611,W0612",', '"--disable=all", "--enable=E",'),
    ("L fix: _wrote pre-binding removed", FS, '            _wrote = False\n            try:\n                path = _get_safe_path(sandbox_dir, filename)', '            try:\n                path = _get_safe_path(sandbox_dir, filename)'),
    ("L fix: canonical_path back inside the try", REGISTRY, '                            canonical_path = _mgr.skills_dir / f"{name}.py"\n                            try:\n                                skill_src = canonical_path.read_text(encoding="utf-8")', '                            try:\n                                canonical_path = _mgr.skills_dir / f"{name}.py"\n                                skill_src = canonical_path.read_text(encoding="utf-8")'),
    ("L fix: Tuple import removed again", TEMPORAL, "from typing import Optional, Tuple   # Tuple: used by `compound_age_parts`", "from typing import Optional"),
    ("L fix: duplicate-char strip argument back", MEMTOOL, "    _probe = _raw_target.rstrip(_PATH_SEPS)", '    _probe = _raw_target.rstrip("/" + os.sep)'),
    ("L enum: descends into nested scopes", LINTTEST, "    if not root and isinstance(node, _SCOPE_NODES):\n        return", "    if False:\n        return"),
]

_R4 = [
    ("R3 CONTROL no-op comment", BROWSER, "#: Never starve a single interact action below this, however many there are.", "#: Never starve one interact action below this, however many there are."),
    ("S1 partial coverage books a strike again", SEARCH, '        return ToolOutcome.ok(_text, world_changed=False,\n                              reason_code="research_sources_partial")', '        return ToolOutcome.partial(_text, world_changed=False,\n                                   reason_code="research_sources_partial")'),
    ("S1 nothing-fetched no longer fails", SEARCH, "    if urls and _n_ok == 0:\n        return ToolOutcome.failed(_text, world_changed=False,", "    if False:\n        return ToolOutcome.failed(_text, world_changed=False,"),
    ("S1 fact_check partial coverage books a strike again", SEARCH, '        return ToolOutcome.ok(\n            f"FACT CHECK COMPLETE (partial source coverage):\\n{verdict}",', '        return ToolOutcome.partial(\n            f"FACT CHECK COMPLETE (partial source coverage):\\n{verdict}",'),
    ("S1 a verify call that never ran is downgraded to OK", SEARCH, '            world_changed=False, reason_code="factcheck_verify_call_failed")', "            world_changed=False)"),
    ("B1 first exec unbounded by the deadline", BROWSER, "timeout=_exec_timeout_for(_deadline, subprocess_timeout) or 1,", "timeout=subprocess_timeout,"),
    ("B1 launch-race retry restarts the budget", BROWSER, '        _retry_t = _exec_timeout_for(_deadline, subprocess_timeout)\n        if not _retry_t:\n            pretty_log("Browser Retry",\n                       "skipped the launch-race retry', '        _retry_t = subprocess_timeout\n        if not _retry_t:\n            pretty_log("Browser Retry",\n                       "skipped the launch-race retry'),
    ("B1 commit retry restarts the budget", BROWSER, '        _retry_t = _exec_timeout_for(_deadline, subprocess_timeout)\n        if not _retry_t:\n            pretty_log("Browser Retry",\n                       "skipped the commit-milestone retry', '        _retry_t = subprocess_timeout\n        if not _retry_t:\n            pretty_log("Browser Retry",\n                       "skipped the commit-milestone retry'),
    ("B1 runner gets the undivided per-op budget", BROWSER, "        timeout_ms=_runner_timeout_ms,", "        timeout_ms=timeout_ms,"),
    ("B1 per-action division removed", BROWSER, "        _runner_action_timeout_ms(_runner_total_ms, _n_actions)", "        int(timeout_ms)"),
    ("B1 action floor removed", BROWSER, "    return max(_MIN_ACTION_MS, int(runner_total_ms) // n)", "    return int(runner_total_ms) // n"),
    ("B1 a spent deadline still issues the exec", BROWSER, "    if rem < _MIN_EXEC_S:\n        return 0", "    if False:\n        return 0"),
    ("D1 darkweb returns a bare string again", DARKWEB, "    return _finish(result)", "    return result"),
    ("D1 darkweb all-sources-down books OK", DARKWEB, '    if urls and _n_ok == 0:\n        _finish = lambda t: ToolOutcome.failed(', '    if False:\n        _finish = lambda t: ToolOutcome.failed('),
]

_R3 = [
    ("R3 CONTROL docker no-op comment", DOCKER, "    #: persistent iptables fault would otherwise re-provision on every call.", "    #: persistent iptables fault would otherwise reprovision on every call."),
    ("R3-J1 nonce sync never called from _save", JOBS, "        if self._sync_nonces(reg):\n            self._save_nonces()\n", ""),
    ("R3-J2 sync keeps terminal rows", JOBS, 'if isinstance(e, dict) and e.get("state") == STATE_RUNNING}', "if isinstance(e, dict)}"),
    ("R3-J3 cap evicts oldest again", JOBS, "        if len(self._nonces) > self._NONCE_STORE_MAX:", "        while len(self._nonces) > self._NONCE_STORE_MAX:\n            self._nonces.pop(next(iter(self._nonces)), None)\n        if False:"),
    ("R3-J4 pending ring ignored", JOBS, "        keep |= set(self._nonce_pending)\n", ""),
    ("R3-D1 cut-off check back after the short-circuit", DOCKER, '        if self._egress_state == "blocked":\n            self._recreate_if_cut_off()\n        if self._ready_is_fresh():', "        if self._ready_is_fresh():"),
    ("R3-D2 recreate keeps the stale readiness stamp", DOCKER, "            # Belt: `_ready_is_fresh` also requires a container, but making the\n            # stamp stale here means the ordering holds even if that changes.\n            self.invalidate_ready()\n", "            # Belt: `_ready_is_fresh` also requires a container, but making the\n            # stamp stale here means the ordering holds even if that changes.\n"),
    ("R3-D3 reason not cleared on a new state", DOCKER, '        self._egress_unavailable_reason = reason if state == "unavailable" else ""', "        if reason:\n            self._egress_unavailable_reason = reason"),
    ("R3-D4 disconnect-failure records the wrong cause", DOCKER, 'self._set_egress_state("unavailable", "cut_off_failed")', 'self._set_egress_state("unavailable", "host_networking")'),
    ("R3-ENUM lifecycle enumeration accepts a transition that never saves", RR2TEST, "        if not saves:\n            offenders.append((fn.name, [w.lineno for w in writes]))", "        if False:\n            offenders.append((fn.name, [w.lineno for w in writes]))"),
    ("R3-ENUM setter enumeration accepts a bypass", EFCTEST, "    return [(fn.name, node.lineno)\n            for fn in ast.walk(tree)", "    return []\n    return [(fn.name, node.lineno)\n            for fn in ast.walk(tree)"),
]

_R1 = [
    ("R3-1 release back under the contended lock", ROUTES, "        if count_fg:\n            llm.foreground_tasks = max(0, int(llm.foreground_tasks) - 1)", "        if count_fg:\n            async with fg_lock:\n                llm.foreground_tasks = max(0, int(llm.foreground_tasks) - 1)"),
    ("R3-2 release removed", ROUTES, "            llm.foreground_tasks = max(0, int(llm.foreground_tasks) - 1)", "            pass"),
    ("R3-4 timeout answers 500 not 504", ROUTES, '            {"error": f"upstream did not answer within {_main_budget:.0f}s"}, 504)', '            {"error": f"upstream did not answer within {_main_budget:.0f}s"}, 500)'),
    ("R3-5 sessions_list back on the loop", ROUTES, "        sessions = await _store_call(store.list, limit=limit)", "        sessions = store.list(limit=limit)"),
    ("R3-6 chat append_turn back on the loop", ROUTES, '            await _store_call(_sess_store.append_turn, str(_session_id),\n                              _new_msgs, str(assistant_text or ""))', '            _sess_store.append_turn(str(_session_id), _new_msgs, str(assistant_text or ""))'),
    ("R3-7 activity read_since back on the loop", ROUTES, "        chunk, new_cursor = await _store_call(\n            log.read_since, cursor, limit=200, severity=SEVERITY_NOTIFY)", "        chunk, new_cursor = log.read_since(cursor, limit=200, severity=SEVERITY_NOTIFY)"),
    ("R3-8 ENUM: helper-bound receivers invisible again", ROUTESTEST, "        if isinstance(val, ast.Call):", "        if False:"),
    ("R3-Z1 zip build walks with a following os.walk", ROUTES, "                for dirpath, file_names, _dir_fd in walk_nofollow(sandbox_dir):", "                for dirpath, file_names, _dir_fd in ((r, f, None) for r, _d, f in os.walk(sandbox_dir)):"),
    ("R3-Z2 zip archives by path instead of the nofollow read", ROUTES, "                        zip_file.writestr(f\"sandbox/{arcname}\", data)", "                        zip_file.write(file_path, f\"sandbox/{arcname}\")"),
    ("R3-Z4 restore writes through a link at the destination", ROUTES, "                    if extracted_path.is_symlink() or extracted_path.parent.is_symlink():\n                        continue\n", ""),
    ("R3-9 ENUM: exemptions silence without a reason", ROUTESTEST, "            if (fn.name, f.attr) in ENUM_EXEMPTIONS:\n                continue", "            if True:\n                continue"),
]

_R2 = [
    ("R3 child dir opened following symlinks", FS, "                    cfd = _open_dir_nofollow(name, dir_fd=dfd)", "                    cfd = os.open(str(dirpath / name), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))"),
    ("R3 regular file opened following symlinks", FS, '                fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)\n                             | getattr(os, "O_NONBLOCK", 0), dir_fd=dfd)', '                fd = os.open(name, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0), dir_fd=dfd)'),
    ("R3 non-regular files copied", FS, '            if not _stat.S_ISREG(st.st_mode):\n                skipped.append(f"{dirpath / name}: not a regular file "', '            if False:\n                skipped.append(f"{dirpath / name}: not a regular file "'),
    ("R3 escaping links recreated", FS, "                if _escapes(dirpath, name, dfd):", "                if False:"),
    ("R3 dir_fd probe fail-open", FS, "def _require_dir_fd(what: str) -> None:\n    if not _DIR_FD_OK:", "def _require_dir_fd(what: str) -> None:\n    if False:"),
    ("R3 projects fork copy follows", PROJECTS, "                copytree_nofollow(src_ws, dst_ws, src_ws, ignore=_ignore,\n                                  dirs_exist_ok=True)", "                shutil.copytree(src_ws, dst_ws, ignore=_ignore, dirs_exist_ok=True)"),
    ("R3 projects clone copy follows", PROJECTS, "                copytree_nofollow(\n                    src_ws, dst_ws, src_ws, dirs_exist_ok=True,", "                shutil.copytree(\n                    src_ws, dst_ws, dirs_exist_ok=True,"),
    ("R3 research reads through the link", RESEARCH, '                    text = read_text_nofollow(fn, dir_fd=_dfd, errors="replace")', '                    text = (Path(dirpath) / fn).read_text(errors="replace")'),
    ("R3 ENUM scoped back to one file", SYMTEST, '    for f in sorted((src_root / "src").rglob("*.py")):', "    for f in [Path(fs.__file__)]:"),
    ("R3 ENUM walk-read rule dropped", SYMTEST, '                offences.setdefault((rel, "walk-read"), []).append(fn.lineno)', "                pass"),
    ("R3 snapshot hashes through a link", CODINGLOOP, "            data = read_bytes_nofollow_fd(fn, dir_fd=dir_fd, max_bytes=5_000_001)", "            data = p.read_bytes()"),
    ("R3 dream skills copy follows", DREAM, "                _skipped = copytree_nofollow(real_skills_dir, temp_skills_dir,\n                                             real_skills_dir)", "                _skipped = shutil.copytree(real_skills_dir, temp_skills_dir) and []"),
    ("R3 isolation seed copy follows", ISOLATION, "                    _sk = copytree_nofollow(src, md / sub, src,\n                                            dirs_exist_ok=True)", "                    _sk = shutil.copytree(src, md / sub, symlinks=False,\n                                          dirs_exist_ok=True) and []"),
    ("R3 cleanup read back to check-then-read", CLEANUP, "                text = read_text_nofollow(\n                    fpath, errors=\"replace\",\n                    max_bytes=_REFERENCE_SCAN_MAX_BYTES + 1)", "                text = fpath.read_text(errors=\"replace\")"),
]

MUTANTS = [
    ("CONTROL no-op comment", GATE, "#: The refusal, per cause.", "#: The refusal text, per cause."),
    ("CONTROL known-bad: merge never replaces", AGENT,
     "        if not standing:\n            v_result = mech", "        if not standing:\n            pass"),
    ("CONTROL pre-fix tree", "PREFIX", "", ""),
    # ── the escalation ordering (mine) ──
    ("GJ mechanical merge dropped at the tool-turn tail", AGENT,
     '        v_result = self._merge_mechanical_refute(v_result, _shape, "reply-shape")\n        # \u00a74FY turn-state override: same merge, same place, after the shape',
     '        v_result = v_result\n        # \u00a74FY turn-state override: same merge, same place, after the shape'),
    ("GJ merge keeps a standing CONFIRMED", AGENT,
     "        if not standing:\n            v_result = mech", "        if not standing:\n            v_result = v_result or mech"),
    # ── the egress cause (mine) ──
    ("GJ host branch does not name its cause", DOCKER, 'self._set_egress_state("unavailable", "host_networking")', 'self._set_egress_state("unavailable")'),

    ("GJ host branch back to WARNING", DOCKER,
     '                "and recreate the sandbox.",\n                level="CRITICAL", icon=Icons.FAIL,',
     '                "and recreate the sandbox.",\n                level="WARNING", icon=Icons.WARN,'),
    ("GJ refusal text ignores the cause", GATE,
     "    return EGRESS_UNAVAILABLE_BY_REASON.get(reason, EGRESS_UNAVAILABLE_MSG)",
     "    return EGRESS_UNAVAILABLE_MSG"),
] + _B + _M + _P + _L + _R4 + _R3 + _R1 + _R2

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def restore():
    for rel in FILES:
        shutil.copyfile(os.path.join(PRISTINE, rel), os.path.join(TREE, rel))


def purge():
    for root, dirs, _ in os.walk(TREE):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)


def _sig(*_):
    restore(); sys.exit("interrupted — tree restored")


for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s, _sig)


def run_tests(tests=None):
    tests = list(dict.fromkeys(tests or TESTS))
    env = dict(os.environ, PYTHONPATH="src")
    env.pop("FORCE_COLOR", None)
    r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider",
                        "-n", "6", "--dist", "loadfile",
                        # two tests bind to the LIVE repo path / scripts dir and fail in
                        # any copied tree regardless of the mutant (no-op control proved it)
                        "--deselect", "tests/test_learning_stack_audit.py::test_gepa_batch_scores_impute_unjudged_at_judged_mean",
                        "--deselect", "tests/test_human_feedback.py::test_the_conftest_live_path_isolation_is_COLLECTION_TIME",
                        *tests],
                       cwd=TREE, env=env, capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    if "error" in out.lower() and ("ERROR collecting" in out or "ImportError" in out or "SyntaxError" in out):
        return "INVALID", out[-600:]
    m = re.search(r"(\d+) failed", out)
    if r.returncode != 0 and m:
        return "KILLED", f"{m.group(1)} failing: " + " ".join(re.findall(r"FAILED (\S+)", out)[:3])
    if r.returncode == 0:
        return "SURVIVED", out.strip().splitlines()[-1][:120]
    return "INVALID", out[-600:]


if "--list" in ARGV:
    for i, m in enumerate(MUTANTS):
        print(i, m[0])
    sys.exit(0)

lo, hi = 0, len(MUTANTS)
if "--slice" in ARGV:
    a, b = ARGV[ARGV.index("--slice") + 1].split(":")
    lo, hi = int(a), int(b)

base = {rel: sha(os.path.join(PRISTINE, rel)) for rel in FILES}
restore(); purge()
try:
    for i in range(lo, hi):
        label, rel, old, new = MUTANTS[i]
        t0 = time.time()
        if rel == "PREFIX":
            if not os.path.isdir(ORIG):
                print(f"[{i:2d}] SKIPPED   {label}  (no ../orig pre-fix copies)"); continue
            for r in FILES:
                shutil.copyfile(os.path.join(ORIG, os.path.basename(r)), os.path.join(TREE, r))
        else:
            p = os.path.join(TREE, rel)
            src = open(p, encoding="utf-8").read()
            if src.count(old) != 1:
                print(f"[{i:2d}] MISSING   {label}  (anchor count {src.count(old)})"); continue
            open(p, "w", encoding="utf-8").write(src.replace(old, new))
        purge()
        c = subprocess.run([PY, "-m", "py_compile", *[os.path.join(TREE, r) for r in FILES]],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print(f"[{i:2d}] INVALID   {label}  (py_compile: {c.stderr[-200:]})")
        else:
            verdict, detail = run_tests(TEST_SETS.get(rel) if rel != "PREFIX" else None)
            print(f"[{i:2d}] {verdict:9s} {label}  ({time.time()-t0:.0f}s) {detail}", flush=True)
        restore(); purge()
        for r, h in base.items():
            assert sha(os.path.join(TREE, r)) == h, f"restore failed for {r}"
finally:
    restore()
print("tree restored + hash-verified")
