#!/usr/bin/env python3
"""Whole-file mutation battery for the 2026-09-13 §4GI fixes (Tier 2 sandbox boundary + Tier 3 API, §R R2).

WHY THIS LIVES IN THE REPO: the numbers quoted in PROJECT_JOURNAL §4GI
(see PROJECT_JOURNAL §4GI for the score) must be re-runnable, not a claim from a scratch directory that is
deleted with the session (the §4DG lesson behind scripts/mutate_fs_ledger.py).

Runs against a COPY of the repo — never the live tree:

    rsync -a --exclude __pycache__ src tests scripts docs pytest.ini /tmp/mtree/
    mkdir -p /tmp/pristine && cp -R /tmp/mtree/src /tmp/pristine/   # sibling of the tree
    PYTHONPATH=src python scripts/mutate_4gi_2026_09_13.py --tree /tmp/mtree [--slice a:b] [--list]

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
# Every file the section touched: restored from the pristine copy after every mutant.
FILES = [
    "src/ghost_agent/tools/file_system.py", "src/ghost_agent/sandbox/docker.py",
    "src/ghost_agent/sandbox/services.py", "src/ghost_agent/sandbox/jobs.py",
    "src/ghost_agent/tools/execute.py", "src/ghost_agent/tools/browser.py",
    "src/ghost_agent/tools/search.py", "src/ghost_agent/api/routes.py",
    "src/ghost_agent/core/learning_health.py", "src/ghost_agent/memory/vector.py",
]
FS, DOCKER, SERVICES, JOBS, EXECUTE, BROWSER, SEARCH, ROUTES, LH, VECTOR = FILES
GUARD = "src/ghost_agent/sandbox/registry_guard.py"
GATE = "src/ghost_agent/sandbox/egress_gate.py"
FILES.extend([GUARD, GATE])

TESTS = [
    "tests/test_4gi_symlink_class.py", "tests/test_browser_wallclock_ceiling.py",
    "tests/test_research_outcome_status.py", "tests/test_4gi_api_routes.py",
    # neighbours that import the touched modules (filled further after fork A2)
    "tests/test_file_system_replace.py", "tests/test_tool_layer_hardening_4dx.py",
    "tests/test_model_writable_root_ops.py", "tests/test_sandbox_services.py",
    "tests/test_sandbox_services_review_fixes.py", "tests/test_jobs_tool_sandbox.py",
    "tests/test_jobs_collect_readmark.py", "tests/test_egress_failclosed_4p.py",
    "tests/test_sandbox_tor_egress.py", "tests/test_sandbox_resume_egress.py",
    "tests/test_egress_guard.py", "tests/test_egress_failclosed_4gi.py",
    "tests/test_registry_guard_4gi.py", "tests/test_sandbox_job_promotion.py",
    "tests/test_egress_gate_consumer_4gi.py", "tests/test_4gi_review_round2.py",
    "tests/test_docker_sudo.py", "tests/test_fact_check.py",
    "tests/test_verdict_fact_recording.py", "tests/test_calibration_audit.py",
]

MUTANTS = [
    ("CONTROL no-op comment", BROWSER, "# a non-numeric timeout_ms/max_chars would otherwise raise ValueError out", "# a non-numeric timeout_ms/max_chars would otherwise raise a ValueError out"),
    ("CONTROL known-bad: research never fails", SEARCH, "    if urls and _n_ok == 0:", "    if False:"),
    ("CONTROL pre-fix tree", "PREFIX", "", ""),
    # ── A1: symlink class ──
    ("A1 copytree follows again", FS, "        str(src), str(dest), symlinks=True, ignore=_ignore,", "        str(src), str(dest), symlinks=False, ignore=_ignore,"),
    ("A1 escaping links recreated", FS, "        return {n for n in names if _symlink_escapes(Path(dirpath) / n, root)}", "        return set()"),
    ("A1 tool_copy_file back to bare copytree", FS, "            await asyncio.to_thread(copytree_nofollow, src_path, dest_path, sandbox_dir)", "            await asyncio.to_thread(shutil.copytree, str(src_path), str(dest_path))"),
    ("A1 repo map reads through a symlinked .py", FS, "                    if f.endswith('.py') and not path_.is_symlink():", "                    if f.endswith('.py'):"),
    ("A1 dir opened following symlinks", FS, "    dflags = os.O_RDONLY | getattr(os, \"O_DIRECTORY\", 0) | getattr(os, \"O_NOFOLLOW\", 0)", "    dflags = os.O_RDONLY | getattr(os, \"O_DIRECTORY\", 0)"),
    ("A1 file opened following symlinks (dir_fd writer)", FS, "        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC\n                 | getattr(os, \"O_NOFOLLOW\", 0) | getattr(os, \"O_NONBLOCK\", 0))\n        try:\n            fd = os.open(name, flags, 0o666, dir_fd=dfd)", "        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC\n                 | getattr(os, \"O_NONBLOCK\", 0))\n        try:\n            fd = os.open(name, flags, 0o666, dir_fd=dfd)"),
    ("A1 spill back to plain write_text", DOCKER, "            from ..tools.file_system import write_text_nofollow_in_dir\n            try:\n                write_text_nofollow_in_dir(\n                    spill_dir, name, capped.encode(\"utf-8\", \"replace\").decode(\"utf-8\"))\n            except ValueError as ve:", "            try:\n                (spill_dir / name).write_text(capped, encoding=\"utf-8\", errors=\"replace\")\n            except ValueError as ve:"),
    ("A1 spill swallows the refusal and returns the pointer", DOCKER, "                pretty_log(\"Sandbox Spill\",\n                           f\"refused to spill run output: {ve}\",\n                           level=\"WARNING\", icon=Icons.WARN)\n                return None", "                pretty_log(\"Sandbox Spill\",\n                           f\"refused to spill run output: {ve}\",\n                           level=\"WARNING\", icon=Icons.WARN)"),
    ("A1 probe matches its own wrapper again", EXECUTE, "KERNEL_LIVENESS_PROBE = \"pgrep -f '[i]pykernel_launcher'\"", "KERNEL_LIVENESS_PROBE = \"pgrep -f ipykernel_launcher\""),
    ("A1 probe ignores the pgrep result", EXECUTE, "    _out, pgrep_code = await asyncio.to_thread(sandbox_manager.execute, KERNEL_LIVENESS_PROBE)\n    return pgrep_code == 0", "    _out, pgrep_code = await asyncio.to_thread(sandbox_manager.execute, KERNEL_LIVENESS_PROBE)\n    return True"),
    ("A1 stateful path ignores the probe", EXECUTE, "        check_code = 0 if await _kernel_alive(sandbox_manager, conn_file) else 1", "        check_code = 0"),
    # ── B: browser ceiling, research status ──
    ("B1 ceiling removed", BROWSER, "    return min(_wallclock_ceiling_s(), want)", "    return want"),
    ("B1 per-op clamp removed", BROWSER, "    timeout_ms = _clamp_runner_timeout_ms(_safe_int(timeout_ms, 30000))", "    timeout_ms = _safe_int(timeout_ms, 30000)"),
    ("B1 floor removed (env 5 → 5 s)", BROWSER, "    return max(60, v)", "    return v"),
    ("B1 runner cap ignores the slack", BROWSER, "    cap_ms = max(1000, (_wallclock_ceiling_s() - _SUBPROCESS_SLACK_S) * 1000)", "    cap_ms = max(1000, _wallclock_ceiling_s() * 1000)"),
    ("B1 enumeration guard: one exec site raw", BROWSER, "                    sandbox_manager.execute, retry_cmd,\n                    timeout=subprocess_timeout, **_wd_kw", "                    sandbox_manager.execute, retry_cmd,\n                    timeout=3600, **_wd_kw"),
    ("B2 nothing-fetched books OK", SEARCH, "    if urls and _n_ok == 0:\n        return ToolOutcome.failed(_text, world_changed=False,\n                                  reason_code=\"research_all_sources_failed\")\n", ""),
    ("B2 partial books OK", SEARCH, "    if _failed or _lost:\n        return ToolOutcome.partial(_text, world_changed=False,\n                                   reason_code=\"research_sources_partial\")\n", ""),
    ("B2 fact_check judges a failed research", SEARCH, "    if _dr.status in (OutcomeStatus.FAILED, OutcomeStatus.REJECTED):", "    if False:"),
    ("B2 fact_check partial coverage reported OK", SEARCH, "    if _dr_partial:\n        # some sources failed", "    if False:\n        # some sources failed"),
    # ── A2: registry guard, nonce, egress ──
    ("A2 services: rows not validated at load", SERVICES, "            return self._validated_rows(data)\n", "            return data if isinstance(data, dict) else {}\n"),
    ("A2 services: validation accepts every row", SERVICES, "            why = _rg.validate_row(entry, require_pid=False)\n", "            why = None\n"),
    ("A2 services: kill bypasses the guard (old -- form)", SERVICES, "        _rg.kill_tree(\n            self._exec, pid,\n            log=lambda msg: pretty_log(\"Service Kill Refused\", msg,\n                                       level=\"ERROR\", icon=Icons.SHIELD),\n            timeout=30)\n", "        self._exec(f\"sh -c 'kill -TERM -- -{int(pid)} 2>/dev/null || kill -TERM {int(pid)} 2>/dev/null'\", timeout=15)\n"),
    ("A2 services: probe two-valued again", SERVICES, "        return _rg.pid_state(self._exec, pid)\n", "        return _rg.pid_state(self._exec, pid) is True\n"),
    ("A2 services: unknown reads as dead for leases", SERVICES, "        return self._entry_state(entry) is not False\n", "        return self._entry_state(entry) is True\n"),
    ("A2 services: allocator uses strict liveness", SERVICES, "        alive_map = {k: self._entry_alive_or_unknown(e) for k, e in reg.items()\n                     if k != self_key}\n", "        alive_map = {k: self._entry_alive(e) for k, e in reg.items()\n                     if k != self_key}\n"),
    ("A2 services: post-launch pops on an inconclusive probe", SERVICES, "            elif _pst is False:\n                reg.pop(key, None)\n", "            if _pst is not True:\n                reg.pop(key, None)\n"),
    ("A2 jobs: floor back to <= 1 only", JOBS, "            if _rg.valid_pid(pid) is None or not math.isfinite(deadline) or deadline <= 0:\n", "            if int(pid) <= 1 or not math.isfinite(deadline) or deadline <= 0:\n"),
    ("A2 jobs: sentinel without a known nonce accepted", JOBS, "        nonce = self._nonces.get(jid)\n        if not nonce:\n", "        nonce = self._nonces.get(jid)\n        if False:\n"),
    ("A2 jobs: nonce not persisted", JOBS, "        self._nonces[jid] = nonce\n        self._save_nonces()\n", "        self._nonces[jid] = nonce\n"),
    ("A2 jobs: nonces not loaded on start", JOBS, "            self._nonces.update(self._load_nonces())\n", "            pass\n"),
    ("A2 jobs: store accepts any key/value", JOBS, "                if valid_job_id(k) and isinstance(v, str) and 8 <= len(v) <= 64}\n", "                }\n"),
    ("A2 guard: floor off (pid 1 valid)", GUARD, "    if isinstance(pid, bool) or p <= 1 or p >= PID_MAX:\n", "    if isinstance(pid, bool) or p < 0 or p >= PID_MAX:\n"),
    ("A2 guard: pid_max not enforced", GUARD, "    if isinstance(pid, bool) or p <= 1 or p >= PID_MAX:\n", "    if isinstance(pid, bool) or p <= 1:\n"),
    ("A2 guard: infra fault reads as dead", GUARD, "    if code != 0 and probe_inconclusive(out, code):\n        return None\n", ""),
    ("A2 guard: kill_tree ignores the floor", GUARD, "    p = valid_pid(pid)\n    if p is None:\n        if log is not None:", "    p = int(pid)\n    if False:\n        if log is not None:"),
    ("A2 guard: the -- form back in the script", GUARD, "        f'sig() {{ kill -\"$1\" -$S 2>/dev/null || '\n", "        f'sig() {{ kill -\"$1\" -- -$S 2>/dev/null || '\n"),
    ("A2 guard: row validation ignores the port", GUARD, "        if valid_port(port, lo, hi) is None:\n            return f\"port {port!r} outside {lo}-{hi}\"\n", ""),
    ("A2 egress: unavailable stays direct (no cut-off)", DOCKER, "                self._block_egress_hard(\n                    \"iptables or tor missing in the image — Tor-only egress NOT enforced \"\n", "                self._egress_state = \"unavailable\"\n                return\n                self._block_egress_hard(\n                    \"iptables or tor missing in the image — Tor-only egress NOT enforced \"\n"),
    ("A2 egress: rules-failed branch stays direct", DOCKER, "            if code != 0:\n                self._block_egress_hard(\n", "            if code != 0:\n                self._egress_state = \"unavailable\"\n                return\n                self._block_egress_hard(\n"),
    ("A2 egress: cut-off claims blocked even when it failed", DOCKER, "        if failed:\n            self._egress_state = \"unavailable\"\n", "        if failed:\n            self._egress_state = \"blocked\"\n"),
    ("A2 egress: cut-off never disconnects", DOCKER, "                self.client.networks.get(name).disconnect(self.container, force=True)\n", "                pass\n"),
    ("A2 egress: exception branch never cuts off", DOCKER, "            if self._egress_state in (\"enforced\", \"blocked\"):\n                pretty_log(\"Sandbox Egress\", f\"enforcement step raised", "            if True:\n                pretty_log(\"Sandbox Egress\", f\"enforcement step raised"),
    ("A2 egress: predicate accepts unavailable", DOCKER, "        return self._egress_state in (\"enforced\", \"blocked\")\n", "        return self._egress_state in (\"enforced\", \"blocked\", \"unavailable\")\n"),
    # ── egress consumer (execute + browser) ──
    ("GATE execute never consults the predicate", EXECUTE, "    _egress_block = _egress_refusal(sandbox_manager)\n    if _egress_block is not None:\n        pretty_log(\"Sandbox Egress\", \"refusing execute — egress unavailable\",", "    _egress_block = None\n    if _egress_block is not None:\n        pretty_log(\"Sandbox Egress\", \"refusing execute — egress unavailable\","),
    ("GATE browser never consults the predicate", BROWSER, "    _egress_block = _egress_refusal(sandbox_manager)\n    if _egress_block is not None:\n        pretty_log(\"Sandbox Egress\", \"refusing browser — egress unavailable\",", "    _egress_block = None\n    if _egress_block is not None:\n        pretty_log(\"Sandbox Egress\", \"refusing browser — egress unavailable\","),
    ("GATE predicate inverted", GATE, "        if probe():\n            return None\n", "        if not probe():\n            return None\n"),
    ("GATE broken probe reads as available", GATE, "    except Exception:  # noqa: BLE001 — a broken probe reads as unavailable\n        pass\n", "    except Exception:  # noqa: BLE001 — a broken probe reads as unavailable\n        return None\n"),
    # ── R3 round 2 (mutate the previous round's fixes first) ──
    ("R2 gate refuses a never-enforced sandbox", GATE, "        if callable(attempted) and not attempted():", "        if False:"),
    ("R2 execute belt removed", DOCKER, '            if (getattr(self, "tor_proxy", None) and getattr(self, "_egress_state", "")\n                    and not self.egress_is_enforced_or_blocked()):', "            if False:"),
    ("R2 already-cut-off tries bridge again", DOCKER, "        if not nets:\n            # Already cut off", "        if not nets:\n            nets = [\"bridge\"]\n        if False:\n            # Already cut off"),
    ("R2 cut-off container never recreated", DOCKER, "        if (self.container is not None and self._container_cut_off(self.container)\n                and self._cut_off_recreate_due()):", "        if False:"),
    ("R2 ensure_running skips the cut-off check", DOCKER, "        self._recreate_if_cut_off()\n        if not (self.container and self._is_container_ready()):", "        if not (self.container and self._is_container_ready()):"),
    ("R2 recreate ignores the backoff", DOCKER, "        return (now - float(self._cut_off_at or 0.0)) >= self._CUT_OFF_RECREATE_BACKOFF_S", "        return True"),
    ("R2 host mode reads as cut off", DOCKER, '            if mode == "host":\n                return False\n', ""),
    ("R2 nonce never dropped", JOBS, "        if jid in self._nonces:\n            self._nonces.pop(jid, None)\n            self._save_nonces()\n", "        return\n"),
    ("R2 nonce store unbounded", JOBS, "        while len(self._nonces) > self._NONCE_STORE_MAX:\n            self._nonces.pop(next(iter(self._nonces)), None)\n", ""),
    ("R2 nonce store read tail-first again", JOBS, "                self._nonce_store, max_bytes=8 << 20).decode(\"utf-8\", \"replace\"))", "                self._nonce_store, max_bytes=1 << 20).decode(\"utf-8\", \"replace\"))"),
    ("R2 fact-check failure structural again", SEARCH, "(timeout, block or \"\n            f\"unreachable — a transient network failure), so the claim was \"", "so the claim was \"\n            f\"\""),
    ("R2 booked response never releases in finally", ROUTES, "        finally:\n            if self._release is not None:\n                await self._release()", "        finally:\n            pass"),
    ("R2 catch_all back to a plain StreamingResponse", ROUTES, "        return _BookedStreamingResponse(", "        return StreamingResponse("),
    ("R2 quarantined rows erased on save", SERVICES, "            _on_disk = {**{k: v for k, v in _q.items() if k not in reg}, **reg}\n", "            _on_disk = dict(reg)\n"),
    ("R2 rows dropped, not quarantined", SERVICES, "                if isinstance(key, str):\n                    quarantined[key] = entry\n", ""),
    # ── C: API routes, logger ──
    ("R store call inline again", ROUTES, "        ok, detail = await _store_call(memory.correct_fragment, match, replacement)\n", "        ok, detail = memory.correct_fragment(match, replacement)\n"),
    ("R store call unbounded", ROUTES, "        return await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=budget)\n", "        return await asyncio.to_thread(fn, *args, **kwargs)\n"),
    ("R quarantine inline", ROUTES, "        n = int(await _store_call(sm.quarantine_lesson, trigger, reason) or 0)\n", "        n = int(sm.quarantine_lesson(trigger, reason) or 0)\n"),
    ("R timeout answers 500 not 504", ROUTES, '    return JSONResponse({"error": str(exc)}, 504)\n', '    return JSONResponse({"error": str(exc)}, 500)\n'),
    ("R catch_all forwards the key", ROUTES, "    headers = _forwardable_headers(request.headers)\n", '    headers = dict(request.headers)\n    headers.pop("host", None)\n    headers.pop("content-length", None)\n'),
    ("R catch_all not booked", ROUTES, "    await _booking.enter_async_context(_main_node_request(_llm, hold_lock=False))\n", ""),
    ("R catch_all booking not released after the stream", ROUTES, "                finally:\n                    await _booking.aclose()\n", "                finally:\n                    pass\n"),
    ("R catch_all booking leaks on a failed send", ROUTES, '        await _booking.aclose()\n        pretty_log("Proxy Failed"', '        pretty_log("Proxy Failed"'),
    ("R api_generate skips the main lock", ROUTES, "        async with _main_node_request(_llm, hold_lock=True):", "        async with _main_node_request(_llm, hold_lock=False):"),
    ("R booking never counts foreground", ROUTES, "    if count_fg:\n        async with fg_lock:\n            llm.foreground_tasks += 1\n", ""),
    ("R inflight not counted", ROUTES, "            if callable(inc) and callable(dec):\n                inc(base)\n                stack.callback(dec, base)\n", ""),
    ("R STORE attrs shrink (enumeration blind to memory_system)", ROUTES, '    "memory_system", "skill_memory", "trajectory_collector", "profile_memory",', '    "skill_memory", "trajectory_collector", "profile_memory",'),
    ("LH logger undefined again", LH, 'logger = logging.getLogger("GhostAgent")\n', ""),
    ("LH handler re-raises", LH, '                logger.debug("lowest-confidence list skipped: %s", _lcx)\n', "                raise\n"),
]


# The full set of test files that touch each CHANGED file (R2): a mutant in
# the API layer does not need the two-minute sandbox suites, and vice versa.
_SANDBOX = ["tests/test_4gi_symlink_class.py", "tests/test_4gi_review_round2.py",
            "tests/test_egress_gate_consumer_4gi.py", "tests/test_egress_failclosed_4gi.py",
            "tests/test_registry_guard_4gi.py", "tests/test_sandbox_services.py",
            "tests/test_sandbox_services_review_fixes.py", "tests/test_sandbox_job_promotion.py",
            "tests/test_sandbox_tor_egress.py", "tests/test_sandbox_resume_egress.py",
            "tests/test_jobs_tool_sandbox.py", "tests/test_jobs_collect_readmark.py",
            "tests/test_egress_failclosed_4p.py", "tests/test_egress_guard.py",
            "tests/test_tool_layer_hardening_4dx.py", "tests/test_model_writable_root_ops.py",
            "tests/test_file_system_replace.py", "tests/test_docker_sudo.py"]
_TOOLS = ["tests/test_browser_wallclock_ceiling.py", "tests/test_research_outcome_status.py",
          "tests/test_egress_gate_consumer_4gi.py", "tests/test_4gi_review_round2.py",
          "tests/test_fact_check.py", "tests/test_4gi_symlink_class.py"]
_API = ["tests/test_4gi_api_routes.py", "tests/test_4gi_review_round2.py",
        "tests/test_verdict_fact_recording.py", "tests/test_calibration_audit.py"]
TEST_SETS = {FS: _SANDBOX, DOCKER: _SANDBOX, SERVICES: _SANDBOX, JOBS: _SANDBOX, GUARD: _SANDBOX,
             GATE: _SANDBOX + _TOOLS, EXECUTE: _SANDBOX + _TOOLS, BROWSER: _TOOLS, SEARCH: _TOOLS,
             ROUTES: _API, LH: _API, VECTOR: _API}


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
