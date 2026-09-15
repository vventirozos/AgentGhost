#!/usr/bin/env python3
"""Whole-file mutation battery for the 2026-09-13 Tier-1 fixes (§4GG, §R R2).

WHY THIS LIVES IN THE REPO: the numbers quoted in PROJECT_JOURNAL §4GG
(40 mutants killed, 0 survivors, no-op control survived, known-bad control
died) must be re-runnable, not a claim from a scratch directory that is
deleted with the session (the §4DG lesson behind scripts/mutate_fs_ledger.py).

Runs against a COPY of the repo — never the live tree:

    rsync -a --exclude __pycache__ src tests scripts docs pytest.ini /tmp/mtree/
    mkdir -p /tmp/pristine && cp -R /tmp/mtree/src /tmp/pristine/   # sibling of the tree
    PYTHONPATH=src python scripts/mutate_tier1_2026_09_13.py --tree /tmp/mtree [--slice a:b] [--list]

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
AGENT = "src/ghost_agent/core/agent.py"
GATE = "src/ghost_agent/core/evidence_gate.py"
OUTC = "src/ghost_agent/tools/outcome.py"

TESTS = [
    "tests/test_evidence_gate_real_rows.py", "tests/test_synthetic_strike_ledger.py",
    "tests/test_late_verdict_ordering.py", "tests/test_repair_reentry_latch.py",
    "tests/test_evidence_gate.py", "tests/test_dispatch_pipeline_extraction.py",
    "tests/test_4ee_label_pins.py", "tests/test_strike_ledger.py",
    "tests/test_core_loop_defect_fixes.py", "tests/test_one_task_per_turn.py",
    "tests/test_tool_outcome_contract.py", "tests/test_selfhood_late_verdict_backfill.py",
    "tests/test_agent_tool_calling_logic.py", "tests/test_learning_stack_audit.py",
    "tests/test_human_feedback.py", "tests/test_probe_origin_never_teaches.py",
    "tests/test_unacknowledged_failure_gate.py", "tests/test_finalize_stream_pins.py",
    "tests/test_futility_breaker_trio.py", "tests/test_4cu_rubric_shadow.py",
    "tests/test_verifier_evidence_window.py", "tests/test_context_compaction.py",
]

# (label, file, old, new)
MUTANTS = [
    ("CONTROL no-op comment", GATE, "#: Below this many characters", "#: Under this many characters"),
    ("CONTROL known-bad: gate never fires", GATE,
     "        return self.consulted > 0 and self.substantive == 0", "        return False"),
    ("CONTROL pre-fix tree (orig agent+gate+outcome)", "PREFIX", "", ""),
    # ── D1 evidence gate rows ──
    ("D1 writes consulted again", GATE,
     "        if name == \"file_system\" and not _fs_op_is_a_read(args):", "        if False:"),
    ("D1 gate ignores the outcome's call_args", GATE,
     'getattr(content_obj, "call_args", None) or {}', "{}"),
    ("D1 every status is an error", GATE,
     'status_is_error = getattr(status, "value", None) in ("failed", "rejected")', "status_is_error = True"),
    ("D1 write is a read op", GATE,
     '"inspect", "read_files", ""})', '"inspect", "read_files", "", "write"})'),
    ("D1 loop records no call_args", AGENT, "                                    call_args=_recorded_args)}", "                                    call_args=None)}"),
    ("D1 outcome drops call_args", OUTC, "        self.call_args = call_args\n", "        self.call_args = None\n"),
    ("D1 pickle drops call_args", OUTC,
     "                 self.reason_code, self.declared, self.call_args))", "                 self.reason_code, self.declared, None))"),
    ("D1 metadata carries empty args", AGENT,
     "_pf_world_mut, dict(t_args) if isinstance(t_args, dict) else {}))", "_pf_world_mut, {}))"),
    # ── D3 synthetic strikes ──
    ("D3 closure skips the binding counter", AGENT,
     "                nonlocal binding_failure_count\n                binding_failure_count += 1\n", "                nonlocal binding_failure_count\n"),
    ("D3 closure skips note_failure", AGENT,
     "                _snote = strikes.note_failure(_sfname, _sreason)\n", "                _snote = None\n"),
    ("D3 closure skips reset_clean_streak", AGENT,
     "                strikes.reset_clean_streak()\n                _snote = strikes.note_failure(_sfname, _sreason)\n",
     "                _snote = strikes.note_failure(_sfname, _sreason)\n"),
    ("D3 unknown-tool site unledgered", AGENT, '                    _strike_synthetic(fname, "unknown_tool")\n', ""),
    ("D3 disabled-tool site unledgered", AGENT, '                    _strike_synthetic(fname, "tool_disabled")\n', ""),
    ("D3 parse-error site unledgered", AGENT, '                    _strike_synthetic("system", "tool_call_parse_error")\n', ""),
    ("D3 bad-json site unledgered", AGENT, '                    _strike_synthetic(fname, "bad_json_arguments")\n', ""),
    ("D3 empty-write site unledgered", AGENT, '                                _strike_synthetic(fname, "empty_write_blocked")\n', ""),
    ("D3 constraint site unledgered", AGENT, '                            _strike_synthetic(fname, "constraint_violation")\n', ""),
    ("D3 invocation-error site unledgered", AGENT,
     "                        _strike_synthetic(fname, describe_invocation_error(fname, e))\n", ""),
    ("D3 decay ignores the binding counter", AGENT,
     "                            and not binding_failure_count:", "                            and True:"),
    # ── D2 late verdict ordering ──
    ("D2 flush never parks the sign", AGENT,
     "                    _park_pending_lesson_sign(self.context, trajectory_id, bool(success))\n", "                    pass\n"),
    ("D2 stash write ignores the parked sign", AGENT,
     "            parked = _take_pending_lesson_sign(self.context, trajectory_id)\n", "            parked = None\n"),
    ("D2 parked booking skips the retained ring", AGENT,
     "                flushed[trajectory_id] = (triggers, bool(parked))\n", "                pass\n"),
    ("D2 backfill never defers", AGENT,
     "            if cached is None and _trajectory_is_in_flight(self.context, trajectory_id):", "            if False:"),
    ("D2 backfill defers on every miss", AGENT,
     "            if cached is None and _trajectory_is_in_flight(self.context, trajectory_id):", "            if cached is None:"),
    ("D2 record never replays", AGENT,
     '        self._replay_deferred_late_backfill(getattr(traj, "id", None))\n', ""),
    ("D2 replay drops the parked verdict", AGENT,
     "            self._backfill_trajectory_outcome(trajectory_id, outcome, reason)\n        except Exception as e:  # noqa: BLE001\n            logger.debug(\"deferred late-verdict replay skipped",
     "            pass\n        except Exception as e:  # noqa: BLE001\n            logger.debug(\"deferred late-verdict replay skipped"),
    ("D2 loop never marks in flight", AGENT,
     "                self._mark_trajectory_in_flight(current_trajectory_id)\n", ""),
    ("D2 take never clears in-flight", AGENT,
     "    if isinstance(ring, dict):\n        ring.pop(trajectory_id, None)\n", ""),
    ("D2 in-flight ring unbounded", AGENT,
     "    ring[trajectory_id] = True\n    ring.move_to_end(trajectory_id)\n    while len(ring) > _TRAJ_IN_FLIGHT_MAX:\n        ring.popitem(last=False)\n",
     "    ring[trajectory_id] = True\n    ring.move_to_end(trajectory_id)\n"),
    ("D2 parked-sign ring unbounded", AGENT,
     "    pend[trajectory_id] = bool(success)\n    pend.move_to_end(trajectory_id)\n    while len(pend) > _PENDING_LESSON_SIGN_MAX:\n        pend.popitem(last=False)\n",
     "    pend[trajectory_id] = bool(success)\n    pend.move_to_end(trajectory_id)\n"),
    # ── round-2 fixes (R3: mutate the previous round's fixes first) ──
    ("R2 with_text drops call_args", OUTC, "                           call_args=res.call_args)", "                           call_args=None)"),
    ("R2 partial counted as an error", GATE,
     'status_is_error = getattr(status, "value", None) in ("failed", "rejected")',
     'status_is_error = getattr(status, "value", None) in ("failed", "rejected", "partial", "unresolved")'),
    ("R2 status never an error", GATE,
     'status_is_error = getattr(status, "value", None) in ("failed", "rejected")', "status_is_error = False"),
    ("R2 deferral skips the immediate lesson flush", AGENT,
     "                self._flush_stashed_lesson_outcome(\n                    trajectory_id, outcome == _Outcome.PASSED.value)\n                _defer_late_backfill(",
     "                _defer_late_backfill("),
    ("R2 deferral flushes the wrong sign", AGENT,
     "                    trajectory_id, outcome == _Outcome.PASSED.value)\n                _defer_late_backfill(",
     "                    trajectory_id, outcome != _Outcome.PASSED.value)\n                _defer_late_backfill("),
    ("R2 synthetic loop-breaker line never printed", AGENT, "                if _sfirst:\n                    # The same operator line", "                if False:\n                    # The same operator line"),
    # ── D4 latch vs repair ──
    ("D4 helper ignores the repair", AGENT,
     "    return bool(task_closed_this_req) and not bool(repair_reentry_active)", "    return bool(task_closed_this_req)"),
    ("D4 latch site ignores the helper", AGENT,
     "                    if _latch_forces_final(_proj_task_closed_this_req,\n                                           _repair_reentry_active):",
     "                    if _proj_task_closed_this_req:"),
    ("D4 repair never arms the flag", AGENT,
     "                                _repair_reentry_active = True\n", ""),
    ("D4 helper always false (latch dead)", AGENT,
     "    return bool(task_closed_this_req) and not bool(repair_reentry_active)", "    return False"),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def restore():
    for rel in (AGENT, GATE, OUTC):
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


def run_tests():
    env = dict(os.environ, PYTHONPATH="src")
    env.pop("FORCE_COLOR", None)
    r = subprocess.run([PY, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider",
                        "-n", "4", "--dist", "loadfile",
                        # two tests bind to the LIVE repo path / scripts dir and fail in
                        # any copied tree regardless of the mutant (no-op control proved it)
                        "--deselect", "tests/test_learning_stack_audit.py::test_gepa_batch_scores_impute_unjudged_at_judged_mean",
                        "--deselect", "tests/test_human_feedback.py::test_the_conftest_live_path_isolation_is_COLLECTION_TIME",
                        *TESTS],
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

base = {rel: sha(os.path.join(PRISTINE, rel)) for rel in (AGENT, GATE, OUTC)}
restore(); purge()
try:
    for i in range(lo, hi):
        label, rel, old, new = MUTANTS[i]
        t0 = time.time()
        if rel == "PREFIX":
            if not os.path.isdir(ORIG):
                print(f"[{i:2d}] SKIPPED   {label}  (no ../orig pre-fix copies)"); continue
            for f, r in (("agent.py", AGENT), ("evidence_gate.py", GATE), ("outcome.py", OUTC)):
                shutil.copyfile(os.path.join(ORIG, f), os.path.join(TREE, r))
        else:
            p = os.path.join(TREE, rel)
            src = open(p, encoding="utf-8").read()
            if src.count(old) != 1:
                print(f"[{i:2d}] MISSING   {label}  (anchor count {src.count(old)})"); continue
            open(p, "w", encoding="utf-8").write(src.replace(old, new))
        purge()
        c = subprocess.run([PY, "-m", "py_compile", os.path.join(TREE, AGENT), os.path.join(TREE, GATE), os.path.join(TREE, OUTC)],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print(f"[{i:2d}] INVALID   {label}  (py_compile: {c.stderr[-200:]})")
        else:
            verdict, detail = run_tests()
            print(f"[{i:2d}] {verdict:9s} {label}  ({time.time()-t0:.0f}s) {detail}", flush=True)
        restore(); purge()
        for r, h in base.items():
            assert sha(os.path.join(TREE, r)) == h, f"restore failed for {r}"
finally:
    restore()
print("tree restored + hash-verified")
