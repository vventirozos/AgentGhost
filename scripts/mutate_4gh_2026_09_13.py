#!/usr/bin/env python3
"""Whole-file mutation battery for the 2026-09-13 §4GH fixes (request e57ad0cf, §R R2).

WHY THIS LIVES IN THE REPO: the numbers quoted in PROJECT_JOURNAL §4GH
(see PROJECT_JOURNAL §4GH for the score) must be re-runnable, not a claim from a scratch directory that is
deleted with the session (the §4DG lesson behind scripts/mutate_fs_ledger.py).

Runs against a COPY of the repo — never the live tree:

    rsync -a --exclude __pycache__ src tests scripts docs pytest.ini /tmp/mtree/
    mkdir -p /tmp/pristine && cp -R /tmp/mtree/src /tmp/pristine/   # sibling of the tree
    PYTHONPATH=src python scripts/mutate_4gh_2026_09_13.py --tree /tmp/mtree [--slice a:b] [--list]

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
SMOOTH = "src/ghost_agent/core/reply_smoothing.py"
SHAPE = "src/ghost_agent/core/reply_shape_check.py"
OUTC = SHAPE   # the third restored file slot

TESTS = [
    "tests/test_forced_final_no_answer.py", "tests/test_navigate_extract_steer.py",
    "tests/test_reply_smoothing.py", "tests/test_stream_treated_view.py",
    "tests/test_4fn_judge_fixes.py", "tests/test_4fl_shape_routing.py",
    "tests/test_no_progress_loop_breaker.py", "tests/test_probe_before_hypothesis.py",
    "tests/test_repair_reentry_latch.py", "tests/test_dispatch_pipeline_extraction.py",
    "tests/test_verifier_auto_repair.py", "tests/test_one_task_per_turn.py",
    "tests/test_verdict_fact_recording.py", "tests/test_agent_tool_calling_logic.py",
    "tests/test_evidence_gate_real_rows.py", "tests/test_late_verdict_ordering.py",
    "tests/test_4da_round10_fixes.py", "tests/test_4da_round11_fixes.py",
]

MUTANTS = [
    ("CONTROL no-op comment", SMOOTH, "# The smoother removes beats that a LATER paragraph supersedes", "# The smoother drops beats that a LATER paragraph supersedes"),
    ("CONTROL known-bad: narration_only always False", SMOOTH,
     "    blocks = [b.strip() for b in _split_blocks(text or \"\") if b.strip()]\n    if not blocks:\n        return False\n    for b in blocks:",
     "    blocks = [b.strip() for b in _split_blocks(text or \"\") if b.strip()]\n    if not blocks:\n        return False\n    return False\n    for b in blocks:"),
    ("CONTROL pre-fix tree (orig agent+smoothing+shape)", "PREFIX", "", ""),
    # ── F1 forced final ──
    ("F1 retry never fires", AGENT,
     "                        if is_final_generation and _forced_final_has_no_answer(\n                                clean_ui, final_ai_content):",
     "                        if False:"),
    ("F1 retry fires on every final (not just forced)", AGENT,
     "                        if is_final_generation and _forced_final_has_no_answer(\n                                clean_ui, final_ai_content):",
     "                        if _forced_final_has_no_answer(\n                                clean_ui, final_ai_content):"),
    ("F1 retry unbounded", AGENT,
     "                                    and turn < effective_max_turns - 1):\n                                _forced_final_retry_used = True\n",
     "                                    and turn < effective_max_turns - 1):\n"),
    ("F1 retry sends no directive", AGENT,
     '                                messages.append({"role": "user",\n                                                 "content": _FORCED_FINAL_ANSWER_DIRECTIVE})\n',
     ""),
    ("F1 fallback keeps the narration", AGENT,
     '                            clean_ui = ui_content.strip("` \\n\\r")\n                            final_ai_content = ""\n',
     '                            clean_ui = ui_content.strip("` \\n\\r")\n'),
    ("F1 fallback drops the evidence body", AGENT,
     '    return (f"{head}\\n\\nLast evidence gathered ({where}):\\n\\n{body}\\n\\n"',
     '    return (f"{head}\\n\\nLast evidence gathered ({where}):\\n\\n\\n\\n"'),
    ("F1 predicate ignores the accumulated narration", SMOOTH,
     '    parts = [p for p in ((accumulated or "").strip(), (this_turn_text or "").strip()) if p]',
     '    parts = [p for p in ((this_turn_text or "").strip(),) if p]'),
    ("F1 predicate: empty is an answer", SMOOTH,
     "    return not body or narration_only(body)", "    return bool(body) and narration_only(body)"),
    ("F1 dropped note counts as an answer", AGENT,
     "    i = text.find(_DROPPED_NOTE_HEAD)\n    if i >= 0:\n        text = text[:i]\n", ""),
    ("F1 fallback uses a private head (shape arm cannot see it)", AGENT,
     '    head = FALLBACK_HEADS["no_answer"]\n', '    head = "I ran out of budget before writing an answer."\n'),
    # ── F2 narration-only shape ──
    ("F2 'let me know' is a beat", SMOOTH, "let me(?!\\s+know)", "let me"),
    ("F2 content check removed", SMOOTH,
     "        if len(b) > _MAX_NARRATION_CHARS or _NARRATION_CONTENT_RE.search(b):", "        if len(b) > _MAX_NARRATION_CHARS:"),
    ("F2 any paragraph with a beat qualifies (glue unbounded)", SMOOTH,
     "        if any(len(s) > _NARRATION_GLUE_MAX_CHARS or _NARRATION_ADDRESSED_RE.search(s)\n               for s in sents if not _is_work_beat(s)):\n            return False\n", ""),
    ("F2 a paragraph without a beat qualifies", SMOOTH,
     "        if not any(_is_work_beat(s) for s in sents):\n            return False\n", ""),
    ("F2 refutation ignores the tools gate", SHAPE, "    if int(n_real_tools or 0) < 1:\n        return []\n", ""),
    ("F2 refutation fires on image turns", SHAPE,
     '    if any(str(n).strip().lower() == "image_generation" for n in (tool_names or ())):\n        return []\n', ""),
    ("F2 wrapper never runs the narration arm", AGENT,
     "                issues = refute_narration_only(\n                    claim, n_real_tools=count_real_tools(tools_run or []),\n                    tool_names=_names)\n",
     "                issues = []\n"),
    ("F2 wrapper mislabels the arm (judge still runs)", AGENT,
     "                reasoning = self._NARRATION_ONLY_REASONING\n", '                reasoning = "reply-shape check (narration)"\n'),
    ("F2 tool-turn exit removed (judge consulted)", AGENT,
     "        if _shape is not None and self._verdict_is_no_claim(_shape):", "        if False:"),
    ("F2 tool-turn exit returns the judge's None", AGENT,
     "            return _mech, last_tool\n        # Replay the active project's explicit user constraints into the",
     "            return None, last_tool\n        # Replay the active project's explicit user constraints into the"),
    ("F2 wrapper not given the rows", AGENT,
     "            _shape = self._reply_shape_refutation(final_ai_content, last_user_content,\n                                                  tools_run_this_turn)",
     "            _shape = self._reply_shape_refutation(final_ai_content, last_user_content)"),
    # ── round-2 fixes (R3: mutate the previous round's fixes first) ──
    ("R2 beat needs no work verb", SMOOTH,
     "            and bool(_NARRATION_WORK_RE.search(sentence))\n", ""),
    ("R2 a question is a beat", SMOOTH,
     "            and not _NARRATION_ADDRESSED_RE.search(sentence))", "            )"),
    ("R2 glue may address the user", SMOOTH,
     "        if any(len(s) > _NARRATION_GLUE_MAX_CHARS or _NARRATION_ADDRESSED_RE.search(s)",
     "        if any(len(s) > _NARRATION_GLUE_MAX_CHARS"),
    ("R2 fallback head back in the raw-dump arm", SHAPE,
     '    + "|".join([re.escape(h) for k, h in FALLBACK_HEADS.items() if k != "no_answer"]',
     '    + "|".join([re.escape(h) for k, h in FALLBACK_HEADS.items()]'),
    ("R2 fallback arm never fires", SHAPE,
     '    if not _NO_ANSWER_HEAD_RE.match(reply or ""):\n        return []\n', "    return []\n"),
    ("R2 no-claim verdicts still repaired (sync)", AGENT,
     "                                        and _vr.confidence >= 0.7\n                                        # §4GH: no claim → no repair\n                                        and not self._verdict_is_no_claim(_vr)\n",
     "                                        and _vr.confidence >= 0.7\n"),
    ("R2 no-claim set misses the fallback", AGENT,
     "    _NO_CLAIM_REASONINGS = frozenset({_NARRATION_ONLY_REASONING, _NO_ANSWER_REASONING})",
     "    _NO_CLAIM_REASONINGS = frozenset({_NARRATION_ONLY_REASONING})"),
    ("R2 dropped names not carried", AGENT, "                        _forced_final_dropped.extend(dropped)\n", ""),
    ("R2 note not appended to what ships", AGENT,
     "                        if _ffd_note and _DROPPED_NOTE_HEAD not in (ui_content or \"\"):\n",
     "                        if False:\n"),
    ("R2 retry on the last budget turn", AGENT,
     "                            if (not _forced_final_retry_used\n                                    and turn < effective_max_turns - 1):",
     "                            if not _forced_final_retry_used:"),
    ("R2 screenshot counts as a load", AGENT, '        elif op in ("navigate", ""):', '        elif op in ("navigate", "", "screenshot"):'),
    ("R2 target not truncated like the breaker's", AGENT,
     '        if str(args.get("url") or "").strip().lower()[:200] != tgt:', '        if str(args.get("url") or "").strip().lower() != tgt:'),
    # ── F3 breaker ──
    ("F3 nav case never detected", AGENT,
     '                    _nav_case = (_afname == "browser"\n                                 and _browser_loaded_but_never_extracted(\n                                     tools_run_this_turn, _atarget))',
     "                    _nav_case = False"),
    ("F3 nav steer also forces the final", AGENT,
     "                        repeated_action_steered.add(_asig)\n                        pretty_log(\n                            \"Loop Breaker\",\n                            f\"No-progress: 'browser' loaded '{_atarget}' {_acnt}x and \"",
     "                        repeated_action_steered.add(_asig)\n                        force_final_response = True\n                        pretty_log(\n                            \"Loop Breaker\",\n                            f\"No-progress: 'browser' loaded '{_atarget}' {_acnt}x and \""),
    ("F3 third load aborts instead of forcing", AGENT,
     "                    elif _acnt >= _hard_n and _nav_case:\n                        force_final_response = True\n",
     "                    elif False:\n                        force_final_response = True\n"),
    ("F3 extracted page still counts as unread", AGENT,
     '        if op == "extract_text":\n            extracted = True\n', '        if op == "extract_text":\n            pass\n'),
    ("F3 target compared case-sensitively", AGENT,
     '        if str(args.get("url") or "").strip().lower()[:200] != tgt:', '        if str(args.get("url") or "").strip()[:200] != tgt:'),
    ("F3 synthetic rows count as loads", AGENT,
     '        if not isinstance(r, dict) or r.get("_synthetic"):\n            continue\n        if str(r.get("name") or "").lower() != "browser":',
     '        if not isinstance(r, dict):\n            continue\n        if str(r.get("name") or "").lower() != "browser":'),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def restore():
    for rel in (AGENT, SMOOTH, SHAPE):
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

base = {rel: sha(os.path.join(PRISTINE, rel)) for rel in (AGENT, SMOOTH, SHAPE)}
restore(); purge()
try:
    for i in range(lo, hi):
        label, rel, old, new = MUTANTS[i]
        t0 = time.time()
        if rel == "PREFIX":
            if not os.path.isdir(ORIG):
                print(f"[{i:2d}] SKIPPED   {label}  (no ../orig pre-fix copies)"); continue
            for f, r in (("agent.py", AGENT), ("reply_smoothing.py", SMOOTH), ("reply_shape_check.py", SHAPE)):
                shutil.copyfile(os.path.join(ORIG, f), os.path.join(TREE, r))
        else:
            p = os.path.join(TREE, rel)
            src = open(p, encoding="utf-8").read()
            if src.count(old) != 1:
                print(f"[{i:2d}] MISSING   {label}  (anchor count {src.count(old)})"); continue
            open(p, "w", encoding="utf-8").write(src.replace(old, new))
        purge()
        c = subprocess.run([PY, "-m", "py_compile", os.path.join(TREE, AGENT), os.path.join(TREE, SMOOTH), os.path.join(TREE, SHAPE)],
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
