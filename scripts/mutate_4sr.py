#!/usr/bin/env python3
"""§4SR battery — gate words as whole words, one classifier, a batch past a gate
(2026-11-01, req 4609b359). Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is
never mutated. No-op control must SURVIVE, known-bad control must DIE.
    python3 scripts/mutate_4sr.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = "src/ghost_agent/core/project_advancer.py"
T = "src/ghost_agent/tools/projects.py"
AG = "src/ghost_agent/core/agent.py"
_SETTLE_GATE = '    if needs_user_gate(description):\n        return "needs_user"\n    cls = str(classification or "").strip().lower()'
_RX = 'return r"(?:(?<![\\w-])" + pre + r")(?:" + alts + r")(?![\\w-])(?!\\.\\w)(?!\\()"'
_CLS = ('            found = [lb for lb in ("needs_user", "coding", "research") if lb in out]\n'
        '            if len(found) == 1:')
MUT = [
    ("CONTROL no-op", A, "def needs_user_gate(description) -> bool:", "def needs_user_gate(description) -> bool:  # no-op"),
    ("CONTROL known-bad", A, '    if needs_user_gate(lower):\n        return "needs_user"\n', ""),
    # r4 fixes first (R3)
    ('R1 substring (r0 shape)', A, '    return bool(_NEEDS_USER_RE.search(_norm(description)))', '    return bool(any(k in _norm(description) for k in _NEEDS_USER_KEYWORDS))'),
    ('R6 no-unicode-map', A, '    d = str(description or "").lower().translate(_NORM_MAP)', '    d = str(description or "").lower()'),
    ('R12 no-lower', A, '    d = str(description or "").lower().translate(_NORM_MAP)', '    d = str(description or "").translate(_NORM_MAP)'),
    ('R13 no-slash-pairs', A, '    return _SLASH_PAIR_RE.sub(_pair, d)', '    return d'),
    # r5 fixes
    ('P1 any-pair-spaced (r5 #1)', A, '    return " / ".join(parts) if all(p in _PAIR_VERBS for p in parts) else m.group(0)', '    return " / ".join(parts)'),
    ('P2 one-verb-enough', A, '    return " / ".join(parts) if all(p in _PAIR_VERBS for p in parts) else m.group(0)', '    return " / ".join(parts) if any(p in _PAIR_VERBS for p in parts) else m.group(0)'),
    # r6 fixes
    ('G3 pair-end-dot (r6 #2)', A, '_SLASH_PAIR_RE = re.compile(r"(?<![/\\w.])[a-z]+(?:/[a-z]+)+(?![/\\w])(?!\\.\\w)")', '_SLASH_PAIR_RE = re.compile(r"(?<![/\\w.])[a-z]+(?:/[a-z]+)+(?![/\\w.])")'),
    ('G4 pair-only-two', A, '_SLASH_PAIR_RE = re.compile(r"(?<![/\\w.])[a-z]+(?:/[a-z]+)+(?![/\\w])(?!\\.\\w)")', '_SLASH_PAIR_RE = re.compile(r"(?<![/\\w.])[a-z]+(?:/[a-z]+)(?![/\\w])(?!\\.\\w)")'),
    ('G5 pair-path-dot', A, '_SLASH_PAIR_RE = re.compile(r"(?<![/\\w.])[a-z]+(?:/[a-z]+)+(?![/\\w])(?!\\.\\w)")', '_SLASH_PAIR_RE = re.compile(r"(?<![/\\w.])[a-z]+(?:/[a-z]+)+(?![/\\w])")'),
    ('G6 no-choice (r6 #3)', A, '"picking one", "choice", "choices",', '"picking one",'),
    ('G7 zero-width-kept', A, '                           "\\u200b": None, "\\u200c": None, "\\u200d": None,\n', ''),
    ('G8 word-joiner-kept', A, '"\\u2060": None, "\\ufeff": None,', '"\\ufeff": None,'),
    # r8 fixes
    ('E1 experiment-counted-ran (r8 #2)', A, '            _not_run.add(res.task_id)             # §4SR r8', '            pass             # §4SR r8'),
    ('E2 text-minus-one', T, "ran if getattr(batch, 'ran', None) is not None else ran - 1", "ran - 1"),
    ('E3 outcome-by-count (r8 #1)', T, '                _mk = _TO.partial if _ran else _TO.failed', '                _mk = _TO.partial if batch.count else _TO.failed'),
    ('E4 world-by-count', T, '                    world_changed=bool(_ran),', '                    world_changed=bool(batch.count),'),
    ('E5 backstop-always-cap (r8 NIT)', A, '        stop_reason = ("count_reached" if _ticks >= limit and max_tasks is not None', '        stop_reason = ("count_reached" if False and max_tasks is not None'),
    ('E6 leading-dot-blocks', A, '    return (r"(?:(?<![\\w/`\\\\@$-])(?<![\\w)\\]\'\\"]\\.)" + pre', '    return (r"(?:(?<![\\w./`\\\\@$-])" + pre'),
    ('E7 leading-dot-free', A, '    return (r"(?:(?<![\\w/`\\\\@$-])(?<![\\w)\\]\'\\"]\\.)" + pre', '    return (r"(?:(?<![\\w/`\\\\@$-])" + pre'),
    # r9 fixes
    ('E8 dot-after-word-only (r9 #1)', A, '(?<![\\w)\\]\'\\"]\\.)" + pre', '(?<!\\w\\.)" + pre'),
    ('E9 no-setting-rule', A, '(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")', '(?!\\()(?!/)")'),
    # r11 fixes
    ('V1 nbsp-not-space (r11 NIT)', A, '(?!/)(?![^\\S\\n]*=(?![=>]))")', '(?!/)(?![ \\t]*=(?![=>]))")'),
    ('V2 loose-right-hyphen (r11)', A, '+ _alts(words) + r")(?!\\w)"', '+ _alts(words) + r")(?![\\w-])"'),
    ('V3 loose-left-free', A, '        return r"(?:(?<![\\w-])" + pre', '        return r"(?:(?<![\\w])" + pre'),
    # r10 fixes
    ('W1 setting-crosses-lines (r10 #1)', A, '(?!/)(?![^\\S\\n]*=(?![=>]))")', '(?!/)(?!\\s*=(?![=>]))")'),
    ('W2 arrow-is-setting', A, '(?!/)(?![^\\S\\n]*=(?![=>]))")', '(?!/)(?![^\\S\\n]*=)")'),
    ('W3 gate-form-strict', A, '    return bool(_GATE_LOOSE_RE.search(_norm(description)))', '    return bool(_GATE_FORMS_RE.search(_norm(description).replace("/", " ")) or _NEEDS_USER_RE.search(_norm(description).replace("/", " ")))'),
    ('W4 loose-no-hyphen', A, '        return r"(?:(?<![\\w-])" + pre + r")(?:" + _alts(words) + r")(?!\\w)"', '        return r"(?:(?<![\\w])" + pre + r")(?:" + _alts(words) + r")(?!\\w)"'),
    ('W5 loose-no-base', A, '_GATE_LOOSE_RE = re.compile(_whole_words(_GATE_FORMS | _NEEDS_USER_KEYWORDS, strict=False))', '_GATE_LOOSE_RE = re.compile(_whole_words(_GATE_FORMS, strict=False))'),
    ('E10 no-sigils', A, '(?<![\\w/`\\\\@$-])(?<!', '(?<![\\w/`\\\\-])(?<!'),
    ('E11 no-backslash-lead', A, '(?<![\\w/`\\\\@$-])(?<!', '(?<![\\w/`@$-])(?<!'),
    ('E12 no-backslash-trail', A, '+ r")(?![\\w`\\\\-])(?!', '+ r")(?![\\w`-])(?!'),
    ('L7 limit-failures-rename (r6 #4)', A, '            stop_reason = ("hard_cap" if max_tasks is None or int(max_tasks) > hard_cap\n                           else "count_reached")', '            stop_reason = _final_reason("hard_cap" if max_tasks is None or int(max_tasks) > hard_cap\n                           else "count_reached")'),
    # r7 fixes
    ('L3 over-cap-is-count (r7 #2)', A, 'if max_tasks is None or int(max_tasks) > hard_cap\n', 'if max_tasks is None\n'),
    ('L4 backstop-failures-rename', A, '                       and int(max_tasks) <= hard_cap else "hard_cap")', '                       and int(max_tasks) <= hard_cap else _final_reason("hard_cap"))'),
    ('L5 count-fail-is-ok (r7 #1)', T, '                                     "repeated_failures") or any(', '                                     "repeated_failures") or False and any('),
    ('L6 minus-not-mapped', A, '                           "\\u2212": "-", "\\ufe63": "-", "\\uff0d": "-"})', '                           "\\ufe63": "-"})'),
    ('P3 no-pair-verbs-extra', A, '    "review", "test", "check", "verify", "reject", "merge", "accept", "deny",', '    "check", "verify", "merge", "accept", "deny",'),
    ('P4 trailing-slash-gates', A, '(?!\\()(?!/)(?![^', '(?!\\()(?![^'),
    ('P5 soft-hyphen-kept', A, '"\\u2011": "-", "\\u00ad": None,\n', '"\\u2011": "-",\n'),
    ('F1 no-plural-forms', A, '    "approvals", "decisions", "confirmations", "authorizations", "authorisations",', '    "decisions", "authorizations", "authorisations",'),
    ('F2 no-pick-one', A, '    "announcements", "picks one", "picked one", "picking one",', '    "announcements",'),
    ('F3 no-sent', A, '"sends", "sending", "sent", "signs off"', '"sends", "sending", "signs off"'),
    ('R14 slash-pairs-in-paths', A, '_SLASH_PAIR_RE = re.compile(r"(?<![/\\w.])[a-z]+(?:/[a-z]+)+(?![/\\w])(?!\\.\\w)")', '_SLASH_PAIR_RE = re.compile(r"[a-z]+(?:/[a-z]+)+")'),
    ('B8 rollup-is-budget', A, '            elif pstatus == "NEEDS_USER" and res.summary.startswith("project is "):', '            elif False:'),
    ('B9 any-block-is-gate (r4 MAJOR)', A, '            elif pstatus == "NEEDS_USER" and res.summary.startswith("project is "):', '            elif pstatus == "NEEDS_USER":'),
    ('M2j paused-gate-unbooked', A, '                _note(res, advanced[-1]["status"])   # r3 n3: a gate here did not run', '                pass'),
    ('M2k safe-mode-unbooked', A, '                if stop_on_fail:\n                    _note(res, status)', '                if stop_on_fail:\n                    pass'),
    ('M2l note-drops-hold', A, '        if not _skip_tick(res) and status == "NEEDS_USER" and res.task_id:\n            _held.append(res.task_id)', '        if not _skip_tick(res) and status == "NEEDS_USER" and res.task_id:\n            pass'),
    ('N3 gate-note-dropped', T, '    gated = ((f"; {_ng} more left waiting at a gate" if _ng else "")', '    gated = (("")'),
    ('N5 prerun-hold-note-dropped', T, '             + (f"; {_nhb} more held before running" if _nhb else ""))', '             + "")'),
    ('N6 ran-hold-note-dropped', T, '    tail += f", {_nha} held for the owner\'s own run" if _nha else ""', '    tail += ""'),
    ('N7 units-mixed', T, '    _nha = sum(1 for h in _held if h not in _nr)', '    _nha = len(_held)'),
    ('N8 payload-ran-count', T, '                "ran": getattr(batch, "ran", batch.count),', '                "ran": batch.count,'),
    # r3 fixes first (R3)
    ('R2 no-lookbehind', A, 'return (r"(?:(?<![\\w/`\\\\@$-])(?<![\\w)\\]\'\\"]\\.)" + pre', 'return (r"(?:(?<=)" + pre'),
    ('R4 hyphen-before-ok', A, 'return (r"(?:(?<![\\w/`\\\\@$-])(?<![\\w)\\]\'\\"]\\.)" + pre', 'return (r"(?:(?<![\\w/`\\\\@$])(?<![\\w)\\]\'\\"]\\.)" + pre'),
    ('R3 no-lookahead', A, '+ r")(?![\\w`\\\\-])(?!\\.\\w)(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")', '+ r")(?!\\.\\w)(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")'),
    ('R5 hyphen-after-ok', A, '+ r")(?![\\w`\\\\-])(?!\\.\\w)(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")', '+ r")(?![\\w`\\\\])(?!\\.\\w)(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")'),
    ('R7 inflections-ok', A, '+ r")(?![\\w`\\\\-])(?!\\.\\w)(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")', '+ r")(?:s|es|d|ed|ion|ation)?(?![\\w`\\\\-])(?!\\.\\w)(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")'),
    ('R8 identifier-ok', A, '+ r")(?![\\w`\\\\-])(?!\\.\\w)(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")', '+ r")(?![\\w`\\\\-])")'),
    ('R10 sentence-dot-blocks', A, '+ r")(?![\\w`\\\\-])(?!\\.\\w)(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")', '+ r")(?![\\w.`\\\\-])(?!\\()(?!/)(?![^\\S\\n]*=(?![=>]))")'),
    ('R11 no-ws-runs', A, '    return "|".join(re.escape(w).replace(r"\\ ", r"\\s+")', '    return "|".join(re.escape(w)'),
    ('F6 no-pre-prefix', A, 'def _whole_words(words, prefixes=("re", "pre"), strict=True) -> str:', 'def _whole_words(words, prefixes=("re",), strict=True) -> str:'),
    ('N1 ran-is-count', A, '        return sum(1 for a in self.advanced if a.get("task_id") not in skip)', '        return len(self.advanced)'),
    ('N2 not-run-not-returned', A, '                             list(_gated), list(_held), sorted(_not_run))', '                             list(_gated), list(_held), [])'),
    ('H5 owner-text-unattended', T, '                    if _by_owner:\n                        # r2 #2', '                    if True:\n                        # r2 #2'),
    ('H6 unattended-text-owner', T, '                    if _by_owner:\n                        # r2 #2', '                    if False:\n                        # r2 #2'),
    ('H7 hold-tail-cut', T, '                    _r = _r if len(_r) <= 200 else _r[:110] + " … " + _r[-85:]', '                    _r = _r[:200]'),
    ('M2b gate-counts', A, '            if _skip_tick(res):\n                continue', '            if _skip_tick(res):\n                _ticks += 1\n                continue'),
    ('M2f ran-held-not-counted', A, '            _held.append(res.task_id)\n        _ticks += 1', '            _held.append(res.task_id)\n            continue\n        _ticks += 1'),
    ('M2g ran-held-unrecorded', A, '            _held.append(res.task_id)\n        _ticks += 1', '            pass\n        _ticks += 1'),
    ('M2h prerun-hold-is-gate', A, '        (_held if str(res.summary or "").endswith(_HELD_MARK) else _gated).append(res.task_id)', '        _gated.append(res.task_id)'),
    ('M2i hold-skip-not-recorded', A, '        _not_run.add(res.task_id)\n        return True', '        return True'),
    ('B1 always-stop', A, '            if stop_on_fail or res.task_id in _not_run:', '            if True:'),
    ('B2 no-repeat-guard', A, '            if stop_on_fail or res.task_id in _not_run:', '            if stop_on_fail:'),
    ('B3 safe-mode-skips', A, '            if stop_on_fail or res.task_id in _not_run:', '            if res.task_id in _not_run:'),
    # r2 fixes first (R3)
    ('F4 no-prefix', A, 'pre = "".join(f"|(?<=\\\\b{p}-)" for p in prefixes)', 'pre = ""'),
    ('F5 no-sign-off', A, '| {"sign-off", "authorise"}', '| {"authorise"}'),
    ('Q3 authorise-not-base', A, '| {"sign-off", "authorise"}', '| {"sign-off"}'),
    ('L1 llm-form-ignored', A, '    if cls == "needs_user" and gate_form(description):\n        return "needs_user"\n', ''),
    ('L2 llm-gates-anything', A, '    if cls == "needs_user" and gate_form(description):', '    if cls == "needs_user":'),
    ('N4 payload-gated-0', T, '                "gated": len(getattr(batch, "gated", None) or []),', '                "gated": 0,'),
    # r1 fixes first (R3)
    ("M1a rollup-hides-failure", A, '"project_failed" if _ledger_failed() else "needs_user")', '"needs_user")'),
    ("M1b ledger-scan-blind", A, '            return any(str(t.get("status", "")).upper() == "FAILED"\n                       for t in store.list_tasks(project_id))',
     "            return False"),
    ("M2a never-counted", A, "        _ticks += 1\n        if status == \"FAILED\":", "        if status == \"FAILED\":"),
    ("M2c no-backstop", A, "    for _ in range(limit + hard_cap):", "    for _ in range(10 ** 6):"),
    ('M2d cap-as-count', A, 'and int(max_tasks) <= hard_cap else "hard_cap")', 'and int(max_tasks) <= hard_cap else "count_reached")'),
    ("M2e gated-as-final", A, '            return "completed_with_failures"\n        return default\n',
     '            return "completed_with_failures"\n        if _gated:\n            return "needs_user"\n        return default\n'),
    ("H1 holds-are-gates", T, '                          if not str(t.get("result_summary") or "").endswith(_HELD_MARK)][:10]',
     "                          if True][:10]"),
    ("H2 hold-text-on-experiment", T, '                    if batch.stop_reason != "model_experiment" and (', "                    if True or ("),
    ("H3 holds-only-text-gone", T, '                elif batch.stop_reason == "needs_user":\n                    # only holds wait', "                elif False:\n                    # only holds wait"),
    ("H4 holds-unlisted", T, '                    _adv_payload["held_tasks"] = _holds', "                    pass"),
    ("C9 sick-worker-30s", A, "require_healthy=True, timeout=10.0", "timeout=30.0"),
    ("C8 two-labels-first-wins", A, _CLS, '            found = [lb for lb in ("needs_user", "coding", "research") if lb in out][:1]\n            if len(found) == 1:'),
    # the gate words
    # settle_gate
    ("S1 gate-ignored", A, _SETTLE_GATE, '    cls = str(classification or "").strip().lower()'),
    ("S2 llm-may-gate", A, '    if cls in ("coding", "research"):\n        return cls',
     '    if cls in ("coding", "research", "needs_user"):\n        return cls'),
    ("S3 default-dropped", A, "        return cls\n    return classify_task(description, default=default)",
     "        return cls\n    return classify_task(description)"),
    ("S4 no-strip", A, '    cls = str(classification or "").strip().lower()', '    cls = str(classification or "").lower()'),
    ("S5 call-site-off", A, "    _settled = settle_gate(nxt.description, classification, default=_default_bucket)",
     "    _settled = classification"),
    # the one classifier
    ("C1 main-slot", A, "            }, use_worker=True, is_background=background, off_main_only=True,",
     "            }, is_background=background, side_main=True,"),
    ("C2 thinking-on", A, '                "chat_template_kwargs": {"enable_thinking": False},\n            }, use_worker=True', "            }, use_worker=True"),
    ("C3 background-ignored", A, "use_worker=True, is_background=background, off_main_only=True",
     "use_worker=True, is_background=False, off_main_only=True"),
    ("C4 hook-not-called", A, "                    on_error(e)\n", "                    pass\n"),
    ("C5 hook-raise-escapes", A, '                except Exception:  # noqa: BLE001\n                    logger.debug("classifier on_error hook failed"',
     '                except ZeroDivisionError:  # noqa: BLE001\n                    logger.debug("classifier on_error hook failed"'),
    ("C6 no-label-research", A, "                    out[:40], str(description)[:80])\n        return classify_task(description)",
     '                    out[:40], str(description)[:80])\n        return "research"'),
    ("C7 error-research", A, "                    logger.debug(\"classifier on_error hook failed\", exc_info=True)\n            return classify_task(description)",
     "                    logger.debug(\"classifier on_error hook failed\", exc_info=True)\n            return \"research\""),
    ("AG1 idle-foreground", AG, "                            ctx, background=True,\n", "                            ctx, background=False,\n"),
    ("AG2 idle-no-hook", AG, '                            on_error=lambda _ce: _idle_phase_failed(self, _idle_ran, "autoadvance", _ce))',
     "                            on_error=None)"),
    # the batch
    # what the model reads
    ("T1 list-only-on-gate", T, "                if _gates:\n", '                if _gates and batch.stop_reason == "needs_user":\n'),
    ("T2 no-addendum", T, '                    if batch.stop_reason != "needs_user":\n', "                    if False:\n"),
    ("T3 gate-text-gone", T, '    if batch.stop_reason == "needs_user":\n        # §4SR', "    if False:\n        # §4SR"),
    ("T4 old-rerun-text", T, 'f"answered (or the fix is in), calling autoadvance again is how to "',
     'f"answered (or the fix is in), re-running autoadvance after a fix is how to "'),
]
PINS = ["tests/test_4sr_gate_words.py"]


def _tests():
    pat = r"classify_task\|needs_user\|advance_many\|default_llm_classifier\|_advance_batch_instruction\|settle_gate"
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


if __name__ == "__main__":
    main()
