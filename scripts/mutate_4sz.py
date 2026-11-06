#!/usr/bin/env python3
"""§4SZ battery — the owner's rules (core/owner_rules.py), their wiring in the agent, the thumbs-down hint,
the idle window that restarts on its own now that the failure-replay slot is removed, and the probe guards
whose tests were restored. Coverage-selected tests in parallel tree copies (scripts/battery_cov.py); the real
src/ is never mutated. No-op control must SURVIVE, the known-bad control must DIE.
    python3 scripts/mutate_4sz.py [check | -j6 all | rerun R1 A2 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OR = "src/ghost_agent/core/owner_rules.py"
AG = "src/ghost_agent/core/agent.py"
EX = "src/ghost_agent/memory/experience.py"
QM = "src/ghost_agent/core/queued_messages.py"
TM = "src/ghost_agent/tools/memory.py"
RT = "src/ghost_agent/api/routes.py"
JS = "interface/static/app.js"
HI = "src/ghost_agent/tools/health_internals.py"
SE = "src/ghost_agent/tools/search.py"
AA = "src/ghost_agent/core/autonomous_activity.py"

# anchors used by more than one mutant (each is ONE exact place in its file)
_QUOTE_IF = "    if len(t) >= 2 and _QUOTE_PAIRS.get(t[0]) == t[-1] and t[0] not in t[1:-1] and t[-1] not in t[1:-1]:"
_CONFIRM_IF = '        if pend.get("n") != n or not _fresh(pend):'
_ALL_IF = '        if pend.get("all") != sorted(r["n"] for r in d["rules"]) or not _fresh(pend):'
_FRESH = '        return 0 <= time.time() - pend["at"] <= FORGET_CONFIRM_S'
_ROW_N = 'type(r.get("n")) is int and 1 <= r["n"] <= 999_999'
_ROW_RULE = 'and isinstance(r.get("rule"), str) and r["rule"].strip()):'
_HINT_RET = '    return RULE_HINT if (_HINT_VERB_RE.search(t) and _HINT_NOUN_RE.search(t)) else ""'
_REFUSE = "            return _refusal(why)\n"
_ADD_START = r'_ADD_START = r"^\s*" + _POLITE + r"(?:learn|remember)\s+this\s+rule\b"'
_ADD_COLON = r'_ADD_RE = re.compile(_ADD_START + r"\s*:", re.I)'
_BODY_IF = "    if len(body.splitlines()) != 1 or _QUESTION_RE.search(body):"
_QRE = "_QUESTION_RE = re.compile(r" + '"' + "[?？" + "\\u037e]|;[" + "\\s" + "\\" + '"' + "'”’»)!.]*$" + '"' + ")"
_WHICH = r'_WHICH = r"(?:all\s+(?:of\s+)?)?my\s+"'
_END = "_END = r" + '"' + r"(?:[\s,]+(?:please|pls))?[\s\"'“”‘’«».!?]*$" + '"'
_END_NO_PLEASE = "_END = r" + '"' + r"[\s\"'“”‘’«».!?]*$" + '"'
_AFFIRM_IF = "    if words <= AFFIRM_MAX_WORDS and _AFFIRM_RE.match(t):"
_TRY = '    try:\n        return _dispatch(str(text or "").strip(), home)\n    except OSError as e:'
_GATE = '            if _rk() != "owner" or reply_is_public():\n                return ""\n            msgs = body.get("messages")'
_HOME = "            home = Path(md).parent if isinstance(md, (str, Path)) and str(md) else None"
_LAST = ('            text = _message_lines(last.get("content")) if last else ""\n'
         "            from ..selfhood.mirror import is_mirror_command")
_NEXT = ('            last = next((m for m in reversed(msgs or []) if isinstance(m, dict) and m.get("role") == "user"), None)\n'
         + _LAST)
_PARTS = ('content = "\\n".join(str(i.get("text") or "") for i in content\n'
          '                            if isinstance(i, dict) and i.get("type") == "text")\n'
          '    return str(content or "")')
_RESTART = "        if idle_secs > self._bio_scaled(3600):\n            self._idle_window_at = datetime.datetime.now()\n            # §4NV: nothing RAN"

MUT = [
    ("CONTROL no-op", OR, "def _fmt(r: dict) -> str:", "def _fmt(r: dict) -> str:  # no-op"),
    ("CONTROL known-bad", OR, '    return sorted(load(home)["rules"], key=lambda r: r["n"])', "    return []"),
    # ── adding ──
    ("R1 too-short-accepted", OR, "    if len(rule) < RULE_MIN_CHARS:", "    if False:"),
    ("R2 too-long-accepted", OR, "    if len(rule) > RULE_MAX_CHARS:", "    if False:"),
    ("R2b min-chars", OR, "RULE_MIN_CHARS = 8", "RULE_MIN_CHARS = 3"),
    ("R2c max-chars", OR, "RULE_MAX_CHARS = 400", "RULE_MAX_CHARS = 4000"),
    ("R3b wordless-rules-are-one-rule", OR, ' or " ".join(t.split())\n', "\n"),
    ("R3 duplicate-added", OR, '        same = next((r for r in d["rules"] if _key(r["rule"]) == _key(rule)), None)', "        same = None"),
    ("R4 cap-ignored", OR, '        if len(d["rules"]) >= RULES_MAX:', "        if False:"),
    ("R4b cap-value", OR, "RULES_MAX = 10", "RULES_MAX = 100"),
    ("R4c cap-off-by-one", OR, '        if len(d["rules"]) >= RULES_MAX:', '        if len(d["rules"]) > RULES_MAX:'),
    ("R5 number-reused", OR, '        d["next_n"] += 1\n', ""),
    ("R5b counter-not-repaired", OR, '    if type(d.get("next_n")) is not int or d["next_n"] <= highest:', "    if False:"),
    ("R5c counter-equal-kept", OR, 'or d["next_n"] <= highest:', 'or d["next_n"] < highest:'),
    ("R5d counter-type-unchecked", OR, '    if type(d.get("next_n")) is not int or d["next_n"] <= highest:', '    if d.get("next_n", 0) <= highest:'),
    ("R6 not-saved", OR, '        d["next_n"] += 1\n        _save(home, d)\n', '        d["next_n"] += 1\n'),
    ("R7 raw-text-stored", OR, "    rule = _clean(text)\n", '    rule = " ".join(str(text).split())\n'),
    ("R7b tokens-removed-once", OR, "    for _ in range(50):\n", "    for _ in range(1):\n"),
    ("R7c leftover-opener-kept", OR, '    t = " ".join(t.replace("<|", "< |").split())', '    t = " ".join(t.split())'),
    ("R7d quote-strip-always", OR, _QUOTE_IF, "    if len(t) >= 2 and t[0] in _QUOTE_PAIRS:"),
    ("R7e quote-strip-never", OR, _QUOTE_IF, "    if False:"),
    ("R7f opener-inside-ignored", OR, " and t[0] not in t[1:-1] and t[-1] not in t[1:-1]:", " and t[-1] not in t[1:-1]:"),
    ("R7g closer-inside-ignored", OR, " and t[0] not in t[1:-1] and t[-1] not in t[1:-1]:", " and t[0] not in t[1:-1]:"),
    # ── forgetting ──
    ("R8 confirm-without-preview", OR, _CONFIRM_IF, "        if False:"),
    ("R8b confirm-any-preview", OR, _CONFIRM_IF, "        if not pend or not _fresh(pend):"),
    ("R8c confirm-never-expires", OR, _CONFIRM_IF, '        if pend.get("n") != n:'),
    ("R8d confirm-window", OR, "FORGET_CONFIRM_S = 15 * 60", "FORGET_CONFIRM_S = 15 * 60 * 60"),
    ("R8e preview-from-the-future-confirms", OR, _FRESH, '        return time.time() - pend["at"] <= FORGET_CONFIRM_S'),
    ("R8f preview-never-expires", OR, _FRESH, '        return 0 <= time.time() - pend["at"]'),
    ("R8g preview-time-too-large-raises", OR, "    except OverflowError:", "    except ZeroDivisionError:"),
    ("R9 preview-removes", OR, '        d["pending"] = {"n": n, "at": time.time()}\n        _save(home, d)', '        d["rules"] = [x for x in d["rules"] if x["n"] != n]\n        _save(home, d)'),
    ("R9b preview-not-armed", OR, '        d["pending"] = {"n": n, "at": time.time()}\n', ""),
    ("R10 forget-removes-nothing", OR, '        d["rules"] = [x for x in d["rules"] if x["n"] != n]\n', ""),
    ("R11 forget-all-without-preview", OR, _ALL_IF, "        if False:"),
    ("R11b forget-all-on-any-preview", OR, _ALL_IF, "        if not pend or not _fresh(pend):"),
    ("R11c forget-all-not-bound-to-shown", OR, 'pend.get("all") != sorted(r["n"] for r in d["rules"])', 'not pend.get("all")'),
    ("R11d forget-all-never-expires", OR, _ALL_IF, '        if pend.get("all") != sorted(r["n"] for r in d["rules"]):'),
    ("R11e forget-all-preview-removes", OR, '        d["pending"] = {"all": [r["n"] for r in rs], "at": time.time()}\n        _save(home, d)', '        d["rules"] = []\n        _save(home, d)'),
    ("R11f forget-all-restarts-numbers", OR, '        n = len(d["rules"])\n        d["rules"] = []\n', '        n = len(d["rules"])\n        d["rules"] = []\n        d["next_n"] = 1\n'),
    ("R11g forget-all-preview-not-armed", OR, '        d["pending"] = {"all": [r["n"] for r in rs], "at": time.time()}\n', ""),
    # ── commands ──
    ("R12 add-without-this", OR, _ADD_START, _ADD_START.replace(r"(?:learn|remember)\s+this\s+rule\b", r"(?:learn|remember|add|new)\s+(?:this\s+|a\s+)?rule\b")),
    ("R12b add-may-open-with-a-quote", OR, _ADD_START, _ADD_START.replace(r'r"^\s*"', "r" + '"' + r"^[\s\"“]*" + '"')),
    ("R12c add-other-separators", OR, _ADD_COLON, _ADD_COLON.replace(r'r"\s*:"', r'r"\s*[:,.;]"')),
    ("R12d add-over-several-lines", OR, _BODY_IF, _BODY_IF.replace("len(body.splitlines()) != 1", "not body")),
    ("R12e add-a-question", OR, _BODY_IF, "    if len(body.splitlines()) != 1:"),
    ("R12f only-an-ascii-question-mark", OR, _QRE, _QRE.replace("[?？" + "\\u037e]", "[?]")),
    ("R12g greek-question-is-a-rule", OR, _QRE, _QRE.replace("|;[", "|#[")),
    ("R12h only-a-final-question-mark", OR, _QRE, _QRE.replace("]|;[", "]$|;[")),
    ("R13 list-all-rules-without-my", OR, _WHICH, _WHICH.replace(r"(?:all\s+(?:of\s+)?)?my\s+", r"(?:all\s+(?:of\s+)?(?:my\s+)?|my\s+)")),
    ("R13a list-without-all-or-my", OR, _WHICH, _WHICH.replace(r"(?:all\s+(?:of\s+)?)?my\s+", r"(?:(?:all\s+(?:of\s+)?)?my\s+)?")),
    ("R13b forget-all-without-my", OR, r'_MY_RULES = r"(?:all\s+(?:of\s+)?)?my\s+rules"', r'_MY_RULES = r"(?:all\s+(?:of\s+)?)?(?:my\s+|the\s+)?rules"'),
    ("R14 end-not-anchored", OR, _END, '_END = r""'),
    ("R14a trailing-please-is-chat", OR, _END, _END_NO_PLEASE),
    ("R14b delete-is-forget", OR, r'_FORGET_RE = re.compile(_LEAD + r"forget\s+" + _N + _END, re.I)', r'_FORGET_RE = re.compile(_LEAD + r"(?:forget|delete|remove)\s+" + _N + _END, re.I)'),
    ("R14c rule-number-any-length", OR, r'(\d{1,6})"' + "\n# \"this rule\"", r'(\d+)"' + "\n# \"this rule\""),
    ("R16 confirm-all-shadowed", OR, "    if _CONFIRM_ALL_RE.match(t):\n        return _confirm_forget_all(home)\n", ""),
    ("R16b forget-all-shadowed", OR, "    if _FORGET_ALL_RE.match(t):\n        return _preview_forget_all(home)\n", ""),
    ("R17 show-rule-unhandled", OR, "    m = _SHOW_RE.match(t)\n    if m:\n        return _show(home, int(m.group(1)))\n", ""),
    # ── the note for a message that is no command ──
    ("R18 hint-for-commands", OR, '    if is_rule_command(t):\n        return ""\n    words = len(t.split())', "    words = len(t.split())"),
    ("R18b hint-any-length", OR, '    if words > HINT_MAX_WORDS:\n        return ""\n', ""),
    ("R18c hint-verb-only", OR, _HINT_RET, '    return RULE_HINT if _HINT_VERB_RE.search(t) else ""'),
    ("R18d hint-noun-only", OR, _HINT_RET, '    return RULE_HINT if _HINT_NOUN_RE.search(t) else ""'),
    ("R18e hint-word-limit", OR, "HINT_MAX_WORDS = 20", "HINT_MAX_WORDS = 21"),
    ("R18f long-dictation-not-hinted", OR, "    if _ADD_START_RE.match(t):\n        return RULE_HINT ", "    if False:\n        return RULE_HINT "),
    ("R18g yes-note-any-length", OR, _AFFIRM_IF, "    if _AFFIRM_RE.match(t):"),
    ("R18h yes-note-any-message", OR, _AFFIRM_IF, "    if words <= AFFIRM_MAX_WORDS:"),
    ("R18i yes-note-for-a-stale-preview", OR, '    if not pend or not _fresh(pend):\n        return ""\n    if isinstance', '    if not pend:\n        return ""\n    if isinstance'),
    ("R18n yes-note-after-the-rule-is-gone", OR, '    if type(pend.get("n")) is int and pend["n"] in have:', '    if type(pend.get("n")) is int:'),
    ("R18o yes-note-after-the-set-changed", OR, '        if pend["all"] != have:\n            return ""\n', ""),
    ("R18p yes-swallows-the-rule-note", OR, "        if waiting:\n            return waiting ", "        if True:\n            return waiting "),
    ("R18q rule-colon-not-hinted", OR, r'r"|\b(?:the|that|last)\s+rule\b|\brule\s*:"', r'r"|\b(?:the|that|last)\s+rule\b"'),
    ("R18j yes-note-misses-forget-all", OR, '    if isinstance(pend.get("all"), list):', "    if False:"),
    ("R18k yes-note-word-limit", OR, "AFFIRM_MAX_WORDS = 6", "AFFIRM_MAX_WORDS = 60"),
    ("R18l old-commands-not-hinted", OR, r'r"|\b(?:proposed|candidate)\s+rules?\b"' + "\n", "\n"),
    ("R18m number-words-not-hinted", OR, r'_NUMBER_WORD = r"(?:\d{1,6}|one|two|three|four|five|six|seven|eight|nine|ten)"', r'_NUMBER_WORD = r"(?:\d{1,6})"'),
    # ── the store never replaces what it cannot read ──
    ("R19 add-over-a-damaged-file", OR, _REFUSE + "        same = next(", "            pass\n        same = next("),
    ("R19b list-of-a-damaged-file", OR, "        return _refusal(why)\n    rs = sorted(", "        pass\n    rs = sorted("),
    ("R19c show-of-a-damaged-file", OR, "        return _refusal(why)\n    r = next(", "        pass\n    r = next("),
    ("R19d preview-on-a-damaged-file", OR, _REFUSE + '        r = next((r for r in d["rules"] if r["n"] == n), None)\n        if r is None:\n            return _note(f"There is no rule {n} — nothing to forget.',
     "            pass\n" + '        r = next((r for r in d["rules"] if r["n"] == n), None)\n        if r is None:\n            return _note(f"There is no rule {n} — nothing to forget.'),
    ("R19e confirm-on-a-damaged-file", OR, _REFUSE + '        pend = d["pending"]\n        r = next(', "            pass\n" + '        pend = d["pending"]\n        r = next('),
    ("R19f preview-all-on-a-damaged-file", OR, _REFUSE + '        if not d["rules"]:\n            return _note("You have no rules — nothing to forget.")',
     "            pass\n" + '        if not d["rules"]:\n            return _note("You have no rules — nothing to forget.")'),
    ("R19g confirm-all-on-a-damaged-file", OR, _REFUSE + '        pend = d["pending"]\n        if not d["rules"]:', "            pass\n" + '        pend = d["pending"]\n        if not d["rules"]:'),
    ("R20 rows-not-a-list-accepted", OR, "    if not isinstance(rows, list):", "    if rows is None:"),
    ("R20b bool-number-accepted", OR, _ROW_N, _ROW_N.replace('type(r.get("n")) is int', 'isinstance(r.get("n"), int)')),
    ("R20c zero-number-accepted", OR, _ROW_N, _ROW_N.replace('1 <= r["n"]', 'r["n"]')),
    ("R20g unnameable-number-accepted", OR, _ROW_N, _ROW_N.replace(' <= 999_999', '')),
    ("R20d non-text-rule-accepted", OR, _ROW_RULE, 'and r.get("rule")):'),
    ("R20h blank-rule-accepted", OR, _ROW_RULE, 'and isinstance(r.get("rule"), str)):'),
    ("R20e duplicate-numbers-accepted", OR, '        if r["n"] in seen:', "        if False:"),
    ("R20f more-than-the-cap-accepted", OR, "    if len(rows) > RULES_MAX:", "    if False:"),
    ("R21 unreadable-read-as-absent", OR, '        return _empty(), f"cannot be read ({type(e).__name__})"', '        return _empty(), ""'),
    ("R21b torn-read-as-absent", OR, '        return _empty(), "is not readable JSON"', '        return _empty(), ""'),
    ("R21c deep-read-as-absent", OR, '        return _empty(), "is nested too deeply to read"', '        return _empty(), ""'),
    ("R21d invalid-read-as-fine", OR, "    if why:\n        return _empty(), why\n", ""),
    ("R21e directory-read-as-absent", OR, "    except FileNotFoundError:", "    except (FileNotFoundError, IsADirectoryError):"),
    ("R21g key-written-twice-accepted", OR, "    if len(d) != len(pairs):", "    if False:"),
    ("R21f byte-order-mark-is-damage", OR, 'encoding="utf-8-sig"', 'encoding="utf-8"'),
    ("R22 bool-preview-time-accepted", OR, 'type(pend.get("at")) in (int, float)', 'isinstance(pend.get("at"), (int, float))'),
    ("R22b preview-without-time-accepted", OR, '    if not (isinstance(pend, dict) and type(pend.get("at")) in (int, float)):', "    if not isinstance(pend, dict):"),
    ("R23 no-home-answers", OR, '    if _path(home) is None:\n        return ""\n    try:', "    try:"),
    ("R24 lock-dropped-on-add", OR, "    with _LOCK:\n        d, why = _read(home)\n        if why:\n" + _REFUSE + "        same = next(",
     "    if True:\n        d, why = _read(home)\n        if why:\n" + _REFUSE + "        same = next("),
    ("R29 failed-save-not-said", OR, _TRY, _TRY.replace("except OSError as e:", "except ZeroDivisionError as e:")),
    # ── what is shown ──
    ("R25 block-when-withheld", OR, "    if not rs or rules_withheld():", "    if not rs:"),
    ("R25b withheld-always", OR, '        return prompt_arm() == "no_lessons"', "        return True"),
    ("R26 block-not-cleaned", OR, "f\"- {_clean(r['rule'])[:RULE_MAX_CHARS]}\" for r in rs", "f\"- {r['rule'][:RULE_MAX_CHARS]}\" for r in rs"),
    ("R27 block-newest-first", OR, '    return sorted(load(home)["rules"], key=lambda r: r["n"])', '    return sorted(load(home)["rules"], key=lambda r: -r["n"])'),
    ("R28 mentions-every-rule", OR, '    return [r for r in rules(home) if mentions(r["rule"], target)]', "    return rules(home)"),
    ("R30 list-shows-a-raw-rule", OR, "{_clean(r['rule'])}\"       # (a hand-edited", "{r['rule']}\"       # (a hand-edited"),
    ("R30b preview-shows-a-raw-rule", OR, "Forget this rule?\\n\\n> {_clean(r['rule'])}", "Forget this rule?\\n\\n> {r['rule']}"),
    ("R30c forgotten-shows-a-raw-rule", OR, "I am no longer shown it:\\n\\n> {_clean(r['rule'])}", "I am no longer shown it:\\n\\n> {r['rule']}"),
    # ── agent wiring ──
    ("A1 block-shown-when-hidden", AG, "        if _owner_context_hidden(context):\n            return \"\"\n        md = getattr(context, \"memory_dir\", None)", "        md = getattr(context, \"memory_dir\", None)"),
    ("A2 block-from-memory-dir", AG, "        return rules_block(Path(md).parent)\n", "        return rules_block(Path(md))\n"),
    ("A3 anyone-but-a-member-runs-commands", AG, _GATE, _GATE.replace('_rk() != "owner"', '_rk() == "member"')),
    ("A3b public-runs-commands", AG, _GATE, _GATE.replace(' or reply_is_public()', '')),
    ("A4 hint-dropped", AG, "                hint = await asyncio.to_thread(rule_command_hint, text, home)\n", "                hint = \"\"\n"),
    ("A4b hint-without-the-store", AG, "asyncio.to_thread(rule_command_hint, text, home)", "asyncio.to_thread(rule_command_hint, text, None)"),
    ("A5 command-home-is-memory-dir", AG, _HOME, _HOME.replace("Path(md).parent", "Path(md)")),
    ("A5b empty-memory-dir-is-the-working-dir", AG, _HOME, _HOME.replace(" and str(md)", "")),
    ("A6 first-user-message-is-the-command", AG, _NEXT, _NEXT.replace("reversed(msgs or [])", "(msgs or [])")),
    ("A7 any-last-message-is-the-command", AG, _NEXT, _NEXT.replace(' and m.get("role") == "user"', "")),
    ("A8 note-not-in-the-state-block", AG, '                        _ds_parts.append(("owner_rule_command", f"### OWNER RULE COMMAND\\n{_owner_rule_note_ctx.get()}\\n\\n"))', "                        pass"),
    ("A9 command-matched-on-collapsed-text", AG, _LAST, _LAST.replace("_message_lines(", "_message_text(")),
    ("A10 text-parts-joined-on-one-line", AG, _PARTS, _PARTS.replace('"\\n".join', '" ".join')),
    # ── the idle window ──
    ("W1 window-never-restarts", AG, _RESTART, _RESTART.replace("if idle_secs > self._bio_scaled(3600):", "if False:")),
    ("W2 restart-writes-user-clock", AG, _RESTART, _RESTART.replace("            # §4NV", "            ctx.last_activity_time = datetime.datetime.now()\n            # §4NV")),
    ("W2b restart-writes-owner-clock", AG, _RESTART, _RESTART.replace("            # §4NV", "            ctx.last_owner_activity_time = datetime.datetime.now()\n            # §4NV")),
    ("W3 restart-every-tick", AG, _RESTART, _RESTART.replace("self._bio_scaled(3600)", "self._bio_scaled(1)")),
    ("W4 restart-line-says-window-age", AG, 'f"no request for {(datetime.datetime.now() - ctx.last_activity_time).total_seconds() / 60.0:.0f}m — "', 'f"no request for {idle_secs / 60.0:.0f}m — "'),
    ("W5 removed-slot-rows-unregistered", AA, 'RETIRED_PHASES = frozenset({"calibration", "self_play", "open_questions"})', 'RETIRED_PHASES = frozenset({"calibration", "open_questions"})'),   # anchor follows §4TB
    # ── neighbours ──
    ("E1 rule-command-not-a-system-command", EX, "    return is_rule_command(request) or is_mirror_command(request)", "    return is_mirror_command(request)"),
    ("E1b mirror-not-a-system-command", EX, "    return is_rule_command(request) or is_mirror_command(request)", "    return is_rule_command(request)"),
    ("Q1 system-command-not-queued", QM, '    if is_system_command(t):\n        return PATH_QUEUE, "a system command"', '    if False:\n        return PATH_QUEUE, "a system command"'),
    ("C1 census-context-without-the-rules", "src/ghost_agent/core/failure_census.py", 'standing += " " + " ".join(str(x.get("rule") or "") for x in _owner_rules(sysd))', 'standing += ""'),
    ("H1 health-misses-a-damaged-rules-file", HI, '_or.damaged(system_dir) or None', 'None'),
    ("T1 privacy-forget-looks-in-the-memory-dir", TM, "rules_mentioning, Path(_pf).parent.parent, target)", "rules_mentioning, Path(_pf).parent, target)"),
    # ── the thumbs-down hint ──
    ("F1 hint-on-a-thumbs-up", RT, '        if signal == "negative" and not result.get("unchanged"):', '        if not result.get("unchanged"):'),
    ("F2 hint-on-a-repeat", RT, '        if signal == "negative" and not result.get("unchanged"):', '        if signal == "negative":'),
    ("F3 no-hint", RT, '            result["hint"] = FEEDBACK_HINT\n', ""),
    ("F4 client-shows-it-every-time", JS, "done.hint && !_ruleHintShown) {", "done.hint) {"),
    ("F5 client-never-shows-it", JS, "            addMessage('system', done.hint);\n", ""),
    ("F6 client-shows-any-value", JS, "if (done && typeof done.hint === 'string' && done.hint && !_ruleHintShown) {", "if (done && done.hint !== undefined && !_ruleHintShown) {"),
    ("F7 client-throws-on-a-bad-body", JS, "const done = await r.json().catch(() => ({}));", "const done = await r.json();"),
    # ── probe guards whose tests were restored ──
    ("G1 probe-records-sources", SE, "    if is_probe_request_id(request_id_context.get()):\n        return 0 ", "    if False:\n        return 0 "),
    ("G2 probe-writes-project-research", SE, "        if is_probe_request_id(request_id_context.get()):\n            return None ", "        if False:\n            return None "),
    ("G3 probe-judged-as-a-correction", AG, "                    if requester_is_member() or _ipr(req_id):", "                    if requester_is_member():"),
    ("G4 probe-uses-up-corrections", AG, "            if _ipr_c(request_id_context.get()):", "            if False:"),
    ("G5 probe-queues-a-caveat", AG, " or _is_probe_id(request_id_context.get()):", ":"),
    ("G6 only-probes-book-a-browser-visit", "src/ghost_agent/tools/browser.py", "            and not _ipr(_ridc.get()) and not _blocked", "            and _ipr(_ridc.get()) and not _blocked"),
]
PINS = ["tests/test_4sz_owner_rules.py", "tests/test_4sz_system_commands.py", "tests/test_4sz_idle_window.py",
        "tests/test_4sz_kept_guards.py", "tests/test_4sz_thumbs_down_hint.py"]


def _tests():
    pat = (r"owner_rules\|_owner_rule_note\|_owner_rules_block\|is_system_command\|queued_messages\|"
           r"_biological_tick\|_idle_window_at\|rule_command\|api/feedback\|activity_liveness\|"
           r"health_internals\|record_loaded_sources\|_record_project_findings\|_queue_source_caveat\|"
           r"_consume_pending_corrections")
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
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [(lab, originals[f].count(old)) for lab, f, old, new in MUT if originals[f].count(old) != 1]
    same = [lab for lab, f, old, new in MUT if old == new]
    dup = sorted({lab.split(" ")[0] for lab, *_ in MUT if [m[0].split(" ")[0] for m in MUT].count(lab.split(" ")[0]) > 1
                  and not lab.startswith("CONTROL")})
    if args[0] == "check":
        ok = not (bad or same or dup)
        print("anchors:", "OK" if ok else (bad, same, dup)); sys.exit(0 if ok else 1)
    if bad or same or dup:
        print("ANCHOR NOT UNIQUE/FOUND, NO CHANGE or DUPLICATE ID:", bad, same, dup); sys.exit(2)
    batch = [m for m in MUT if m[0].split(" ")[0] in set(args[1:]) or m[0].startswith("CONTROL")] if args[0] == "rerun" else MUT
    full = _tests()
    cov = BC.build_map([f for f in srcs if f.endswith(".py")], full, log=lambda m: print(m, flush=True))
    # a .js file has no coverage map: its mutants run the pins
    sel = {lab: (BC.tests_for(cov, f, old, originals[f]) if f.endswith(".py") else None) for lab, f, old, new in batch}
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
