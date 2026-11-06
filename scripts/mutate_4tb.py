#!/usr/bin/env python3
"""§4TB battery — four tools reviewed: `replan` removed (the tables that named it, the watchdog's note, the
unparsed-call scrub, the name healer), `self_state` folded into `introspect` as `set_mood` (the write, the
clear path, who may write, the turn loop treating it as a write, old files, the retired parts),
`fact_check` returning its sources under one wall clock (the section, its bounds, its neutralisation, the
titles, the memory switch), and `notify_operator` reporting delivery truthfully (who is polling now, the
digest, the wording); then the independent review's items (the comment above each group names the item),
and the second review's ("review r2": the memory switch and the broad link layer are deleted — their mutants with them). Coverage-selected tests in parallel tree copies (scripts/battery_cov.py); the real
src/ is never mutated. No-op control must SURVIVE, the known-bad control must DIE.
(§4TC, 2026-11-06: the mood action was rebuilt — `scripts/mutate_4tc.py` is its battery. The set_mood mutants
here follow the new lines; M4, M5, M16 and M17 went with their subjects — the derived mood, the self / derived
provenance, an old stored mood being kept — and `tests/test_selfhood_derived_mood.py` left the pins with them.)
    python3 scripts/mutate_4tb.py [check | -j6 all | rerun R1 M2 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = "src/ghost_agent/core/agent.py"
RS = "src/ghost_agent/core/reply_smoothing.py"
RG = "src/ghost_agent/tools/registry.py"
IT = "src/ghost_agent/tools/introspect.py"
IS = "src/ghost_agent/core/isolation.py"
MD = "src/ghost_agent/selfhood/mood.py"
SC = "src/ghost_agent/selfhood/schema.py"
AA = "src/ghost_agent/core/autonomous_activity.py"
MN = "src/ghost_agent/main.py"
SE = "src/ghost_agent/tools/search.py"
NT = "src/ghost_agent/tools/notify_tool.py"
RT = "src/ghost_agent/api/routes.py"
LG = "src/ghost_agent/core/link_grounding.py"
SM = "src/ghost_agent/selfhood/model.py"

# anchors used by more than one mutant (each is ONE exact place in its file)
_SET = '            _before, after = state.set_owner_mood(label, direction, level, note=why)'     # §4TC: the write
_WRITE_FIRST = ('    _blocked = _member_block()\n'
                '    if _blocked is not None:\n'
                '        return _blocked\n'
                '    # §4TB: the one write — after the member block, before every read path\n'
                '    # (it has its own gates, and must not need a rendered view to refuse)\n'
                '    if raw_action in WRITE_ACTIONS:\n'
                '        return _set_mood(self_model, context, mood, evidence, intensity=intensity, manner=manner)\n')
_ASK_TITLES = '        if getattr(deep_research_callable, "accepts_source_titles", False) is True:'
_DIGEST = "        return digest_shows(rec, severities=(SEVERITY_NOTIFY,))"
_REQUIRED = ('    if not word and not man_word:\n'
             '        return MOOD_REQUIRED\n'
             '    clearing = word.lower() in _MOOD_CLEAR_WORDS\n')
_GUARDS = ('\n'
           '    refusal, probe = _mood_guard(context)\n'
           '    if refusal is not None:\n'
           '        return refusal\n')
_NO_CONTAIN = ('        return _mood_refusal("SYSTEM BLOCK: set_mood could not check whether this is a contained "\n'
               '                             "run, so nothing was set.", "guard_unavailable"), False\n')
_NO_PROV = ('        return _mood_refusal("SYSTEM BLOCK: set_mood could not check what this request has "\n'
            '                             "read, so nothing was set.", "guard_unavailable"), False\n')
_NO_KIND = ('        return _mood_refusal("SYSTEM BLOCK: set_mood could not tell what kind of request this "\n'
            '                             "is, so nothing was set.", "guard_unavailable"), False\n')
_FLAT = '    t = " ".join(str(text or "").split())\n'
_WRITTEN = "    t = _FC_WRITTEN_URL_RE.sub(_FC_LINK, t)\n"
_MARKED = '    lines = [l.rstrip() for l in _fc_mark_heads(unify_line_breaks(verdict)).split("\\n")]'
_HIDDEN_HEAD = ('    if len(_FC_RESULT_HEAD_RE.findall(plain)) > len(_FC_RESULT_HEAD_RE.findall(text)):\n'
                '        text = plain\n')
_STORED = "            if isinstance(n, str) and v is not None and int(v) >= 0:"
_RECORDED = ("        if not isinstance(consumer, str) or not (acknowledged or valid_consumer_name(consumer)):\n"
             "            return\n")
_MORE = '                 + (f"; +{more} more" if more > 0 else "") + ".")'
_HYDRATE = "                should_fetch_memory = (not self._is_strict_trivial_chat(lc)"
_RESEARCH_WAIT = "        _dr_raw = await _fc_wait_for(_research_task, max(0.05, min(_research_cap, _total)))"
_VERIFY_WAIT = "            max(0.05, min(_FACTCHECK_VERIFY_S + _fc_hook_s, _left)))"
_LIVE = "            if n in registered and live:"
_EVICT = ('        while len(polls) > _CONSUMER_POLLS_MAX:\n'
          '            victim = next((n for n, (_t, ack) in polls.items() if not ack), None)\n'
          '            polls.pop(victim if victim is not None else next(iter(polls)))\n')
_UNREG = "            elif live and readable and named:"

MUT = [
    ("CONTROL no-op", IT, 'WRITE_ACTIONS = frozenset({"set_mood"})', 'WRITE_ACTIONS = frozenset({"set_mood"})  # no-op'),
    ("CONTROL known-bad", IT, _SET, "            _before = after = state.mood()"),
    # ── change 1: `replan` is removed ──
    ("R1 members-get-replan-again", AG, '    "abort_attempt",\n})', '    "abort_attempt", "replan",\n})'),
    ("R2 replan-is-a-bookkeeping-tool-again", AG, '    "abort_attempt",\n    "knowledge_base",  # insert_fact', '    "replan", "abort_attempt",\n    "knowledge_base",  # insert_fact'),
    ("R3 watchdog-writes-a-replan-call", AG, "                            break_text = WATCHDOG_BREAK_NOTE\n",
     '                            break_text = "\\n</think>\\n<tool_call>\\n<function=replan>\\n<parameter=reason>\\nSYSTEM OVERRIDE\\n</parameter>\\n</function>\\n</tool_call>"\n'),
    ("R4 watchdog-note-leaves-the-think-block-open", AG, 'WATCHDOG_BREAK_NOTE = ("\\n</think>\\n[SYSTEM OVERRIDE:', 'WATCHDOG_BREAK_NOTE = ("\\n[SYSTEM OVERRIDE:'),
    ("R5 replan-call-exempt-from-the-scrub", RS, "        if any(a <= m.start() < b for a, b in fences):\n            continue\n        out.append(m)",
     '        if any(a <= m.start() < b for a, b in fences):\n            continue\n        if "replan" in m.group(0):\n            continue\n        out.append(m)'),
    ("R6 name-healer-takes-a-half-match", AG, "n=1, cutoff=0.7)", "n=1, cutoff=0.5)"),
    ("R7 replan-dispatchable-again", RG, '        "abort_attempt": _abort_attempt,\n', '        "abort_attempt": _abort_attempt,\n        "replan": _abort_attempt,\n'),
    ("R8 replan-advertised-again", RG, '    {"type": "function", "function": {"name": "abort_attempt",',
     '    {"type": "function", "function": {"name": "replan", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}},\n'
     '    {"type": "function", "function": {"name": "abort_attempt",'),
    # (review 1) a description that points at a removed tool
    ("R9 abort-attempt-points-at-replan-again", RG, "or to escape a hard problem. After this tool runs,",
     "or to escape a hard problem (use `replan` for that). After this tool runs,"),
    # ── change 2: `set_mood`, the one write of `introspect` ──
    ("M1 no-write-action", IT, 'WRITE_ACTIONS = frozenset({"set_mood"})', "WRITE_ACTIONS = frozenset()"),
    ("M2 predicate-reads-the-raw-action", IT, '    return str(a.get("action") or "").strip().lower() in WRITE_ACTIONS',
     '    return str(a.get("action") or "") in WRITE_ACTIONS'),
    # (review 4) a missing mood must not clear; only the two words do
    ("M3 a-missing-mood-clears", IT, _REQUIRED, '    clearing = (not word) or word.lower() in _MOOD_CLEAR_WORDS\n'),
    # (§4TC: M4 "a clear removes a derived mood" and M5 "the write is booked as derived" went with the derived
    #  mood and the self / derived provenance; who a change is booked to is pinned in mutate_4tc.py)
    ("M6 evidence-dropped", IT, _SET, _SET.replace("note=why", 'note=""')),
    ("M7 contained-run-may-write", IT, "    if contained:\n", "    if False:\n"),
    ("M8 written-after-outside-content", IT, '    if ref is not None:\n        try:\n            pretty_log("Introspect"',
     '    if False:\n        try:\n            pretty_log("Introspect"'),
    ("M9 a-probe-writes", IT, "    if probe:\n        from .outcome import ToolOutcome", "    if False:\n        from .outcome import ToolOutcome"),
    ("M10 an-owner-turn-is-a-dry-run", IT, '        probe = request_kind() == "probe"', '        probe = request_kind() in ("probe", "owner")'),
    ("M11 write-runs-before-the-member-block", IT, _WRITE_FIRST,
     '    if raw_action in WRITE_ACTIONS:\n'
     '        return _set_mood(self_model, context, mood, evidence, intensity=intensity, manner=manner)\n'
     '    _blocked = _member_block()\n'
     '    if _blocked is not None:\n'
     '        return _blocked\n'),
    # (review 10) what a label may carry into the store
    ("M12 label-keeps-escape-and-control-characters", IT, '    t = _MOOD_CTRL_RE.sub(" ", _MOOD_ESC_RE.sub(" ", value))', "    t = value"),
    ("M13 set-mood-collapses-like-a-read", AG, '                              (_cname == "introspect" and _introspect_is_write_call(t_args)) or \\\n', ""),
    ("M14 unclassifiable-call-is-a-read", AG, "    except Exception:  # noqa: BLE001\n        return True\n\n\ndef _run_is_read_call",
     "    except Exception:  # noqa: BLE001\n        return False\n\n\ndef _run_is_read_call"),
    ("M15 any-context-attribute-is-containment", IS, '    return isinstance(getattr(context, "_replay_forbidden_tools", None), (frozenset, set))',
     '    return getattr(context, "_replay_forbidden_tools", None) is not None'),
    # (§4TC: M16 — the derived vocabulary — and M17 — an old state file keeps its mood — went with their
    #  subjects: there is no derived mood, and a mood stored before §4TC is ignored on purpose)
    ("M18 old-question-rows-are-an-unregistered-phase", AA, 'RETIRED_PHASES = frozenset({"calibration", "self_play", "open_questions"})',
     'RETIRED_PHASES = frozenset({"calibration", "self_play"})'),
    ("M19 principle-gate-flag-rejected", MN,
     '    parser.add_argument("--principle-gate", action="store_true", default=False, help=argparse.SUPPRESS)   # §4TB: deprecated no-op, see _DEPRECATED_NOOP_ARGS\n', ""),
    ("M20 definition-does-not-offer-the-write", RG, '"activity", "learning", "mood", "set_mood"]', '"activity", "learning", "mood"]'),
    ("M21 the-word-none-does-not-clear", MD, 'CLEAR_WORDS = frozenset({"clear", "none", "neutral"})', 'CLEAR_WORDS = frozenset({"clear", "neutral"})'),
    ("M22 a-mood-that-is-not-text-is-stored-as-its-repr", IT,
     '    if not isinstance(value, str):\n        return ""\n    from ..core.owner_rules import _clean',
     '    value = "" if value is None else str(value)\n    from ..core.owner_rules import _clean'),
    ("M23 containment-check-fails-open", IT, _NO_CONTAIN, "        contained = False\n"),
    ("M24 outside-content-check-fails-open", IT, _NO_PROV, "        ref = None\n"),
    ("M25 request-kind-check-fails-open", IT, _NO_KIND, "        probe = False\n"),
    ("M26 a-probe-of-a-disabled-model-is-a-dry-run", IT,
     '    if self_model is None or not getattr(self_model, "enabled", False):\n        return _mood_refusal(\n            "The mood is unavailable — the selfhood module is disabled "',
     '    if (not probe) and (self_model is None or not getattr(self_model, "enabled", False)):\n        return _mood_refusal(\n            "The mood is unavailable — the selfhood module is disabled "'),
    ("M27 outside-content-refusal-is-not-the-shared-one", IT, '        ref = refuse_if_untrusted("the mood change")',
     '        from ..utils.provenance import untrusted_seen\n'
     '        ref = _mood_refusal("SYSTEM BLOCK: set_mood is refused after outside content.", "untrusted_content") if untrusted_seen() else None'),
    # (review 3) the same mood set again and again is a no-progress repeat
    ("M28 a-repeated-set-mood-is-never-stopped", AG, '                    if (fname and (not is_mutating or fname == "introspect")\n',
     "                    if (fname and not is_mutating\n"),
    ("M29 label-keeps-chat-template-tokens", IT, "    return _cap_text(_clean(t), cap)", '    return _cap_text(" ".join(t.split()), cap)'),
    ("M30 guards-asked-before-the-argument", IT, _REQUIRED, _GUARDS.lstrip("\n") + _REQUIRED),
    # (review 12) a facade with no caller
    ("M31 the-uncalled-facade-is-back", SM, "    def build_wakeup_prefix(\n",
     "    def note_referenced_experiences(self, *a, **k):\n        return 0\n\n    def build_wakeup_prefix(\n"),
    # ── review r2 ──
    ("M32 format-characters-stay-in-a-label", IT, '    t = "".join(c for c in t if unicodedata.category(c) != "Cf")\n', ""),
    ("M33 the-contained-run-refusal-is-booked-as-a-failure", IT, "    if contained:\n        return _mood_refusal(\n",
     '    if contained:\n        return _TO_failed("SYSTEM BLOCK: set_mood is not available in a contained run."), False\n        return _mood_refusal(\n'),
    # ── change 3: `fact_check` returns its sources, under one clock ──
    ("F1 sources-not-returned", SE, '    _tail = f"\\n\\n{_sources}" if _sources else ""', '    _tail = ""'),
    ("F2 a-source-without-facts-is-listed", SE, "        if not m or not distilled_block_has_facts(block):", "        if not m:"),
    ("F3 no-limit-on-sources", SE, "FACTCHECK_MAX_SOURCES = 8", "FACTCHECK_MAX_SOURCES = 80"),
    ("F4 evidence-not-cut", SE, "FACTCHECK_EVIDENCE_CHARS = 300", "FACTCHECK_EVIDENCE_CHARS = 3000"),
    ("F5 evidence-keeps-its-line-breaks", SE, _FLAT, '    t = str(text or "").strip()\n'),
    ("F6 result-heads-in-page-text-left-as-they-are", SE, "    t = _fc_mark_heads(t)\n    if len(t) > limit:", "    if len(t) > limit:"),
    ("F7 evidence-skips-the-reports-neutraliser", SE, "    t = neutralise_report_grammar(t)\n    t = _fc_mark_heads(t)", "    t = _fc_mark_heads(t)"),
    # (review 6) every line of the verdict after the first is quoted
    ("F8 later-verdict-lines-are-not-quoted", SE,
     "                     + [(_FC_QUOTE + l) if l.strip() else _FC_QUOTE.rstrip() for l in lines[1:]])", "                     + lines[1:])"),
    ("F9 titles-never-asked-for", SE, _ASK_TITLES, "        if False:"),
    ("F10 titles-asked-of-any-callable", SE, _ASK_TITLES, "        if True:"),
    ("F11 research-has-no-clock", SE, _RESEARCH_WAIT, "        _dr_raw = await _research_task"),
    ("F12 verify-ignores-the-whole-call-clock", SE, _VERIFY_WAIT, "            _FACTCHECK_VERIFY_S + _fc_hook_s)"),
    ("F13 verify-started-with-no-time-left", SE, "        if _left < 1.0:\n", "        if False:\n"),
    ("F16 research-hands-out-no-titles", SE, "                source_titles.setdefault(_u, _t)\n", "                pass\n"),
    ("F17 page-text-may-write-a-partial-head", SE, r'r"FACT[^\S\n]+CHECK[^\S\n]+(?:COMPLETE|FAILED|PARTIAL)|SOURCES',
     r'r"FACT[^\S\n]+CHECK[^\S\n]+(?:COMPLETE|FAILED)|SOURCES'),
    ("F18 verdict-line-separators-not-unified", SE, _MARKED, _MARKED.replace("unify_line_breaks(verdict)", 'str(verdict or "")')),
    ("F19 heads-in-the-verdict-not-marked", SE, _MARKED, _MARKED.replace("_fc_mark_heads(unify_line_breaks(verdict))", "unify_line_breaks(verdict)")),
    ("F24 page-text-may-write-the-sources-head", SE, r'|SOURCES[^\S\n]+the[^\S\n]+verdict|"', '|"'),
    ("F25 the-neutraliser-keeps-other-line-separators", SE, "    t = unify_line_breaks(t)\n    t = _EXTRACT_MARK_RE", "    t = _EXTRACT_MARK_RE"),
    ("F26 grammar-hidden-behind-other-leading-characters", SE, r'_GRAMMAR_LEAD = "(?:[^\\S\\n]|[\u200b-\u200d\u2060\ufeff]|[*_`~])*"',
     r'_GRAMMAR_LEAD = "[ \\t]*"'),
    ("F27 head-marking-is-case-sensitive", SE, r'|SOURCES\s*the\s*verdict", re.IGNORECASE)', r'|SOURCES\s*the\s*verdict")'),
    ("F28 grammar-quoting-is-case-sensitive", SE, 'r"(?im)^(?=" + _GRAMMAR_LEAD', 'r"(?m)^(?=" + _GRAMMAR_LEAD'),
    ("F29 failed-and-partial-heads-not-marked", SE, r'_FC_RESULT_HEAD_RE = re.compile(r"FACT\s*CHECK\s*(?:COMPLETE|FAILED|PARTIAL)',
     r'_FC_RESULT_HEAD_RE = re.compile(r"FACT\s*CHECK\s*(?:COMPLETE)'),
    # (review 2) the links of a result: the rows' own, and only those
    ("F30 an-address-written-out-in-a-title-or-evidence-is-kept", SE, _WRITTEN, ""),
    ("F33 a-long-url-is-cut", SE, "    return u if len(u) <= FACTCHECK_URL_CHARS else FACTCHECK_URL_TOO_LONG", "    return u[:FACTCHECK_URL_CHARS]"),
    ("F34 a-source-that-is-not-a-web-address-is-listed", SE, "        return FACTCHECK_URL_NOT_WEB\n", "        return u\n"),
    ("F35 listed-sources-do-not-count-as-fetched", LG, "            out.extend(u for u in listed if u not in out)\n", ""),
    ("F36 rows-read-from-a-result-that-did-not-complete", SE, '    if not lines or not lines[0].startswith("FACT CHECK COMPLETE"):', "    if not lines:"),
    ("F37 any-line-with-a-url-is-a-row", SE, "        m = _FC_ROW_RE.fullmatch(line)", '        m = re.search(r"(https?://\\S+)", line)'),
    ("F38 rows-read-from-the-verdict-too", SE, "    for line in lines[start + 1:] if start is not None else ():", "    for line in lines:"),
    # (review 11) the clock's settings and the research's own waits
    ("F39 any-number-is-a-number-of-seconds", SE,
     '    return v if (v == v and v not in (float("inf"), float("-inf")) and v >= 0) else float(default)', "    return v"),
    ("F40 a-word-in-the-setting-raises", SE, "    except (TypeError, ValueError):\n        return float(default)\n    return v if",
     "    except TypeError:\n        return float(default)\n    return v if"),
    ("F41 per-source-wait-drops-a-cancel", SE, "                return await _cancel_safe_wait_for(process_url(url, budget),",
     "                return await asyncio.wait_for(process_url(url, budget),"),
    ("F42 fetch-wait-drops-a-cancel", SE, "                res = await _cancel_safe_wait_for(      # §4TB", "                res = await asyncio.wait_for(      # §4TB"),
    ("F43 verify-wait-forgets-the-side-hook", SE, _VERIFY_WAIT, "            max(0.05, min(_FACTCHECK_VERIFY_S, _left)))"),
    ("F44 research-cut-ignores-a-smaller-total", SE, _RESEARCH_WAIT, "        _dr_raw = await _fc_wait_for(_research_task, max(0.05, _research_cap))"),
    ("F45 default-total-forgets-the-side-hook", SE, "        _research_cap + _FACTCHECK_VERIFY_S + _fc_hook_s)", "        _research_cap + _FACTCHECK_VERIFY_S)"),
    # ── review r2 ──
    ("F46 sources-built-from-the-copy-cut-for-the-verifier", SE, "    _sources = factcheck_sources(_whole_report, _titles)",
     "    _sources = factcheck_sources(dr_result, _titles)"),
    ("F47 a-fact-check-is-no-fetch-for-the-source-check", AG,
     '    _FETCHERS = ("browser", "web_search", "deep_research", "darkweb_search", "darkweb_research",\n                 "fact_check")',
     '    _FETCHERS = ("browser", "web_search", "deep_research", "darkweb_search", "darkweb_research")'),
    ("F48 a-fact-check-message-runs-without-memory-again", AG, _HYDRATE,
     '                should_fetch_memory = ("fact-check" not in lc and "verify" not in lc and not self._is_strict_trivial_chat(lc)'),
    ("F49 an-executed-fact-check-row-vouches-with-its-whole-text", LG,
     '    seen = _fact_check_seen(content) if row.get("name") == "fact_check" else str(content or "")', '    seen = str(content or "")'),
    ("F50 a-fact-check-message-row-vouches-with-its-whole-text", LG,
     '        if m.get("role") == "tool" and m.get("name") == "fact_check":\n            c = _fact_check_seen(c)\n', ""),
    ("F51 the-loop-hands-the-grounding-raw-rows", AG, "                    _lg.tool_row_text(t) for t in tools_run_this_turn if isinstance(t, dict)))",
     "                    str(t.get('content') or '') for t in tools_run_this_turn if isinstance(t, dict)))"),
    ("F52 a-check-that-did-not-complete-vouches-for-nothing", LG, '    if not text.startswith("FACT CHECK COMPLETE"):\n        return text',
     '    if not text.startswith("FACT CHECK COMPLETE"):\n        return ""'),
    ("F53 a-completed-check-is-read-whole", LG, '        return "\\n".join(factcheck_listed_urls(text))', "        return text"),
    ("F54 a-verdict-of-zero-width-characters-completes", SE, "    shows = [bool(_fc_visible(l).strip()) for l in lines]", "    shows = [bool(l.strip()) for l in lines]"),
    ("F55 a-head-behind-format-characters-is-not-marked", SE, _HIDDEN_HEAD, ""),
    ("F56 ordinary-text-is-rewritten-to-its-compatibility-form", SE, _HIDDEN_HEAD, "    text = plain\n"),
    ("F57 an-upper-case-scheme-is-no-web-address", SE, '    if not re.fullmatch(r"(?i)https?://\\S+", u):', '    if not re.fullmatch(r"https?://\\S+", u):'),
    ("F58 a-row-with-an-upper-case-scheme-is-not-read", SE, '_FC_ROW_RE = re.compile(r"\\d+\\. (https?://\\S+)", re.IGNORECASE)',
     '_FC_ROW_RE = re.compile(r"\\d+\\. (https?://\\S+)")'),
    ("F59 an-abandoned-research-keeps-its-exception", SE,
     "        _research_task.add_done_callback(lambda t: t.cancelled() or t.exception())\n", ""),
    ("F60 a-www-address-in-a-title-is-kept", SE, r'_FC_WRITTEN_URL_RE = re.compile(r"(?:https?://|\bwww\.)\S+", re.IGNORECASE)',
     r'_FC_WRITTEN_URL_RE = re.compile(r"(?:https?://)\S+", re.IGNORECASE)'),
    ("F61 a-verify-budget-of-zero-stands", SE, '_FACTCHECK_VERIFY_S = _env_seconds("GHOST_FACTCHECK_VERIFY_S", 150.0) or 150.0',
     '_FACTCHECK_VERIFY_S = _env_seconds("GHOST_FACTCHECK_VERIFY_S", 150.0)'),
    # ── change 4: `notify_operator` reports delivery truthfully ──
    ("N1 polling-window-is-days", AA, "CONSUMER_POLLING_WINDOW_S = 300.0", "CONSUMER_POLLING_WINDOW_S = 3000000.0"),
    ("N2 any-client-that-ever-acked-is-a-channel", NT, _LIVE, "            if n in registered:"),
    ("N3 a-client-that-never-acked-is-a-channel", NT, _LIVE, "            if live:"),
    ("N4 digest-always-named", NT, _DIGEST, "        return True"),
    ("N5 record-asked-about-without-its-request", NT, '            meta=({"req_id": str(req_id)} if req_id and req_id != "SYSTEM" else {}))', "            meta={})"),
    ("N6 a-poll-is-not-noted", RT, "    note_consumer_poll(agent.context, consumer, acknowledged=offset is not None)\n", ""),
    ("N7 poll-table-unbounded", AA, _EVICT, ""),
    ("N8 digest-shows-what-the-owner-already-read", AA, "                and not _seen_in_conversation(r))", "                and True)"),
    ("N9 result-says-delivered", NT, '        head = (f"Notification queued for the operator via: "', '        head = (f"Notification delivered to the operator via: "'),
    ("N10 quiet-clients-not-named", NT, "    if quiet:\n        more = ", "    if False:\n        more = "),
    ("N11 digest-listed-beside-a-live-client-always", NT, "(['next-turn digest'] if in_digest else [])", "['next-turn digest']"),
    ("N12 unconfigured-push-is-a-channel", NT, '        if notifier is not None and getattr(notifier, "configured", False):', "        if notifier is not None:"),
    # (review 9) names, the table's bound, the quiet line, an unreadable file
    ("N13 a-full-table-evicts-the-oldest-whoever-it-is", AA, "            polls.pop(victim if victim is not None else next(iter(polls)))",
     "            polls.pop(next(iter(polls)))"),
    ("N14 the-endpoint-never-says-acknowledged", RT, "acknowledged=offset is not None)", "acknowledged=False)"),
    ("N15 quiet-line-names-every-client", NT, "'; '.join(quiet[:_QUIET_NAMES_MAX])", "'; '.join(quiet)"),
    ("N16 the-rest-of-the-quiet-clients-not-counted", NT, _MORE, '                 + ".")'),
    ("N17 any-name-is-recorded", AA, _RECORDED, "        if not isinstance(consumer, str):\n            return\n"),
    ("N18 a-name-that-is-no-plain-token-is-shown", NT, "            named = valid_consumer_name(n)", "            named = True"),
    ("N19 a-name-may-be-any-text", AA, 'CONSUMER_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,32}")', 'CONSUMER_NAME_RE = re.compile(r"(?s).{1,3200}")'),
    ("N20 unreadable-client-list-not-said", NT, '    if not readable:\n        head += "\\nClients: the client list could not be read."\n', ""),
    ("N21 a-damaged-file-reads-as-no-client", NT, "        return set(), False\n    if not isinstance(data, dict):", "        return set(), True\n    if not isinstance(data, dict):"),
    ("N22 a-file-that-is-not-a-table-reads-as-no-client", NT, "    if not isinstance(data, dict):\n        return set(), False", "    if not isinstance(data, dict):\n        return set(), True"),
    ("N23 a-name-is-listed-twice", NT, "        for n in sorted(registered | set(polls)):", "        for n in sorted(registered) + sorted(polls):"),
    # (review 7, 8) what the text claims
    ("N24 no-registered-client-reads-as-delivered-on-reconnect", NT, "    elif registered:\n        head = (", "    elif True:\n        head = ("),
    ("N25 a-poll-cadence-is-claimed", NT, "client (last polled {age}s ago)", "client (poller, ≤30s)"),
    ("N26 an-away-client-is-not-said-to-get-it-later", NT, "; it receives this at its next poll)\")", ")\")"),
    ("N27 a-connected-unregistered-client-is-not-named", NT, _UNREG, "            elif False:"),
    ("N28 digest-not-mentioned-without-a-channel", NT, '    if not channels and in_digest:\n        head += " It will also show in the next-turn digest."\n', ""),
    # ── review r2 ──
    ("N29 a-client-under-an-odd-name-is-not-registered", NT, _STORED, "            if valid_consumer_name(n) and v is not None and int(v) >= 0:"),
    ("N30 a-null-position-is-a-registration", NT, _STORED, "            if isinstance(n, str) and (v is None or int(v) >= 0):"),
    ("N31 a-registered-client-under-an-odd-name-is-not-seen-polling", AA, _RECORDED,
     "        if not valid_consumer_name(consumer):\n            return\n"),
    ("N32 polling-clients-under-odd-names-are-dropped", NT, "                else:\n                    live_unnamed.append(age)",
     "                else:\n                    pass"),
    ("N33 quiet-clients-under-odd-names-are-dropped", NT, "                    quiet_unnamed.append(when)", "                    pass"),
    ("N34 six-quiet-clients-end-with-plus-zero-more", NT, _MORE, '                 + f"; +{more} more.")'),
]
PINS = ["tests/test_4tb_replan_removed.py", "tests/test_4tb_set_mood.py", "tests/test_4tb_fact_check.py",
        "tests/test_4tb_notify_delivery.py"]


def _tests():
    pat = (r"replan\|WATCHDOG_BREAK_NOTE\|unparsed_call_markup_present\|strip_unparsed_tool_calls\|"
           r"_canonicalise_tool_name\|Unknown tool\|_MEMBER_ALLOWED_TOOLS\|_BOOKKEEPING_TOOL_NAMES\|"
           r"tool_introspect\|set_mood\|is_write_call\|_COLLAPSE_READSAFE\|is_contained_context\|"
           r"restrict_tool_surface\|DERIVED_MOOD_VOCAB\|SelfState\|RETIRED_PHASES\|activity_liveness\|"
           r"parse_args\|tool_fact_check\|factcheck_sources\|neutralise_report_grammar\|tool_deep_research\|"
           r"unopened_citation_note\|haystack_from\|ground_links\|fetched_urls\|hydrate_context\|tool_notify_operator\|note_consumer_poll\|"
           r"notifications/pending\|render_activity_digest\|digest_shows\|get_available_tools\|"
           r"get_active_tool_definitions")
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
