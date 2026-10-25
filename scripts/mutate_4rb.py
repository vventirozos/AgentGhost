"""§4RB battery — a reasoning effort the operator picks per request in the web UI (2026-10-24).
Reuses the §4QA harness.   python3 scripts/mutate_4rb.py [all|check|rerun R1 R2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

EF = "src/ghost_agent/core/effort.py"
AG = "src/ghost_agent/core/agent.py"
RT = "src/ghost_agent/api/routes.py"
MN = "src/ghost_agent/main.py"
SV = "interface/server.py"
JS = "interface/static/effort.js"
APP = "interface/static/app.js"
FSY = "src/ghost_agent/tools/file_system.py"
ISO = "src/ghost_agent/core/isolation.py"
m.PINS = ["tests/test_4rb_reasoning_effort.py"]


def _tests():
    import glob, subprocess
    pat = (r"ghost_agent.core import effort\|effort_override_context\|_with_turn_effort\|_effort_header\|effort.js"
           r"\|_effort_escalate\|_effort_note_due\|_STEER_HEAD_RE")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    # the cache-bust test kills any static-file mutant for an unmoved ?v=, not for its behaviour
    return sorted(({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS)) - {"tests/test_interface_cachebust.py"})


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", EF, "def thinking_off(payload) -> bool:", "# ctl\ndef thinking_off(payload) -> bool:"),
    ("CONTROL known-bad (must DIE): always on", EF,
     '    return os.environ.get("GHOST_REASONING_EFFORT", "0").strip() == "1"', "    return True"),
    ("R1 the switch defaults on", EF, 'os.environ.get("GHOST_REASONING_EFFORT", "0")', 'os.environ.get("GHOST_REASONING_EFFORT", "1")'),
    ("R2 junk levels pass", EF, '    return v if v in LEVELS else ""', "    return v"),
    ("R3 x-high not normalised", EF, '.strip().lower().replace("-", "")', ".strip().lower()"),
    ("R4 thinking-off turns get it", EF,
     "    if level not in LEVELS or not isinstance(payload, dict) or thinking_off(payload):",
     "    if level not in LEVELS or not isinstance(payload, dict):"),
    ("R5 the caller's value loses", EF, '    kw.setdefault("reasoning_effort", level)', '    kw["reasoning_effort"] = level'),
    ("R6 apply mutates the caller's payload", EF,
     "    kw = dict(kw) if isinstance(kw, dict) else {}\n    kw.setdefault(\"reasoning_effort\", level)\n    return {**payload, \"chat_template_kwargs\": kw}",
     "    kw = kw if isinstance(kw, dict) else {}\n    kw.setdefault(\"reasoning_effort\", level)\n    payload[\"chat_template_kwargs\"] = kw\n    return payload"),
    ("R7 the hook ignores the switch", AG, "            if not _effort.enabled():\n                return payload\n            from .llm import EFFORT_ARMS",
     "            from .llm import EFFORT_ARMS"),
    ("R8 no pick → a level is sent anyway", AG, "            if not level:\n                return payload\n            out = _effort.apply(payload, level)",
     "            level = level or \"xhigh\"\n            out = _effort.apply(payload, level)"),
    ("R9 the probe arm loses", AG, "            if prompt_arm() in EFFORT_ARMS:\n                return payload\n            # r4: only",
     "            # r4: only"),
    ("R10 the pick is not recorded", AG, '                _tf.record(self.context, rid, reasoning_effort=level)\n', ""),
    ("R10b the line every turn", AG,
     '                if _tf.facts_for(self.context, rid).get("reasoning_effort") != level:   # once per request',
     "                if True:"),
    ("R23 the warm-up ignores the last level", AG, "            payload = self._warm_up_effort(payload)        # §4RB r3: the level the last turn used\n", ""),
    ("R27 members pick a level", AG,
     '            level = _effort.parse_override(effort_override_context.get()) if (owner or inherited) else ""',
     "            level = _effort.parse_override(effort_override_context.get())"),
    ("R28 background turns move the warm-up", AG,
     "            if owner and not _effort.thinking_off(payload):", "            if not _effort.thinking_off(payload):"),
    ("R29 delegated sub-agents lose the pick", AG,
     '            inherited = kind == "background" and rid_s.startswith("sub-")', "            inherited = False"),
    ("R30 sub-agents move the warm-up", AG,
     "            if owner and not _effort.thinking_off(payload):", "            if (owner or inherited) and not _effort.thinking_off(payload):"),
    ("R31 a probe's or member's sub-agent inherits", AG,
     '            inherited = kind == "background" and rid_s.startswith("sub-")', '            inherited = rid_s.startswith("sub-")'),
    ("R24 the last level is not kept", AG, "                self._main_effort_level = level\n", "                pass\n"),
    ("R25 thinking-off turns move the warm-up", AG,
     "            if owner and not _effort.thinking_off(payload):\n                # r3 M", "            if owner:\n                # r3 M"),
    ("R26 the warm-up ignores the switch", AG,
     "            if not _effort.enabled():\n                return payload\n            return _effort.apply(payload, str(getattr(self",
     "            return _effort.apply(payload, str(getattr(self"),
    ("R11 the hook is not called", AG, "payload = self._with_turn_effort(payload, req_id)", "payload = payload"),
    ("R12 routes ignore the header", RT,
     '    effort_override_context.set(_parse_effort(request.headers.get("X-Ghost-Effort")))',
     '    effort_override_context.set("")'),
    ("R13 health always off", RT, "        return bool(enabled())", "        return False"),
    ("R14 autonomous turns keep a web pick", MN, '(effort_override_context, ""))]', '(effort_override_context, "low"))]'),
    ("R15 interface passes junk", SV, '    return {"X-Ghost-Effort": v} if v in _EFFORT_LEVELS else {}', '    return {"X-Ghost-Effort": v} if v else {}'),
    ("R16 interface leaves the body field", SV, '        _effort = _effort_header(body.pop("reasoning_effort", None))',
     '        _effort = _effort_header(body.get("reasoning_effort", None))'),
    ("R17 the non-streamed path drops the header", SV, "                         **_effort},                                            # §4RB",
     "                         },                                            # §4RB"),
    ("R18 hidden selector still sends", JS, "    if (!sel || sel.classList.contains('hidden')) return '';", "    if (!sel) return '';"),
    ("R19 hiding keeps a stale pick", JS, "    if (!on) sel.value = '';", ""),
    ("R20 the selector never shows", JS, "    const on = health.reasoning_effort === true;", "    const on = false;"),
    ("R21 a missed health poll hides it", JS, "    if (!sel || !health) return;", "    if (!sel) return;\n    health = health || {};"),
    ("R22 an early poll is lost", JS, "    applyEffortSelector(sel, win.__ghostHealth || null);", ""),
    # xhigh after the first failure — the appended note (operator 2026-10-24)
    ("R32 any failed run escalates", EF, "    if fname == \"execute\" and ran_written_failed:", "    if fname == \"execute\":"),
    ("R33 a refused write's syntax note escalates", EF,
     'or (not refused and "SYNTAX CHECK FAILED"', 'or ("SYNTAX CHECK FAILED"'),
    ("R34 a read's syntax note escalates", EF, "    if o in _WRITE_OPS and (code ==", "    if (code =="),
    ("R35 a refused edit does not escalate", EF, '    if o in ("edit", "replace") and refused and code in WRONG_EDIT_CODES:', "    if False:"),
    ("R36 any refused file op escalates", EF, '    if o in ("edit", "replace") and refused and code in WRONG_EDIT_CODES:',
     "    if refused and code in WRONG_EDIT_CODES:"),
    ("R36b any refusal code escalates", EF, '    if o in ("edit", "replace") and refused and code in WRONG_EDIT_CODES:',
     '    if o in ("edit", "replace") and refused:'),
    ("R36c the written-but-failed code is missed", EF, '(code == "written_but_syntax_failed"\n                            or ', "(False\n                            or "),
    ("R36d the reason code is not passed", AG,
     '                                                      reason_code=getattr(_outcome, "reason_code", "") or "")',
     '                                                      reason_code="")'),
    ("R37 the note loses the template sentence", EF, "{str(why or 'a step failed').rstrip('.')}. {XHIGH_SENTENCE}\"",
     "{str(why or 'a step failed').rstrip('.')}.\""),
    ("R38 the note is not a loop row", EF, '    return f"SYSTEM: {', '    return f"NOTE: {'),
    ("R39 escalation ignores the switch", AG, "            if not why or not _effort.enabled():\n                return\n",
     "            if not why:\n                return\n"),
    ("R40 the owner's pick does not win", AG,
     "                    and _effort.parse_override(effort_override_context.get()):\n                return",
     "                    and False:\n                return"),
    ("R41 a member's pick stops it", AG,
     '            if (kind == "owner" or (kind == "background" and rid.startswith("sub-"))) \\\n',
     "            if True \\\n"),
    ("R42 the probe arm loses", AG, "            if not rid or prompt_arm() in EFFORT_ARMS:\n                return\n            kind",
     "            if not rid:\n                return\n            kind"),
    ("R43 a second failure re-books", AG,
     '            if rid in due or _tf.facts_for(self.context, rid).get("effort_escalated"):\n                return\n', ""),
    ("R44 the note comes every turn", AG, "            why = due.pop(rid, None) if isinstance(due, dict) else None",
     "            why = due.get(rid) if isinstance(due, dict) else None"),
    ("R44b a placed note is not recorded", AG, "            _tf.record(self.context, rid, effort_escalated=why)\n", ""),
    ("R44c a thinking-off turn takes the note", AG, "            if not thinks or not _effort.enabled():", "            if not _effort.enabled():"),
    ("R44d the reset keeps a booking", AG, "            if isinstance(due, dict):\n                due.pop(rid, None)\n", ""),
    ("R44e the reset keeps the record", AG, '                _tf.record(self.context, rid, effort_escalated="")', "                pass"),
    ("R44f the reset is not called", AG, "                self._effort_reset(req_id)                   # §4RB", "                pass                   # §4RB"),
    ("R45 the note is never appended", AG,
     '                        messages.append({"role": "user", "content": _effort_note})', "                        pass"),
    ("R46 the tool trigger is not wired", AG,
     "                            self._effort_escalate(request_id_context.get(), _stuck)", "                            pass"),
    ("R47 a run of any file it named escalates", AG,
     "                                       and bool(set(futility_keys_in_command(a_hash, written_keys=_written.keys()))\n                                                & set(_written.keys())))",
     "                                       and bool(futility_keys_in_command(a_hash, written_keys=_written.keys())))"),
    ("R48 the note-due check ignores the switch", AG, "            if not thinks or not _effort.enabled():", "            if not thinks:"),
    ("R50 a partly-missed replace is missed", EF, '    if o == "replace" and code == "some_replace_blocks_failed":', "    if False:"),
    ("R51 an edit to a file in parts escalates", EF, '    if code == "syntax_failed_file_in_parts":          # unfinished by design (§4NU parts)\n        return ""\n', ""),
    ("R52 the tool books an in-parts edit as broken", FSY,
     '                                   reason_code=("syntax_failed_file_in_parts" if _is_in_progress(path)\n                                                else "written_but_syntax_failed"))',
     '                                   reason_code="written_but_syntax_failed")'),
    ("R53 the multi-block miss is not listed", EF, '"search_block_not_found", "no_blocks_matched", ', '"search_block_not_found", '),
    ("R54 the edit-ambiguous twin is not listed", EF, '"old_string_not_found", "old_string_ambiguous", ', '"old_string_not_found", '),
    ("R49 report turns take the note", AG,
     "                        req_id, thinks=not ((is_final_generation and getattr(self.context, \"_breaker_forced_final\", False))\n                                            or loop_retry_is_no_think(messages, last_user_content)))",
     "                        req_id, thinks=not (False\n                                            or loop_retry_is_no_think(messages, last_user_content)))"),
]

if __name__ == "__main__":
    m.main()
