#!/usr/bin/env python3
"""§4JR battery — reply language: the classifier, the turn-state rule, the
in-loop regeneration (2026-09-22).

    python3 scripts/mutate_4jr.py <copy-root>

Whole-file mutants on a COPY of the tree (never the deployed one); every
mutant is a world the pins claim to distinguish. No-op and known-bad
controls bracket the batch.
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RL = os.path.join(ROOT, "src/ghost_agent/core/reply_language.py")
TS = os.path.join(ROOT, "src/ghost_agent/core/turn_state_check.py")
AG = os.path.join(ROOT, "src/ghost_agent/core/agent.py")
FILES = [RL, TS, AG]
TESTS = [
    "tests/test_reply_language_guard_2026_09_22.py",
    "tests/test_reply_language_rule_2026_09_21.py",
    "tests/test_4fy_turn_state_check.py",
    "tests/test_repair_directive_pending_request.py",
]

MUTANTS = [
    ("CONTROL no-op comment", RL,
     "def script_share(", "# ctl\ndef script_share("),

    ("M1 the Latin-request threshold slides back to 0.5", RL,
     "EN_TO_EL_MIN_SHARE = 0.6", "EN_TO_EL_MIN_SHARE = 0.5"),

    ("M2 indented continuation lines count as prose", RL,
     '_SKIP_LINE = re.compile(r"^(?:\\s{2,}|\\t|\\s*(?:[-*•]|\\d+[.)]|\\||#{1,6}\\s|\\*\\*[^*]*«|«))")',
     '_SKIP_LINE = re.compile(r"^\\s*(?:[-*•]|\\d+[.)]|\\||#{1,6}\\s|\\*\\*[^*]*«|«)")'),

    ("M3 a request that names a language is judged", RL,
     "    if _letters(text) < MIN_REQUEST_LETTERS or _LANGUAGE_ASK.search(text):",
     "    if _letters(text) < MIN_REQUEST_LETTERS:"),

    ("M4 the runtime's abort notes are judged", RL,
     '        if _ABORT_MARKER.search(str(reply or "")) or refute_no_answer_fallback(reply):',
     "        if refute_no_answer_fallback(reply):"),

    ("M5 Greeklish reads as English", RL,
     '        return None if _greeklish(text) else "latin"',
     '        return "latin"'),

    ("M6 a mixed-script request reads as Latin", RL,
     "    return None  # mixed — the user chose both; nothing to enforce",
     '    return "latin"'),

    ("M7 fenced code is prose", RL,
     "        if _FENCE_LINE.match(ln):\n            in_fence = not in_fence\n            continue\n",
     ""),

    ("M8 any reply is long enough to judge", RL,
     "MIN_REPLY_LETTERS = 60", "MIN_REPLY_LETTERS = 0"),

    ("M9 the rule leaves refute_turn_state", TS,
     '        msg = _check_reply_language(request, body)\n        if msg:\n            issues.append(("reply_language", msg))\n',
     ""),

    ("M10 the vocabulary entry is dropped — a content complaint, not a shape one", AG,
     'r"|empty_evidence|reply_language):\\s"', 'r"|empty_evidence):\\s"'),

    ("M11 the guard rides the verifier's clean-turn gate", AG,
     "                if (repair_round < self._MAX_VERIFIER_REPAIRS\n"
     '                        and os.getenv("GHOST_REPLY_LANGUAGE_REPAIR", "1")',
     "                if (repair_round < self._MAX_VERIFIER_REPAIRS\n"
     "                        and execution_failure_count == 0\n"
     '                        and os.getenv("GHOST_REPLY_LANGUAGE_REPAIR", "1")'),

    ("M12 the round is free — a Greek model is asked again and again", AG,
     "                        repair_round += 1\n"
     "                        # Text-only re-entry:",
     "                        # Text-only re-entry:"),

    ("M13 the Greek draft is kept in front of the English answer", AG,
     "                        _repair_reentry_active = True\n"
     '                        final_ai_content = ""\n'
     "                        _verdict_is_fresh = False\n",
     "                        _repair_reentry_active = True\n"
     "                        _verdict_is_fresh = False\n"),

    ("M14 the kill switch is ignored", AG,
     '                        and os.getenv("GHOST_REPLY_LANGUAGE_REPAIR", "1")\n'
     '                        .strip().lower() not in ("0", "false", "no", "off")):',
     "                        ):"),

    ("M15 the planner's stop is left set — the loop breaks on an EMPTY reply", AG,
     "                        force_final_response = True\n"
     "                        force_stop = False\n"
     "                        _repair_reentry_active = True\n",
     "                        force_final_response = True\n"
     "                        _repair_reentry_active = True\n"),

    ("M23 the regeneration turn gets its tools back", AG,
     "                        force_final_response = True\n"
     "                        force_stop = False\n",
     "                        force_final_response = False\n"
     "                        force_stop = False\n"),

    ("M24 the guard is gated on force_stop again (the probe4jr world)", AG,
     "                if (repair_round < self._MAX_VERIFIER_REPAIRS\n"
     '                        and os.getenv("GHOST_REPLY_LANGUAGE_REPAIR", "1")',
     "                if (repair_round < self._MAX_VERIFIER_REPAIRS and not force_stop\n"
     '                        and os.getenv("GHOST_REPLY_LANGUAGE_REPAIR", "1")'),

    ("M16 the issue stops saying which language to answer in", TS,
     '    return (f"the user wrote in {expected} but the reply\'s prose is in {got} "\n'
     '            f"— answer in {expected}")',
     '    return f"the user wrote in {expected} but the reply\'s prose is in {got}"'),

    ("M17 the language site decides shape on its own (a second rule)", AG,
     "                            shape_only=GhostAgent._delivery_shape_only(_lang_vr),",
     "                            shape_only=True,"),

    ("M18 a Greek ask answered in English is never a drift", RL,
     "EL_TO_EN_MAX_SHARE = 0.2", "EL_TO_EN_MAX_SHARE = 0.0"),

    ("M19 the directive stops naming the current request", AG,
     "                            shape_only=GhostAgent._delivery_shape_only(_lang_vr),\n"
     "                            pending_request=last_user_content,",
     "                            shape_only=GhostAgent._delivery_shape_only(_lang_vr),"),

    ("M20 the alert leaks: no standalone suffix on the language directive", AG,
     "                        )\n"
     "                        _directive += _REPAIR_STANDALONE_SUFFIX\n"
     "                        messages.append(msg)\n",
     "                        )\n"
     "                        messages.append(msg)\n"),

    ("M21 a standing language instruction earlier in the conversation is ignored", RL,
     "        if any(_LANGUAGE_ASK.search(str(m or \"\")) for m in (prior_user_messages or ())):\n"
     "            return None  # a standing language instruction earlier in the conversation\n",
     ""),

    ("M22 the site stops handing the conversation to the classifier", AG,
     "                        _lang_mismatch = reply_language_mismatch(\n"
     "                            last_user_content, final_ai_content,\n"
     "                            prior_user_messages=[\n"
     '                                str(m.get("content") or "")\n'
     "                                for m in (messages or [])[:-1]\n"
     '                                if isinstance(m, dict) and m.get("role") == "user"\n'
     '                                and isinstance(m.get("content"), str)])\n',
     "                        _lang_mismatch = reply_language_mismatch(\n"
     "                            last_user_content, final_ai_content)\n"),

    ("KNOWN-BAD control: the classifier never fires", RL,
     "    try:\n        want = request_script(request)",
     "    return None\n    try:\n        want = request_script(request)"),
]


def main():
    saved = {f: f + ".pristine" for f in FILES}
    for f, p in saved.items():
        shutil.copy2(f, p)
    env = dict(os.environ, GHOST_API_KEY="x",
               PYTHONPATH=os.path.join(ROOT, "src"))
    env.pop("FORCE_COLOR", None)
    killed = survived = 0
    try:
        for n, (label, target, old, new) in enumerate(MUTANTS):
            for f, p in saved.items():
                shutil.copy2(p, f)
            src = open(saved[target]).read()
            if src.count(old) != 1:
                print(f"[{n:2d}] ANCHOR-MISS ({src.count(old)}x) {label}", flush=True)
                continue
            open(target, "w").write(src.replace(old, new, 1))
            for d, _, fs in os.walk(os.path.join(ROOT, "src")):
                if d.endswith("__pycache__"):
                    shutil.rmtree(d, ignore_errors=True)
            c = subprocess.run([sys.executable, "-m", "py_compile", target],
                               capture_output=True, text=True)
            if c.returncode != 0:
                print(f"[{n:2d}] NO-COMPILE {label}: {c.stderr.strip()[-120:]}", flush=True)
                continue
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, "-m", "pytest", *TESTS, "-x", "-q",
                 "-p", "no:randomly", "--timeout=300"],
                cwd=ROOT, env=env, capture_output=True, text=True)
            verdict = "SURVIVED" if p.returncode == 0 else "KILLED  "
            if not label.startswith("CONTROL"):
                killed += p.returncode != 0
                survived += p.returncode == 0
            tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            print(f"[{n:2d}] {verdict} {label}  ({time.time()-t0:.0f}s) {tail[:70]}",
                  flush=True)
    finally:
        for f, p in saved.items():
            shutil.copy2(p, f)
            os.unlink(p)
    print(f"=== 4JR BATTERY DONE: {killed} killed / {survived} survived "
          f"(known-bad control counted)", flush=True)


if __name__ == "__main__":
    main()
