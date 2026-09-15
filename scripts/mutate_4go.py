#!/usr/bin/env python3
"""§4GO mutation battery — the trailing-beat narration rule.

    python3 scripts/mutate_4go.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
RS = os.path.join(ROOT, "src/ghost_agent/core/reply_smoothing.py")
AG = os.path.join(ROOT, "src/ghost_agent/core/agent.py")
TESTS = [
    "tests/test_4go_trailing_beat_narration.py",
    "tests/test_reply_smoothing.py",
    "tests/test_reply_pass1_temporal.py",
    "tests/test_reply_stale_announcement.py",
    "tests/test_reply_toolcall_scrub.py",
    "tests/test_forced_final_no_answer.py",
]

MUTANTS = [
    ("CONTROL no-op comment",
     "def _trailing_beat(block: str) -> bool:",
     "# a no-op edit\ndef _trailing_beat(block: str) -> bool:"),

    ("M1 the trailing beat is invisible again",
     "    sents = [x for x in _SENTENCE_SPLIT_RE.split(block.strip()) if x.strip()]",
     "    return False\n    sents = [x for x in _SENTENCE_SPLIT_RE.split(block.strip()) if x.strip()]"),

    ("M2 a one-sentence beat also counts as a TRAILING one",
     "    if len(sents) < 2:\n        return False                      # one sentence: that is shape 1",
     "    if len(sents) < 1:\n        return False                      # one sentence: that is shape 1"),

    ("M3 the offer is treated as a beat",
     "    if _OFFER_RE.match(last):\n        return False",
     "    if False:\n        return False"),

    ("M4 the beat needs no evidence at all",
     "    if _trailing_beat(stripped):\n        return bool(later_prose) and _restated_anywhere_later(stripped,\n                                                              later_prose)",
     "    if _trailing_beat(stripped):\n        return True"),

    ("M5 the evidence goes back to ONE paragraph",
     "        return bool(later_prose) and _restated_anywhere_later(stripped,\n                                                              later_prose)",
     "        return bool(later_prose) and _restated_later(stripped, later_prose)"),

    ("M6 a LONE beat is swept up with the run",
     "        adjacent = (i and drop[i - 1]) or (i + 1 < len(blocks) - 1 and drop[i + 1])\n        if not adjacent:\n            continue",
     "        adjacent = True\n        if not adjacent:\n            continue"),

    ("M7 the run no longer checks that an answer follows",
     "        if any(_is_answer_block(blocks[j])\n               for j in range(i + 1, len(blocks)) if not drop[j]):\n            drop[i] = True",
     "        drop[i] = True"),

    ("M8 the system note counts as an answer",
     "    if not s or s == UNPARSED_TOOL_CALL_NOTE.strip():\n        return False",
     "    if not s:\n        return False"),

    ("M9 the inverted-trim guard is length-only again",
     "    return narration_only(s)",
     "    return False"),

    ("M10 the run rule ignores the paragraph-length bound",
     "        if len(blocks[i].strip()) > _MAX_NARRATION_CHARS:\n            continue",
     "        if False:\n            continue"),

    ("KNOWN-BAD control: smoothing returns nothing",
     "def smooth_reply(text: str) -> str:",
     "def smooth_reply(text: str) -> str:\n    return \"\""),
]

#: Mutants against agent.py (the abort note), same battery, same test set.
AGENT_MUTANTS = [
    ("M11 the abort note is dropped when text accumulated",
     "    body = (final_ai_content or \"\").strip()\n    if not body:\n        return note",
     "    body = (final_ai_content or \"\").strip()\n    if body:\n        return final_ai_content\n    if not body:\n        return note"),

    ("M12 the strike cap keeps its own spelling",
     "                        final_ai_content = _with_abort_note(\n                            final_ai_content,\n                            \"[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit after \"",
     "                        final_ai_content = final_ai_content or (\n                            \"[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit after \""),

    ("M13 the note stacks on re-entry",
     "    if note[:48] in body:                      # idempotent across re-entry\n        return final_ai_content",
     "    if False:\n        return final_ai_content"),
]


def main():
    pristine = RS + ".pristine"
    shutil.copy2(RS, pristine)
    env = dict(os.environ, GHOST_API_KEY="x",
               PYTHONPATH=os.path.join(ROOT, "src"))
    env.pop("FORCE_COLOR", None)
    try:
        for n, (label, old, new) in enumerate(MUTANTS):
            src = open(pristine).read()
            if src.count(old) != 1:
                print(f"[{n:2d}] ANCHOR-MISS ({src.count(old)}x) {label}", flush=True)
                continue
            open(RS, "w").write(src.replace(old, new, 1))
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, "-m", "pytest", *TESTS, "-x", "-q",
                 "-p", "no:randomly", "--timeout=300"],
                cwd=ROOT, env=env, capture_output=True, text=True)
            verdict = "SURVIVED" if p.returncode == 0 else "KILLED  "
            tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            print(f"[{n:2d}] {verdict} {label}  ({time.time()-t0:.0f}s) {tail[:70]}",
                  flush=True)
        ag_pristine = AG + ".pristine"
        shutil.copy2(AG, ag_pristine)
        try:
            for n, (label, old, new) in enumerate(AGENT_MUTANTS, start=len(MUTANTS)):
                src = open(ag_pristine).read()
                if src.count(old) != 1:
                    print(f"[{n:2d}] ANCHOR-MISS ({src.count(old)}x) {label}", flush=True)
                    continue
                open(AG, "w").write(src.replace(old, new, 1))
                t0 = time.time()
                p = subprocess.run(
                    [sys.executable, "-m", "pytest", *TESTS, "-x", "-q",
                     "-p", "no:randomly", "--timeout=300"],
                    cwd=ROOT, env=env, capture_output=True, text=True)
                verdict = "SURVIVED" if p.returncode == 0 else "KILLED  "
                tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
                print(f"[{n:2d}] {verdict} {label}  ({time.time()-t0:.0f}s) {tail[:70]}",
                      flush=True)
        finally:
            shutil.copy2(ag_pristine, AG)
            os.unlink(ag_pristine)
    finally:
        shutil.copy2(pristine, RS)
        os.unlink(pristine)
    print("=== 4GO BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
