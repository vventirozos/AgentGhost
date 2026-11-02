#!/usr/bin/env python3
"""Slack bot maintenance-mode battery (2026-11-01). Coverage-selected, parallel
(scripts/battery_cov.py); the real tree is never mutated. No-op control must SURVIVE,
known-bad must DIE.    python3 scripts/mutate_slack_maint.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = "interface/externals/slack_bot/main.py"
_MENTION = """    _maint = maintenance_text()
    if _maint:
        logger.info("maintenance: mention from %s not sent to the agent", event.get("user"))"""
_DM = """    _maint = maintenance_text()
    if _maint:
        logger.info("maintenance: DM not sent to the agent")"""
MUT = [
    ("CONTROL no-op", B, "def maintenance_text() -> str | None:", "def maintenance_text() -> str | None:  # no-op"),
    ("CONTROL known-bad", B, _MENTION, _MENTION.replace("    if _maint:", "    if False:")),
    ("M2 dm-unguarded", B, _DM, _DM.replace("    if _maint:", "    if False:")),
    ("M3 reaction-unguarded", B, "        if maintenance_text():\n            # nothing reaches the agent in maintenance",
     "        if False:\n            # nothing reaches the agent in maintenance"),
    ("T1 empty-file-off", B, "    return body or MAINTENANCE_DEFAULT_TEXT", "    return body or None"),
    ("T2 unreadable-off", B, "maintenance stays ON\", MAINTENANCE_FILE, e)\n        return MAINTENANCE_DEFAULT_TEXT",
     "maintenance stays ON\", MAINTENANCE_FILE, e)\n        return None"),
    ("T3 read-empty-path (equivalent: open(\"\") is FileNotFoundError)", B, "    if not MAINTENANCE_FILE:\n        return None\n    try:", "    try:"),
    ("T4 no-read-cap", B, "            body = fh.read(_MAINTENANCE_TEXT_MAX).strip()", "            body = fh.read().strip()"),
    ("S1 off-keeps-file", B, "            os.remove(MAINTENANCE_FILE)\n", "            pass\n"),
    ("S2 set-empty-path", B, "    if not MAINTENANCE_FILE:\n        raise OSError", "    if False:\n        raise OSError"),
    ("S3 no-write-cap", B, "        fh.write(str(text or \"\").strip()[:_MAINTENANCE_TEXT_MAX])", "        fh.write(str(text or \"\").strip())"),
    ("S4 tmp-left", B, "    os.replace(tmp, MAINTENANCE_FILE)", "    import shutil; shutil.copy(tmp, MAINTENANCE_FILE)"),
    ("C1 dash-unspaced (r1 MAJOR)", B, "    if (after[0] in _MAINTENANCE_SPACE and lead[:1] in", "    if (True and lead[:1] in"),
    ("C1b dash-after-unspaced", B, "            and lead[1:2] and lead[1] in _MAINTENANCE_SPACE):", "            and lead[1:2]):"),
    ("C1c no-dash-form", B, "        return \"on\", lead[1:].strip(_MAINTENANCE_SPACE)\n    return None", "        return None\n    return None"),
    ("C2 case-sensitive", B, "    if len(t.splitlines()) != 1 or not t.lower().startswith(\"maintenance\"):", "    if len(t.splitlines()) != 1 or not t.startswith(\"maintenance\"):"),
    ("C2b multiline-ok (r2)", B, "    if len(t.splitlines()) != 1 or not t.lower().startswith(\"maintenance\"):", "    if not t.lower().startswith(\"maintenance\"):"),
    ("C2e only-newline (r3)", B, "    if len(t.splitlines()) != 1 or not t.lower().startswith(\"maintenance\"):", "    if \"\\n\" in t or \"\\r\" in t or not t.lower().startswith(\"maintenance\"):"),
    ("C2c space-after-word", B, "    if not rest[:1] or rest[0] not in _MAINTENANCE_SPACE:\n        return None", "    if not rest[:1]:\n        return None"),
    ("C2d newline-is-space (equivalent: the one-line check)", B, '_MAINTENANCE_SPACE = " \\t"', '_MAINTENANCE_SPACE = " \\t\\n"'),
    ("C3 no-command-in-maint", B, "    _cmd = parse_maintenance_command(user_text) if not event.get(\"files\") else None",
     "    _cmd = None if maintenance_text() else (parse_maintenance_command(user_text) if not event.get(\"files\") else None)"),
    ("C4 files-parsed", B, "    _cmd = parse_maintenance_command(user_text) if not event.get(\"files\") else None",
     "    _cmd = parse_maintenance_command(user_text)"),
    ("C5 on-question", B, "    if after.rstrip(\".!\") == \"\":\n        return \"on\", \"\"", "    if after.rstrip(\".!?\") == \"\":\n        return \"on\", \"\""),
    ("C5b off-question", B, "    if low.rstrip(\".!\") == \"off\":", "    if low.rstrip(\".!?\") == \"off\":"),
    ("C6 status-no-question", B, "    if low.rstrip(\".!?\") == \"status\":", "    if low == \"status\":"),
    ("C6b off-no-punct", B, "    if low.rstrip(\".!\") == \"off\":", "    if low == \"off\":"),
    ("C7 no-unwrap", B, "    if len(t) >= 2 and t[0] == t[-1] and t[0] in \"*_`\" and t[0] not in t[1:-1]:", "    if False:"),
    ("C8 wrap-unbalanced", B, "    if len(t) >= 2 and t[0] == t[-1] and t[0] in \"*_`\" and t[0] not in t[1:-1]:", "    if len(t) >= 2 and t[0] in \"*_`\":"),
    ("C8b wrap-inside (r3)", B, "    if len(t) >= 2 and t[0] == t[-1] and t[0] in \"*_`\" and t[0] not in t[1:-1]:", "    if len(t) >= 2 and t[0] == t[-1] and t[0] in \"*_`\":"),
    ("C9 colon-text-dropped", B, "    if after.startswith(\":\"):\n        return \"on\", after[1:].strip(_MAINTENANCE_SPACE)", "    if after.startswith(\":\"):\n        return \"on\", \"\""),
    ("C9b spaced-colon (r3)", B, "    if after.startswith(\":\"):", "    if lead.startswith(\":\"):"),
    ("C10 on-prefix-loose", B, "    if not low.startswith(\"on\"):\n        return None", "    if not low.startswith(\"o\"):\n        return None"),
    ("S6 surrogate-raises", B, '        with open(tmp, "w", encoding="utf-8", errors="replace") as fh:', '        with open(tmp, "w", encoding="utf-8") as fh:'),
    ("S5 tmp-left-on-failure", B, "        try:                     # r1: a failed write left its temp file behind\n            os.remove(tmp)", "        try:                     # r1: a failed write left its temp file behind\n            pass"),
    ("F1 flag-none-clears", B, "    if not value:\n        return\n    try:\n        set_maintenance(value == \"on\")", "    try:\n        set_maintenance(value == \"on\")"),
    ("E1 no-escape", B, "    return _slack_escape(text)\n\n\ndef apply_maintenance_flag", "    return text\n\n\ndef apply_maintenance_flag"),
    ("W1 write-error-uncaught", B, "    except OSError as e:\n        logger.error(\"maintenance %s failed: %s\", verb, e)",
     "    except ZeroDivisionError as e:\n        logger.error(\"maintenance %s failed: %s\", verb, e)"),
    ("W2 off-is-on", B, "            set_maintenance(False)\n            msg = \"✅", "            set_maintenance(True)\n            msg = \"✅"),
    ("R1 mention-not-threaded", B, _MENTION + "\n        await say(text=_maintenance_reply(_maint), thread_ts=thread_ts)",
     _MENTION + "\n        await say(text=_maintenance_reply(_maint), thread_ts=None)"),
]
PINS = ["tests/test_slack_bot_maintenance.py"]


def _tests():
    return sorted(set(PINS) | {"tests/test_slack_bot_owner_lock.py", "tests/test_slack_bot_feedback.py",
                               "tests/test_slack_bot_dedup.py", "tests/test_slack_bot_shadow_ban.py"})


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
        # only the bot's own test files load this file; the coverage fallback for
        # module-level anchors pulled in ~26k unrelated tests whose tree-environment
        # failures killed the no-op (2026-11-01)
        return full, f"bot files ({len(full)})"

    def _cov_select(label, f, old):
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
