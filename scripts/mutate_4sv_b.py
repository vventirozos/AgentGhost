#!/usr/bin/env python3
"""§4SV-b battery — the Mirror's new look (history, three replies, no repeat picks, row shapes) and the one
`introspect` line. Coverage-selected tests in parallel tree copies (scripts/battery_cov.py); the real src/ is
never mutated. No-op control must SURVIVE, the known-bad control must DIE.
    python3 scripts/mutate_4sv_b.py [check | -j6 all | rerun M1 I2 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MI = "src/ghost_agent/selfhood/mirror.py"
IN = "src/ghost_agent/tools/introspect.py"
MUT = [
    ("CONTROL no-op", MI, "def row_replies(row: dict) -> List[dict]:", "def row_replies(row: dict) -> List[dict]:  # no-op"),
    ("CONTROL known-bad", MI, "    return (rng or random).sample(pool, min(REPLIES_PER_LOOK, len(pool)))", "    return []"),
    ("M1 one-reply", MI, "REPLIES_PER_LOOK = 3", "REPLIES_PER_LOOK = 1"),
    ("M2 reply-cut", MI, "REPLY_MAX_CHARS = 600", "REPLY_MAX_CHARS = 1500"),
    ("M3 since-last-look-off", MI, "    since = max(now_utc.timestamp() - REPLY_WINDOW_DAYS * 86400, _look_epoch(after))", "    since = now_utc.timestamp() - REPLY_WINDOW_DAYS * 86400"),
    ("M3b window-floor-off", MI, "    since = max(now_utc.timestamp() - REPLY_WINDOW_DAYS * 86400, _look_epoch(after))", "    since = _look_epoch(after)"),
    ("M4 boundary-inclusive", MI, '        if not (since < _epoch(t.get("timestamp")) <= now_utc.timestamp() + 60):', '        if not (since <= _epoch(t.get("timestamp")) <= now_utc.timestamp() + 60):'),
    ("M5 look-first-row", MI, '    after = rows[-1].get("at") if rows else ""', '    after = rows[0].get("at") if rows else ""'),
    ("M5b look-no-after", MI, '    after = rows[-1].get("at") if rows else ""', '    after = ""'),
    ("M5c not-random", MI, "    return (rng or random).sample(pool, min(REPLIES_PER_LOOK, len(pool)))", "    return pool[:min(REPLIES_PER_LOOK, len(pool))]"),
    ("M5d bad-time-raises", MI, "    except (ValueError, OverflowError, OSError):\n        return 0.0", "    except (OverflowError, OSError):\n        return 0.0"),
    ("M6 collapse-off", MI, "        if runs[-1][2] == sent:", "        if False:"),
    ("M7 history-unbounded", MI, "for a, b, sent in runs[-SHOW_LAST:])", "for a, b, sent in runs)"),
    ("M8 run-end-not-moved", MI, "            runs[-1][1] = day\n", "            pass\n"),
    ("M9 always-plural", MI, "    many = len(replies) > 1", "    many = True"),
    ("M9c two-is-one", MI, "    many = len(replies) > 1", "    many = len(replies) > 2"),
    ("M9b always-singular", MI, "    many = len(replies) > 1", "    many = False"),
    ("M10 old-row-unread", MI, '    if str(row.get("reply_excerpt") or "").strip():', "    if False:"),
    ("M11 row-filter-off", MI, '        return [r for r in got if isinstance(r, dict) and str(r.get("excerpt") or "").strip()]', "        return list(got)"),
    ("M12 prev-always-seed", MI, '    prev = str(shown[-1]["sentence"]) if shown else SEED_SENTENCE', "    prev = SEED_SENTENCE"),
    ("C1 old-rows-shown", MI, '    return [r for r in rows if isinstance(r.get("replies"), list)]', "    return list(rows)"),
    ("C2 old-rows-in-prompt", MI, "    shown = current_rows(rows)", "    shown = rows"),
    ("C3 old-rows-in-introspect", MI, "    rows = current_rows(load(home))\n    if not rows:\n        return \"\"\n    last = rows[-1]", "    rows = load(home)\n    if not rows:\n        return \"\"\n    last = rows[-1]"),
    ("C4 after-from-shown-rows", MI, '    after = rows[-1].get("at") if rows else ""', '    after = current_rows(rows)[-1].get("at") if current_rows(rows) else ""'),
    ("M13 excerpt-uncut", MI, '"excerpt": " ".join(t.split())[:200]}', '"excerpt": " ".join(t.split())}'),
    ("M14 image-first-row", MI, '    last = rows[-1]\n    return (f"self-image:', '    last = rows[0]\n    return (f"self-image:'),
    ("M15 image-label-gone", MI, ' "A self-description, not a report of work done and not a checked fact)")', ' ")")'),
    ("M15b banner-first-row", MI, "    seen = row_replies(last)", "    seen = row_replies(rows[0])"),
    ("M15c banner-one-paragraph", MI, '        tail += "\\n" + "\\n>\\n".join(quoted)', '        tail += "\\n" + "\\n".join(quoted)'),
    ("M16 banner-no-times", MI, '            when = _local_time(r.get("at"))', '            when = ""'),
    ("M17 banner-singular", MI, "{'these replies' if len(seen) > 1 else 'this reply'}", "{'this reply'}"),
    ("M18 history-not-shown", MI, "    return PROMPT.format(history=history_lines(rows), replies=shown, **_WORDS[many])", "    return PROMPT.format(history=history_lines(rows[-1:]), replies=shown, **_WORDS[many])"),
    ("I1 overview-line-off", IN, '        image = _self_image(context, indent="  ")\n', '        image = ""\n'),
    ("I2 summary-line-off", IN, "    image = _self_image(context)\n", '    image = ""\n'),
    ("I3 overview-context-dropped", IN, "             _overview_selfhood(self_model, context),", "             _overview_selfhood(self_model),"),
    ("I4 summary-context-dropped", IN, "        return _render_summary(self_model, context)\n", "        return _render_summary(self_model)\n"),
    ("I8 public-gate-off", IN, "        if reply_is_public():\n            return \"\"\n        line = self_image_line(", "        line = self_image_line("),
    ("I9 cap-back", IN, "_OVERVIEW_MAX_CHARS = 3720", "_OVERVIEW_MAX_CHARS = 3200"),
    ("I5 reader-raises", IN, '        logger.debug("introspect self-image read skipped: %s", e)\n        return ""', "        raise"),
    ("I6 indent-lost", IN, '        return indent + line if line else ""', '        return line if line else ""'),
    # (I7 "line-after-carried" — the self-image line placed after the carried-state lines — is removed:
    # §4TB took the carried-state lines out of the overview with open questions and unfinished threads.)
]
PINS = ["tests/test_4sv_mirror.py"]


def _tests():
    pat = r"selfhood import mirror\|selfhood.mirror\|tools import introspect\|tools.introspect\|_overview_selfhood\|_render_summary"
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
    if args[0] == "check":
        print("anchors:", "OK" if not bad else bad); sys.exit(1 if bad else 0)
    if bad:
        print("ANCHOR NOT UNIQUE/FOUND:", bad); sys.exit(2)
    batch = [m for m in MUT if m[0].split(" ")[0] in set(args[1:]) or m[0].startswith("CONTROL")] if args[0] == "rerun" else MUT
    full = _tests()
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
    for lab, st, tier, tail in res:
        if st != "KILLED":
            print(f"  {st:8} {lab} — [{tier}] {tail}", flush=True)


if __name__ == "__main__":
    main()
