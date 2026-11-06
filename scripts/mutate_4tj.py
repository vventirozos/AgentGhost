#!/usr/bin/env python3
"""§4TJ battery — head saves recover from "off" without an agent restart, a side call with no save hook is
logged, and deleting an unread file is refused. Coverage-selected tests in parallel tree copies
(scripts/battery_cov.py); the real src/ is never mutated. No-op control must SURVIVE, known-bad must DIE.
    python3 scripts/mutate_4tj.py [check | -j6 all | rerun D1 H2 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS = "src/ghost_agent/core/head_saves.py"
AG = "src/ghost_agent/core/agent.py"
LL = "src/ghost_agent/core/llm.py"
FS = "src/ghost_agent/tools/file_system.py"

MUT = [
    ("CONTROL no-op", FS, "def _delete_loses(path: Path) -> bool:", "def _delete_loses(path: Path) -> bool:  # no-op"),
    ("CONTROL known-bad", FS, '        _unread = await asyncio.to_thread(unread_delete_refusal, path, filename)      # §4TJ', '        _unread = ""'),
    # recheck_supported
    ("H1 asks-unmarked", HS, "    if base not in _unsupported:\n        return False\n    try:\n        _http(f\"{base}/slots/{SLOT_ID}?action=save\", {\"filename\": \"\"}", "    try:\n        _http(f\"{base}/slots/{SLOT_ID}?action=save\", {\"filename\": \"\"}"),
    ("H2 real-name", HS, '{"filename": ""}, timeout=timeout)', '{"filename": "probe.bin"}, timeout=timeout)'),
    ("H3 any-http-lifts", HS, "        if e.code != 400:\n            return False\n", "        pass\n"),
    ("H4 no-http-lifts", HS, "    except Exception:  # noqa: BLE001\n        return False\n    _unsupported.discard(base)", "    except Exception:  # noqa: BLE001\n        pass\n    _unsupported.discard(base)"),
    ("H5 mark-stays", HS, "    _unsupported.discard(base)\n    return True", "    return True"),
    ("H6 says-no", HS, "    _unsupported.discard(base)\n    return True", "    _unsupported.discard(base)\n    return False"),
    # the agent's tick
    ("A1 tick-no-probe", AG, "            if await self._head_saves_back():\n                last_total = None\n", ""),
    ("A2 tick-no-warm", AG, "            if await self._head_saves_back():\n                last_total = None\n", "            await self._head_saves_back()\n"),
    ("A3 flag-off-probes", AG, "        if not _head_saves.enabled():\n            return False\n        llm = getattr(getattr(self, \"context\", None), \"llm_client\", None)\n        base = _head_saves.server_base", "        llm = getattr(getattr(self, \"context\", None), \"llm_client\", None)\n        base = _head_saves.server_base"),
    ("A5 always-back", AG, "        if not await asyncio.to_thread(_head_saves.recheck_supported, base):\n            return False\n", "        await asyncio.to_thread(_head_saves.recheck_supported, base)\n"),
    # the log line
    ("L1 silent", LL, "            if conv_keep_enabled():\n                # §4TJ", "            if False:\n                # §4TJ"),
    ("L2 always", LL, "            if conv_keep_enabled():\n                # §4TJ", "            if True:\n                # §4TJ"),
    ("L3 unnamed", LL, """what = side_call_what() or "this request's picture look"\n                pretty_log""", """what = "this request's picture look"\n                pretty_log"""),
    ("L4 no-default", LL, """what = side_call_what() or "this request's picture look"\n                pretty_log""", """what = side_call_what()\n                pretty_log"""),
    ("L5 wrong-icon", LL, '"warm-up yet); a conversation too big to park is re-read from token 0",\n                           icon=Icons.HEAD_FAIL)', '"warm-up yet); a conversation too big to park is re-read from token 0",\n                           icon=Icons.HEAD_SAVE)'),
    # delete: one file
    ("D1 empty-refused", FS, "    if st.st_size == 0 or not unread_rewrite_refusal(path, path.name, count_in_progress=False):", "    if not unread_rewrite_refusal(path, path.name, count_in_progress=False):"),
    ("D2 read-ignored", FS, "    if st.st_size == 0 or not unread_rewrite_refusal(path, path.name, count_in_progress=False):", "    if st.st_size == 0:"),
    ("D3 no-birth-rule", FS, '    return elapsed is None or getattr(st, "st_birthtime", st.st_mtime) < time.time() - elapsed', "    return True"),
    ("D4 mtime-not-birth", FS, 'getattr(st, "st_birthtime", st.st_mtime) < time.time() - elapsed', "st.st_mtime < time.time() - elapsed"),
    ("D5 untracked-is-new", FS, '    return elapsed is None or getattr(st, "st_birthtime"', '    return elapsed is not None and getattr(st, "st_birthtime"'),
    # delete: a directory
    ("D6 dir-ignored", FS, "    if path.is_dir():\n        for root, dirs, names in os.walk(path):", "    if False:\n        for root, dirs, names in os.walk(path):"),
    ("D7 links-judged", FS, "if not p.is_symlink() and p.is_file():", "if p.is_file():"),
    ("D8 walk-uncut", FS, "                if len(lost) >= _DELETE_WALK_MAX:\n                    cut = True\n                    break\n", ""),
    ("D9 cut-may-pass", FS, "asked[1] == sig.digest() and not cut\n", "asked[1] == sig.digest()\n"),
    ("D10 same-request-passes", FS, "    if (asked and asked[0] != rid and asked[1]", "    if (asked and asked[1]"),
    ("D11 changed-file-passes", FS, "    if (asked and asked[0] != rid and asked[1] == sig.digest() and not cut", "    if (asked and asked[0] != rid and not cut"),
    ("D12 no-expiry", FS, "            and now - asked[2] <= _DELETE_ASK_TTL_S):", "            ):"),
    ("D19 sig-unread-only", FS, """        sig.update(f"{p}\\0{_file_stamp(p)}\\n".encode("utf-8", "surrogateescape"))\n        if _delete_loses(p):\n            lost.append(p)""", """        if _delete_loses(p):\n            lost.append(p)\n            sig.update(f"{p}\\0{_file_stamp(p)}\\n".encode("utf-8", "surrogateescape"))"""),
    ("D20 sig-no-stamp", FS, """f"{p}\\0{_file_stamp(p)}\\n".encode""", """f"{p}\\n".encode"""),
    ("D21 scratch-gone", FS, "    if path.name in _SYSTEM_SCRATCH:\n        return False\n", ""),
    ("D22 cut-promises", FS, "    if cut:\n        return (f\"'{filename}' holds at least", "    if False:\n        return (f\"'{filename}' holds at least"),
    ("C1 identity-silent", AG, """                    pretty_log("Conversation", f"not saved before {why} — llama runs another model than the "\n                               f"last save plan; its next request re-reads it", icon=Icons.HEAD_FAIL)\n""", ""),
    ("C2 identity-saves", AG, """                               f"last save plan; its next request re-reads it", icon=Icons.HEAD_FAIL)\n                    return False\n""", """                               f"last save plan; its next request re-reads it", icon=Icons.HEAD_FAIL)\n"""),
    ("C3 identity-always", AG, """                if ident != hs.get("identity"):\n                    # round 2""", """                if True:\n                    # round 2"""),
    ("C4 no-plan-silent", AG, """                pretty_log("Conversation", f"not saved before {why} — no save plan yet; its next request "\n                           f"re-reads it", icon=Icons.HEAD_FAIL)                      # §4TJ: was silent\n""", ""),
    ("C5 slot-silent", AG, """                pretty_log("Conversation", f"not saved before {why} — llama did not report the slot; its next "\n                           f"request re-reads it", icon=Icons.HEAD_FAIL)              # §4TJ: was silent\n""", ""),
    ("D23 append-unlocks", FS, "not unread_rewrite_refusal(path, path.name, count_in_progress=False):", "not unread_rewrite_refusal(path, path.name):"),
    ("D24 writes-lose-parts", FS, "    if count_in_progress and k in _IN_PROGRESS:\n        return \"\"\n    now = _file_stamp(path)", "    now = _file_stamp(path)"),
    ("D25 record-kept-on-allow", FS, "    if not lost:\n        _DELETE_ASKED.pop(key, None)\n        return \"\"", "    if not lost:\n        return \"\""),
    ("D26 dir-text", FS, "Read those files whole first (operation='read' on each), then delete the folder.", "Read it whole first (operation='read'), then delete it."),
    ("C6 no-owner-logged", "src/ghost_agent/core/agent.py", "            if not owner:\n                return False\n            if not isinstance(hs, dict):", "            if not isinstance(hs, dict):"),
    ("D27 birth-after-mutation", FS, "    if _progress_key(path) in _MUTATED:\n        return True\n", ""),
    ("D28 not-recorded", FS, "        _MUTATED[k] = True\n", "        pass\n"),
    ("D29 own-file-recorded", FS, "        wrec[\"stamp\"] = _file_stamp(path)\n    else:\n", "        wrec[\"stamp\"] = _file_stamp(path)\n    if True:\n"),
    ("D30 not-forgotten", FS, "for table in (_READ, _WRITTEN, _IN_PROGRESS, _MUTATED):", "for table in (_READ, _WRITTEN, _IN_PROGRESS):"),
    ("D31 mutated-unbounded", FS, "            for old in list(_MUTATED)[: _IN_PROGRESS_MAX // 4]:\n                _MUTATED.pop(old, None)\n", "            pass\n"),
    ("D13 conversation-not-keyed", FS, '    key = (str(slot_session_context.get() or ""), str(path.resolve()))', '    key = ("x" if slot_session_context.get() else "", str(path.resolve()))'),
    ("D14 record-kept-after-pass", FS, "            and now - asked[2] <= _DELETE_ASK_TTL_S):\n        _DELETE_ASKED.pop(key, None)\n", "            and now - asked[2] <= _DELETE_ASK_TTL_S):\n        pass\n"),
    ("D15 no-record-unattended", FS, "    if key[0]:\n        if len(_DELETE_ASKED) >= _IN_PROGRESS_MAX", "    if True:\n        if len(_DELETE_ASKED) >= _IN_PROGRESS_MAX"),
    ("D16 unbounded", FS, "            for old in list(_DELETE_ASKED)[: _IN_PROGRESS_MAX // 4]:\n                _DELETE_ASKED.pop(old, None)\n", "            pass\n"),
    ("D17 ask-unattended", FS, '           if key[0] else "leave it and say so in your report")', '           if True else "leave it and say so in your report")'),
    ("D18 code", FS, 'world_changed=False, reason_code="unread_delete")', 'world_changed=False, reason_code="unread_rewrite")'),
]


def _tests():
    r = subprocess.run("grep -rlE 'tool_delete_file|operation=.delete.|_side_main_dispatch|head_saves|rewarm_main_prefix_loop' tests --include='test_*.py'",
                       shell=True, cwd=ROOT, capture_output=True, text=True)
    return sorted(set(r.stdout.split()))


PINS = ["tests/test_4tj_saves_back_and_unread_delete.py"]


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
