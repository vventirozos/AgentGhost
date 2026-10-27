#!/usr/bin/env python3
"""§4PM battery — the census's new instruments (core/failure_census.py,
core/owner_seeds.py, the agent hooks), 2026-10-16. Same harness as §4PL: a
COPY of the tree (`BATTERY_ROOT`), one mutant per run, fast pins then the
full selection for survivors; the no-op control must SURVIVE, the known-bad
control must DIE.

    BATTERY_ROOT=/path/to/copy python3 scripts/mutate_4pm.py [all|rerun N1 …]
"""
import filecmp, glob, os, shutil, signal, subprocess, sys

ROOT = os.environ.get("BATTERY_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FC = "src/ghost_agent/core/failure_census.py"
OS_ = "src/ghost_agent/core/owner_seeds.py"
AG = "src/ghost_agent/core/agent.py"
PINS = ["tests/test_4pm_observability.py", "tests/test_4pl_failure_census.py"]
_FULL_PAT = r"failure_census\|owner_seeds\|judge_reactions\|_census_fast"

MUT = [
    ("CONTROL no-op (must SURVIVE)", FC, "def _tname(t: dict) -> str:", "# ctl\ndef _tname(t: dict) -> str:"),
    ("CONTROL known-bad (must DIE): every call ok", FC, "def tool_class(t: dict) -> str:\n", "def tool_class(t: dict) -> str:\n    return \"ok\"\n"),
    ("N1 idempotency text not a guard", FC, "    r\"SYSTEM IDEMPOTENCY\\b|SYSTEM BLOCK\\b", "    r\"SYSTEM BLOCK\\b"),
    ("N3 banner prefix not stripped", FC, "    r\"^\\s*(?:\\[FAILURE BANNER\\]\\s*)?(?:ERROR:\\s*)?(?:\"", "    r\"^\\s*(?:ERROR:\\s*)?(?:\""),
    ("N8 guards reset the run", FC, "        if c in (\"guard\", \"asked_user\"):\n            continue\n", ""),
    ("N9 live classes ignored", FC, "        classes = (list(live_classes) if isinstance(live_classes, list)\n                   else [tool_class(tc) for tc in tools])", "        classes = [tool_class(tc) for tc in tools]"),
    ("N11 never-run build is ok", FC, "            out = \"last_run_failed\" if failed else (\"never_run\" if never else \"ran_ok\")", "            out = \"last_run_failed\" if failed else \"ran_ok\""),
    ("N12 research by status not evidence", FC, "                if a.consulted:\n", "                if False:\n"),
    ("N13 all-guard is ok", FC, "    if all(c == \"guard\" for c in classes):\n        return \"blocked\"", "    if False:\n        return \"blocked\""),
    ("N14 kind order: memory before image", FC, "    if imgs or any(n in _IMAGE_TOOLS for n in names):", "    if (imgs or any(n in _IMAGE_TOOLS for n in names)) and not any(n in _MEMORY_TOOLS for n in names):"),
    ("N15 live row drops classes", FC, "                row[\"classes\"] = [tool_class(t) for t in rows]", "                pass"),
    ("N16 live task not written", FC, "            row[\"task\"] = {\"kind\": to[\"kind\"], \"outcome\": to[\"outcome\"]}", "            pass"),
    ("N17 fast row for probes", FC, "    try:\n        if not _owner_request(req_id):\n            return False\n        return _append({\"kind\": \"fast\"", "    try:\n        return _append({\"kind\": \"fast\""),
    ("N19 no-tool judge cache not read", FC, "            reactions.setdefault(k, v)", "            pass"),
    ("N20 no-tool cache overrides tool cache", FC, "            reactions.setdefault(k, v)", "            reactions[k] = v"),
    ("N21 cause unparsed", FC, "    return (m.group(1) or m.group(2) or \"\").strip().rstrip(\".)_ \") or \"interrupted\"", "    return \"interrupted\""),
    ("N22 gold packets carry the id", FC, "            \"id\": gid, \"asked_on\"", "            \"id\": gid, \"tid\": str(t.get(\"id\")), \"asked_on\""),
    ("N23 gold draw unseeded", FC, "    pick = random.Random(today.isoformat()).sample(turns, min(n, len(turns)))", "    pick = random.sample(turns, min(n, len(turns)))"),
    ("N24 bad grades ingested", FC, "        if not tid or g not in GRADES:", "        if not tid:"),
    ("N25 gold never due", FC, "    out = {\"last_graded\": last, \"due\": age is None or age > GOLD_EVERY_DAYS,", "    out = {\"last_graded\": last, \"due\": age is None,"),
    ("N26 guard counts wrong tool", FC, "                     \"guards\": [n for n, c in zip(cnames, classes) if c == \"guard\"],", "                     \"guards\": [n for n, c in zip(cnames, classes) if c == \"failed\"],"),
    ("N27 no-tool judge writes tool cache", OS_, "    return Path(home_system) / \"selfplay\" / (REACTIONS_NOTOOL_FILENAME if no_tool else REACTIONS_FILENAME)", "    return Path(home_system) / \"selfplay\" / REACTIONS_FILENAME"),
    ("N28 no-tool judge judges tool turns", OS_, "bool(getattr(t, \"tool_calls\", None)) == no_tool:", "not getattr(t, \"tool_calls\", None):"),
    ("N30 command reply writes nothing", AG, "                    await self._census_fast(req_id, _sys_reply, kind=\"command\")", "                    pass"),
    ("N5 designed stop is a failure", FC, "    if is_designed_stop(c):                 # before anything", "    if False:                 # before anything"),
    ("N7 live status ignored", FC, "    if st is not None:\n        # a live row", "    if False:\n        # a live row"),
    ("N18 fast rows out of window counted", FC, "            fast += cutoff < str(r.get(\"at\") or \"\")[:10] <= today.isoformat()", "            fast += 1"),
    ("N29 hook drops the loop rows", AG, "                                tools=list(tools_run or ()), delivered_reply=_shown)", "                                tools=(), delivered_reply=_shown)"),
    ("N32 synthetic short-circuit back", FC, "    if is_designed_stop(c):                 # before anything", "    if t.get(\"_synthetic\"):\n        return \"guard\"\n    if is_designed_stop(c):                 # before anything"),
    ("N34 kind from real rows only", FC, "    names = [_tname(t) for t in rows]", "    names = [_tname(t) for t in real]"),
    ("N35 delivered by bare filename", FC, "any(f\"/api/download/{i}\" in str(reply or \"\") for i in imgs)", "any(i in str(reply or \"\") for i in imgs)"),
    ("N36 delivered view ignored", FC, "            to = task_outcome(rows, reply if delivered_reply is None else delivered_reply)", "            to = task_outcome(rows, reply)"),
    ("N37 hook skips the image note", AG, "                    _shown = (reply or \"\") + _unshown_image_note(reply or \"\", tools_run,\n                                                                 getattr(self.context, \"sandbox_dir\", None))", "                    _shown = reply or \"\""),
    ("N38 gold redrawn", FC, "    if keyp.exists() or d.exists():\n", "    if False:\n"),
    ("N40 key inside the folder", FC, "    keyp = base / GOLD_KEYS_REL / f\"{today.isoformat()}.json\"", "    keyp = base / GOLD_DIR_REL / today.isoformat() / \"key.json\""),
    ("N41 machine lines reach packets", FC, "    for rx in _BLIND_STRIP:\n        reply = rx.sub(\"\", reply)\n", ""),
    ("N42 yours and judge folded", FC, "                \"yours\": neg and r[\"source\"] == \"human\",", "                \"yours\": neg,"),
    ("N43 fast window unbounded above", FC, "            fast += cutoff < str(r.get(\"at\") or \"\")[:10] <= today.isoformat()", "            fast += cutoff < str(r.get(\"at\") or \"\")[:10]"),
    ("N44 search cap text not a guard", FC, "|SYSTEM SEARCH CAP\\b|SYSTEM PREFLIGHT\\b", "|SYSTEM PREFLIGHT\\b"),
    ("N33 guard codes ignored live", FC, "        if rc in GUARD_REASONS or (st in (\"rejected\", \"failed\") and _GUARD_HEAD.search(text)):", "        if st in (\"rejected\", \"failed\") and _GUARD_HEAD.search(text):"),
    ("N45 live guard head ignored", FC, "        if rc in GUARD_REASONS or (st in (\"rejected\", \"failed\") and _GUARD_HEAD.search(text)):", "        if rc in GUARD_REASONS:"),
    ("N46 browser_blocked a guard again", FC, "GUARD_REASONS = frozenset({\"idempotency_guard\",", "GUARD_REASONS = frozenset({\"browser_blocked\", \"idempotency_guard\","),
    ("N48 gold rows only in the window", FC, "            \"gold\": gold_agreement(base, all_rows, today),", "            \"gold\": gold_agreement(base, rows, today),"),
    ("N49 sample days not read", FC, "    extra_days = [d for d in _graded_sample_days(base) if d not in window]", "    extra_days = []"),
    ("N50 typo wipes grades", FC, "    if not good:                       # r2 m2", "    if False:                       # r2 m2"),
    ("N51 empty window locks the date", FC, "    if not turns:\n        return {\"error\": \"no delivered owner turns", "    if False:\n        return {\"error\": \"no delivered owner turns"),
    ("N52 quote strip end-anchored", FC, "sources read: [^\\n]*\\._(?=\\n|$)\")", "sources read: [^\\n]*\\._$\")"),
    ("N6 any rejection is a guard", FC, "        if st == \"rejected\":\n            return \"failed\"", "        if st == \"rejected\":\n            return \"guard\""),
    ("N53 ledger lock dropped", FC, "    with _LEDGER_LOCK:\n        return _append_locked(row, base)", "    return _append_locked(row, base)"),
    ("N54 owner confirm read as a guard (stored)", FC, "    if _ASK_HEAD.search(text):\n        return \"asked_user\"\n    if _GUARD_HEAD", "    if _GUARD_HEAD"),
    ("N55 owner confirm codes ignored live", FC, "        if rc in ASK_REASONS or (st in (\"rejected\", \"failed\") and _ASK_HEAD.search(text)):", "        if False:"),
    ("N56 partial not a failure", FC, "        run = run + 1 if c in (\"failed\", \"partial\") else 0", "        run = run + 1 if c == \"failed\" else 0"),
    ("N57 ending on a question ignored", FC, "    if classes[-1] == \"asked_user\":\n        return \"asked_user\"        # r3 M2", "    if False:\n        return \"asked_user\"        # r3 M2"),
    ("N58 per-signal false alarms dropped", FC, "    out[\"false_alarms_on_right\"].update({k: sum(1 for r in right if sig(r)[k]) for k in keys})   # r3 M4", ""),
    ("N59 packet tool results dropped", FC, "\"returned\": _redact_packet(str(c.get(\"result\") or c.get(\"error\") or \"\"))[:1500]}", "\"returned\": \"\"}"),
    ("N60 packet conversation dropped", FC, "            \"earlier_turns\": _earlier(t),", "            \"earlier_turns\": [],"),
    ("N61 stash for every turn", AG, "                        if not _census_owner(req_id):", "                        if False:"),
    ("N62 packets not redacted", FC, "        return redact_profile_values(redact_text(str(text or \"\")))", "        return str(text or \"\")"),
    ("N63 second pass judges tool turns", AG, "getattr(ctx.args, 'model', 'default'), no_tool=True, cap=_nt_cap),", "getattr(ctx.args, 'model', 'default'), cap=_nt_cap),"),
    ("N64 second pass ignores the owner stop", AG, "        if getattr(self, \"_last_idle_stop\", \"\") != \"owner\":\n            try:\n                from .owner_seeds import judge_reactions as _judge_reactions_nt", "        if True:\n            try:\n                from .owner_seeds import judge_reactions as _judge_reactions_nt"),
    ("N31 fast helper writes nothing", AG, "            await _wf(asyncio.to_thread(_fc.record_fast, str(req_id or \"\"), str(reply or \"\")), 2.0)", "            pass"),
    ("N39 gold window includes today", FC, "    turns, _ = _owner_turns(base / \"system\", _window(today - datetime.timedelta(days=1), days))", "    turns, _ = _owner_turns(base / \"system\", _window(today, days))"),
    ("N47 confirm-wait wording narrowed", FC, "_ASK_HEAD = re.compile(r\"^\\s*(?:\\[FAILURE BANNER\\]\\s*)?(?:ERROR:\\s*)?NOT (?:deleted|done|executed): \"", "_ASK_HEAD = re.compile(r\"^\\s*(?:\\[FAILURE BANNER\\]\\s*)?(?:ERROR:\\s*)?NOT (?:deleted|done): \""),
]

def _grep(pat):
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return {os.path.relpath(t, ROOT) for t in out}


def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY="x")
    env.pop("FORCE_COLOR", None)
    try:
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x",
                            "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]


def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:])
        batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    fast = sorted(set(PINS) | {"tests/test_4mt_selfplay_useful.py"})
    full = sorted(_grep(_FULL_PAT) | set(PINS))
    files = sorted({m[1] for m in batch})
    backups = {f: f + ".4pm.bak" for f in files}
    for f, b in backups.items():
        shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))

    def restore(*_):
        for f, b in backups.items():
            shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
                       capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
    print(f"ROOT={ROOT}; fast {len(fast)} files, full {len(full)} files; {len(batch)} mutants", flush=True)
    try:
        for label, f, old, new in batch:
            restore()
            path = os.path.join(ROOT, f)
            src = open(path).read()
            if src.count(old) != 1:
                print(f"SKIP     {label}: anchor count {src.count(old)}", flush=True)
                continue
            open(path, "w").write(src.replace(old, new))
            if subprocess.run([PY, "-m", "py_compile", path], capture_output=True).returncode != 0:
                print(f"NOCOMP   {label}", flush=True)
                continue
            sel = full if label.startswith("CONTROL no-op") else fast
            rc, tail = run_tests(sel)
            tier = f"fast {len(sel)}"
            if rc == 0 and sel is not full:
                rc, tail = run_tests(full)
                tier += " → full"
            status = "SURVIVED" if rc == 0 else "KILLED"
            killed += rc != 0
            survived += rc == 0
            print(f"{status:8} {label} — [{tier}] {tail}", flush=True)
    finally:
        restore()
        ok = all(filecmp.cmp(os.path.join(ROOT, f), os.path.join(ROOT, b), shallow=False) for f, b in backups.items())
        for b in backups.values():
            os.remove(os.path.join(ROOT, b))
        print(f"RESTORED identical={ok} killed={killed} survived={survived}", flush=True)


if __name__ == "__main__":
    main()
