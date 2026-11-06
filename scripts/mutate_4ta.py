#!/usr/bin/env python3
"""§4TA battery — the macro engine is removed: the `youtube_transcribe` built-in that replaced the one real
macro (its definition, its handler, the tables that name it, containment), the acquired-skill listing and
creation paths the cut touched, the request kinds that book a skill run (moved out of the engine), the
loop-yield / learning report rows, and the unknown-tool path an old macro name now takes. Coverage-selected
tests in parallel tree copies (scripts/battery_cov.py); the real src/ is never mutated. No-op control must
SURVIVE, the known-bad control must DIE.
    python3 scripts/mutate_4ta.py [check | -j6 all | rerun Y1 I2 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RG = "src/ghost_agent/tools/registry.py"
AS = "src/ghost_agent/tools/acquired_skills.py"
AG = "src/ghost_agent/core/agent.py"
IS = "src/ghost_agent/core/isolation.py"
ST = "src/ghost_agent/core/strikes.py"
LV = "src/ghost_agent/core/liveness.py"
LH = "src/ghost_agent/core/learning_health.py"
PV = "src/ghost_agent/utils/provenance.py"
FC = "src/ghost_agent/core/failure_census.py"

# anchors used by more than one mutant (each is ONE exact place in its file)
_CALL = '        return await tools[YOUTUBE_TRANSCRIBE_INNER_TOOL](action="transcribe", filename=url)'
_NO_URL = "        if not isinstance(url, str) or not url.strip():"
_URL_PROP = '                "url": {"type": "string", "description": "The YouTube link of the video to transcribe."},'
_INNER = 'YOUTUBE_TRANSCRIBE_INNER_TOOL = "knowledge_base"'
_CONTAIN = "        if YOUTUBE_TRANSCRIBE_INNER_TOOL in forbidden:"
_DEFS_REFRESH = ('                        _cname == "manage_skills"\n'
                 '                        and str(t_args.get("action") or "").strip().lower() == "delete"\n'
                 "                    ):")
_UNKNOWN = ('                        _unknown_tool_message(fname, self.available_tools), world_changed=False,\n'
            '                        reason_code="unknown_tool")}')
_UNKNOWN_STRIKE = '                    execution_failure_count += 1\n                    _strike_synthetic(fname, "unknown_tool")'
_BOOK_KINDS = '        if _k in ("probe", "member", "test"):'
_BOOK_SIM = '            return not str(request_id_context.get() or "").startswith("sim-")'
_SKILL_ROW = '    YieldProbe("skills.acquired",'
_SKILL_STORE = '    data = _read_json(home, "system/memory/acquired_skills/skills_registry.json")'

MUT = [
    ("CONTROL no-op", RG, _INNER, _INNER + "  # no-op"),
    ("CONTROL known-bad", RG, _CALL, '        return "nothing"'),
    # ── the youtube_transcribe built-in: advertised ──
    ("Y1 not-advertised", RG, "TOOL_DEFINITIONS.append(YOUTUBE_TRANSCRIBE_TOOL_DEFINITION)\n", ""),
    ("Y2 url-not-required", RG, _URL_PROP + '\n            },\n            "required": ["url"],',
     _URL_PROP + '\n            },\n            "required": [],'),
    ("Y3 parameter-named-link", RG, _URL_PROP, _URL_PROP.replace('"url": {', '"link": {')),
    ("Y4 description-reworded", RG, '''            "knowledge_base(action='transcript'). Input: url."''',
     '''            "knowledge_base(action='transcript'). Input: url. (1 steps)"'''),
    # ── …and dispatched ──
    ("Y5 not-dispatchable", RG, '        "youtube_transcribe": _youtube_transcribe,\n', ""),
    ("Y6 wrong-action", RG, _CALL, _CALL.replace('action="transcribe"', 'action="transcript"')),
    ("Y7 url-passed-as-url", RG, _CALL, _CALL.replace("filename=url", "url=url")),
    ("Y8 url-not-passed", RG, _CALL, _CALL.replace("filename=url", 'filename=""')),
    ("Y9 result-cut-like-a-macro-step", RG, _CALL,
     _CALL.replace("return await ", "return str(await ").replace("filename=url)", "filename=url))[:4000]")),
    ("Y10 raw-tool-past-the-egress-guard", RG, _CALL,
     _CALL.replace("tools[YOUTUBE_TRANSCRIBE_INNER_TOOL](", "tools[YOUTUBE_TRANSCRIBE_INNER_TOOL].__wrapped__(")),
    ("Y11 extra-arguments-forwarded", RG, _CALL, _CALL.replace("filename=url)", "filename=url, **_ignored)")),
    ("Y12 inner-tool-is-another", RG, _INNER, 'YOUTUBE_TRANSCRIBE_INNER_TOOL = "recall"'),
    # ── …and refused without a link ──
    # ── review fixes ──
    ("H1 removed-name-healed-onto-a-neighbour", AG, "        if normalised in _REMOVED_TOOL_NAMES:\n            return None\n", ""),
    ("H2 no-removed-names", AG, '_REMOVED_TOOL_NAMES = frozenset({"managecomposedskills", "managecomposedskill",\n                                 "learnskill", "listlessons"})', "_REMOVED_TOOL_NAMES = frozenset()"),
    ("S1 transcript-condensed-by-the-shield", AG, '                                                                 "youtube_transcribe"]:   # §4TA', '                                                                 "youtube_transcribe_"]:   # §4TA'),
    ("O1 skill-name-not-registered-as-outside-content", RG, "                        register_opaque(name)\n", ""),
    ("O2 runner-not-marked", RG, "                        _run._ghost_opaque = True ", "                        _run._ghost_opaque = False "),
    ("O3 provenance-ignores-registered-names", PV, "    if name in extra_tools or name in _OPAQUE:", "    if name in extra_tools:"),
    ("O4 provenance-ignores-the-callers-hint", PV, "    if name in extra_tools or name in _OPAQUE:", "    if name in _OPAQUE:"),
    ("O5 dispatch-ignores-the-runners-mark", AG, '(self.available_tools or {}).get(fname), "_ghost_opaque", False)', '(self.available_tools or {}).get(fname), "_ghost_opaque_", False)'),
    ("B9 a-members-run-is-booked", AS, '("probe", "member", "test")', '("probe", "test")'),
    ("Y13 no-url-guard-off", RG, _NO_URL, "        if False:"),
    ("Y14 non-text-url-accepted", RG, _NO_URL, "        if not url or not str(url).strip():"),
    # ── the tables that name it ──
    ("T1 a-replay-may-call-it", IS, '    "youtube_transcribe", "news_headlines",\n', '    "news_headlines",\n'),
    ("T2 a-transcript-is-not-outside-content", PV,
     '    "darkweb_research", "browser", "fact_check", "youtube_transcribe",\n',
     '    "darkweb_research", "browser", "fact_check",\n'),
    ("T3 a-transcript-is-not-a-lookup", FC, '"system_utility", "youtube_transcribe",\n', '"system_utility",\n'),
    ("T4 members-get-it", AG, '    "abort_attempt",\n})', '    "abort_attempt", "youtube_transcribe",\n})'),   # anchor follows §4TB (the `replan` tool is gone)
    ("T5 two-calls-collapse-to-one", AG, '    "file_system", "knowledge_base",\n})',
     '    "file_system", "knowledge_base", "youtube_transcribe",\n})'),
    # ── containment: forbidden wherever the tool it calls is ──
    ("I1 kept-where-knowledge-base-is-forbidden", IS, _CONTAIN, "        if False:"),
    ("I2 forbidden-everywhere", IS, _CONTAIN, "        if True:"),
    ("I3 fork-copies-the-old-registry", IS, '            for sub in ("acquired_skills",):',
     '            for sub in ("acquired_skills", "composed_skills"):'),
    # ── the removed tool is nothing special; an old macro name is an unknown tool ──
    ("N1 removed-tool-exempt-from-the-loop-breaker", ST, 'READWRITE_LOOP_TOOLS = frozenset({\n    "manage_tasks",',
     'READWRITE_LOOP_TOOLS = frozenset({\n    "manage_composed_skills",\n    "manage_tasks",'),
    ("N2 any-tool-resets-the-preflight-guard", AG,
     "        return not _run_is_read_call(t_args)\n    return False", "        return not _run_is_read_call(t_args)\n    return True"),
    ("N3 padded-delete-leaves-the-tool-list-stale", AG, _DEFS_REFRESH, _DEFS_REFRESH.replace(".strip().lower()", ".lower()")),
    ("U1 unknown-tool-not-declared", AG, _UNKNOWN, _UNKNOWN.replace('reason_code="unknown_tool"', 'reason_code="invocation_error"')),
    ("U2 unknown-tool-is-no-strike", AG, _UNKNOWN_STRIKE, '                    _strike_synthetic(fname, "unknown_tool")'),
    ("U3 unknown-tool-reply-lists-nothing", AG,
     """    return f"Error: Unknown tool '{n}'. Available tools: {', '.join(names)}.\"""",
     """    return f"Error: Unknown tool '{n}'.\""""),
    # ── acquired skills: listing and creation ──
    ("K1 list-mentions-composed-skills", AS, '            body = "No custom skills have been acquired yet."',
     '            body = "No custom skills have been acquired or composed yet."'),
    ("K2 a-built-in-name-is-accepted", AS, "        if name in _builtin:", "        if False:"),
    ("K3 stored-skill-replaces-a-built-in-handler", RG, "                    if skill_name in _BUILTIN_TOOL_NAMES:",
     "                    if False:"),
    ("K4 stored-skill-doubles-a-built-in-definition", RG, "                    if skill_name in _existing_names:",
     "                    if False:"),
    ("K5 manage-skills-description-names-macros", RG, "the custom tools created via `create_skill` — and is the AUTHORITATIVE",
     "acquired tools (created via `create_skill`) AND composed macros — and is the AUTHORITATIVE"),
    # ── who books a skill run (moved here from the engine) ──
    ("B1 a-probe-books", AS, _BOOK_KINDS, '        if _k in ("member", "test"):'),
    ("B2 self-play-books", AS, _BOOK_SIM, "            return True"),
    ("B3 the-owners-scheduled-work-does-not-book", AS, _BOOK_SIM, "            return False"),
    ("B4 telemetry-not-gated", AS, "        if not _usage_is_bookable():\n            return\n", ""),
    # ── the reports ──
    ("L1 the-skill-row-reads-the-old-macro-registry", LV, _SKILL_STORE,
     '    data = _read_json(home, "system/memory/composed_skills/composed_skills.json")'),
    ("L2 a-macro-row-is-back", LV, _SKILL_ROW, '    YieldProbe("macros.auto_mined",'),
    ("L3 learning-report-reads-the-wrong-home", LH, '        lines.append("\\n" + _yield_render(Path(memory_dir).parent.parent))',
     '        lines.append("\\n" + _yield_render(Path(memory_dir).parent))'),
]
PINS = ["tests/test_4ta_macro_engine_removed.py", "tests/test_4ta_kept_guards.py"]


def _tests():
    pat = (r"youtube_transcribe\|manage_skills\|create_skill\|acquired_skills\|restrict_tool_surface\|"
           r"yield_all\|render_yield\|learning_health\|_usage_is_bookable\|log_telemetry\|_call_mutated_world\|"
           r"is_readwrite_loop_exempt\|_unknown_tool_message\|Unknown tool\|UNTRUSTED_SOURCE_TOOLS\|"
           r"_LOOKUP_TOOLS\|invalidate_tool_defs\|_COLLAPSE_READSAFE\|_MEMBER_ALLOWED_TOOLS\|"
           r"get_available_tools\|get_active_tool_definitions\|_fork_memory_dir\|REPLAY_FORBIDDEN_TOOLS")
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
