#!/usr/bin/env python3
"""§4TD battery — the lesson store is removed. Concentrated on what the removal could weaken without anyone
noticing: the one fail-closed flag every simulation guard is re-based on (`core/isolation.py`: the strict read,
the stamp at both places that build a simulated context, every guard in `core/agent.py` in both directions), the
stale calls of the two removed tools (healer and the ordinary unknown-tool reply), the lesson twins that stay on
disk (never served: `memory/vector.py`), the thumb's route (label, the rule hint, the mood hook, the banner drop),
the health and learning views, the acquired-skills paths the cut touched (registry, `skills.acquired` yield),
the lists it edited (self-report, bookkeeping, pre-tool bookkeeping, read-safe collapse, mutating), the hydration
tiers left in the bus, the population rules moved out of the removed module, and, after the review, the pins that
went with lesson tests and the deploy purge (`scripts/lesson_purge_4td.py`). Coverage-selected tests in
parallel tree copies (scripts/battery_cov.py); the real src/ is never mutated. No-op control must SURVIVE, the
known-bad control must DIE.
    python3 scripts/mutate_4td.py [check | -j6 all | rerun D1 E4 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = "src/ghost_agent/core/agent.py"
IS = "src/ghost_agent/core/isolation.py"
CL = "src/ghost_agent/core/coding_loop.py"
FB = "src/ghost_agent/core/feedback.py"
BU = "src/ghost_agent/core/bus.py"
VE = "src/ghost_agent/memory/vector.py"
PO = "src/ghost_agent/memory/population.py"
RT = "src/ghost_agent/api/routes.py"
HI = "src/ghost_agent/tools/health_internals.py"
IT = "src/ghost_agent/tools/introspect.py"
RG = "src/ghost_agent/tools/registry.py"
RQ = "src/ghost_agent/memory/request_text.py"
PG = "scripts/lesson_purge_4td.py"

_READ = "        return getattr(context, SIMULATED_ATTR, False) is True\n"
_FAIL = "    except Exception:  # noqa: BLE001\n        return True\n\n\nclass SafeCollection"
_THUMB = ("                    if (not unchanged and is_owner_label_source(source)\n"
          "                            and turn_is_recent_enough(traj)):\n")
_NEG = '        if signal == "negative" and not result.get("unchanged"):\n'

MUT = [
    ("CONTROL no-op", VE, 'RETIRED_ROW_TYPES = ("skill",)\n', 'RETIRED_ROW_TYPES = ("skill",)  # same\n'),
    ("CONTROL known-bad", IS, _READ, "        return False\n"),

    # ── the flag: the strict read, fail closed, the stamp
    ("F1 any truthy value counts as simulated", IS, _READ,
     "        return bool(getattr(context, SIMULATED_ATTR, False))\n"),
    ("F2 a missing flag reads as simulated", IS, _READ,
     "        return getattr(context, SIMULATED_ATTR, True) is True\n"),
    ("F3 an unreadable flag reads as live", IS, _FAIL,
     "    except Exception:  # noqa: BLE001\n        return False\n\n\nclass SafeCollection"),
    ("F4 no context reads as simulated", IS, "    if context is None:\n        return False\n    try:\n        return getattr(context, SIMULATED_ATTR",
     "    if context is None:\n        return True\n    try:\n        return getattr(context, SIMULATED_ATTR"),
    ("F5 mark_simulated does not read the flag back", IS, "        stamped = getattr(context, SIMULATED_ATTR, None) is True\n",
     "        stamped = True\n"),
    ("F6 mark_simulated stamps False", IS, "        setattr(context, SIMULATED_ATTR, True)\n",
     "        setattr(context, SIMULATED_ATTR, False)\n"),
    ("F7 replay copy not stamped", IS, "        mark_simulated(iso)\n        iso.sandbox_dir = workspace",
     "        iso.sandbox_dir = workspace"),
    ("F8 coding leaf not stamped", CL, "    mark_simulated(iso)\n", ""),
    ("F9 a live context starts simulated", AG, "        self.simulated = False\n", "        self.simulated = True\n"),

    # ── every guard, flipped and dropped
    ("G1 turn_origin: simulated turn is the owner's", AG, '    if is_simulated_context(context):\n        return "sim"\n', ""),
    ("G2 turn_origin: flipped", AG, '    if is_simulated_context(context):\n        return "sim"\n',
     '    if not is_simulated_context(context):\n        return "sim"\n'),
    ("G3 verifier verdict: simulated turn traced", AG,
     "        _is_sim = is_simulated_context(self.context)\n        _trace = None if _is_sim else {",
     "        _is_sim = False\n        _trace = None if _is_sim else {"),
    ("G4 verifier verdict: live turn untraced", AG,
     "        _is_sim = is_simulated_context(self.context)\n        _trace = None if _is_sim else {",
     "        _is_sim = not is_simulated_context(self.context)\n        _trace = None if _is_sim else {"),
    ("G5 pending corrections: simulated writes the real file", AG, "                or is_simulated_context(ctx)):", "                or False):"),
    ("G6 pending corrections: live gets no file", AG, "                or is_simulated_context(ctx)):", "                or not is_simulated_context(ctx)):"),
    ("G7 solution short-circuit: guard dropped", AG, "                            _is_sim = is_simulated_context(self.context)",
     "                            _is_sim = False"),
    ("G8 solution short-circuit: flipped", AG, "                            _is_sim = is_simulated_context(self.context)",
     "                            _is_sim = not is_simulated_context(self.context)"),
    ("G9 finalize: never a simulation", AG, "        is_simulation = is_simulated_context(self.context)", "        is_simulation = False"),
    ("G11 Perfect-It runs in a simulation", AG, "        if not is_simulation and tools_run_this_turn and heavy_tools_used",
     "        if tools_run_this_turn and heavy_tools_used"),
    ("G12 Perfect-It only in a simulation", AG, "        if not is_simulation and tools_run_this_turn and heavy_tools_used",
     "        if is_simulation and tools_run_this_turn and heavy_tools_used"),
    ("G13 notify backstop runs in a simulation", AG, "                    and not is_simulation and not _promise_active):", "                    and not _promise_active):"),
    ("G14 notify backstop only in a simulation", AG, "                    and not is_simulation and not _promise_active):",
     "                    and is_simulation and not _promise_active):"),
    ("G15 stored notify fires in a simulation", AG, "            if _np_pid and not is_simulation:", "            if _np_pid:"),
    ("G16 stored notify only in a simulation", AG, "            if _np_pid and not is_simulation:", "            if _np_pid and is_simulation:"),
    ("G17 notify capture runs in a simulation", AG, "                            and not is_simulated_context(self.context)):", "                            ):"),
    ("G18 notify capture only in a simulation", AG, "                            and not is_simulated_context(self.context)):",
     "                            and is_simulated_context(self.context)):"),
    ("G19 head check runs in a simulation", AG, "                        _iso = (is_simulated_context(self.context)", "                        _iso = (False"),
    ("G21 stream tail: never simulated", AG, "                    _sim = is_simulated_context(self.context)", "                    _sim = False"),
    ("G23 stream backstop runs in a simulation", AG, "                            and not _sim and not _promise_active):", "                            and not _promise_active):"),
    ("G24 stream stored notify fires in a simulation", AG, "                    if _np_pid and not _sim:", "                    if _np_pid:"),

    # ── the two removed tools: stale calls
    ("T1 learnskill healed again", AG, '"managecomposedskill",\n                                 "learnskill", "listlessons"})',
     '"managecomposedskill",\n                                 "listlessons"})'),
    ("T2 listlessons healed again", AG, '"managecomposedskill",\n                                 "learnskill", "listlessons"})',
     '"managecomposedskill",\n                                 "learnskill"})'),
    ("T3 removed names not refused", AG, "        if normalised in _REMOVED_TOOL_NAMES:\n            return None\n", ""),
    ("T4 stale call booked as another reason", AG, '                        reason_code="unknown_tool")}', '                        reason_code="tool_error")}'),

    # ── the lists the cut edited
    ("L1 self-report list loses introspect", AG, '_SELF_REPORT_TOOLS = frozenset({"introspect"})', "_SELF_REPORT_TOOLS = frozenset()"),
    ("Z1 the nudge offers the rule command on a public surface", AG, 'if _nudge_kind() == "owner" and not reply_is_public() else "")', 'if _nudge_kind() == "owner" else "")'),
    ("Z2 the placeholder is stored as a rule", "src/ghost_agent/core/owner_rules.py", 'if re.fullmatch(r"<[^<>]*>", rule):', 'if False:'),
    ("Z3 a re-exported row overwrites the first text", "scripts/lesson_purge_4td.py", "merged.setdefault(i, row)", "merged[i] = row"),
    ("Z4 the read-back ignores metadata", "scripts/lesson_purge_4td.py", "if back != {i: (v[0], v[1]) for i, v in merged.items()}:", "if {k: v[0] for k, v in back.items()} != {i: v[0] for i, v in merged.items()}:"),
    ("Z5 a failed delete is a traceback", "scripts/lesson_purge_4td.py", "except (OSError, RuntimeError) as exc:", "except ZeroDivisionError as exc:"),
    ("L2 pre-tool bookkeeping loses update_profile", AG, '_PRE_TOOL_BOOKKEEPING_TOOLS = frozenset({"remember", "update_profile", "forget", "reset_all"})',
     '_PRE_TOOL_BOOKKEEPING_TOOLS = frozenset({"remember", "forget", "reset_all"})'),
    ("L3 pre-tool bookkeeping loses forget", AG, '_PRE_TOOL_BOOKKEEPING_TOOLS = frozenset({"remember", "update_profile", "forget", "reset_all"})',
     '_PRE_TOOL_BOOKKEEPING_TOOLS = frozenset({"remember", "update_profile", "reset_all"})'),
    ("L4 bookkeeping loses knowledge_base", AG, '    "abort_attempt",\n    "knowledge_base",  # insert_fact', '    "abort_attempt",\n    # insert_fact'),
    ("L5 update_profile is not mutating", AG, 'is_mutating = _cname in ["execute", "image_generation", "manage_tasks", "update_profile", "vision_analysis"]',
     'is_mutating = _cname in ["execute", "image_generation", "manage_tasks", "vision_analysis"]'),
    ("L6 read-safe collapse loses introspect", AG, '    "darkweb_search", "darkweb_research", "introspect",\n', '    "darkweb_search", "darkweb_research",\n'),

    # ── lesson twins stay on disk and are never served
    ("V1 no retired type", VE, 'RETIRED_ROW_TYPES = ("skill",)\n', "RETIRED_ROW_TYPES = ()\n"),
    ("V2 a row that cannot be read is served", VE, "    except Exception:  # noqa: BLE001 — a row that cannot be read is not served\n        return True\n",
     "    except Exception:  # noqa: BLE001 — a row that cannot be read is not served\n        return False\n"),
    ("V3 no filter stays no filter", VE, '        return {"type": {"$nin": retired}}\n', "        return where\n"),
    ("V4 asking for the retired type is served", VE, "            return None if cond.strip().lower() in RETIRED_ROW_TYPES else where\n", "            return where\n"),
    ("V5 search_advanced does not narrow", VE, "        where = without_retired_rows(where)\n        if where is None:\n            return []\n", ""),
    ("V6 search_advanced does not post-filter", VE, "                if is_retired_row(results['metadatas'][0][i]):\n                    continue\n", ""),
    ("V7 ambient search does not narrow", VE, 'where=without_retired_rows({"type": {"$nin": AMBIENT_EXCLUDED_TYPES}}),', 'where={"type": {"$nin": AMBIENT_EXCLUDED_TYPES}},'),
    ("V8 ambient search does not post-filter", VE, "                        if is_retired_row(meta):      # §4TD", "                        if False:      # §4TD"),
    ("V9 exclusion list loses the retired types", VE, '            return {"type": {"$nin": list(cond["$nin"]) + [t for t in retired if t not in cond["$nin"]]}}\n',
     '            return where\n'),
    ("V10 a compound filter is not wrapped", VE, '    return {"$and": [where, {"type": {"$nin": retired}}]}\n', "    return where\n"),

    # ── the thumb's route
    ("R1 thumbs-down hint on every signal", RT, _NEG, '        if not result.get("unchanged"):\n'),
    ("R2 hint repeats on an unchanged label", RT, _NEG, '        if signal == "negative":\n'),
    ("R3 hint dropped", RT, "            result[\"hint\"] = FEEDBACK_HINT\n", "            pass\n"),
    ("R4 banner not dropped on a label", RT, '                drop(result.get("trajectory_id"))\n', "                pass\n"),
    ("R5 source not passed to the label", RT, 'str(body.get("source") or "")[:80],', '"",'),
    ("R6 thumb moves the mood on an unchanged label", FB, _THUMB,
     "                    if (is_owner_label_source(source)\n                            and turn_is_recent_enough(traj)):\n"),
    ("R7 thumb moves the mood from any source", FB, _THUMB,
     "                    if (not unchanged\n                            and turn_is_recent_enough(traj)):\n"),
    ("R8 thumb moves the mood on an old turn", FB, _THUMB,
     "                    if (not unchanged and is_owner_label_source(source)\n                            ):\n"),

    # ── health and the learning views
    ("H1 health loses the Rules line", HI, '    ("Rules", _rules, True),\n', ""),
    ("H2 a damaged rules file reads as a count", HI, 'if not out.damaged("owner rules", _or.damaged(system_dir) or None):', "if True:"),
    ("H4 overview loses the warnings", IT, '             if l.lstrip().startswith("⚠")]', "             if False]"),
    ("H5 overview shows every line", IT, '             if l.lstrip().startswith("⚠")]', "             ]"),

    # ── acquired skills, untouched by the removal
    ("A1 create_skill runs the other tool", RG, '"create_skill": lambda **kwargs: tool_create_skill(', '"create_skill": lambda **kwargs: tool_manage_skills('),
    ("A2 manage_skills runs the other tool", RG, '"manage_skills": lambda **kwargs: tool_manage_skills(', '"manage_skills": lambda **kwargs: tool_create_skill('),

    # ── the hydration tiers that remain
    ("B1 graph tier lost", BU, "            self._fetch_graph(query),\n", "            _none(),\n"),
    ("B2 experience tier lost", BU, "            self._fetch_experience(scope_request or query) if extras else _none(),\n", "            _none(),\n"),
    ("B3 history tier lost", BU, "            self._fetch_history(scope_request or query) if extras else _none(),\n", "            _none(),\n"),

    # ── the population rules kept from the removed module
    ("P1 a member's write is the owner's", PO, "        if requester_is_member():\n            return ORIGIN_MEMBER\n", "        if requester_is_member():\n            return ORIGIN_USER\n"),
    ("P2 a probe's write is the owner's", PO, "        if is_probe_request_id(rid):\n            return ORIGIN_PROBE\n", "        if is_probe_request_id(rid):\n            return ORIGIN_USER\n"),
    ("P3 idle writes are the owner's", PO, '    if not rid or rid == "SYSTEM":\n        return ORIGIN_AUTO\n', '    if not rid or rid == "SYSTEM":\n        return ORIGIN_USER\n'),
    ("P4 internal request is the owner's", PO, "        if is_internal_request(rid):\n            return ORIGIN_AUTO\n", "        if is_internal_request(rid):\n            return ORIGIN_USER\n"),
    ("P5 a probe earns usage credit", PO, "        if write_origin() in (ORIGIN_PROBE, ORIGIN_MEMBER):\n", "        if write_origin() in (ORIGIN_MEMBER,):\n"),
    ("P6 a member earns usage credit", PO, "        if write_origin() in (ORIGIN_PROBE, ORIGIN_MEMBER):\n", "        if write_origin() in (ORIGIN_PROBE,):\n"),
    ("P7 an internal turn earns usage credit", PO, '        return bool(is_internal_request(str(request_id_context.get() or "")))\n', "        return False\n"),
    ("P8 a member's trajectory teaches", PO, '        if _role == "member":\n            return False\n', ""),
    ("P9 a probe's reflection copy teaches", PO, '        if str(extra.get("source_task_kind") or "") == "probe":     # r2: a probe\'s reflection copy\n            return False\n', ""),
    ("P10 an interrupted turn teaches", PO, '        if "[TURN_INTERRUPTED]" in str(getattr(traj, "final_response", "") or ""):\n            return False\n', ""),
    ("P11 non-request rows teach a consumer", PO, '            if str(getattr(traj, "task_kind", "") or "") != "user_request":\n                return False\n', "            pass\n"),
    # ── review round: pins that went with lesson tests
    ("W1 the turn does not publish the request", AG, '                    _cur_req.set(str(last_user_content or ""))\n', "                    pass\n"),
    ("W2 Perfect-It keeps tool-call markup", AG, "        p_msg = re.sub(r'<tool_call>.*?</tool_call>', '', p_msg, flags=re.DOTALL | re.IGNORECASE).strip()\n", "        p_msg = p_msg.strip()\n"),
    ("W3 no accent folding", RQ, '    t = "".join(ch for ch in t if not unicodedata.combining(ch))\n', ""),
    ("W4 the nudge does not name the rule command", AG, 'f"call \'update_profile\' now{_rule_clause}. If it is NOT — "', '"call \'update_profile\' now. If it is NOT — "'),

    # ── review round: the purge (`scripts/lesson_purge_4td.py`)
    ("PG2 the row's own type is not checked and the filter is gone", PG, """        got = collection.get(where={"type": ROW_TYPE}, include=["documents", "metadatas"], limit=BATCH, offset=offset)
        ids = got.get("ids") or []
        if not ids:
            return rows
        for i, doc, meta in zip(ids, got["documents"], got["metadatas"]):
            if isinstance(meta, dict) and meta.get("type") == ROW_TYPE:
                rows[i] = (doc, meta)
""", """        got = collection.get(include=["documents", "metadatas"], limit=BATCH, offset=offset)
        ids = got.get("ids") or []
        if not ids:
            return rows
        for i, doc, meta in zip(ids, got["documents"], got["metadatas"]):
            rows[i] = (doc, meta)
"""),
    ("PG3 a look-alike type goes too", PG, 'ROW_TYPE = "skill"\n', 'ROW_TYPE = "skills"\n'),
    ("PG4 delete before the export", PG, """    exported = write_export(backup / EXPORT_NAME, rows) if rows else 0
    if rows:
        delete_rows(collection, list(rows))
""", """    if rows:
        delete_rows(collection, list(rows))
    exported = write_export(backup / EXPORT_NAME, rows) if rows else 0
"""),
    ("PG5 no read back", PG, '    if back != {i: (v[0], v[1]) for i, v in merged.items()}:\n', "    if False:\n"),
    ("PG6 read back checks the count only", PG, '    if back != {i: (v[0], v[1]) for i, v in merged.items()}:\n', "    if len(back) != len(merged):\n"),
    ("PG7 a dry run purges", PG, '    if dry_run:\n        return {"dry_run": True,', '    if False:\n        return {"dry_run": True,'),
    ("PG8 a live agent is no obstacle", PG, "    if agent_answers():\n", "    if False:\n"),
    ("PG9 a timeout counts as nobody there", PG, "    except OSError:\n        return True\n", "    except OSError:\n        return False\n"),
    ("PG10 the backup dir is not checked", PG, "    check_backup_dir(backup, memory_dir)\n", ""),
    ("PG11 a rerun overwrites the export", PG, "    merged = {}\n    if path.exists():\n", "    merged = {}\n    if False:\n"),
    ("PG12 a rerun replaces a moved file", PG, "        while dst.exists():\n", "        while False:\n"),
    ("PG13 the lock stays behind", PG, 'FILES = ("skills_playbook.json", "skills_playbook.lock", "skills_pruned_archive.jsonl")', 'FILES = ("skills_playbook.json", "skills_pruned_archive.jsonl")'),
    ("PG14 the archive stays behind", PG, 'FILES = ("skills_playbook.json", "skills_playbook.lock", "skills_pruned_archive.jsonl")', 'FILES = ("skills_playbook.json", "skills_playbook.lock")'),
    ("PG15 a store that shrank by more passes", PG, "    if left or before - after != len(ids):\n", "    if False:\n"),
    ("PG16 a torn earlier export is overwritten", PG, """            raise Refused(f"the export already in the backup dir cannot be read ({exc})") from exc\n""", "            merged = {}\n"),
    ("PG17 another writer is ignored", PG, """        acquire_writer_lock(memory_dir, "lesson_purge_4td")\n""", "        pass\n"),
]
PINS = ["tests/test_4td_lesson_store_removed.py", "tests/test_4td_lesson_purge.py", "tests/test_4og_reply_state.py", "tests/test_isolation_replay.py", "tests/test_4tb_replan_removed.py",
        "tests/test_verifier_escalation_ledger.py", "tests/test_4sy_health_internals.py", "tests/test_learning_health.py",
        "tests/test_human_feedback.py", "tests/test_4tc_mood_hooks.py", "tests/test_notify_promise.py",
        "tests/test_vector_memory_integrity_fixes.py"]


def _tests():
    pat = (r"is_simulated_context\|mark_simulated\|simulated\|isolated_replay_context\|build_leaf_context\|"
           r"_canonicalise_tool_name\|_unknown_tool_message\|_BOOKKEEPING_TOOL_NAMES\|_PRE_TOOL_BOOKKEEPING\|"
           r"_COLLAPSE_READSAFE\|_SELF_REPORT_TOOLS\|without_retired_rows\|search_advanced\|apply_human_label\|"
           r"FEEDBACK_HINT\|check_health\|health_internals\|learning_report\|_learning_report\|YieldProbe\|"
           r"create_skill\|manage_skills\|_fetch_all_tiers\|write_origin\|usage_credit_blocked\|"
           r"trajectory_may_teach\|iter_teachable\|memory.population")
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
