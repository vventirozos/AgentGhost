"""§4RA battery — coding requests earn deadline extensions (2026-10-24). Reuses the §4QA harness.
    python3 scripts/mutate_4ra.py [all|check|rerun R1 R2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

LG = "src/ghost_agent/utils/logging.py"
AG = "src/ghost_agent/core/agent.py"
WC = "src/ghost_agent/core/write_continuation.py"
PA = "src/ghost_agent/core/project_advancer.py"
SB = "interface/externals/slack_bot/main.py"
m.PINS = ["tests/test_4ra_deadline_extension.py", "tests/test_client_deadline_report_2026_09_21.py",
          "tests/test_4mf_items.py", "tests/test_4mq_long_horizon.py"]


def _tests():
    import glob, subprocess
    pat = r"request_deadline_s\|request_remaining_s\|effective_report_floor\|note_code_progress\|base_deadline_s"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", LG, "def note_code_progress(req_id: str, what: str = \"\") -> None:",
     "# ctl\ndef note_code_progress(req_id: str, what: str = \"\") -> None:"),
    ("CONTROL known-bad (must DIE): never extend", LG,
     "        if ceil <= base or pt is None:\n            return have", "        if True:\n            return have"),
    ("R1 the client header wins again", LG, "        return min(client, cap)", "        return client"),
    ("R2 the cap wins over a lower client", LG, "        return min(client, cap)", "        return cap"),
    # R3/R4 are EQUIVALENT (battery 1): the ceiling is min(ceil, client) and the base is the client
    # when there is no server cap, so neither clause changes an observable value — kept as clarity
    ("R3 no-header clients are extended", LG,
     "    if base <= 0 or ceil <= 0 or client <= 0 or server_request_cap_s(req_id) <= 0:",
     "    if base <= 0 or ceil <= 0 or server_request_cap_s(req_id) <= 0:"),
    ("R4 background kinds are extended", LG,
     "    if base <= 0 or ceil <= 0 or client <= 0 or server_request_cap_s(req_id) <= 0:",
     "    if base <= 0 or ceil <= 0 or client <= 0:"),
    ("R5 the ceiling ignores the client", LG, "    return max(base, min(ceil, client))", "    return max(base, ceil)"),
    ("R6 stale progress still earns", LG,
     "        if now - float(pt) > PROGRESS_WINDOW_S:\n            return have",
     "        if False:\n            return have"),
    ("R7 the window ignores the reserve", LG,
     '        window = EXTENSION_STEP_S + float(s.get("floor_s") or 0.0)', "        window = EXTENSION_STEP_S"),
    ("R8 earn anywhere, not near the deadline", LG,
     "            if have >= room or base + have - elapsed >= window:", "            if have >= room:"),
    ("R9 the last step overshoots the ceiling", LG,
     "            have = room if room - have <= EXTENSION_STEP_S else have + EXTENSION_STEP_S",
     "            have = have + EXTENSION_STEP_S"),
    ("R10 the extension is not kept", LG, '        s["extended_s"] = have\n        return have', "        return have"),
    ("R11 the deadline is not capped", LG,
     "    return min(base + _earn_extension(req_id, base, ceil), max(base, ceil))",
     "    return base + _earn_extension(req_id, base, ceil)"),
    ("R12 remaining ignores extensions", LG,
     "    deadline = request_deadline_s(req_id)\n    if deadline <= 0:\n        return None",
     "    deadline = base_deadline_s(req_id)\n    if deadline <= 0:\n        return None"),
    ("R13 progress is not recorded", LG,
     '            s["progress_t"] = time.monotonic()', "            pass"),
    ("R14 the reserve is not recorded", LG, '            s["floor_s"] = f', "            pass"),
    ("R15 effective_report_floor does not note it", AG,
     "        _glog.note_report_floor(str(req_id), floor)          # §4RA r2 M-A: the extension window sees it\n",
     ""),
    ("R16 no second read at turn start", AG,
     "                    # §4RA r2 M-A: read again with THIS reserve noted — a reserve that grew\n"
     "                    # this turn must widen the extension window before the report check\n"
     "                    _remaining_s = _glog.request_remaining_s(str(req_id or \"\"))\n",
     "                    # §4RA r2 M-A: read again with THIS reserve noted — a reserve that grew\n"
     "                    # this turn must widen the extension window before the report check\n"),
    ("R17 prose files count", AG, "    return str(target or \"\").lower().endswith(CODE_PROGRESS_EXTS)",
     "    return True"),
    ("R18 any file_system op counts", AG,
     "    if fname != \"file_system\" or str(op or \"\").lower() not in CODE_WRITE_OPS:",
     "    if fname != \"file_system\":"),
    ("R19 errors count", AG, "    if is_error:\n        return False\n    if fname == \"execute\":",
     "    if fname == \"execute\":"),
    ("R20 refused calls count", AG,
     '                        _refused = _res_unresolved or bool(getattr(_outcome, "is_rejection", False))',
     "                        _refused = False"),
    ("R21 a failing write counts", AG,
     "                                            _refused or (_res_is_error and fname != \"execute\"),",
     "                                            _refused,"),
    ("R22 any run counts", AG,
     "                                if (isinstance(_r, dict) and _r.get(\"progress\") is True\n",
     "                                if (isinstance(_r, dict)\n"),
    ("R23 a run counts without a new write", AG,
     "                                        and _r.get(\"writes\", 0) > _r.get(\"writes_at_earn\", 0)):",
     "                                        ):"),
    ("R24 the seen-write count is not kept", AG,
     "                                _r[\"writes_at_earn\"] = _r.get(\"writes\", 0)\n",
     "                                pass\n"),
    ("R24b a refused run uses up the earn", AG,
     "                                    _earned_recs.append(_r)\n",
     "                                    _earned_recs.append(_r)\n                                    _r[\"writes_at_earn\"] = _r.get(\"writes\", 0)\n"),
    ("R24c writes enter the run-earn loop", AG,
     "                        if fname == \"execute\" and isinstance(_written, dict):\n                            for _k in futility_keys_in_command(",
     "                        if isinstance(_written, dict):\n                            for _k in futility_keys_in_command("),
    ("R24d an earned count earns again", AG,
     "                                        and _r.get(\"writes\", 0) > _r.get(\"writes_at_earn\", 0)):",
     "                                        and _r.get(\"writes\", 0) >= _r.get(\"writes_at_earn\", 0)):"),
    ("R25 the hook never notes progress", AG,
     "                            _glog.note_code_progress(request_id_context.get() or \"\",",
     "                            (lambda *a: None)(request_id_context.get() or \"\","),
    ("R26 the budget line hides the ceiling", AG,
     "        if float(ceiling_s or 0.0) > float(deadline_s or 0.0):", "        if False:"),
    ("R27 the continuation's first part is not progress", WC,
     "        _note_progress(req_id, path)              # §4RA r2 M-B: the saved first part is progress\n", ""),
    ("R28 appended parts are not progress", WC,
     "                    _note_progress(req_id, path)  # §4RA r2 M-B: and so is each appended part\n", ""),
    ("R29 continued prose counts", WC,
     '        if is_code_progress("file_system", "append", path, False):', "        if True:"),
    ("R30 the continuation uses the unsized reserve", WC,
     "        floor = effective_report_floor(_glog.request_deadline_s(req_id), req_id)",
     "        floor = effective_report_floor(_glog.request_deadline_s(req_id))"),
    ("R31 the advancer uses the unsized reserve", PA,
     "effective_report_floor(request_deadline_s(rid), rid)", "effective_report_floor(request_deadline_s(rid))"),
    ("R32 Slack waits the old hour", SB, '"GHOST_SLACK_AGENT_TIMEOUT", "15300"', '"GHOST_SLACK_AGENT_TIMEOUT", "3600"'),
    ("R33 Slack's client ignores the setting", SB, "httpx.AsyncClient(timeout=AGENT_TIMEOUT_S)",
     "httpx.AsyncClient(timeout=3600.0)"),
]

if __name__ == "__main__":
    m.main()
