"""§4RD battery — req e2c7c7f4 fixes: literal tool arguments, attribute-only
decode, sandbox-named syntax notes, stopped streams booked (2026-10-24).
Reuses the §4QA harness.   python3 scripts/mutate_4rd.py [all|check|rerun R1 R2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

AG = "src/ghost_agent/core/agent.py"
FS = "src/ghost_agent/tools/file_system.py"
m.PINS = ["tests/test_4rd_tool_args_literal.py", "tests/test_4rd_stopped_stream_record.py",
          "tests/test_4ec_dispatch_pins.py", "tests/test_4ec_parser_dialects.py",
          "tests/test_core_loop_defect_fixes.py"]


def _tests():
    return list(m.PINS)


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", AG, "def _html_attr(value: str) -> str:", "# ctl\ndef _html_attr(value: str) -> str:"),
    ("CONTROL known-bad (must DIE): attributes not decoded", AG,
     "    return _html.unescape(value)\n", "    return value\n"),
    ("D1 the blanket decode is back", AG,
     "            # §4RD: tool arguments reach the tool EXACTLY as the model wrote\n",
     "            if tool_calls:\n"
     "                import html as _h\n"
     "                for _tc in tool_calls:\n"
     "                    try:\n"
     "                        _a = json.loads(_tc['function']['arguments'], strict=False)\n"
     "                        _tc['function']['arguments'] = json.dumps({k: (_h.unescape(v) if isinstance(v, str) else v) for k, v in _a.items()})\n"
     "                    except Exception:\n"
     "                        pass\n"),
    ("D2 the backslash-quote rewrite is back", AG,
     "            # §4RD: tool arguments reach the tool EXACTLY as the model wrote\n",
     r"""            if tool_calls:
                for _tc in tool_calls:
                    try:
                        _a = json.loads(_tc['function']['arguments'], strict=False)
                        _tc['function']['arguments'] = json.dumps({k: (v.replace('\\"', '"') if isinstance(v, str) else v) for k, v in _a.items()})
                    except Exception:
                        pass
"""),
    ("D3 function-tag attributes literal", AG,
     "                                args_val[a.group(1)] = _html_attr(a.group(2))\n",
     "                                args_val[a.group(1)] = a.group(2)\n"),
    ("D4 Format 2 value literal", AG,
     "                            args_val[p.group(1)] = _html_attr(p.group(3))\n",
     "                            args_val[p.group(1)] = p.group(3)\n"),
    ("D5 Format 4 attribute literal", AG,
     "                                args_val[a.group(1)] = _html_attr(a.group(3))\n",
     "                                args_val[a.group(1)] = a.group(3)\n"),
    ("D6 Format 5 attribute literal", AG,
     "                                args_val[d.group(1)] = _html_attr(d.group(3))\n",
     "                                args_val[d.group(1)] = d.group(3)\n"),
    ("D7 host path kept in the syntax note", FS,
     "        err = err.replace(_host, str(filename))\n", "        pass\n"),
    ("S1 a closed drain books nothing", AG,
     '                    if not _drain_flags.get("recorded"):\n                        # §4RD: closed before the tail ran',
     '                    if False:\n                        # §4RD: closed before the tail ran'),
    ("S2 the tail does not mark itself recorded", AG,
     '                    _drain_flags["recorded"] = True                 # §4RD: the tail decided; the finally must not book again\n',
     ""),
    ("S3 the partial is not read", AG,
     '                            _sp_text = str((_drain_flags.get("partial") or (lambda: ""))() or "")\n',
     '                            _sp_text = ""\n'),
    ("S4 the partial accessor is never set", AG,
     '            _drain_flags["partial"] = lambda: full_content\n', ""),
    ("S5 a Stop reads as a disconnect", AG,
     "                            _sp_stop = bool(_cancelled_mid_drain or _stop_seen)\n",
     "                            _sp_stop = bool(_cancelled_mid_drain)\n"),
    ("S6 the Stop is never read", AG,
     '                    _stop_seen = bool(getattr(_active_turn, "cancelled", False))   # identity, not the id (r1)\n',
     "                    _stop_seen = False\n"),
    ("R2 the record runs outside the request context", AG,
     "                            _sp_toks = (request_id_context.set(_stream_req_id),\n",
     "                            _sp_toks = None and (request_id_context.set(_stream_req_id),\n"),
    ("R6 regex fallback keeps escaped quotes", AG,
     """                                        args_dict["content"] = raw_content.replace('\\\\"', '"')   # §4RD: a JSON slice\n""",
     """                                        args_dict["content"] = raw_content\n"""),
    ("R7 the deadline is read after the clock closed", AG,
     "                                    req_id, (_req_deadline - (time.monotonic() - _req_t0))\n                                    if _req_deadline > 0 else None)}),",
     "                                    req_id)}),"),
    ("R8 the aborted row loses the trajectory id", AG,
     "                                trajectory_id=current_trajectory_id,       # r2: the id the late verdict keys on\n", ""),
    ("R1 a disconnect books FAILED", AG,
     '                                **({} if _sp_stop else {"marker": interruption_marker(',
     '                                **({} if True else {"marker": interruption_marker('),
    ("R3 no duration from the drain", AG,
     "                                elapsed_s=time.monotonic() - _req_t0,\n                                trajectory_id=",
     "                                elapsed_s=None,\n                                trajectory_id="),
    ("R4 aborted rows drop the snapshots", AG,
     "                project_id=project_id, elapsed_s=elapsed_s,\n", "                \n"),
    ("R5 shorter host path first", FS,
     "    for _host in sorted({str(path), _resolved} - {\"\"}, key=len, reverse=True):",
     "    for _host in sorted({str(path), _resolved} - {\"\"}, key=len):"),
]

if __name__ == "__main__":
    m.main()
