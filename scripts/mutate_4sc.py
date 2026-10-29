"""§4SC battery — the cut-write continuation keeps the conversation
(2026-10-29). Reuses the §4QA harness (in place, backup + restore + byte
compare, two-tier, controls).
    python3 scripts/mutate_4sc.py [all|check|rerun K1 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

AG = "src/ghost_agent/core/agent.py"; LL = "src/ghost_agent/core/llm.py"
WC = "src/ghost_agent/core/write_continuation.py"
ST = "src/ghost_agent/utils/stylometry.py"; SE = "src/ghost_agent/tools/search.py"
PM = "src/ghost_agent/tools/probe_model.py"; PR = "src/ghost_agent/core/project_research.py"
m.PINS = ["tests/test_4sc_cut_write_slot.py", "tests/test_4rwg_side_call.py",
          "tests/test_4ob_write_continuation.py", "tests/test_4sa_head_review_fixes.py"]


def _tests():
    import glob, subprocess
    pat = (r"_side_call_before\|before_side_main_dispatch\|continue_cut_write\|_call_watching_stop\|"
           r"_slot_switch_before\|side_main")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", WC,
     "async def _call_watching_stop(llm, payload, cancelled):",
     "# ctl\nasync def _call_watching_stop(llm, payload, cancelled):"),
    ("CONTROL known-bad (must DIE): a side call never lets the next head call restore", AG,
     "        if kind in done.get(rid, ()) and not level_moved and not _after_side:",
     "        if kind in done.get(rid, ()) and not level_moved:"),
    ("K1 the continuation is a plain main call", WC,
     '                                                     task_label="write-continuation", side_main=True))',
     '                                                     task_label="write-continuation"))'),
    ("K2 side_main ignored by the client", LL,
     "                if side_main and not off_main_only and not (use_swarm or use_worker or use_vision\n",
     "                if False and not off_main_only and not (use_swarm or use_worker or use_vision\n"),
    ("K3 side_main runs the MAIN hook instead", LL,
     "                    try:\n                        await self._before_side_main_dispatch(payload)\n                    finally:\n                        _side_call_what_ctx.reset(_what_tok)",
     "                    try:\n                        await self._before_main_dispatch(payload)\n                    finally:\n                        _side_call_what_ctx.reset(_what_tok)"),
    ("K4 the hook's name never cleared", LL,
     "                    finally:\n                        _side_call_what_ctx.reset(_what_tok)",
     "                    finally:\n                        pass"),
    ("K5 the save names the look for a continuation", AG,
     '        what = what if isinstance(what, str) and what else "this request\'s picture look"',
     '        what = "this request\'s picture look"'),
    # §4SC-b (review r1): the same hole on the other foreground calls
    ("K6 neutralize_query is a plain main call", ST,
     "        res = await llm_client.chat_completion(\n" + "            payload, use_worker=True, off_main_only=True, require_healthy=True,\n"
     "            timeout=_NEUTRALIZE_TIMEOUT_S, total_budget=_NEUTRALIZE_BUDGET_S)\n",
     "        res = await llm_client.chat_completion(payload)\n"),
    ("K7 neutralize_query falls back to main when the worker is down", ST,
     "            payload, use_worker=True, off_main_only=True, require_healthy=True,\n"
     "            timeout=_NEUTRALIZE_TIMEOUT_S, total_budget=_NEUTRALIZE_BUDGET_S)\n",
     "            payload, use_worker=True, off_main_only=False, require_healthy=True,\n"
     "            timeout=_NEUTRALIZE_TIMEOUT_S, total_budget=_NEUTRALIZE_BUDGET_S)\n"),
    ("K8 the fact-check verify is a plain main call", SE,
     '            side_main=True, task_label="fact-check-verify"), _FACTCHECK_VERIFY_S + _fc_hook_s)\n',
     '            task_label="fact-check-verify"), _FACTCHECK_VERIFY_S + _fc_hook_s)\n'),
    ("K9 a foreground probe on main is a plain main call", PM,
     '                side_main=(not _bg and node == "main"),\n',
     '                side_main=False,\n'),
    ("K9b a background probe is a side call too", PM,
     '                side_main=(not _bg and node == "main"),\n',
     '                side_main=(node == "main"),\n'),
    ("K10 the inline research summary is a plain main call", PR,
     '           side_main=not is_background, task_label="research-summary")\n',
     '           task_label="research-summary")\n'),
    ("K11 the continuation drops the tools (a non-head call)", AG,
     '                    if cont_payload.get("tools"):\n                        cont_payload["tool_choice"] = "none"\n',
     '                    if cont_payload.pop("tools", None):\n                        cont_payload.pop("tool_choice", None)\n'),
    ("K11b the continuation keeps the tools but invites a call", AG,
     '                    if cont_payload.get("tools"):\n                        cont_payload["tool_choice"] = "none"\n',
     '                    if cont_payload.get("tools"):\n                        cont_payload["tool_choice"] = "auto"\n'),
    ("K12 a given window is capped at the state's estimate", AG,
     "                save_s = late - time.monotonic()\n",
     "                save_s = min(_head_saves.save_timeout_s(n), late - time.monotonic())\n"),
    ("K13 every side call is named the continuation", LL,
     '                        if str(task_label or "") in ("", "write-continuation")\n',
     '                        if True\n'),
    # §4SC-c (review r2)
    ("K14 neutralize_query is unbounded (the worker client's 1200 s default)", ST,
     "            payload, use_worker=True, off_main_only=True, require_healthy=True,\n"
     "            timeout=_NEUTRALIZE_TIMEOUT_S, total_budget=_NEUTRALIZE_BUDGET_S)\n",
     "            payload, use_worker=True, off_main_only=True, require_healthy=True)\n"),
    ("K14b neutralize_query calls a node with its breaker open", ST,
     "            payload, use_worker=True, off_main_only=True, require_healthy=True,\n",
     "            payload, use_worker=True, off_main_only=True, require_healthy=False,\n"),
    ("K15 no worker pool → the call runs on main", ST,
     "    if _pool is not None and not _pool:\n",
     "    if False:\n"),
    ("K16 a continuation's <tool_call> text is appended", AG,
     '                        if ((cont_choice.get("message", {}) or {}).get("tool_calls")\n'
     '                                or "<tool_call" in _cont_low or "<function" in _cont_low\n'
     '                                # §4SC-d: the closing forms too — a cut inside\n'
     '                                # the opening tag continues as "ll>\\n{…}</tool_call>"\n'
     '                                or "tool_call>" in _cont_low or "</function" in _cont_low):\n',
     '                        if False:\n'),
    ("K17 the fact-check wait leaves out the side hook's save", SE,
     '            side_main=True, task_label="fact-check-verify"), _FACTCHECK_VERIFY_S + _fc_hook_s)\n',
     '            side_main=True, task_label="fact-check-verify"), _FACTCHECK_VERIFY_S + 5.0)\n'),
    ("K18 the restore line always says picture look", AG,
     "        self._side_call_what_last = what         # §4SC-c: for the restore line\n",
     ""),
    ("K19 a tool-less continuation is a plain main call", AG,
     '                            side_main=not cont_payload.get("tools"))\n',
     '                            side_main=False)\n'),
    # §4SC-d (review r3)
    ("K20 the scrub's whole budget is the route bound (a 9 s permit gate)", ST,
     "            timeout=_NEUTRALIZE_TIMEOUT_S, total_budget=_NEUTRALIZE_BUDGET_S)\n",
     "            timeout=_NEUTRALIZE_TIMEOUT_S, total_budget=_NEUTRALIZE_TIMEOUT_S)\n"),
    ("K21 a cut inside an opening tag is continued as prose", AG,
     "                    and _tail_m is None\n",
     ""),
    ("K22 only the opening call forms are gated on the continuation", AG,
     '                                or "tool_call>" in _cont_low or "</function" in _cont_low):\n',
     '                                ):\n'),
    ("K23 an off-main-only pool call on the main box skips the side hook", LL,
     "                elif self._pool_is_main_box(use_swarm, use_worker, use_vision, use_coding, use_critic):\n",
     "                elif not off_main_only and self._pool_is_main_box(use_swarm, use_worker, use_vision, use_coding, use_critic):\n"),
    ("K24 a cut inside an opening tag ships the fragment", AG,
     "                if _tag_tail_cut:\n",
     "                if False:\n"),
    # §4SC-e (review r4)
    ("K25 the cut note lands inside an open code fence", AG,
     '                    _fence = "\\n```" if _in_open_fence(full_content) else ""\n',
     '                    _fence = ""\n'),
    ("K26 the tail pattern is not line-anchored (any `<` at the end)", AG,
     '    r"(?:^|\\n)<(?:t(?:o(?:o(?:l(?:_(?:c(?:a(?:l(?:l)?)?)?)?)?)?)?)?"\n',
     '    r"<(?:t(?:o(?:o(?:l(?:_(?:c(?:a(?:l(?:l)?)?)?)?)?)?)?)?"\n'),
    # §4SC-f (review r5)
    ("K27 the substring exclusion reads the tail itself (a cut at '<tool_call' ships raw)", AG,
     "                _rest_low = _stripped[:_tail_m.start()].lower() if _tail_m else _full_low\n",
     "                _rest_low = _full_low\n"),
    ("K28 a leading `>` is glued unconditionally", AG,
     '                            _glue_gt = cont_text[:1] == ">" and (_last.count("<") > _last.count(">")\n'
     '                                                                  or _last.endswith(("-", "=", ">")))\n',
     '                            _glue_gt = cont_text[:1] == ">"\n'),
    ("K29 a `<t…` line inside an open code fence is read as a call tag", AG,
     "                _tail_m = None if _in_fence else _CALL_TAG_TAIL.search(_stripped)\n",
     "                _tail_m = _CALL_TAG_TAIL.search(_stripped)\n"),
    # §4SC-g (review r6)
    ("K30 a continuation round cut inside a tag is continued", AG,
     '                        if stream_finish_reason == "length" and not _in_open_fence(full_content):\n',
     '                        if False:\n'),
    ("K31 a dangling tail after a complete call block reaches the parser", AG,
     "                elif _tail_m is not None:\n",
     "                elif False:\n"),
    ("K32 the fence count reads mid-line ``` mentions", AG,
     '    return len(re.findall(r"(?m)^\\s{0,3}```", _strip_think_blocks(text))) % 2 == 1\n',
     '    return text.count("```") % 2 == 1\n'),
]

if __name__ == "__main__":
    m.main()
