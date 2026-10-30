"""§4SD battery — no time ceiling for a progressing build, a per-turn output
cap of min(64k, the window's room), and EAGAIN/EBUSY write retries
(2026-10-29). Reuses the §4QA harness (in place, backup + restore + byte
compare, two-tier, controls). bin/ghost and the launcher are NOT mutated in
place (the operator's live client / launchd's launcher): see the journal.
    python3 scripts/mutate_4sd.py [all|check|rerun K1 …]
"""
import glob
import importlib.util
import os
import subprocess

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

AG = "src/ghost_agent/core/agent.py"; LG = "src/ghost_agent/utils/logging.py"
FS = "src/ghost_agent/tools/file_system.py"; IF = "interface/server.py"
m.PINS = ["tests/test_4sd_no_ceiling_budget.py", "tests/test_4sd_tool_turn_room.py",
          "tests/test_4sd_transient_write_retry.py", "tests/test_4ra_deadline_extension.py",
          "tests/test_client_deadline_report_2026_09_21.py", "tests/test_4rf_cut_in_thinking.py"]
_HEAD = ["tests/test_4ro_head_saves.py", "tests/test_4rp_slot_keep.py", "tests/test_4rv_conv_keep.py",
         "tests/test_4rwg_side_call.py", "tests/test_4sa_head_review_fixes.py", "tests/test_4sc_cut_write_slot.py"]
_FAST = {AG: m.PINS, LG: m.PINS,
         FS: m.PINS + ["tests/test_edit_operation.py", "tests/test_file_system_write_corruption_guards.py"],
         IF: m.PINS + ["tests/test_interface_proxy_auth.py"]}


def _tests():
    pat = (r"_retry_transient\|_write_edit_nofollow\|_earn_extension\|extension_ceiling_s\|request_deadline_s\|"
           r"base_deadline_s\|output_cap_for_tool_turn\|_tool_turn_room\|DEFAULT_TOOL_TURN_MAX_TOKENS\|"
           r"client_deadline_context\|X-Ghost-Client-Timeout\|tool_replace_text\|tool_edit_text\|"
           r"head_saves\|_head_save_plan")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS) | set(_HEAD))


m._tests = _tests
m._fast_tests = lambda f, pins: sorted(set(_FAST.get(f, pins)))
MUT_R1 = [
    ("CONTROL no-op (must SURVIVE)", LG,
     "def note_turn_seconds(req_id: str, seconds: float) -> None:",
     "# ctl\ndef note_turn_seconds(req_id: str, seconds: float) -> None:"),
    ("CONTROL known-bad (must DIE): the CLI-less extension (no ceiling ignored)", LG,
     '    if raw in ("inf", "none", "unlimited"):\n        return math.inf\n',
     '    if raw in ("none", "unlimited"):\n        return 0.0\n'),
    # ── the time budget (utils/logging.py)
    ("B1 none/unlimited not understood", LG,
     '    if raw in ("inf", "none", "unlimited"):\n', '    if raw in ("inf",):\n'),
    ("B2 an inf client with no server cap becomes an inf deadline", LG,
     "    if math.isinf(client):\n        return 0.0          # §4SD", "    if False:\n        return 0.0          # §4SD"),
    ("B3 the stall window does not grow with the longest turn", LG,
     '    return max(PROGRESS_WINDOW_S, 2.0 * float(s.get("longest_turn_s") or 0.0))',
     "    return PROGRESS_WINDOW_S"),
    ("B4 the stall window is one turn, not two", LG,
     '    return max(PROGRESS_WINDOW_S, 2.0 * float(s.get("longest_turn_s") or 0.0))',
     '    return max(PROGRESS_WINDOW_S, 1.0 * float(s.get("longest_turn_s") or 0.0))'),
    ("B5 the steps are floored (the deadline lands short of the window)", LG,
     "math.ceil(short / EXTENSION_STEP_S)", "int(short / EXTENSION_STEP_S)"),
    ("B6 the extension is not clamped to the ceiling", LG,
     "            have = min(room, have + math.ceil(short / EXTENSION_STEP_S) * EXTENSION_STEP_S)",
     "            have = have + math.ceil(short / EXTENSION_STEP_S) * EXTENSION_STEP_S"),
    ("B7 the stall check reads the old constant", LG,
     "        if now - float(pt) > _progress_window_s(s):", "        if now - float(pt) > PROGRESS_WINDOW_S:"),
    ("B8 the longest turn is the last turn", LG,
     '            s["longest_turn_s"] = max(float(s.get("longest_turn_s") or 0.0), sec)',
     '            s["longest_turn_s"] = sec'),
    ("B9 the budget line reads 'extendable to no limit'", AG,
     "        if math.isinf(float(ceiling_s or 0.0)):\n", "        if False:\n"),
    ("B10 _span(inf) is 'inf min'", AG,
     '    if math.isinf(secs):\n        return "no limit"', '    if False:\n        return "no limit"'),
    ("B11 the turn loop never records its turn", AG,
     '                    _glog.note_turn_seconds(request_id_context.get() or "", thinking_duration)   # §4SD: sizes the stall window\n',
     ""),
    ("B12 the web stream sends its byte-gap timeout", IF,
     '                                   "X-Ghost-Client-Timeout": "inf",',
     '                                   "X-Ghost-Client-Timeout": str(int(CHAT_TIMEOUT_S)),'),
    # ── the per-turn output cap (core/agent.py)
    ("C1 no 64k ceiling (the whole room)", AG,
     "    return max(lo, min(hi, win - used - TURN_ROOM_MARGIN_TOKENS))",
     "    return max(lo, win - used - TURN_ROOM_MARGIN_TOKENS)"),
    ("C2 no margin", AG,
     "    return max(lo, min(hi, win - used - TURN_ROOM_MARGIN_TOKENS))",
     "    return max(lo, min(hi, win - used))"),
    ("C3 no 16k floor", AG,
     "    return max(lo, min(hi, win - used - TURN_ROOM_MARGIN_TOKENS))",
     "    return min(hi, win - used - TURN_ROOM_MARGIN_TOKENS)"),
    ("C4 an unknown window gets 64k", AG,
     "    if win <= 0:\n        return lo\n    return max(lo,", "    if win <= 0:\n        return hi\n    return max(lo,"),
    ("C5 the room never reaches the payload", AG,
     '                        payload["max_tokens"] = await self._tool_turn_room(payload)\n',
     "                        pass\n"),
    ("C6 an explicit operator cap is overridden", AG,
     '                    if payload.get("max_tokens") == DEFAULT_TOOL_TURN_MAX_TOKENS:\n',
     "                    if True:\n"),
    ("C7 tool schemas not counted", AG,
     "                if tools:\n                    n += estimate_tokens(", "                if False:\n                    n += estimate_tokens("),
    ("C8 the window is read once per process", AG,
     "        if isinstance(c, tuple) and len(c) == 2 and now < c[1]:",
     "        if isinstance(c, tuple) and len(c) == 2:"),
    ("C9 a failed read is cached ten minutes", AG,
     "now + (600.0 if n > 0 else 60.0)", "now + 600.0"),
    ("C10 n_ctx read from the top level", AG,
     '                n = int(((p or {}).get("default_generation_settings") or {}).get("n_ctx") or 0)',
     '                n = int((p or {}).get("n_ctx") or 0)'),
    ("C11 the window read asks the /v1 path", AG,
     '                p = await asyncio.to_thread(_head_saves._http, base + "/props", None, 5.0)',
     '                p = await asyncio.to_thread(_head_saves._http, base + "/v1/props", None, 5.0)'),
    # ── EAGAIN/EBUSY retries (tools/file_system.py)
    ("R1 every OSError is retried", FS,
     "            if e.errno not in _TRANSIENT_IO_ERRNOS:\n                raise\n", "            pass\n"),
    ("R2 EBUSY not retried", FS,
     "_TRANSIENT_IO_ERRNOS = frozenset({_errno.EAGAIN, _errno.EBUSY})", "_TRANSIENT_IO_ERRNOS = frozenset({_errno.EAGAIN})"),
    ("R3 no last attempt after the backoff", FS,
     "        time.sleep(_pause)\n    return fn(*args, **kwargs)\n",
     "        time.sleep(_pause)\n    raise OSError(_errno.EAGAIN, 'gave up')\n"),
    ("R4 edit open not retried", FS,
     "        fd = _retry_transient(os.open, str(real), flags)", "        fd = os.open(str(real), flags)"),
    ("R5 edit ftruncate not retried", FS,
     "        _retry_transient(os.ftruncate, fd, 0)", "        os.ftruncate(fd, 0)"),
    ("R6 edit write not retried", FS,
     "            n = _retry_transient(os.write, fd, view)", "            n = os.write(fd, view)"),
    ("R7 write not retried", FS,
     "        _swapped = await asyncio.to_thread(_retry_transient, _atomic_write)",
     "        _swapped = await asyncio.to_thread(_atomic_write)"),
    ("R8 append not retried", FS,
     "        total = await asyncio.to_thread(_retry_transient, _atomic_append)",
     "        total = await asyncio.to_thread(_atomic_append)"),
    ("R9 streaming replace build not retried", FS,
     "                    _retry_transient, _streaming_replace)", "                    _streaming_replace)"),
    ("R10 streaming rename not retried", FS,
     "                        await asyncio.to_thread(_retry_transient, os.replace, _tmp, path)",
     "                        await asyncio.to_thread(os.replace, _tmp, path)"),
]

RD = "src/ghost_agent/api/routes.py"; PRR = "src/ghost_agent/core/project_research.py"
TD = "src/ghost_agent/core/task_deliverable.py"
_FAST.update({RD: m.PINS, PRR: m.PINS, TD: m.PINS})
_FAST[AG] = m.PINS + ["tests/test_finalize_stream_pins.py", "tests/test_thinking_loop_guards.py"]
# round 2 (R3: the previous round's fixes first) — the review's fixes + R10
m.MUT = [MUT_R1[0], MUT_R1[1],
    ("D1 a length abort is back (200K chars)", AG,
     "                                # §4SD: no length abort — the turn's max_tokens bounds length.\n",
     "                                if len(guard_buf) > 200000:\n"
     "                                    thinking_loop_detected = True\n"
     "                                    break\n"),
    ("D2 final answers take the room", AG,
     "    return (not is_final_generation and isinstance(payload, dict)",
     "    return (isinstance(payload, dict)"),
    ("D3 any cap takes the room", AG,
     "            and payload.get(\"max_tokens\") == DEFAULT_TOOL_TURN_MAX_TOKENS)",
     "            and bool(payload.get(\"max_tokens\")))"),
    ("N1 the switch also unblocks a FIFO", FS,
     "    if stat_mod.S_ISREG(st.st_mode):\n        os.set_blocking(fd, True)",
     "    if True:\n        os.set_blocking(fd, True)"),
    ("N2 the switch never unblocks", FS,
     "    if stat_mod.S_ISREG(st.st_mode):\n        os.set_blocking(fd, True)",
     "    if stat_mod.S_ISREG(st.st_mode):\n        os.set_blocking(fd, False)"),
    ("N3 the streaming read stays non-blocking", FS,
     "                            _block_if_regular(_sfd, _sst)       # §4SD: a buffered EAGAIN read = EOF\n",
     "                            pass\n"),
    ("N4 the edit read stays non-blocking", FS,
     "            raise _FileChangedUnderneath()\n        _block_if_regular(fd, st)                               # §4SD\n        chunks, remaining",
     "            raise _FileChangedUnderneath()\n        chunks, remaining"),
    ("N5 the zip read stays non-blocking", RD,
     "            _block_if_regular(fd, fst)                          # §4SD: virtiofs EAGAIN\n", ""),
    ("N6 the brief hash stays non-blocking", PRR,
     "        _block_if_regular(fd, st)                               # §4SD: virtiofs EAGAIN\n", ""),
    ("N7 the deliverable read checks the path only", TD,
     "        if not _stat.S_ISREG(_block_if_regular(fd).st_mode):   # §4SD: virtiofs EAGAIN; the fd, not the path\n            return None\n",
     ""),
    ("P1 a performed rename is retried (ENOENT)", FS,
     "            if e.errno in _TRANSIENT_IO_ERRNOS and not os.path.lexists(tmp):\n                return\n",
     ""),
    ("P2 a rename that did not happen is called done", FS,
     "            if e.errno in _TRANSIENT_IO_ERRNOS and not os.path.lexists(tmp):",
     "            if e.errno in _TRANSIENT_IO_ERRNOS:"),
    ("P3 the commit is not retried", FS,
     "    _retry_transient(_once)\n", "    _once()\n"),
    ("P4 the mutation stamp runs for a refused duplicate", FS,
     "            if isinstance(r, int) and not _state.get(\"dup\"):", "            if isinstance(r, int):"),
    ("P5 the append is not retried at all", FS,
     "            r = _retry_transient(_atomic_append)                # §4SD: re-runs only before the commit",
     "            r = _atomic_append()"),
    ("R10 the streaming commit is a plain rename", FS,
     "                        await asyncio.to_thread(_commit_replace, _tmp, path)   # §4SD",
     "                        await asyncio.to_thread(os.replace, _tmp, path)"),
    ("R7b the write commit is a plain rename", FS,
     "                _commit_replace(_tmp, path)                     # §4SD\n            except BaseException:",
     "                os.replace(_tmp, path)\n            except BaseException:"),
    ("R12 the streaming fstat leaks its fd", FS,
     "                        except BaseException:\n                            os.close(_sfd)\n                            raise\n                        if not stat_mod.S_ISREG(_sst.st_mode):",
     "                        except BaseException:\n                            raise\n                        if not stat_mod.S_ISREG(_sst.st_mode):"),
]

# round 3 — the loop probes (operator: "evaluate the n-gram loop probe, make
# sure that it's good") + the round-2 mutants the probe change interrupted
SG = "src/ghost_agent/core/stream_guards.py"
_FAST[SG] = ["tests/test_4sd_loop_probes.py", "tests/test_4kv_slack12_fixes.py", "tests/test_thinking_loop_guards.py",
             "tests/test_stream_guards_module.py", "tests/test_4rk_model_fit.py",
             "tests/test_coding_executor_stream_guard.py", "tests/test_finalize_stream_pins.py"]
_FAST[AG] = _FAST[AG] + ["tests/test_4sd_loop_probes.py"]
_tests_r2 = m._tests
m._tests = lambda: sorted(set(_tests_r2()) | set(_FAST[SG]))
_R2_DONE = 7          # controls, D1-D3, N1, N2 ran before the stop
m.MUT = m.MUT[:2] + m.MUT[_R2_DONE:] + [
    ("G1 every long line counts again", SG,
     '    _stripped = [l.strip() if _is_text_line(l) else "" for l in lines[:-1]]',
     "    _stripped = [l.strip() for l in lines[:-1]]"),
    ("G2 an indented line is a text line", SG,
     '    return (raw_line[:1] not in (" ", "\\t")', "    return (True"),
    ("G3 no word minimum", SG,
     "            and len(_ALPHA_WORD_RE.findall(raw_line)) >= TEXT_LINE_MIN_WORDS)",
     "            and len(_ALPHA_WORD_RE.findall(raw_line)) >= 0)"),
    ("G4 the n-gram code gate is off", SG,
     '        if _code_shaped(a + "\\n" + b) and not _periods_match(a, b):\n            return False\n    return True',
     "        if False:\n            return False\n    return True"),
    ("G5 the n-gram gate compares prose too", SG,
     '        if _code_shaped(a + "\\n" + b) and not _periods_match(a, b):',
     "        if not _periods_match(a, b):"),
    ("G6 the block probe never fires", SG,
     "        if not _periods_match(a, b):\n            return False\n    return True\n",
     "        if not _periods_match(a, b):\n            return False\n    return False\n"),
    ("G7 two occurrences are a block loop", SG,
     "BLOCK_LOOP_OCCURRENCES = 3", "BLOCK_LOOP_OCCURRENCES = 2"),
    ("G8 the look-back is unbounded", SG,
     "        lo = max(0, pos[-1] - BLOCK_LOOP_MAX_PERIOD)", "        lo = 0"),
    ("G9 the similarity bar is loose", SG,
     "BLOCK_LOOP_SIMILAR = 0.9", "BLOCK_LOOP_SIMILAR = 0.5"),
    ("G10 only verbatim periods match", SG,
     "    if a == b:\n        return True\n    return difflib", "    return a == b\n    return difflib"),
    ("G11 a short varying period is a block loop", SG,
     '        if max(a.count("\\n"), b.count("\\n")) < 2 and a != b:\n            return False                 # under three lines: the n-gram probe\'s\n',
     ""),
    ("G12 alternation is not checked", SG,
     "               _alternating_frame_run(reasoning_buf)) >= SENTENCE_RUN_THRESHOLD",
     "               0) >= SENTENCE_RUN_THRESHOLD"),
    ("G13 two sentences are an alternation", SG,
     "    return run if run >= 3 else 0", "    return run"),
    ("G14 a one-frame run also counts as an alternation", SG,
     "    if len(frames) < 3 or frames[0] == frames[1]:", "    if len(frames) < 3:"),
    ("G15 the block probe is not wired", AG,
     "                                    if reasoning_content and _detect_block_loop(reasoning_content):",
     "                                    if False:"),
]

# round 4 — the operator's "disable the coding executor's 30k-char abort"
# went in ahead of round 3 (stopped after R7b: 11 killed, P5 re-pinned, R7b
# equivalent — the whole write step is retried and a re-run writes the same
# bytes); the probe mutants G1–G15 had not run yet
CE = "src/ghost_agent/core/coding_executor.py"
_FAST[CE] = ["tests/test_coding_executor_stream_guard.py", "tests/test_coding_executor_think_policy_and_append_guards.py"]
_names = {x[0].split()[0]: x for x in m.MUT}
m.MUT = m.MUT[:2] + [_names["P5"]] + [x for x in m.MUT if x[0].startswith("G")] + [
    ("CE1 the spec length ceiling is back", CE,
     '                if _detect_thinking_loop(buf):\n                    aborted = "loop"\n                    break\n',
     '                if _detect_thinking_loop(buf):\n                    aborted = "loop"\n                    break\n'
     '                if r_len >= 30_000:\n                    aborted = "loop"\n                    break\n'),
]

# round 5 — the second fresh-eye review (3 readers): probe fixes (n-gram gate
# reverted, Unicode/bullet text lines, non-overlapping block search, linear
# period match), commit proved by inode, no nested retry, read-side retries,
# writer fd close, SSE keepalive, stall window = turn + step, same write is no
# progress, overflow retry at 16k
RT = "src/ghost_agent/api/routes.py"
_FAST[RT] = ["tests/test_4sd_stream_keepalive.py", "tests/test_stream_turn_usage.py",
             "tests/test_feedback_stream_id_restamp.py", "tests/test_4mp_reply_tap.py"]
_FAST[SG] = _FAST[SG]
_FAST[FS] = _FAST[FS] + ["tests/test_edit_ladder_ledger.py"]
_FAST[AG] = _FAST[AG] + ["tests/test_4ra_deadline_extension.py", "tests/test_second_cap_reports.py",
                         "tests/test_logic_and_network_fixes.py"]
_tests_r4 = m._tests
m._tests = lambda: sorted(set(_tests_r4()) | set(_FAST[RT]) | {"tests/test_edit_ladder_ledger.py",
                                                               "tests/test_second_cap_reports.py"})
m.MUT = m.MUT[:2] + [
    ("V1 the block search overlaps again", SG,
     "        p = reasoning_buf.rfind(tail, lo, pos[-1])",
     "        p = reasoning_buf.rfind(tail, lo, pos[-1] + THINKING_LOOP_WINDOW - 1)"),
    ("V2 text words are ASCII only", SG,
     '_WORD_RE = re.compile(r"[^\\W\\d_]{2,}")', '_WORD_RE = re.compile(r"[A-Za-z]{2,}")'),
    ("V3 an indented bullet is code", SG,
     '    if raw_line[:1] in (" ", "\\t") and not _INDENTED_BULLET_RE.match(raw_line):',
     '    if raw_line[:1] in (" ", "\\t"):'),
    ("V4 the period match is loose", SG,
     "BLOCK_LOOP_SIMILAR = 0.9", "BLOCK_LOOP_SIMILAR = 0.5"),
    ("V5 only verbatim periods match", SG,
     "    shared = sum((Counter(la) & Counter(lb)).values())", "    shared = 0"),
    ("V6 the n-gram code gate is back", SG,
     "    return recent.count(tail) >= THINKING_LOOP_THRESHOLD\n",
     "    if recent.count(tail) < THINKING_LOOP_THRESHOLD:\n        return False\n"
     "    pos = [len(buf) - THINKING_LOOP_WINDOW]\n"
     "    while len(pos) < THINKING_LOOP_THRESHOLD:\n"
     "        pos.append(buf.rfind(tail, 0, pos[-1] + THINKING_LOOP_WINDOW - 1))\n"
     "    periods = [buf[pos[i + 1]:pos[i]] for i in range(len(pos) - 1)]\n"
     "    for a, b in zip(periods, periods[1:]):\n"
     "        _ls = [l for l in (a + chr(10) + b).splitlines() if l.strip()]\n"
     "        if len(_ls) >= 3 and sum(map(_is_text_line, _ls)) * 2 < len(_ls) and not _periods_match(a, b):\n"
     "            return False\n"
     "    return True\n"),
    ("G1 every long line counts again", SG,
     '    _stripped = [l.strip() if _is_text_line(l) else "" for l in lines[:-1]]',
     "    _stripped = [l.strip() for l in lines[:-1]]"),
    ("G6 the block probe never fires", SG,
     "        if not _periods_match(a, b):\n            return False\n    return True\n",
     "        if not _periods_match(a, b):\n            return False\n    return False\n"),
    ("G12 alternation is not checked", SG,
     "               _alternating_frame_run(reasoning_buf)) >= SENTENCE_RUN_THRESHOLD",
     "               0) >= SENTENCE_RUN_THRESHOLD"),
    ("G15 the block probe is not wired", AG,
     "                                    if reasoning_content and _detect_block_loop(reasoning_content):",
     "                                    if False:"),
    ("I1 a landed rename is assumed, not proved", FS,
     "        return (st.st_dev, st.st_ino) == tmp_id\n", "        return True\n"),
    ("I2 an lstat error reads as renamed", FS,
     "        except OSError:\n            return False                     # cannot tell: not proved",
     "        except OSError:\n            return True                      # cannot tell: not proved"),
    ("I3 ENOENT after a landed rename is an error", FS,
     "            if e.errno in _TRANSIENT_IO_ERRNOS | {_errno.ENOENT} and _landed():",
     "            if e.errno in _TRANSIENT_IO_ERRNOS and _landed():"),
    ("I4 the writer's retry re-runs a failed commit", FS,
     "        e._ghost_retried = True              # the caller's retry must not re-run the write",
     "        pass"),
    ("I5 the writer leaks its fd when the switch fails", FS,
     "        _close_quietly(fd)                                      # review: not yet owned by fdopen\n"
     "        raise\n    with os.fdopen(fd, \"w\", encoding=encoding) as fh:\n        fh.write(text)\n        fh.flush()",
     "        raise\n    with os.fdopen(fd, \"w\", encoding=encoding) as fh:\n        fh.write(text)\n        fh.flush()"),
    ("I6 the edit read open is not retried", FS,
     "    fd = _retry_transient(os.open, str(real), flags)            # §4SD review: reads too\n"
     "    try:\n        st = os.fstat(fd)\n        if not stat_mod.S_ISREG(st.st_mode):\n            raise _NotRegular()",
     "    fd = os.open(str(real), flags)\n"
     "    try:\n        st = os.fstat(fd)\n        if not stat_mod.S_ISREG(st.st_mode):\n            raise _NotRegular()"),
    ("K1 no keepalive on the tap path", RT,
     "                                yield STREAM_KEEPALIVE_FRAME\n                                continue",
     "                                continue"),
    ("K2 no keepalive on the plain path", RT,
     "                            yield STREAM_KEEPALIVE_FRAME\n                        content, created_time, req_id = _plain.result()",
     "                            pass\n                        content, created_time, req_id = _plain.result()"),
    ("K3 every tick makes a new get (a frame is lost)", RT,
     "                            if _get is None:\n                                _get = asyncio.ensure_future(_tap.queue.get())",
     "                            if True:\n                                _get = asyncio.ensure_future(_tap.queue.get())"),
    ("S1 the stall window is 2x the longest turn again", LG,
     '    return max(PROGRESS_WINDOW_S, float(s.get("longest_turn_s") or 0.0) + EXTENSION_STEP_S)',
     '    return max(PROGRESS_WINDOW_S, 2.0 * float(s.get("longest_turn_s") or 0.0))'),
    ("S2 the same write counts again", LG,
     "            if key in seen:\n                return False", "            if False:\n                return False"),
    ("S3 the turn loop passes no key", AG,
     '                                                          if fname == "file_system" else ""))',
     '                                                          if False else ""))'),
    ("O1 the overflow retry keeps the room", AG,
     "                        if int(payload.get(\"max_tokens\") or 0) > DEFAULT_TOOL_TURN_MAX_TOKENS:\n"
     "                            payload[\"max_tokens\"] = DEFAULT_TOOL_TURN_MAX_TOKENS\n",
     ""),
    [x for x in m.MUT if x[0].startswith("CE1")][0],
]

# round 6 — the operator's turn-cap removal + 30-min Slack, and the third
# review round's fixes (plain-path context, held tap frame, plain hard stop,
# lstat retry, earn on new writes only, bullet/YAML rule, identical-vs-alike
# block periods, alternation threshold)
SB = "interface/externals/slack_bot/main.py"
_FAST[SB] = ["tests/test_4ra_deadline_extension.py"]
_FAST[AG] = _FAST[AG] + ["tests/test_4sd_turn_cap_and_slack_budget.py", "tests/test_4sd_tool_turn_room.py",
                         "tests/test_4sd_no_ceiling_budget.py"]
_FAST[LG] = _FAST[LG] + ["tests/test_4sd_turn_cap_and_slack_budget.py"]
_tests_r5 = m._tests
m._tests = lambda: sorted(set(_tests_r5()) | {"tests/test_4sd_turn_cap_and_slack_budget.py"})
m.MUT = m.MUT[:2] + [
    ("T1 a request with no deadline has no count either", AG,
     "    return sys.maxsize if timed else TURN_CAP_WITHOUT_DEADLINE", "    return sys.maxsize"),
    ("T2 a request with no clock has no count", AG,
     "        timed = float(deadline_s or 0.0) > 0 and elapsed_s is not None",
     "        timed = float(deadline_s or 0.0) > 0"),
    ("T3 the loop keeps the literal 40", AG,
     '                effective_max_turns = main_turn_cap(getattr(self, "max_turns_override", None),\n'
     '                                                    _glog.request_deadline_s(_rid_cap),\n'
     '                                                    _glog.request_elapsed_s(_rid_cap))',
     '                effective_max_turns = getattr(self, "max_turns_override", None) or 40'),
    ("T4 the room ignores the time left", AG,
     "            if not turn_room_fits_time(_glog.extension_ceiling_s(rid), _glog.request_elapsed_s(rid)):",
     "            if False:"),
    ("T5 a 25-minute budget may take a 64k turn", AG,
     "TURN_ROOM_MIN_TIME_LEFT_S = 4500.0", "TURN_ROOM_MIN_TIME_LEFT_S = 1500.0"),
    ("T6 the time spent is not subtracted", AG,
     "    return ceil - spent >= TURN_ROOM_MIN_TIME_LEFT_S", "    return ceil >= TURN_ROOM_MIN_TIME_LEFT_S"),
    ("SL1 Slack gets 60 minutes", SB,
     '"GHOST_SLACK_REQUEST_S", "1800"', '"GHOST_SLACK_REQUEST_S", "3600"'),
    ("P1 the plain path's context is not put back", RT,
     "                    _restore_turn_context(_plain_ctx)\n", "                    pass\n"),
    ("P2 a frame the held get took is dropped", RT,
     "                        if _get is not None and _get.done() and not _get.cancelled():",
     "                        if False:"),
    ("P3 a hard stop on the plain path ends with no [DONE]", RT,
     "                        if _plain.cancelled():", "                        if False:"),
    ("P4 a hang-up on the plain path leaves the turn running", RT,
     "                            _plain.cancel()    # the client went away mid-turn", "                            pass    # the client went away mid-turn"),
    ("L1 the commit's identity lstat is not retried", FS,
     "    tmp_id = _retry_transient(os.lstat, tmp)", "    tmp_id = os.lstat(tmp)"),
    ("E1 a run earns against every write again", AG,
     '                                        and _r.get("new_writes", 0) > _r.get("writes_at_earn", 0)):',
     '                                        and _r.get("writes", 0) > _r.get("writes_at_earn", 0)):'),
    ("E2 every write counts as new", AG,
     "                                    if _wk not in _seen_w:", "                                    if True:"),
    ("B1 a JSDoc star line is text again", SG,
     r'_INDENTED_BULLET_RE = re.compile(r"[ \t]+(?:[-+•]|', r'_INDENTED_BULLET_RE = re.compile(r"[ \t]+(?:[-*+•]|'),
    ("B2 a YAML key item is text again", SG,
     r'\s+(?![^\s:]+:(?:\s|$))")', r'\s")'),
    ("B3 alike periods need only three occurrences", SG,
     "BLOCK_LOOP_OCCURRENCES_ALIKE = 4", "BLOCK_LOOP_OCCURRENCES_ALIKE = 3"),
    ("B4 verbatim periods need a fourth too", SG,
     "    if periods[0] == periods[1]:\n        return True", "    if False:\n        return True"),
    ("B5 the alike bar is loose", SG, "BLOCK_LOOP_SIMILAR = 0.9", "BLOCK_LOOP_SIMILAR = 0.8"),
    ("B6 the alike bar is strict", SG, "BLOCK_LOOP_SIMILAR = 0.9", "BLOCK_LOOP_SIMILAR = 0.97"),
    ("A1 two alternating frames die at 20 again", SG,
     "ALTERNATING_RUN_THRESHOLD = 40", "ALTERNATING_RUN_THRESHOLD = 20"),
    ("A2 alternation is not checked", SG,
     "            or _alternating_frame_run(reasoning_buf) >= ALTERNATING_RUN_THRESHOLD)",
     "            or False)"),
]

# round 6b — re-run the three survivors after their pins
_r6b = {"L1", "B5", "B6"}
m.MUT = m.MUT[:2] + [x for x in m.MUT if x[0].split()[0] in _r6b]

# round 7 — review round 4's fixes: the verbatim shortcut's period bound, the
# identity lstat's retried mark, the bullet regex's spacing
m.MUT = m.MUT[:2] + [
    ("V7 a file-sized verbatim period is a loop again", SG,
     "    if periods[0] == periods[1] and len(periods[0]) <= BLOCK_LOOP_VERBATIM_MAX_PERIOD:",
     "    if periods[0] == periods[1]:"),
    ("V8 the verbatim bound below the logged loops", SG,
     "BLOCK_LOOP_VERBATIM_MAX_PERIOD = 4_000", "BLOCK_LOOP_VERBATIM_MAX_PERIOD = 1_000"),
    ("L2 the identity lstat's give-up is not marked", FS,
     "        e._ghost_retried = True              # the caller's whole-step retry must not nest it (round 4)",
     "        pass"),
    ("R1 the bullet's spaces backtrack again", SG,
     r"[ \t]+(?=\S)(?![^\s:]+:(?:\s|$))", r"\s+(?![^\s:]+:(?:\s|$))"),
]


# round 8 — the coding executor writes a big NEW file in its own call (§4SD),
# after six review rounds (r9–r14)
CE = "src/ghost_agent/core/coding_executor.py"; PA = "src/ghost_agent/core/project_advancer.py"
PJ = "src/ghost_agent/tools/projects.py"
_EXEC_PINS = ["tests/test_4sd_executor_generate.py", "tests/test_coding_executor.py"]
_FAST.update({CE: _EXEC_PINS, PA: _EXEC_PINS + ["tests/test_project_advancer.py"], PJ: _EXEC_PINS})
m.MUT = m.MUT[:2] + [
    ("X1 generate may rewrite an existing file", CE,
     "        if exists:\n            return files, (f\"{path} already exists", "        if False:\n            return files, (f\"{path} already exists"),
    ("X2 a later file is blind to the earlier ones", CE, "        generated[path] = text            # the later files'", "        pass            # the later files'"),
    ("X3 a generate entry reaches _apply_file unconverted", CE,
     "        out.append(entry)\n        done_entries.append(entry)", "        out.append(f)\n        done_entries.append(entry)"),
    ("X4 an empty stop after a cut ends the file", CE,
     "        if not cont.strip() and not done:\n            bad += 0 if yielded else 1", "        if not cont.strip() and not done and finish != \"stop\":\n            bad += 0 if yielded else 1"),
    ("X5 a restart round is appended", CE, "        if restart or chatter:", "        if chatter:"),
    ("X6 one bad round fails the file", CE, "            if bad >= 2:\n                break\n            err = \"\"",
     "            if bad >= 1:\n                break\n            err = \"\""),
    ("X7 the overlap is not stripped", CE,
     "            cont = strip_tail_overlap(text[-_GENERATE_TAIL_CHARS:], cont)", "            pass"),
    ("X8 the call ignores the time left", CE,
     "    return min(cap, by_time)", "    return cap"),
    ("X9 the file call never takes the room", CE,
     "            cap = output_cap_for_tool_turn(used, win)", "            pass"),
    ("X10 a round starts with no time", CE,
     "    if by_time < GENERATE_MIN_TOKENS:\n        return 0", "    if False:\n        return 0"),
    ("X11 the attempt loop ignores the time", CE,
     "        if _attempt and _time_left_s(deadline) < GENERATE_MIN_ATTEMPT_S:", "        if False:"),
    ("X12 thinking stays on in the file call", CE,
     "        _disable_thinking(payload)            # the plan", "        pass            # the plan"),
    ("X13 a caller's deadline is ignored", CE,
     "    return min(out, outer) if outer > 0 else out", "    return out"),
    ("X14 the request clock is ignored", CE,
     "    return min(left, req) if req is not None else left", "    return left"),
    ("X15 the streamed finish reason is dropped", CE,
     '                finish = ch.get("finish_reason")', "                pass"),
    ("X16 the grammar forbids generate", CE, '                           "generate": {"type": "string"},\n', ""),
    ("X17 the unattended step's deadline is not set", PA,
     "    tok = leaf_deadline_context.set(deadline)", "    tok = leaf_deadline_context.set(0.0)"),
    ("X18 a generated-only spec is unusable", CE,
     '                    or str(f.get("generate") or "").strip()', "                    or False"),
    ("X19 out of time is FAILED again", PA, '    if getattr(cres, "out_of_time", False):', "    if False:"),
    ("X20 the batch re-picks an out-of-time task", PA,
     "        if res.task_id and res.summary.startswith(OUT_OF_TIME_PREFIX):", "        if False:"),
    ("X21 out of time does not write the finished files", CE,
     "                if path and not reason:\n                    kept.append(path)", "                if False:\n                    kept.append(path)"),
    ("X22 the own-file rule covers any entry", CE,
     "    if own and old is not None and fspec.get(_GENERATED_KEY) and _norm_path(path) in own:",
     "    if own and old is not None and _norm_path(path) in own:"),
    ("X23 prose after the end marker is written", CE,
     "        head = \"\\n\".join(lines[:i] + ([before] if before.strip() else []))", "        head = \"\\n\".join(lines[:i] + ([before] if before.strip() else []) + lines[i + 1:])"),
    ("X26 a code line opening a continuation is cut", CE,
     "    return bool(_CHATTER_RE.match(cont) or _MORE_OPENERS_RE.match(cont)) and not _CODE_CHAR_RE.search(first)",
     "    return bool(_CHATTER_RE.match(cont) or _MORE_OPENERS_RE.match(cont))"),
    ("X27 a background stream never yields to the owner", CE,
     "                elif finish is None and (now_req > owner_base", "                elif False and (now_req > owner_base"),
    ("X28 a stale pending file is written later", CE,
     "            pending.pop(_k, None)             # r23", "            pass             # r23"),
    ("X29 the owner's edit between attempts is regenerated over", CE,
     "            if live is None or live == own[np_]:\n                continue", "            if True:\n                continue"),
    ("X30 a template partial is refused as truncated", CE,
     '            and "<html" in content.lower() and "</html>" not in content.lower()):',
     '            and "</html>" not in content.lower()):'),
    ("X24 a pending file is reused for any brief", CE,
     "        for _k in [k for k, (b, _t) in pending.items() if _real_plan and _plan.get(k) != b]:",
     "        for _k in [k for k, (b, _t) in pending.items() if _real_plan and k not in _plan]:"),
    ("X25 kept rounds are not progress", CE,
     "        if cont.strip():\n            _note_round_progress(path)", "        if False:\n            _note_round_progress(path)"),
]



# round 9 — the §4SD open items (10-30): the spec call yields, an owner's build
# takes the side path, rewrites of one file stop counting, a finite client
# timeout is a hard limit; after review rounds r1–r2
LL = "src/ghost_agent/core/llm.py"
CL = "src/ghost_agent/core/coding_loop.py"
_OPEN_PINS = ["tests/test_4sd_open_items.py"]
for _f in (CE, PA, AG, LG):
    _FAST[_f] = _OPEN_PINS + _FAST.get(_f, [])
_FAST[PA] = _FAST[PA] + ["tests/test_4mq_long_horizon.py"]
_FAST[LL] = _OPEN_PINS + _HEAD
_FAST[CL] = ["tests/test_4fg_coding_loop.py"]
_FAST[RT] = _FAST.get(RT, []) + ["tests/test_4sb_queued_wording.py"]
_tests_r8 = m._tests
m._tests = lambda: sorted(set(_tests_r8()) | set(_OPEN_PINS) | set(_EXEC_PINS) | {"tests/test_4mq_long_horizon.py", "tests/test_4sb_queued_wording.py", "tests/test_4fg_coding_loop.py"})
m.MUT = m.MUT[:2] + [
    ("Y1 a waiting foreground call does not make it yield", CE,
     '        return now > self.base or int(getattr(self.llm, "foreground_tasks", 0) or 0) > 0', "        return now > self.base"),
    ("Y2 the yield baseline is not the first chunk", CE,
     "        if self.base is None:\n            self.base = now\n            return False",
     "        if self.base is None:\n            self.base = 0\n            return False"),
    ("Y3 the last spec call yields too", CE,
     "    watch = _OwnerWatch(llm, is_background and may_yield)", "    watch = _OwnerWatch(llm, is_background)"),
    ("Y4 the spec loop never stops yielding", CE,
     "            llm, payload, is_background, may_yield=n < _MAX_YIELDS)", "            llm, payload, is_background, may_yield=True)"),
    ("Y5 a yielded spec is used as the answer", CE,
     "        if aborted != YIELDED:\n            return content, reasoning, aborted", "        if True:\n            return content, reasoning, aborted"),
    ("Y6 the thinking spec call does not loop on yields", CE,
     "            content, reasoning, loop_aborted = await _spec_stream(llm, {",
     "            content, reasoning, loop_aborted = await _stream_spec_completion(llm, {"),
    ("Y7 the background no-think call is not streamed", CE,
     "    if is_background:\n        # §4SD: streamed, so it can give", "    if False:\n        # §4SD: streamed, so it can give"),
    ("Y8 no retry without the grammar after an empty answer", CE,
     "        if grammar and not content.strip() and not reasoning.strip():", "        if False:"),
    ("Y9 a foreground call takes no side path", CE,
     '    return {} if is_background else {"side_main": True, "task_label": what}', "    return {}"),
    ("Y10 the spec stream takes no side path", CE,
     '    agen = streamer(payload, is_background=is_background,\n                    **_main_call_kw(is_background, "coding spec"))',
     "    agen = streamer(payload, is_background=is_background)"),
    ("Y11 the file stream takes no side path", CE,
     '    agen = streamer(payload, is_background=is_background,\n                    **_main_call_kw(is_background, "generated file"))',
     "    agen = streamer(payload, is_background=is_background)"),
    ("Y12 a finished spec still yields", CE,
     "            if not finished and watch.owner_waits():", "            if watch.owner_waits():"),
    ("Y13 a deferral is retried as a grammar failure", CE,
     "            if not grammar or isinstance(e, _Deferred):      # review r1", "            if not grammar:      # review r1"),
    ("Y14 the file stream never yields", CE,
     "            if finish is None and watch.owner_waits():", "            if False:"),
    ("Y15 a side stream runs the head's hook", LL, "                    if _on_main and side_main:\n", "                    if False:\n"),
    ("Y16 a failed side stream is not marked displaced", LL,
     '            if _on_main and (_state["answered"] or (side_main and not is_background and _state["open"])):',
     '            if _on_main and _state["answered"]:'),
    ("Y17 the advancer's classify call takes no side path", PA,
     '            }, side_main=True, task_label="project-classify")', '            }, task_label="project-classify")'),
    ("Y18 one rewrite past the cap counts", LG,
     "            if rewrites.get(rewrite_of, 0) >= MAX_COUNTED_REWRITES:", "            if rewrites.get(rewrite_of, 0) > MAX_COUNTED_REWRITES:"),
    ("Y19 rewrites are never counted", LG,
     "            rewrites[rewrite_of] = rewrites.get(rewrite_of, 0) + 1\n", "            pass\n"),
    ("Y20 an append=\"true\" part is a rewrite", AG,
     '    return not (app is True or str(app).strip().lower() in ("true", "1", "yes"))', "    return not (app is True)"),
    ("Y21 the loop never passes rewrite_of", AG,
     '                                                     rewrite_of=(_fs_norm(ptarget or "") if fname == "file_system"',
     '                                                     rewrite_of=("" if fname == "file_system"'),
    ("Y22 spellings of one file count apart", AG,
     '                                                     rewrite_of=(_fs_norm(ptarget or "") if fname == "file_system"',
     '                                                     rewrite_of=(str(ptarget or "") if fname == "file_system"'),
    ("Y23 a capped rewrite re-arms a run", AG,
     '                                        if not _whole or _rec["rewrites"] <= _glog.MAX_COUNTED_REWRITES:', "                                        if True:"),
    ("Y24 web/CLI get a hard limit", AG, "    return 0.0 < d < math.inf\n", "    return 0.0 < d\n"),
    ("Y25 the tools are stopped after the cancel", AG,
     '    stopped = registry.interrupt_tools(req_id)\n    res = registry.cancel(req_id, reason=f"{HARD_LIMIT_REASON} ({_span(limit)})")\n',
     '    res = registry.cancel(req_id, reason=f"{HARD_LIMIT_REASON} ({_span(limit)})")\n    stopped = registry.interrupt_tools(req_id)\n'),
    ("Y27 unread queued messages are lost", AG,
     '        registry.add_leftovers(turn.session_id, turn.kind, res.get("undelivered_items") or [])', "        pass"),
    ("Y28 the watch is never started", AG, "                if hard_limit_applies(_cdl):\n", "                if False:\n"),
    ("Y29 the watch outlives its turn", AG,
     "            if _hw is not None and not _hw.done():\n                _hw.cancel()", "            if _hw is not None and not _hw.done():\n                pass"),
    ("Y30 the stopped reply names no files", AG,
     '            if str(_tc.reason or "").startswith(HARD_LIMIT_REASON):\n', "            if False:\n"),
    ("Y31 the watch stops before the client's own timeout", AG,
     "            rem = max(rem, float(client_s) - el)", "            pass"),
    ("Y33 a stop after the claim leaves the task claimed", PA,
     "    except asyncio.CancelledError:\n        if _own:", "    except ZeroDivisionError:\n        if _own:"),
    ("Y35 the unattended step's claim is reopened inside its own timeout", PA,
     '    _own = kwargs.get("claim_sink") is None', "    _own = True"),
    ("Y36 a stop reopens a leaf that already left IN_PROGRESS", PA,
     '        if store is not None and _task_status(store, project_id, tid) == "IN_PROGRESS":', "        if store is not None:"),
    ("Y34 the reply names the server cap", AG,
     "    limit = max(_glog.request_deadline_s(req_id), float(client_s or 0.0))", "    limit = _glog.request_deadline_s(req_id)"),
    ("Y37 a stopped build loses its finished files", CE,
     "            await asyncio.wait_for(asyncio.shield(_write_pending()), STOP_FLUSH_S)", "            pass"),
    ("Y38 a stop inside an attempt skips the flush", CE,
     "    except asyncio.CancelledError:\n        # §4SD open items r3/r4", "    except ZeroDivisionError:\n        # §4SD open items r3/r4"),
    ("Y41 a stopped background build flushes behind its reopen", CE,
     "        if not is_background and caller_waits_context.get():", "        if not is_background:"),
    ("Y42 the unattended step says it waits", PA,
     "    tok_w = caller_waits_context.set(False)", "    tok_w = caller_waits_context.set(True)"),
    ("Y43 a verify the stop killed is a failed attempt", CE,
     "        if _stop_requested():\n            raise asyncio.CancelledError()\n    except asyncio.CancelledError:",
     "        pass\n    except asyncio.CancelledError:"),
    ("Y44 the stop is never read", CE,
     "        return _ts.requested()", "        return False"),
    ("Y45 a stopped step's time is not booked", PA,
     '                _bump_meta(store, project_id, lambda m: {**m, "unattended_runtime_seconds": float(   # r5 m3',
     '                (lambda *a: None)(store, project_id, lambda m: {**m, "unattended_runtime_seconds": float(   # r5 m3'),
    ("Y40 an unattended step stopped from outside stays claimed", PA,
     '                _after_abort(store, project_id, claimed, t0, "stopped", "step was stopped")', "                pass"),
    ("Y39 the stop's files line is stored as a reply", RT,
     '                        r"(?:\\s*Files written so far: [^\\n]*\\.)?",', '                        r"",'),
    ("Y46 the agentic leaf books a stop-killed verify FAILED", CL,
     "    if _stop_requested():\n        raise asyncio.CancelledError()\n    return CodingResult(False, f\"agentic leaf failed after",
     "    return CodingResult(False, f\"agentic leaf failed after"),
    ("Y47 the hard limit cuts a running build", AG,
     "    return [n for n in names if n in HARD_LIMIT_SPARED_TOOLS]", "    return []"),
    ("Y48 the build's report gets no grace", AG,
     "        if spared:\n            last_spared = now\n", "        pass\n"),
    ("Y49 a wedged build is never cut", AG,
     "if spared and now - past_since < float(spare_max_s):", "if spared:"),
    ("Y50 the grace is capped at the constant", AG,
     "report_grace_s = max(HARD_LIMIT_REPORT_GRACE_S, effective_report_floor(",
     "report_grace_s = min(HARD_LIMIT_REPORT_GRACE_S, effective_report_floor("),
    ("Y51 the spare cap restarts every poll", AG,
     "            past_since = now if past_since is None else past_since\n", "            past_since = now\n"),
    ("Y52 the owner's stop does not end the watch", AG,
     "        if turn is None or turn.cancelled:\n            return False\n        now",
     "        if turn is None:\n            return False\n        now"),
    ("Y32 the watch spins on a closed request", AG,
     "        rem = _glog.request_remaining_s(req_id)\n        if rem is None:\n            return False",
     "        rem = _glog.request_remaining_s(req_id)\n        if rem is None:\n            rem = 0.0"),
]

if __name__ == "__main__":
    m.main()
