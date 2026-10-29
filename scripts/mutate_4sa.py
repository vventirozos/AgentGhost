"""§4SA battery — head/conversation review fixes (2026-10-29): shutdown save,
level-change save, "moved on" vs "no head", probe idle clock, deferral INFO
lines, the 3072-token save floor, the growing save bound, the head-first gate.
Reuses the §4QA harness (in place, backup + restore + byte compare, two-tier,
controls).
    python3 scripts/mutate_4sa.py [all|check|rerun S1 S2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

AG = "src/ghost_agent/core/agent.py"; LL = "src/ghost_agent/core/llm.py"
HS = "src/ghost_agent/core/head_saves.py"; MN = "src/ghost_agent/main.py"
m.PINS = ["tests/test_4sa_head_review_fixes.py", "tests/test_4rv_conv_keep.py", "tests/test_4ro_head_saves.py",
          "tests/test_4rp_slot_keep.py", "tests/test_4rwg_side_call.py", "tests/test_outside_call_4rq.py",
          "tests/test_shutdown_line.py"]


def _tests():
    import glob, subprocess
    pat = (r"head_saves\|_slot_switch_before\|note_main_slot_prompt\|slot_kept\|_conv_save\|_conv_restore\|"
           r"conv_keep_enabled\|_head_defer\|_heartbeat\|save_slot_conversation_at_shutdown\|last_activity_time")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", HS, "def save_timeout_s(n_tokens: int) -> float:",
     "# ctl\ndef save_timeout_s(n_tokens: int) -> float:"),
    ("CONTROL known-bad (must DIE): the save ignores the token floor", AG,
     "            if n is not None and n < n_head + conv_min_past_tokens() and still_it():",
     "            if False:"),
    # ── the shutdown save ──
    ("S1 shutdown saves nothing", AG,
     "                return bool(await _wait_for(self._conv_save(llm, \"the agent's shutdown\", late),\n                                            SHUTDOWN_CONV_SAVE_S + 1.0))",
     "                return False"),
    ("S2 shutdown saves during a running turn", AG,
     '            if (int(getattr(llm, "foreground_requests", 0) or 0) > 0\n'
     '                    or getattr(self, "_turn_lock_rid", None)):',
     '            if (int(getattr(llm, "foreground_requests", 0) or 0) > 0):'),
    ("S3 shutdown save unbounded", AG,
     "            late = time.monotonic() + SHUTDOWN_CONV_SAVE_S\n            try:",
     "            late = time.monotonic() + 3600.0\n            try:"),
    ("S4 main.py no longer calls the shutdown save", MN,
     "            await agent.save_slot_conversation_at_shutdown()", "            pass"),
    # ── the level-change save ──
    ("L1 level change saves nothing", AG,
     '            await self._conv_save(llm, f"this conversation\'s change to {cur_level} effort", _late)',
     "            pass"),
    ("L2 level change saves inside the same request too", AG,
     "        if (_conv and level_moved and owner and owner == key and isinstance(held_by, str) and held_by\n"
     "                and held_by != rid and self._turn_holder_ok()):",
     "        if (_conv and level_moved and owner and owner == key and isinstance(held_by, str) and held_by\n"
     "                and self._turn_holder_ok()):"),
    ("L3 level-change save after the level was updated", AG,
     "        if (_conv and level_moved and owner and owner == key and isinstance(held_by, str) and held_by\n"
     "                and held_by != rid and self._turn_holder_ok()):\n"
     "            # §4SA: a NEW request of this conversation at another level — the slot\n"
     "            # holds the conversation at the OLD level: to disk first, with that\n"
     "            # level (live 10-28: the head went over it unsaved, and when the level\n"
     "            # came back the file was three requests old). Not inside a request\n"
     "            # (the thinking-off turn flips the level and back): one file per\n"
     "            # conversation, and two multi-GB saves per flip would cost more than\n"
     "            # the re-read they could spare.\n"
     '            await self._conv_save(llm, f"this conversation\'s change to {cur_level} effort", _late)\n'
     "        if cur_level is not None:\n"
     "            self._slot_level = cur_level             # what this call leaves in the slot\n",
     "        if cur_level is not None:\n"
     "            self._slot_level = cur_level             # what this call leaves in the slot\n"
     "        if (_conv and level_moved and owner and owner == key and isinstance(held_by, str) and held_by\n"
     "                and held_by != rid and self._turn_holder_ok()):\n"
     "            # §4SA: a NEW request of this conversation at another level — the slot\n"
     "            # holds the conversation at the OLD level: to disk first, with that\n"
     "            # level (live 10-28: the head went over it unsaved, and when the level\n"
     "            # came back the file was three requests old). Not inside a request\n"
     "            # (the thinking-off turn flips the level and back): one file per\n"
     "            # conversation, and two multi-GB saves per flip would cost more than\n"
     "            # the re-read they could spare.\n"
     '            await self._conv_save(llm, f"this conversation\'s change to {cur_level} effort", _late)\n'),
    # ── "moved on" vs "no head" ──
    ("M1 moved-on never set", LL,
     "        llm._slot_moved_on = bool(background or requester_is_member()\n                                  or (llm._head_displaced and not shares_head_tools(payload, ref)))",
     "        llm._slot_moved_on = False"),
    ("M2 tools-shared calls count as moved on", LL,
     "        llm._slot_moved_on = bool(background or requester_is_member()\n                                  or (llm._head_displaced and not shares_head_tools(payload, ref)))",
     "        llm._slot_moved_on = bool(background or requester_is_member() or llm._head_displaced)"),
    ("M5 still_it reads the head mark", AG,
     "                        and not slot_moved_on(llm))",
     '                        and getattr(llm, "_head_displaced", False) is not True)'),
    ("M6 switch early-return reads the head mark", AG,
     "                _conv and slot_moved_on(llm)):",
     '                _conv and getattr(llm, "_head_displaced", False) is True):'),
    ("M7 tools calls never restore a conversation file", AG,
     '        if (_conv and key and kind in ("head", "tools") and not _long_followup',
     '        if (_conv and key and kind == "head" and not _long_followup'),
    ("M8 head_in_slot not recorded", AG,
     '            _extra = {"head_in_slot": getattr(llm, "_head_displaced", False) is not True,\n                      "calls": int(getattr(llm, "_slot_request_calls", 0) or 0),   # §4SA: for the head-first gate',
     '            _extra = {"calls": int(getattr(llm, "_slot_request_calls", 0) or 0),   # §4SA: for the head-first gate'),
    ("M9 restore ignores the meta", AG,
     '        llm._head_displaced = _head_saves.meta_field(hs["dir"], ckey, "head_in_slot") is False',
     "        llm._head_displaced = False"),
    ("M10 forget does not mark moved on", AG,
     "        llm._slot_moved_on = True                      # §4SA", "        pass"),
    ("M11 side call why reads the head mark", AG,
     '"the slot was already displaced" if slot_moved_on(llm) else',
     '"the slot was already displaced" if getattr(llm, "_head_displaced", False) is True else'),
    # ── the idle clock ──
    ("P1 probes move the gap clock", AG,
     "            if not is_probe_request_id(_rid):\n                self.context.last_real_activity_time = datetime.datetime.now()",
     "            if True:\n                self.context.last_real_activity_time = datetime.datetime.now()"),
    ("P2 the gap clock reads the scheduler's", AG,
     "            last = getattr(ctx, \"last_real_activity_time\", None)\n            if not isinstance(last, datetime.datetime):\n                last = ctx.last_activity_time",
     "            last = ctx.last_activity_time"),
    # ── deferral lines ──
    ("D1 deferral always INFO", AG,
     '            if why == getattr(self, "_head_defer_last_why", None):', "            if False:"),
    ("D2 deferral never INFO", AG,
     '            if why == getattr(self, "_head_defer_last_why", None):', "            if True:"),
    ("D3 memory never reset", AG, "        self._head_defer_last_why = None", "        pass"),
    ("D4 kept-why words", AG,
     "                if conv_keep_enabled() else", "                if False else"),
    # ── the save floor ──
    ("T1 threshold back to 1024", AG, "CONV_MIN_PAST_TOKENS = 3072", "CONV_MIN_PAST_TOKENS = 1024"),
    ("T2 env ignored", AG,
     "        return max(0, int(raw)) if raw else CONV_MIN_PAST_TOKENS", "        return CONV_MIN_PAST_TOKENS"),
    ("T3 save uses the constant", AG,
     "            if n is not None and n < n_head + conv_min_past_tokens() and still_it():",
     "            if n is not None and n < n_head + CONV_MIN_PAST_TOKENS and still_it():"),
    # ── the save bound ──
    ("B1 save bound flat", AG,
     "            save_s = min(_head_saves.save_timeout_s(n), late - time.monotonic())",
     "            save_s = min(_head_saves.SAVE_TIMEOUT_S, late - time.monotonic())"),
    ("B2 bound does not grow", HS,
     "    gb = (n * STATE_BYTES_PER_TOKEN + CHECKPOINTS_BYTES) / 1e9\n"
     "    return max(SAVE_TIMEOUT_S, RESTORE_S_PER_GB * gb)",
     "    gb = (n * STATE_BYTES_PER_TOKEN + CHECKPOINTS_BYTES) / 1e9\n"
     "    return SAVE_TIMEOUT_S"),
    # ── the head-first gate ──
    ("G1 gate ignored (always off)", AG,
     "                          and held_by != rid and 0 < _head_first <= _calls)",
     "                          and False)"),
    ("G2 gate off-by-one", AG,
     "                          and held_by != rid and 0 < _head_first <= _calls)",
     "                          and held_by != rid and 0 < _head_first < _calls)"),
    ("G3 gate restores the older file too", AG,
     '        if (_conv and key and kind in ("head", "tools") and not _long_followup',
     '        if (_conv and key and kind in ("head", "tools")'),
    ("G4 calls not counted", LL,
     '                llm._slot_request_calls = (int(getattr(llm, "_slot_request_calls", 0) or 0) + 1\n'
     '                                           if rid and rid == getattr(llm, "_slot_request", None) else 1)',
     "                llm._slot_request_calls = 1"),
    ("G5 env ignored", LL,
     "        return max(0, int(raw)) if raw else 0", "        return 0"),
]


# review round 1 (the gate reviewer): the counter and the file
m.MUT += [
    ("G6 release keeps the count", LL,
     "        llm._slot_request_calls = 0           # §4SA review: a count never outlives its request\n", ""),
    ("G7 a restored conversation inherits the count", AG,
     "        llm._slot_request_calls = 0                 # §4SA review: this request's calls, not the last one's\n", ""),
    ("G8 the file's count is not gated", AG,
     "            if 0 < _gate <= _calls_saved and _saved_rid != rid:", "            if False:"),
    ("G9 the count is not saved", AG,
     '                      "calls": int(getattr(llm, "_slot_request_calls", 0) or 0),   # §4SA: for the head-first gate',
     '                      "calls": 0,'),
]


# review round 1 (the ownership reviewer)
m.MUT += [
    ("A1 after-side lift head-only again", AG,
     "        if kind in done.get(rid, ()) and not level_moved and not _after_side:\n            return\n        if _after_side:\n            self._side_call_rid = None",
     '        if kind in done.get(rid, ()) and not level_moved and not (_after_side and kind == "head"):\n            return\n        if kind == "head":\n            self._side_call_rid = None'),
    ("A2 after-side on the head mark", AG,
     '        _after_side = (getattr(self, "_side_call_rid", None) == rid and bool(rid) and slot_moved_on(llm))',
     '        _after_side = (getattr(self, "_side_call_rid", None) == rid and bool(rid) and getattr(llm, "_head_displaced", False) is True)'),
    ("A3 no stamp on release", LL,
     "        if stamp:\n            llm._slot_changed_at = time.monotonic()", "        pass"),
    ("A4 moved-on not role-gated", LL,
     "        llm._slot_moved_on = bool(background or requester_is_member()\n                                  or (llm._head_displaced and not shares_head_tools(payload, ref)))",
     "        llm._slot_moved_on = bool(background\n                                  or (llm._head_displaced and not shares_head_tools(payload, ref)))"),
    ("A5 the tick names a kept conversation", AG,
     "                if getattr(llm, \"_head_displaced\", False) is True and not _kept():\n                    return str(getattr(llm, \"_head_displaced_by\", \"\") or \"unlabelled\")\n                return \"\"",
     "                if getattr(llm, \"_head_displaced\", False) is True:\n                    return str(getattr(llm, \"_head_displaced_by\", \"\") or \"unlabelled\")\n                return \"\""),
    ("A6 restore keeps the early mark", AG,
     "        llm._head_displaced_early = False           # (review: a background call's marks must not outlive it)\n", ""),
    ("B3 a background save keeps the 60 s cap", AG,
     "            if not late_given:\n                late = max(late, time.monotonic() + min(_head_saves.save_timeout_s(n),",
     "            if False:\n                late = max(late, time.monotonic() + min(_head_saves.save_timeout_s(n),"),
    ("K1 one file per conversation again (save)", AG,
     '            key = _head_saves.conv_key(hs.get("identity", ""), owner, getattr(self, "_slot_level", None))',
     '            key = _head_saves.conv_key(hs.get("identity", ""), owner)'),
    ("K2 the restore ignores the level file", AG,
     '        ckey = _head_saves.conv_key(hs.get("identity", ""), session, level) if level else None',
     '        ckey = None'),
    ("S5 shutdown save during a draining stream", AG,
     "            if int(getattr(llm, \"_main_streams_open\", 0) or 0) > 0 or slot_moved_on(llm):",
     "            if False:"),
]


# review round 2 (fresh readers on the round-1 fixes)
m.MUT += [
    ("M3 ownership back on the head only", LL,
     "            if key and (key == owner or not llm._slot_moved_on):",
     "            if key and (key == owner or not llm._head_displaced):"),
    ("N1 a refused late stream still writes the marks", LL,
     "            if isinstance(changed_at, (int, float)) and changed_at > opened_at:\n                # §4SA review r2",
     "            if False:\n                # §4SA review r2"),
    ("N2 the file gate does not exempt its own request", AG,
     "            if 0 < _gate <= _calls_saved and _saved_rid != rid:", "            if 0 < _gate <= _calls_saved:"),
    ("N3 the claim ignores which file", AG,
     "                    and (len(_ls) < 4 or _ls[3] == ckey)         # §4SA r2: this level's file\n", ""),
    ("N4 a keyless look over an owned slot saves nothing", AG,
     '                await self._conv_displace(llm, "a request\'s picture look", _late)', "                pass"),
    ("N5 the first tick names a kept conversation", AG,
     "                    and getattr(llm, \"_head_displaced\", False) is True and not _kept()):",
     "                    and getattr(llm, \"_head_displaced\", False) is True):"),
    ("N6 the shutdown waits on the lock without a bound", AG,
     "                return bool(await _wait_for(self._conv_save(llm, \"the agent's shutdown\", late),\n                                            SHUTDOWN_CONV_SAVE_S + 1.0))",
     "                return bool(await self._conv_save(llm, \"the agent's shutdown\", late))"),
    ("N7 prune counts files, not conversations", HS,
     "            same_conv = kind == CONV and cid in conv_seen", "            same_conv = False"),
    ("N8 the re-warm's release stamps a change of hands", AG,
     "                release_slot_session(llm, stamp=False)   # §4RP: the head owns the slot again (§4SA r2: no stamp)",
     "                release_slot_session(llm)   # §4RP: the head owns the slot again (§4SA r2: no stamp)"),
    ("N9 the level file is not tried first", AG,
     '        ckey = _head_saves.conv_key(hs.get("identity", ""), session, level) if level else None\n        if not (ckey and _head_saves.stored_tokens(hs["dir"], ckey)):\n            ckey = _head_saves.conv_key(hs.get("identity", ""), session)',
     '        ckey = _head_saves.conv_key(hs.get("identity", ""), session)'),
    ("N10 the background bound margin back to 5 s", AG,
     "                                                        BEFORE_MAIN_DISPATCH_S - CONV_SAVE_WAIT_S - 15.0))",
     "                                                        BEFORE_MAIN_DISPATCH_S - CONV_SAVE_WAIT_S - 5.0))"),
    ("N11 the meta forgets its request", AG,
     '                      "rid": str(getattr(llm, "_slot_request", "") or ""),       # (r2: the gate exempts this request)',
     '                      "rid": "",'),
    ("N12 the meta forgets its conversation", AG,
     '                      "conv": _head_saves.conv_key(hs.get("identity", ""), owner)}  # (r2: prune counts conversations)',
     '                      "conv": ""}'),
]

if __name__ == "__main__":
    m.main()
