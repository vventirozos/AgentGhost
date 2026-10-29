"""§4SB battery — queued-messages fixes (2026-10-29): the extra step when a
message lands while the reply is generated, the read marker + drained-items
record, the hand-back of drained-but-unread rows (finally + persisted rows),
the stop that reaches tool batches (turn cancel, inbox stop, Stop button),
every running turn cancelled, the leftovers cap log + return, the mid-turn
refusal wordings, the cancel-stub not persisted, inbox text normalised.
Reuses the §4QA harness (in place, backup + restore + byte compare, two-tier,
controls).
    python3 scripts/mutate_4sb.py [all|check|rerun X1 X2 …]
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "mutate_4qa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutate_4qa.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

AG = "src/ghost_agent/core/agent.py"; RT = "src/ghost_agent/api/routes.py"
TU = "src/ghost_agent/core/turns.py"; TS = "src/ghost_agent/utils/tool_stop.py"
PV = "src/ghost_agent/utils/provenance.py"
m.PINS = ["tests/test_4sb_extra_step.py", "tests/test_4sb_queue_routes.py", "tests/test_4sb_queued_wording.py",
          "tests/test_queued_messages.py", "tests/test_tool_interrupt.py", "tests/test_sessions_and_cancel.py"]


def _tests():
    import glob, subprocess
    pat = (r"queued_messages\|turn_registry\|TurnRegistry\|tool_stop\|interrupt_tools\|_persist_session\|"
           r"turn/inbox\|turn/cancel\|_hand_back_unread\|refuse_unless_user_said\|status_only")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(m.ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, m.ROOT) for t in out} | set(m.PINS))


m._tests = _tests
m.MUT = [
    ("CONTROL no-op (must SURVIVE)", AG,
     "    def _hand_back_unread(turn_reg, turn, delivered_rows) -> int:",
     "    # ctl\n    def _hand_back_unread(turn_reg, turn, delivered_rows) -> int:"),
    ("CONTROL known-bad (must DIE): post() ignores the inbox cap", TU,
     "            if (turn is None or not turn.accepting or turn._cancel or len(turn.inbox) >= MAX_PENDING\n",
     "            if (turn is None or not turn.accepting or turn._cancel or False\n"),
    # ── the extra step ──
    ("X1 no extra step", AG,
     "                if (_q_pending and clean_ui and not getattr(_q_turn, \"_q_extra_used\", False)   # (r2: an empty 200 is not an answer)",
     "                if (False and clean_ui and not getattr(_q_turn, \"_q_extra_used\", False)   # (r2: an empty 200 is not an answer)"),
    ("X2 extra step unbounded", AG,
     "                    _q_turn._q_extra_used = True\n",
     "                    pass\n"),
    ("X3 extra step on the last budget turn", AG,
     "                        and not is_final_generation and not force_final_response and not force_stop\n"
     "                        and turn < effective_max_turns - 1\n",
     "                        and not is_final_generation and not force_final_response and not force_stop\n"),
    ("X4 answer so far dropped", AG,
     "                        final_ai_content += _q_kept\n",
     "                        pass\n"),
    ("X5 the model's own row not appended before the extra step", AG,
     "                    _q_turn._q_extra_used = True\n                    messages.append(msg)\n",
     "                    _q_turn._q_extra_used = True\n"),
    # ── the read marker / drained record ──
    ("R1 read marker never advanced (streamed path)", AG,
     "                    _q_mark_read(_q_t)       # (r2/r3/r4: an empty 200 or a thinking-only reply — split or inline — answered nothing)",
     "                    pass"),
    ("R2 drained items not recorded", AG,
     '                                _active_turn._q_items = list(getattr(_active_turn, "_q_items", ()) or ()) + [dict(_q)]\n',
     "                                pass\n"),
    # ── the hand-back ──
    ("H1 no hand-back in the finally", AG,
     '            self._hand_back_unread(_turn_reg, _active_turn, locals().get("_delivered_rows"))\n',
     "            pass\n"),
    ("H2 hand-back keeps the rows persisted", AG,
     "                            del delivered_rows[i]\n",
     "                            pass\n"),
    ("H3 hand-back hands everything (ignores read)", AG,
     "            unread = [items[i] for i in skipped] + items[read:]", "            unread = items"),
    # ── the stop reaches the tools ──
    ("G1 turn cancel no longer requests the stop", AG,
     "        # thread's docker exec / yt-dlp / browser runner; only the killers do\n"
     "        stop.request()\n"
     "        for t in tasks:\n",
     "        # thread's docker exec / yt-dlp / browser runner; only the killers do\n"
     "        for t in tasks:\n"),
    ("T1 killed text loses the files sentence", TS,
     '                "running, and it was stopped. Its output is not available." + files + tail)\n',
     '                "running, and it was stopped. Its output is not available." + tail)\n'),
    ("I1 inbox stop does not interrupt tools", RT,
     "            interrupted += reg.interrupt_tools(t.req_id)\n",
     "            interrupted += []\n"),
    ("I2 Stop button does not interrupt", RT,
     "        _interrupted = _reg.interrupt_tools(_turn.req_id)\n",
     "        _interrupted = []\n"),
    ("I3 cancel False still answers cancel", RT,
     '            path, reason = qm.PATH_QUEUE, "the turn is finishing"   # §4SB: not "cancel"\n',
     "            pass\n"),
    ("I4 only the newest running turn cancelled", TU,
     "        return sorted(hits, key=lambda t: t.started_at)\n",
     "        return sorted(hits, key=lambda t: t.started_at)[-1:]\n"),
    ("I5 Stop hold not gated", RT,
     '    if _turn is not None and result.get("cancelled") and qm.enabled():\n',
     '    if _turn is not None and result.get("cancelled"):\n'),
    # ── the leftovers ──
    ("L1 leftovers drop silent", TU,
     "            if dropped:\n"
     "                pretty_log(\"Queued Message\",\n"
     "                           f\"{head} {dropped} older unread message(s) dropped (leftovers full)\",\n"
     "                           icon=Icons.QUEUED_MSG)\n",
     "            if dropped:\n"
     "                pass\n"),
    ("L2 add_leftovers ignores the cap", TU,
     "        self._leftovers[session_id] = kept[-MAX_PENDING:]\n",
     "        self._leftovers[session_id] = kept[:]\n"),
    ("L3 add_leftovers returns 0 always", TU,
     "        return min(n, MAX_PENDING)        # the cap drops the OLDEST: earlier rows first\n",
     "        return 0\n"),
    # ── the wordings ──
    ("W1 provenance mid-turn wording lost", PV,
     "    if mid_turn:\n",
     "    if False:\n"),
    ("W2 status-only wording ignores delivered", AG,
     "    if not added:\n        return (\"Not run: the user asked for a STATUS / progress report. Answer from \"\n",
     "    if False:\n        return (\"Not run: the user asked for a STATUS / progress report. Answer from \"\n"),
    # ── the routes ──
    ("P1 the stub is persisted again", RT,
     '        if re.fullmatch(r"_\\(Turn cancelled:[^\\n]*?\\.\\)_"\n'
     '                        r"(?:\\s*No output was produced before the cancellation\\.)?",\n'
     '                        str(assistant_text or "").strip()):\n',
     "        if False:\n"),
    ("N1 inbox keeps \\r", RT,
     '    text = str(body.get("text") or "").replace("\\r", "").strip()   # §4SB: as /api/chat strips it\n',
     '    text = str(body.get("text") or "").strip()   # §4SB: as /api/chat strips it\n'),
]


# review round 1 (routes/wording reviewer)
m.MUT += [
    ("W3 any mid-turn message is a statement", PV,
     '            mid_turn = owner_said(value, owner_statements("\\n".join(added))) == "stated"',
     '            mid_turn = True'),
    ("I6 the oldest cancelled turn is reported", RT,
     "            out.update(request_id=stopped[-1].req_id, request_ids=[t.req_id for t in stopped],",
     "            out.update(request_id=stopped[0].req_id, request_ids=[t.req_id for t in stopped],"),
    ("I7 interrupted tools dropped from a queue reply", RT,
     '            if interrupted:\n                out["interrupted"] = interrupted     # r2: the tools WERE stopped — the client must know',
     '            if False:\n                out["interrupted"] = interrupted     # r2: the tools WERE stopped — the client must know'),
]


# review round 1 (loop reviewer)
m.MUT += [
    ("C1 the carried answer lost on an upstream HTTP error", AG,
     '                    final_ai_content = _with_carry(self.context, request_id_context.get(),\n                                                   f"CRITICAL: Upstream error {e.response.status_code} — the model server could "',
     '                    final_ai_content = (\n                                                   f"CRITICAL: Upstream error {e.response.status_code} — the model server could "'),
    ("C2 the carry helper returns the text alone", AG,
     '        if kept and kept not in text:\n            return kept + "\\n\\n" + text', '        if False:\n            return kept + "\\n\\n" + text'),
    ("S1 the streamed final does not mark the drained rows read", AG,
     "                            _q_mark_read(_active_turn)     # carries every drained row (r4: the overflow's left-out set too)\n",
     "                            pass\n"),
    ("O1 an empty or thinking-only reply marks the drained rows read", AG,
     '                if _q_t is not None and (_strip_think_blocks(str(full_content or "")).strip() or msg.get("tool_calls")):   # drained message',
     '                if _q_t is not None:   # drained message'),
    ("E1 an empty 200 takes the extra step", AG,
     "                if (_q_pending and clean_ui and not getattr(_q_turn, \"_q_extra_used\", False)   # (r2: an empty 200 is not an answer)",
     "                if (_q_pending and not getattr(_q_turn, \"_q_extra_used\", False)   # (r2: an empty 200 is not an answer)"),
    ("H4 handed-back rows lose their accept time", AG,
     "            kept = int(turn_reg.add_leftovers(turn.session_id, turn.kind, [dict(m) for m in unread]) or 0)",
     "            kept = int(turn_reg.add_leftovers(turn.session_id, turn.kind, [dict(m, ts=time.time()) for m in unread]) or 0)"),
]


# review round 2 (server reviewer)
m.MUT += [
    ("K1 the TTL reads the accept time", TU,
     '            fresh = [m for m in self._leftovers[sid] if m.get("kept_at", m.get("ts", 0)) >= cutoff]',
     '            fresh = [m for m in self._leftovers[sid] if m.get("ts", 0) >= cutoff]'),
    ("K2 the carry survives a reused request id", AG,
     '                try:                                               # §4SB r2: and no stale §4PO carry (sched-/probe ids)\n                    getattr(self.context, "_po_carries", {}).pop(str(req_id or ""), None)',
     '                try:                                               # §4SB r2: and no stale §4PO carry (sched-/probe ids)\n                    pass'),
    ("K3 a refuted carry comes back on an error exit", AG,
     '                            try:        # §4SB r2: a refuted carry must not come back on an error exit\n                                getattr(self.context, "_po_carries", {}).pop(str(request_id_context.get() or ""), None)',
     '                            try:        # §4SB r2: a refuted carry must not come back on an error exit\n                                pass'),
    ("K4 skipped overflow rows are not handed back", AG,
     "            unread = [items[i] for i in skipped] + items[read:]", "            unread = items[read:]"),
    ("K5 the overflow retry notes no skipped rows", AG,
     "                                _q_note_overflow(_q_t, _q_rows, _q_sent)    # (r2/r3: the OLDEST rows left out are unread;",
     "                                _q_mark_read(_q_t)    # ("),
]


# review round 3 (server reviewer)
m.MUT += [
    ("Q1 a completed reply keeps the overflow's left-out set", AG,
     "        turn._q_read = len(getattr(turn, \"_q_items\", ()) or ())\n        turn._q_skipped = ()",
     "        turn._q_read = len(getattr(turn, \"_q_items\", ()) or ())"),
    ("Q2 the overflow notes rows an earlier reply had read", AG,
     "            if isinstance(i, int) and i >= read:", "            if isinstance(i, int):"),
    ("Q3 the overflow reads the prompt position, not the drained one", AG,
     "            i = r.get(\"_q_i\") if isinstance(r, dict) else None", "            i = rows.index(r)"),
    # Q4 ("the drain stamps no position") RETIRED in the final battery: equivalent since r4 — a left-out
    # row without its stamp is placed by text among the unread items, and the left-out rows are the
    # drained prefix, so the text match yields the stamped indices; the stamp is the fast path only.
]

# review round 4 (server reviewer)
m.MUT += [
    ("V1 the streamed final keeps the overflow's left-out set", AG,
     "                            _q_mark_read(_active_turn)     # carries every drained row (r4: the overflow's left-out set too)",
     "                            _active_turn._q_read = len(getattr(_active_turn, \"_q_items\", ()) or ())"),
    ("V2 the overflow notes rows read on an EMPTY retry reply", AG,
     "                            if _q_t is not None and not _raw_reply_empty:",
     "                            if _q_t is not None:"),
    ("V3 an inline <think>-only reply counts as an answer", AG,
     '                if _q_t is not None and (_strip_think_blocks(str(full_content or "")).strip() or msg.get("tool_calls")):   # drained message',
     '                if _q_t is not None and (str(full_content or "").strip() or msg.get("tool_calls")):   # drained message'),
    ("V4 an unplaced left-out row books the marker read", AG,
     "        turn._q_read = read if unidentified else len(items)", "        turn._q_read = len(items)"),
    ("V5 a row without its stamp is never placed by text", AG,
     "                i = next((k for k in range(read, len(items))\n"
     "                          if k not in skipped and str((items[k] or {}).get(\"text\") or \"\") == text), None)",
     "                i = None"),
]

# review round 5 (server reviewer)
m.MUT += [
    ("W1 a final that emitted nothing keeps its drained rows booked read", AG,
     "                if not _q_vis:\n                    _active_turn._q_read, _active_turn._q_skipped = _q_pre",
     "                if False:\n                    _active_turn._q_read, _active_turn._q_skipped = _q_pre"),
    ("W3 the final's visible-text check reads the system's fallback as an answer", AG,
     '                    str((_drain_flags.get("model") or (lambda: ""))() or ""))).strip()',
     '                    str((_drain_flags.get("partial") or (lambda: ""))() or ""))).strip()'),
    ("W2 the overflow fallback compares the labelled row text", AG,
     '                if text.startswith(_ql + "\\n"):\n                    text = text[len(_ql) + 1:]\n',
     ''),
]

# review round 6 (server reviewer)
m.MUT += [
    ("Y1 the forced-final retry's answer is not model text", AG,
     "                        _model_text += _ff_candidate      # §4SB r6: the retry ANSWERED — model text, not the system's\n",
     ""),
    ("Y2 the route persists the cut rows before the drain's hand-back", "src/ghost_agent/api/routes.py",
     "                                _cl = content.aclose()\n                                try:\n                                    _cl.send(None)",
     "                                _cl = None\n                                try:\n                                    raise StopIteration"),
    ("Y3 a final closed before its first step runs no tail", AG,
     "        if not self._started:\n            self._started = True\n            try:\n                self._tail()",
     "        if False:\n            self._started = True\n            try:\n                self._tail()"),
    ("Y4 a leaked raw reasoning prefix counts as an answer", AG,
     "                _q_vis = strip_raw_orphan_reasoning(_strip_think_blocks(\n                    str((_drain_flags.get(\"model\") or (lambda: \"\"))() or \"\"))).strip()",
     "                _q_vis = (_strip_think_blocks(\n                    str((_drain_flags.get(\"model\") or (lambda: \"\"))() or \"\"))).strip()"),
    ("Y5 the marker never advances after the hand-back", AG,
     "                                           getattr(_active_turn, \"_q_delivered_rows\", None))\n                    _q_mark_read(_active_turn)",
     "                                           getattr(_active_turn, \"_q_delivered_rows\", None))"),
]

# review round 7 (server reviewer): W3/Y4 re-anchored to the r7 gate text
m.MUT = [mm for mm in m.MUT if mm[0].split(" ")[0] not in ("W3", "Y4")]
m.MUT += [
    ("W3 the final's visible-text check reads the system's fallback as an answer", AG,
     '                    str((_drain_flags.get("model") or (lambda: ""))() or "")))).strip()',
     '                    str((_drain_flags.get("partial") or (lambda: ""))() or "")))).strip()'),
    ("Y4 a leaked raw reasoning prefix counts as an answer", AG,
     '                _q_vis = _MODULE_SCRUB_RE.sub("", strip_raw_orphan_reasoning(_strip_think_blocks(\n',
     '                _q_vis = _MODULE_SCRUB_RE.sub("", (_strip_think_blocks(\n'),
    ("Z1 scrubbed call markup counts as an answer", AG,
     '                _q_vis = _MODULE_SCRUB_RE.sub("", strip_raw_orphan_reasoning(_strip_think_blocks(\n'
     '                    str((_drain_flags.get("model") or (lambda: ""))() or "")))).strip()',
     '                _q_vis = (strip_raw_orphan_reasoning(_strip_think_blocks(\n'
     '                    str((_drain_flags.get("model") or (lambda: ""))() or "")))).strip()'),
]

# review round 8 (server reviewer): the retract yields back OUTSIDE the guarded try (r6's shape)
m.MUT += [
    ("Z3 the tap's retract frames yield outside the guarded try", "src/ghost_agent/api/routes.py",
     """                    _stream_ended = False
                    try:
                        # §4SB r7: the tap's retract frames yield INSIDE the guarded
                        # region — a hang-up at one of them closes `content` below
                        # (unstarted: its tail hands the drained rows back)
                        if _tap is not None:
                            for _rf in _tap.retract_before_stream():
                                yield _rf
""", """                    if _tap is not None:
                        for _rf in _tap.retract_before_stream():
                            yield _rf
                    _stream_ended = False
                    try:
                        if _tap is not None:
                            pass
"""),
]

# review round 9 (server reviewer): the tap finally ignores a finished, untaken turn
m.MUT += [
    ("Z4 a hang-up at a tap frame after the turn finished closes nothing", "src/ghost_agent/api/routes.py",
     "                        elif not _taken and not _turn.cancelled() and _turn.exception() is None:",
     "                        elif False:"),
]

# review round 10 (server reviewer): the finished str reply stored whole
m.MUT += [
    ("Z5 a finished non-stream reply is stored as the tap's partial under a cut note", "src/ghost_agent/api/routes.py",
     "                                if isinstance(_c, str) and _c.strip():\n",
     "                                if False:\n"),
]

if __name__ == "__main__":
    m.main()
