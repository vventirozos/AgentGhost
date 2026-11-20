#!/usr/bin/env python3
r"""§4UW battery — character / mood / tone review fixes (2026-11-18): the bound of 3, failed and cut turns, an
autoadvance batch's tasks, the displeased reading in front of the correction judge, the shade of the reply
that reports the work, the return lift and a complaint, the journal's rest.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated. A no-op control must
SURVIVE, a known-bad control must DIE.      python3 scripts/mutate_4uw.py [-jN] [all|rerun M1 M2 …]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MO = "src/ghost_agent/selfhood/mood.py"; ST = "src/ghost_agent/selfhood/state.py"
MD = "src/ghost_agent/selfhood/model.py"; SE = "src/ghost_agent/selfhood/sentiment.py"
UC = "src/ghost_agent/distill/user_correction.py"; AG = "src/ghost_agent/core/agent.py"
MUT = [
    ("CONTROL no-op", MO, r'''def sounds_displeased(emotions) -> bool:''', r'''def sounds_displeased(emotions) -> bool:  # no-op'''),
    ("CONTROL known-bad", MO, r'''    if not isinstance(emotions, dict):
        return False
    return any(k in SENTIMENT_COLD''', r'''    return True
    return any(k in SENTIMENT_COLD'''),
    # B — the bound of 3 (the first cut, an eased push, is removed: see the journal)
    ("B1 old-bound", MO, r'''INTENSITY_THRESHOLDS = (0.20, 0.45, 0.90)''', r'''INTENSITY_THRESHOLDS = (0.20, 0.45, 0.75)'''),
    ("B2 old-anchor", MO, r'''ANCHOR_MAGNITUDE = {1: 0.32, 2: 0.60, 3: 0.96}''', r'''ANCHOR_MAGNITUDE = {1: 0.32, 2: 0.60, 3: 0.90}'''),
    ("B3 moved-always", ST, r'''            if moved is not None and (abs(moved[0] - float(dp)) >= 5e-5 or abs(moved[1] - float(de)) >= 5e-5):''', r'''            if moved is not None:'''),
    ("B4 moved-never", ST, r'''            if moved is not None and (abs(moved[0] - float(dp)) >= 5e-5 or abs(moved[1] - float(de)) >= 5e-5):''', r'''            if False:'''),
    # T — what one turn's events are
    ("T1 lost-not-counted", MO, r''' + guards + pressure + lost)''', r''' + guards + pressure)'''),
    ("T3 failed-adds", MO, r'''        units = max(units, 1.0)''', r'''        units = units + 1.0'''),
    ("T3b failed-no-weight", MO, r'''        units = max(units, 1.0)''', r'''        pass'''),
    ("T4 failed-on-a-cut", MO, r'''    turn_failed = bool(getattr(sig, "failed", False)) and bool(sig.completed)''', r'''    turn_failed = bool(getattr(sig, "failed", False))'''),
    ("T5 reward-with-lost", MO, r'''             and not lost and not turn_failed)''', r'''             and not turn_failed)'''),
    ("T6 reward-when-failed", MO, r'''             and not lost and not turn_failed)''', r'''             and not lost)'''),
    ("T7 cut-is-a-whole-turn", MO, r'''    if not bool(sig.completed):
        # §4UW''', r'''    if False:
        # §4UW'''),
    ("T8 cut-drops-tasks", MO, r'''nothing more
        if done:''', r'''nothing more
        if False:'''),
    ("T9 cut-marker-alone", MO, r'''        if out:
            out.append(Event(0.0, 0.0, CUT_SHORT_CAUSE, KIND_TURN))''', r'''        if True:
            out.append(Event(0.0, 0.0, CUT_SHORT_CAUSE, KIND_TURN))'''),
    ("T10 lost-not-said", MO, r'''        if lost:
            parts.append''', r'''        if False:
            parts.append'''),
    ("T11 failed-not-said", MO, r'''        if turn_failed:
            parts.append''', r'''        if False:
            parts.append'''),
    # A — the agent's hooks
    ("A1 batch-done", AG, r'''        return (sum(1 for v in last.values() if v == "DONE"), sum(1 for v in last.values() if v == "FAILED"))''', r'''        return (0, sum(1 for v in last.values() if v == "FAILED"))'''),
    ("A2 batch-failed", AG, r'''        return (sum(1 for v in last.values() if v == "DONE"), sum(1 for v in last.values() if v == "FAILED"))''', r'''        return (sum(1 for v in last.values() if v == "DONE"), 0)'''),
    ("A3 shape-any-advanced", AG, r'''(not act and isinstance(advanced, list) and "stop_reason" in payload)''', r'''(not act and isinstance(advanced, list))'''),
    ("A4 any-action", AG, r'''    if act not in ("", "task_update", "autoadvance"):
        return 0, 0
    advanced''', r'''    advanced'''),
    ("A5 failed-update-counts", AG, r'''(act == "task_update" and getattr(call, "error", None))''', r'''False'''),
    ("A6 batch-row-a-failure", AG, r'''        failed_calls = [c for c in calls if c.error and not mood_tasks_failed(c)]''', r'''        failed_calls = [c for c in calls if c.error]'''),
    ("A7 cut-as-finished", AG, r'''                                completed=not aborted,''', r'''                                completed=True,'''),
    ("A8 cut-blamed-lockdown", AG, r'''                                lockdown=bool(_plock) and not aborted,''', r'''                                lockdown=bool(_plock),'''),
    ("A9 failed-not-passed", AG, r'''                                failed=(traj.outcome == Outcome.FAILED.value)),''', r'''                                failed=False),'''),
    ("A10 shade-heard-not-passed", AG, r'''                            shade_heard=_shade_heard)''', r'''                            shade_heard="")'''),
    ("A11 loop-does-not-mark", AG, r'''                    mood_note_task_result(str(request_id_context.get() or ""), fname, str_res)   # §4UW''', r'''                    pass'''),
    ("A12 strict-json", AG, r'''        return json.JSONDecoder().raw_decode(str(text or "").lstrip())[0]''', r'''        return json.loads(str(text or ""))'''),
    ("A13 frozen-empty", AG, r'''                if got[0]:
                    return got[0]            # a line once heard''', r'''                if True:
                    return got[0]            # a line once heard'''),
    ("A14 pride-under-hold", AG, r'''            if not text and mood_tasks_done_so_far(req_id) > 0 and sm.may_shade_now():''', r'''            if not text and mood_tasks_done_so_far(req_id) > 0:'''),
    ("A15 pride-without-task", AG, r'''            if not text and mood_tasks_done_so_far(req_id) > 0 and sm.may_shade_now():''', r'''            if not text and sm.may_shade_now():'''),
    ("A16 reading-not-asked", AG, r'''            return _sd(await _sentiment.reading_for(str(request_id_context.get() or "")))''', r'''            return False'''),
    ("A17 slot-for-queued", AG, r'''                    if not queued:
                        # §4UW''', r'''                    if True:
                        # §4UW'''),
    ("A18 no-complaint", AG, r'''                    complaint = bool(_hcp(self._msg_text({"content": text})))''', r'''                    complaint = False'''),
    ("A19 other-tool-marks", AG, r'''        if not rid or str(tool_name or "").strip() != "manage_projects":
            return 0''', r'''        if not rid:
            return 0'''),
    ("A20 task-marker-kept", AG, r'''            _MOOD_TASK_DONE_REQ.pop(str(req_id or ""), None)    # §4UW''', r'''            pass    # §4UW'''),
    ("A21 gate-second-ask", AG, r'''            if not is_correction_candidate(current_user_text, prev_assistant,
                                           displeased=await self._mood_message_displeased()):
                return None''', r'''            return None'''),
    # M — the self-model
    ("M1 never-heard", MD, r'''heard=bool(_name) and _name == str(shade_heard or ""))''', r'''heard=False)'''),
    ("M2 always-heard", MD, r'''heard=bool(_name) and _name == str(shade_heard or ""))''', r'''heard=bool(_name))'''),
    ("M3 shade-under-hold", MD, r'''            return bool(self.state is not None and not self.state.hold_reason(now=now))''', r'''            return bool(self.state is not None)'''),
    ("M4 lift-for-a-complaint", MD, r'''        if not complaint:
            self._apply(M.return_event(quiet), now=now)''', r'''        if True:
            self._apply(M.return_event(quiet), now=now)'''),
    ("M5 heard-not-taken", MD, r'''            if heard:
                self.shade.take(now=self._now_of(now))''', r'''            if False:
                self.shade.take(now=self._now_of(now))'''),
    # M6 shade-when-off: EQUIVALENT (a switched-off self-model has no state) — the `self.enabled and` it removed is gone
    # S — the reading memo
    ("S1 overwritten", SE, r'''        if rid and rid in _READINGS and _READINGS[rid] is _PENDING:''', r'''        if rid and rid in _READINGS:'''),
    ("S2 unasked-kept", SE, r'''        if rid and rid in _READINGS and _READINGS[rid] is _PENDING:''', r'''        if rid and _READINGS.get(rid) is _PENDING:'''),
    ("S3 answer-not-noted", SE, r'''        _note_reading(request_id, (answer or {}).get("emotions"))''', r'''        pass'''),
    ("S4 failure-not-noted", SE, r'''        _note_reading(request_id, None)
        return False''', r'''        return False'''),
    ("S5 waits-for-ever", SE, r'''            if time.monotonic() >= deadline:
                return None''', r'''            if False:
                return None'''),
    ("S6 empty-is-a-reading", SE, r'''                return dict(got) or None''', r'''                return dict(got)'''),
    ("S7 not-bounded", SE, r'''        while len(_READINGS) > READINGS_MAX:
            _READINGS.popitem(last=False)''', r'''        while False:
            _READINGS.popitem(last=False)'''),
    # U / D — the gate
    ("U1 displeased-ignored", UC, r'''            or bool(displeased))''', r'''            )'''),
    ("U2 empty-message", UC, r'''    if not (isinstance(current_user_text, str) and current_user_text.strip()):
        return False
    return (has_correction_phrase''', r'''    return (has_correction_phrase'''),
    ("D1 any-emotion", MO, r'''    return any(k in SENTIMENT_COLD and isinstance(v, (int, float)) and not isinstance(v, bool)''', r'''    return any(isinstance(v, (int, float)) and not isinstance(v, bool)'''),
    ("D2 any-score", MO, r'''               and v >= SENTIMENT_EMOTION_MIN for k, v in emotions.items())''', r'''               and v > 0 for k, v in emotions.items())'''),
    ("D3 bool-score", MO, r'''    return any(k in SENTIMENT_COLD and isinstance(v, (int, float)) and not isinstance(v, bool)''', r'''    return any(k in SENTIMENT_COLD and isinstance(v, (int, float))'''),
    # R — the journal
    ("R1 rest-never", ST, r'''            if rp or re_:
                # §4UW''', r'''            if False:
                # §4UW'''),
    ("R2 rest-always", ST, r'''            if rp or re_:
                # §4UW''', r'''            if True:
                # §4UW'''),
    # N — §4UW-b: an undo and the ledger's epochs
    ("N1 undo-voids-the-ledger", ST, r'''                self.pair_epoch, self._epoch_before_set = self._epoch_before_set, None
                self.tone_epoch += 1''', r'''                self._epoch_before_set = None
                self._new_epoch_locked()'''),
    ("N2 undo-keeps-the-tones", ST, r'''                self.pair_epoch, self._epoch_before_set = self._epoch_before_set, None
                self.tone_epoch += 1''', r'''                self.pair_epoch, self._epoch_before_set = self._epoch_before_set, None'''),
    ("N3 epoch-reused", ST, r'''        self._epoch_seq += 1
        self.pair_epoch = self._epoch_seq''', r'''        self.pair_epoch += 1'''),
    ("N4 set-does-not-void", ST, r'''            self._epoch_before_set = self.pair_epoch
            self._new_epoch_locked()''', r'''            self._epoch_before_set = self.pair_epoch'''),
    # N5: EQUIVALENT (a "neutral" clears `prev`, so no undo can read the stale way back) — the line is removed
    ("N6 tones-by-the-pair-epoch", MD, r'''            return int(getattr(self.state, "tone_epoch", 0) or 0)''', r'''            return int(getattr(self.state, "pair_epoch", 0) or 0)'''),
    ("N7 set-keeps-the-tones", ST, r'''        self.pair_epoch = self._epoch_seq
        self.tone_epoch += 1''', r'''        self.pair_epoch = self._epoch_seq'''),
    ("N8 stale-way-back", ST, r'''            if self._epoch_before_set is not None:
                # the pair''', r'''            if True:
                # the pair'''),
    ("N9 neutral-on-nothing-keeps-the-ledger", ST, r'''                self._new_epoch_locked()
                return before, before            # nothing on file to reset''', r'''                return before, before            # nothing on file to reset'''),
    ("N10 neutral-keeps-the-tone-books", ST, r'''            self._new_epoch_locked()             # (nothing is left to undo''', r'''            self._epoch_seq += 1; self.pair_epoch = self._epoch_seq             # (nothing is left to undo'''),
    # G — read once, forget, the days' sums (review r1)
    ("G1 cut-read-again", AG, r'''            return rid in _MOOD_TURN_READ        # (a new''', r'''            return False        # (a new'''),
    ("G2 finished-skipped", AG, r'''        if aborted:
            return rid in _MOOD_TURN_READ''', r'''        if True:
            return rid in _MOOD_TURN_READ'''),
    ("G3 guard-off", AG, r'''                    if not _mood_turn_already_read(str(req_id or ""), bool(aborted)):''', r'''                    if True:'''),
    ("G4 reading-not-forgotten", AG, r'''                _sent.forget_reading(str(request_id_context.get() or ""))''', r'''                pass'''),
    ("G5 forgets-for-queued", AG, r'''            if not queued:
                # §4UW (review r1)''', r'''            if True:
                # §4UW (review r1)'''),
    ("G6 days-sum-the-size", "src/ghost_agent/selfhood/days.py", r'''            mp, me = float(mv[0]), float(mv[1])''', r'''            pass'''),
    ("G7 arrival-text-not-passed", AG, r'''        self._mood_owner_arrived(                   # §4TC: an owner's request ends the quiet
            ((body.get("messages") or [{}])[-1] or {}).get("content")''', r'''        self._mood_owner_arrived(                   # §4TC: an owner's request ends the quiet
            None and ((body.get("messages") or [{}])[-1] or {}).get("content")'''),
    ("G9 read-memo-kept", AG, r'''                _MOOD_TURN_READ.pop(str(request_id_context.get() or ""), None)''', r'''                pass'''),
    ("G10 inf-moved", "src/ghost_agent/selfhood/days.py", r'''and not isinstance(v, bool) and math.isfinite(v) for v in mv)):''', r'''and not isinstance(v, bool) and v == v for v in mv)):'''),
    ("G8 judge-does-not-wait", SE, r'''            await asyncio.sleep(_READING_POLL_S)''', r'''            return None'''),
]
#: module-level code (a constant): coverage maps no test to it
FORCE = {"B1 old-bound": ["test_three_is_for_when_nearly_everything_went_one_way",
                         "test_a_good_afternoon_reads_two_and_the_automatic_mood_still_reaches_three"],
         "B2 old-anchor": ["test_three_is_for_when_nearly_everything_went_one_way"]}
PINS = ["tests/test_4uw_mood_fixes.py", "tests/test_4tc_mood_engine.py", "tests/test_4tc_mood_hooks.py",
        "tests/test_4tc_set_mood.py", "tests/test_4tx_mood_review.py", "tests/test_4ua_character_mood_tone.py",
        "tests/test_4ub_mood_fixes.py", "tests/test_4ud_mood.py", "tests/test_4ug_character.py",
        "tests/test_4uh_personality.py", "tests/test_4up_character_eval.py", "tests/test_4us_mood_visible.py",
        "tests/test_4tw_sentiment.py", "tests/test_user_correction.py", "tests/test_user_correction_rebuttal.py",
        "tests/test_selfhood_state_thread.py", "tests/test_selfhood_model.py"]


def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    cov = BC.build_map(srcs, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        if label in FORCE:
            ids = sorted(set(ids or []) | {f"tests/test_4uw_mood_fixes.py::{t}" for t in FORCE[label]})
        if ids:
            return ids, f"cov {len(ids)} tests"
        return list(PINS), f"fallback {len(PINS)} files"
    print(f"{len(PINS)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
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
