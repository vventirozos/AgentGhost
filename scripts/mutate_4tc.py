#!/usr/bin/env python3
"""§4TC battery — the mood becomes a real one: two numbers that decay (`selfhood/mood.py`: the half-life,
the intensity thresholds and their margin, the eight regions and theirs, the "curious" tag), what moves
them (each event's sign, size and cap; boredom; the owner's anchors, synonyms and manners), the stored
pair and its journal (`selfhood/state.py`: what a read may write, the rows, who a change is booked to, the
bound, old files), `introspect` (`set_mood`'s parsing and refusals, the manner, the `mood` story, the view
lines), the turn loop (who MOVES the mood and who HEARS it, the end-of-turn signals, the settle before the
owner clock is reset, the line's place — and the head staying the same bytes under every mood), the
owner's thumbs (`core/feedback.py`), `/api/health` and the face's hue. §4TB's guards of `set_mood` (who
may call it, fail closed, the probe dry run, the argument before the guards) are pinned in
`scripts/mutate_4tb.py`, whose mutants follow the new lines. After the independent review (the comment
above each group names the item): `set_mood` is the owner's alone; boredom is read-time only — never in the
pair, measured from a persisted owner stamp, journalled once per level per quiet; praise is a short message
of thanks and filler only; the journal survives a bad byte; an anchored mood line that changed says so; what
the mood does not look at; a thumb on a recent turn only; context pressure named as such; the "curious" tag.
The first boredom rule's mutants (B2–B7) and the word-list praise mutants (E17, E18) went with their
subjects. Review r2 (marked "review r2"): the journal's order during a quiet, a declared-ok `set_mood`, praise
is no question, thumbs move pleasantness only and never from the future, the tag belongs to the move that set
it, no causes on a public surface, lines that are no rows. Coverage-selected tests in parallel tree
copies (scripts/battery_cov.py); the real src/ is never mutated. No-op control must SURVIVE, the known-bad
control must DIE.
    python3 scripts/mutate_4tc.py [check | -j6 all | rerun D1 E4 …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = "src/ghost_agent/core/agent.py"
FB = "src/ghost_agent/core/feedback.py"
IT = "src/ghost_agent/tools/introspect.py"
MD = "src/ghost_agent/selfhood/mood.py"
ST = "src/ghost_agent/selfhood/state.py"
SC = "src/ghost_agent/selfhood/schema.py"
NA = "src/ghost_agent/selfhood/narrative.py"
RT = "src/ghost_agent/api/routes.py"
JS = "interface/static/matrix_graph.js"

# anchors used by more than one mutant (each is ONE exact place in its file)
_TAG = ('    if novelty and p > 0 and e > 0 and region(p, e) == _CURIOUS_BASE:\n'
        '        return CURIOUS, n, True\n')
_CLEAN = '    clean = guards == 0 and not pressure and bool(sig.completed) and share <= CLEAN_TURN_MAX_SHARE\n'
_UNITS = '    units = min(float(TROUBLE_CAP), min(failures, TROUBLE_CAP) * weight + guards + pressure)\n'
_ROW = ('            if changed:\n'
        '                self._append_history_locked(new, str(kind or ""), prev=before.describe())\n')
_MOVES = ('        return (_rk() == "owner" and turn_origin(context) == "user"\n'
          '                and not _mood_run_is_contained(context))\n')
_HEARD = '        if kind in ("owner", "member", "probe", "job"):\n'
_SCHED = '        return kind == "background" and str(request_id_context.get() or "").startswith("sched-")\n'
_THUMB = ("                    if (not unchanged and is_owner_label_source(source)\n"
          "                            and turn_is_recent_enough(traj)):\n")
_BORED_GATE = "        if n == 0 and not for_reply:\n            # boredom: only over a NEUTRAL mood"
_QUIET_END = "            if was > 0:\n"
_ARRIVED = ("            if not mood_moves_on_this_request(self.context):\n"
            "                return\n"
            "            sm = _mood_model(self.context)\n"
            "            if sm is not None:\n"
            "                sm.owner_arrived(request_id=str(request_id_context.get() or \"\"),\n"
            "                                 conversation=_mood_conversation())")
_OWNERS = "            owners = mood_moves_on_this_request(context) is True\n"
_VIEW = '        out.append(f"{indent}{head}: {label}, {n} of 3" + (f" — {cause}" if cause else ""))\n'
_HEALTH = ('    mood = {"label": "neutral", "intensity": 0, "manner": "", "set_at": "", "source": "auto", "cause": "",\n'
           '            "held_until": ""}\n    try:\n')
_SYS = '                stable_system_prompt = (base_prompt + stable_skill_instruction).replace("\\r", "")\n'

MUT = [
    ("CONTROL no-op", MD, "HALF_LIFE_S = 2 * 3600.0", "HALF_LIFE_S = 2 * 3600.0  # no-op"),
    ("CONTROL known-bad", AG, _SYS,
     '                stable_system_prompt = (base_prompt + stable_skill_instruction\n'
     '                                        + _mood_tone_block(self.context)).replace("\\r", "")\n'),
    # ── decay ──
    ("D1 the-half-life-is-four-hours", MD, "HALF_LIFE_S = 2 * 3600.0", "HALF_LIFE_S = 4 * 3600.0"),
    ("D2 a-value-grows-with-time", MD, "    return float(value) * (0.5 ** (t / HALF_LIFE_S))", "    return float(value) * (2.0 ** (t / HALF_LIFE_S))"),
    ("D3 the-stored-pair-is-not-decayed-when-read", ST, "            p, e = M.decay(stored.p, age), M.decay(stored.e, age)", "            p, e = stored.p, stored.e"),
    # ── intensity: thresholds and their margin ──
    ("T1 the-third-threshold-is-lower", MD, "INTENSITY_THRESHOLDS = (0.20, 0.45, 0.75)", "INTENSITY_THRESHOLDS = (0.20, 0.45, 0.65)"),
    ("T2 no-margin-on-the-way-down", MD, "INTENSITY_MARGIN = 0.04", "INTENSITY_MARGIN = 0.0"),
    ("T3 the-margin-works-against-a-held-level", MD, "        if m >= INTENSITY_THRESHOLDS[k - 1] - INTENSITY_MARGIN:", "        if m >= INTENSITY_THRESHOLDS[k - 1] + INTENSITY_MARGIN:"),
    ("T4 only-the-held-level-keeps-its-margin", MD, "    for k in range(h, raw, -1):", "    for k in range(h, max(raw, h - 1), -1):"),
    ("T5 a-worn-off-mood-is-dated-when-it-was-noticed", ST, "        cur.set_at = self._wore_off_at(stored, n, now)", "        cur.set_at = cur.updated_at"),
    # ── the regions ──
    ("G1 near-zero-is-wider", MD, "AXIS_RATIO = 0.40", "AXIS_RATIO = 0.60"),
    ("G2 a-held-label-has-no-margin", MD, "AXIS_MARGIN = 0.10", "AXIS_MARGIN = 0.0"),
    ("G3 content-and-down-swapped", MD, '            return "content" if p > 0 else "down"', '            return "down" if p > 0 else "content"'),
    ("G5 a-held-label-survives-a-sign-flip", MD, '        return ap >= ae and (p > 0) == (held == "content") and ratio < AXIS_RATIO + AXIS_MARGIN',
     '        return ap >= ae and ratio < AXIS_RATIO + AXIS_MARGIN'),
    # (review 12) the tag: pleasant AND energetic — trouble wins the sign of P
    ("G6 curious-outside-its-region", MD, _TAG, '    if novelty:\n        return CURIOUS, n, True\n'),
    ("G7 novelty-never-tags", MD, _TAG, '    if False:\n        return CURIOUS, n, True\n'),
    # G8: region(p, e) == cheerful already implies e > 0, so the energy test AND the region test go together
    ("G8 the-tag-survives-a-lost-energy", MD, _TAG, '    if novelty and p > 0:\n        return CURIOUS, n, True\n'),
    # G9 was "the-tag-needs-the-diagonal-again": §4TX made that the rule (only where the plain region is
    # cheerful), so the mutant now drops it again — research at high energy with trouble (restless) reads curious
    ("G9 the-tag-ignores-the-plain-region", MD, _TAG, '    if novelty and p > 0 and e > 0:\n        return CURIOUS, n, True\n'),
    # ── quiet → boredom ──
    # ── (review, MAJOR 2) quiet → boredom: read-time only ──
    ("B8 boredom-after-three-hours", MD, "BOREDOM_AFTER_S = 6 * 3600.0", "BOREDOM_AFTER_S = 3 * 3600.0"),
    ("B9 the-second-level-after-a-day", MD, "BOREDOM_DEEP_AFTER_S = 12 * 3600.0", "BOREDOM_DEEP_AFTER_S = 24 * 3600.0"),
    ("B10 boredom-never-reaches-two", MD, "    return 2 if q >= BOREDOM_DEEP_AFTER_S else 1", "    return 1"),
    ("B11 boredom-replaces-a-live-mood", ST, _BORED_GATE, "        if not for_reply:\n            # boredom: only over a NEUTRAL mood"),
    ("B12 boredom-is-never-read", ST, _BORED_GATE, "        if False:\n            # boredom: only over a NEUTRAL mood"),
    ("B13 the-owner-is-answered-bored", ST, "        if _owner_is_asking():\n            return None", "        if False:\n            return None"),
    ("B14 the-process-clock-beats-the-stored-stamp", ST, "        if age is not None:\n            return age\n        fn = self._quiet_fn", "        if False:\n            return age\n        fn = self._quiet_fn"),
    ("B15 the-owner-stamp-is-not-loaded", SC,
     '            last_owner_turn_at=(d.get("last_owner_turn_at")\n'
     '                                if isinstance(d.get("last_owner_turn_at"), str) else ""),',
     '            last_owner_turn_at="",'),
    ("B16 a-bored-row-per-read", ST, "                for k in range(logged + 1, level + 1):", "                for k in range(1, level + 1):"),
    ("B17 the-journal-is-not-asked-what-the-quiet-has", ST, "                    logged = self._open_boredom(start.timestamp())", "                    logged = 0"),
    ("B18 a-boredom-is-dated-now", ST, "        at = start + datetime.timedelta(seconds=M.BOREDOM_THRESHOLDS[level])", "        at = _aware(now)"),
    ("B19 a-bored-reading-is-since-now", ST, "                                set_at=_iso(self._bored_since(level, start, now)),", "                                set_at=_iso(now),"),
    # (review r2, 1) the journal's order during a quiet
    ("B29 a-bored-row-may-precede-the-wearing-off", ST, "            if neutral_at is not None and neutral_at > at:\n                at = neutral_at", "            if False:\n                at = neutral_at"),
    ("B30 the-memo-skips-the-settle", ST,
     "                if stored is not None and stored.intensity > 0:\n"
     "                    self._settle_locked(now)\n"
     "                seen = self._bored_seen\n",
     "                seen = self._bored_seen\n"
     "                if ((seen is None or abs(seen[0] - start.timestamp()) >= _SAME_QUIET_S)\n"
     "                        and stored is not None and stored.intensity > 0):\n"
     "                    self._settle_locked(now)\n"),
    ("B20 the-quiet-ended-at-every-arrival", ST, _QUIET_END, "            if True:\n"),
    ("B21 the-quiet-never-ends-in-the-journal", ST, _QUIET_END, "            if False:\n"),
    ("B22 an-arrival-does-not-stamp-the-owner", ST,
     "                self._append_history_locked(row, M.KIND_QUIET_END, prev=f\"{M.BORED}, {was} of 3\")\n            self.mark_owner_turn(now=now)",
     "                self._append_history_locked(row, M.KIND_QUIET_END, prev=f\"{M.BORED}, {was} of 3\")"),
    ("B23 the-end-of-an-owners-turn-is-not-stamped", AG,
     "                    _sm_mood.owner_turn_ended(request_id=str(req_id or \"\"),\n"
     "                                              conversation=_mood_conversation())\n", ""),
    ("B24 anyones-arrival-ends-the-quiet", AG, _ARRIVED, _ARRIVED.split("                return\n", 1)[1]),
    ("B25 the-arrival-is-never-noted", AG, "        self._mood_owner_arrived()                  # §4TC: an owner's request ends the quiet\n", ""),
    ("B26 a-thumb-row-hides-the-open-quiet", ST, "            if kind != M.KIND_QUIET:\n                continue", "            if kind != M.KIND_QUIET:\n                break"),
    # §4TX: the diary also leaves out the manner rows — the mutant lets only the boredom rows back in
    ("B27 the-diary-reads-the-boredom-rows", NA,
     "                    if row.get(\"kind\") not in (KIND_QUIET, KIND_QUIET_END, KIND_MANNER)]",
     "                    if row.get(\"kind\") not in (KIND_MANNER,)]"),
    ("B28 the-diary-reads-the-mood-of-the-instant", NA, "        block = state.format_as_prefix(include_mood=False)", "        block = state.format_as_prefix(include_mood=True)"),
    # ── events: one owner turn ──
    ("E1 a-failure-lifts-the-pleasantness", MD, "EV_TROUBLE = (-0.12, +0.08)", "EV_TROUBLE = (+0.12, +0.08)"),
    ("E2 trouble-has-no-cap", MD, "TROUBLE_CAP = 4", "TROUBLE_CAP = 40"),
    ("E3 a-guard-event-is-not-trouble", MD, _UNITS, _UNITS.replace(" + guards + pressure)", " + pressure)")),
    # §4TX: a clean turn may have up to CLEAN_TURN_MAX_SHARE of its calls failed (both `clean` and
    # `_clean_reward` read it); the mutant lets any share of failures through
    ("E4 a-turn-with-a-failure-is-a-clean-turn", MD, "CLEAN_TURN_MAX_SHARE = 0.10", "CLEAN_TURN_MAX_SHARE = 1.0"),
    ("E5 a-turn-that-was-cut-is-a-clean-turn", MD, _CLEAN, _CLEAN.replace(" and bool(sig.completed)", "")),
    # §4TX: a long CLEAN turn now keeps its reward on purpose; the mutant makes EVERY long turn (failures, a
    # cut, nothing but excused calls) a clean long one — full reward, energy-only cost
    ("E7 a-long-turn-is-also-a-clean-turn", MD,
     '    long_clean = clean and calls > 0 and calls >= max(0, int(getattr(sig, "excused", 0) or 0))\n',
     "    long_clean = True\n"),
    ("E8 one-page-is-a-research-turn", MD, "RESEARCH_MIN_SOURCES = 3", "RESEARCH_MIN_SOURCES = 1"),
    ("E9 research-carries-no-novelty", MD, "KIND_TURN, novelty=True))", "KIND_TURN, novelty=False))"),
    ("E10 a-long-turn-raises-the-energy", MD, "EV_LONG_TURN = (-0.15, -0.30)", "EV_LONG_TURN = (-0.15, +0.30)"),
    ("E12 the-call-count-does-not-make-a-turn-long", MD, "    long_turn = dur >= LONG_TURN_S or all_calls >= LONG_TURN_CALLS", "    long_turn = dur >= LONG_TURN_S"),
    # ── events: the owner's reactions ──
    ("E13 a-thumbs-up-lowers-it", MD, "EV_THUMB_UP = (+0.25, 0.0)", "EV_THUMB_UP = (-0.25, 0.0)"),
    # (review r2, 4) thumbs move the pleasantness only
    ("E32 a-thumbs-up-pumps-the-energy", MD, "EV_THUMB_UP = (+0.25, 0.0)", "EV_THUMB_UP = (+0.25, +0.05)"),
    ("E14 a-thumbs-down-lifts-it", MD, "EV_THUMB_DOWN = (-0.30, 0.0)", "EV_THUMB_DOWN = (+0.30, 0.0)"),
    ("E15 a-correction-lifts-it", MD, "EV_CORRECTION = (-0.20, 0.0)", "EV_CORRECTION = (+0.20, 0.0)"),
    ("E16 thanks-is-as-strong-as-a-thumb", MD, "EV_PRAISE = (+0.10, 0.0)", "EV_PRAISE = (+0.30, 0.0)"),
    # (review 4) praise: a short message of thanks and filler only
    ("E19 praise-has-no-length-limit", MD, "PRAISE_MAX_WORDS = 6", "PRAISE_MAX_WORDS = 60"),
    ("E20 a-word-that-is-neither-is-skipped", MD, "        else:\n            return False\n    return found > 0", "        else:\n            i += 1\n    return found > 0"),
    ("E21 filler-alone-is-praise", MD, "    return found > 0", "    return True"),
    ("E22 a-curly-apostrophe-breaks-a-filler-word", MD, """.replace("\\u2019", "'").replace("\\u2018", "'").lower()""", ".lower()"),
    ("E23 not-is-filler", MD, 'PRAISE_FILLER = frozenset({"ok", "okay",', 'PRAISE_FILLER = frozenset({"not", "no", "ok", "okay",'),
    # (review r2, 3) a question is not praise; "is" / "that" are not filler; the missing forms
    ("E27 a-question-is-praise", MD, "    if _is_question(t):\n        return False", "    if False:\n        return False"),
    ("E28 is-and-that-are-filler-again", MD, 'PRAISE_FILLER = frozenset({"ok", "okay", "ghost", "mate", "that\'s",', 'PRAISE_FILLER = frozenset({"is", "that", "ok", "okay", "ghost", "mate", "that\'s",'),
    ("E29 a-closing-quote-hides-the-question-mark", MD, "    for ch in reversed(text):", "    for ch in reversed(text[-1:]):"),
    ("E30 cheers-is-no-thanks", MD, '{"thanks", "thx", "ty", "cheers", "great",', '{"thanks", "thx", "ty", "great",'),
    ("E31 many-thanks-is-no-thanks", MD, '("many", "thanks"), ', ""),
    # (review 10) context pressure is named as that
    ("E24 context-pressure-is-called-a-loop-guard", MD, '            parts.append("the context ran out of room (context pressure)")', '            parts.append("a loop guard stepped in once")'),
    ("E25 context-pressure-is-no-trouble", MD, _UNITS, _UNITS.replace(" + guards + pressure)", " + guards + 0 * pressure)")),
    # (review 8) what the mood does not look at; the owner's word holds for the turn
    ("E26 the-owners-word-does-not-hold-for-the-turn", MD, "    if bool(sig.owner_set):\n        return out", "    if False:\n        return out"),
    # ── who may move it ──
    ("W1 a-probe-moves-the-mood", AG, _MOVES,
     '        return (_rk() in ("owner", "probe") and turn_origin(context) in ("user", ORIGIN_PROBE)\n'
     '                and not _mood_run_is_contained(context))\n'),
    ("W2 a-member-moves-the-mood", AG, _MOVES,
     '        return (_rk() in ("owner", "member") and turn_origin(context) == "user"\n'
     '                and not _mood_run_is_contained(context))\n'),
    ("W3 a-scheduled-task-or-a-job-moves-the-mood", AG, _MOVES,
     '        return (_rk() in ("owner", "background", "job") and turn_origin(context) in ("user", "internal")\n'
     '                and not _mood_run_is_contained(context))\n'),
    ("W4 a-contained-run-moves-the-mood", AG, _MOVES, '        return (_rk() == "owner" and turn_origin(context) == "user")\n'),
    ("W5 a-turn-cut-from-outside-moves-the-mood", AG, "                    if not aborted:\n                        # See the docstring: None = in-semaphore caller", "                    if True:\n                        # See the docstring: None = in-semaphore caller"),
    ("W6 the-live-lockdown-flag-beats-the-snapshot", AG,
     "                        _plock = (pressure_lockdown\n"
     "                                  if pressure_lockdown is not None\n"
     "                                  else bool(getattr(\n"
     "                                      self.context, \"_ctx_pressure_lockdown\",\n"
     "                                      False)))\n"
     "                        _sm_mood.note_owner_turn(",
     "                        _plock = bool(getattr(self.context, \"_ctx_pressure_lockdown\", False))\n"
     "                        _sm_mood.note_owner_turn("),
    ("W7 earlier-turns-of-the-history-are-counted", AG, "        turn = list(msgs or [])[self._this_turn_boundary(msgs, real_request) + 1:]", "        turn = list(msgs or [])"),
    ("W8 every-page-load-is-a-source", AG, "            sources = len(fetched_urls(rows))", "            sources = len(rows)"),
    ("W9 a-breaker-close-is-no-guard-event", AG, "    note_guard_event(context, req_id, kind)        # §4TC: a breaker that closes a request is a guard event\n", ""),
    ("W10 the-reaction-hook-has-no-gate", AG,
     "            if not mood_moves_on_this_request(self.context):\n"
     "                return\n"
     "            sm = _mood_model(self.context)\n"
     "            if sm is not None:\n"
     "                sm.note_owner_reaction(kind, request_id=str(request_id_context.get() or \"\"))",
     "            sm = _mood_model(self.context)\n"
     "            if sm is not None:\n"
     "                sm.note_owner_reaction(kind, request_id=str(request_id_context.get() or \"\"))"),
    ("W11 any-verdict-of-the-judge-is-a-correction", AG, "            if _contradicts is True:\n                from ..selfhood.mood import KIND_CORRECTION", "            if _contradicts is not None:\n                from ..selfhood.mood import KIND_CORRECTION"),
    ("W12 a-label-from-any-source-moves-the-mood", FB, _THUMB, "                    if (not unchanged\n                            and turn_is_recent_enough(traj)):\n"),
    ("W13 the-same-thumb-again-counts-again", FB, _THUMB, "                    if (is_owner_label_source(source)\n                            and turn_is_recent_enough(traj)):\n"),
    # (review 9) a thumb on a recent turn only
    ("W15 a-thumb-on-an-old-turn-moves-the-mood", FB, _THUMB, "                    if (not unchanged and is_owner_label_source(source)\n                            and True):\n"),
    ("W16 a-day-is-a-month", FB, "THUMB_MAX_TURN_AGE_S = 24 * 3600.0", "THUMB_MAX_TURN_AGE_S = 30 * 24 * 3600.0"),
    ("W17 a-turn-of-unreadable-age-is-recent", FB, "        if at is None:\n            return False", "        if at is None:\n            return True"),
    # (review r2, 4) a turn dated in the future is not recent
    ("W23 a-future-dated-turn-is-recent", FB, "    return 0.0 <= age <= THUMB_MAX_TURN_AGE_S", "    return age <= THUMB_MAX_TURN_AGE_S"),
    # (review 8) what the mood does not look at
    ("W18 an-introspect-call-is-trouble", AG, '    if str(m.get("name") or "").strip() == "introspect":\n        return True', '    if str(m.get("name") or "").strip() == "introspect":\n        return False'),
    ("W19 a-designed-refusal-is-trouble", AG, 'MOOD_BLIND_REASONS = frozenset({"untrusted_content", "owner_data_blocked", "owner_only"})', "MOOD_BLIND_REASONS = frozenset()"),
    ("W20 the-turn-never-knows-the-owner-set-the-mood", AG, "            owner_set=bool(owner_set),", "            owner_set=False,"),
    ("W21 a-refused-set-mood-holds-the-turn", AG, "                        and c.result and not c.error", "                        and (c.result or c.error)"),
    ("W22 the-lockdown-is-not-passed-on", AG, "            pressure=bool(lockdown),", "            pressure=False,"),
    ("W14 another-persons-reaction-is-the-owners", FB, 'OWNER_LABEL_SOURCES = frozenset({"web", "clockwork", "cli", "slack:owner"})', 'OWNER_LABEL_SOURCES = frozenset({"web", "clockwork", "cli", "slack:owner", "slack:requester"})'),
    # ── the stored pair and its journal ──
    ("J1 a-row-for-every-nudge", ST, _ROW, _ROW.replace("if changed:", "if True:")),
    ("J2 no-row-for-a-change", ST, _ROW, _ROW.replace("if changed:", "if False:")),
    ("J3 the-owners-change-is-booked-as-automatic", ST, "                novelty=(label == M.CURIOUS), source=M.SOURCE_OWNER,", "                novelty=(label == M.CURIOUS), source=M.SOURCE_AUTO,"),
    ("J4 an-event-keeps-the-owners-name-on-a-new-mood", ST, "                source=(M.SOURCE_AUTO if changed else base.source),", "                source=base.source,"),
    ("J5 neutral-keeps-the-pair", ST, "                p=0.0, e=0.0, updated_at=at, label=M.NEUTRAL, intensity=0, novelty=False,", "                p=base.p, e=base.e, updated_at=at, label=M.NEUTRAL, intensity=0, novelty=False,"),
    # J6 (repaired): "neutral" still clears the manner; only the §4TX step-down (keep_manner=True) keeps it
    ("J6 neutral-keeps-the-manner", ST,
     "                **(self._kept_manner(base, now) if keep_manner else {}),\n",
     "                **self._kept_manner(base, now),\n"),
    ("J7 setting-a-manner-resets-the-mood", ST, "                p=base.p, e=base.e, updated_at=base.updated_at or _iso(now), label=base.label,", "                p=0.0, e=0.0, updated_at=base.updated_at or _iso(now), label=base.label,"),
    ("J8 the-journal-has-no-cap", ST, "MOOD_HISTORY_MAX_ROWS = 1200", "MOOD_HISTORY_MAX_ROWS = 120000"),
    ("J10 a-row-from-before-the-pair-is-read", ST, '            if not isinstance(d, dict) or "at" not in d or "p" not in d:', "            if not isinstance(d, dict):"),
    ("J11 an-old-stored-mood-is-loaded", SC, '        if not isinstance(d, dict) or "p" not in d or "e" not in d:', "        if not isinstance(d, dict):"),
    ("J13 the-diary-gets-no-timeline", NA, "            if steps:\n                block = (block", "            if False:\n                block = (block"),
    # (review 6) a byte that is not UTF-8
    ("J14 a-bad-byte-costs-the-whole-journal", ST, 'self.mood_history_path.read_text(encoding="utf-8", errors="replace").splitlines()', 'self.mood_history_path.read_text(encoding="utf-8").splitlines()'),
    # §4TX: the compaction is shared by the journal and the events log (`_compact_locked(path, …)`)
    ("J15 a-bad-byte-switches-the-cap-off", ST, '            with path.open("rb") as f:\n                lines = f.readlines()', '            with path.open("r", encoding="utf-8") as f:\n                lines = [ln.encode("utf-8") for ln in f.readlines()]'),
    # (review r2, 5) the tag belongs to the move that set it; (7) lines that are no rows
    ("J18 the-tag-survives-a-later-event", ST, "                                       novelty=bool(novelty))\n            changed = (label, n) != (base.label, base.intensity)\n            cause = _cap_text", "                                       novelty=bool(base.novelty or novelty))\n            changed = (label, n) != (base.label, base.intensity)\n            cause = _cap_text"),
    ("J19 the-owners-word-keeps-an-old-tag", ST, "                novelty=(label == M.CURIOUS), source=M.SOURCE_OWNER,", "                novelty=(base.novelty or label == M.CURIOUS), source=M.SOURCE_OWNER,"),
    ("J20 a-json-line-that-is-no-object-is-read", ST, '            if not isinstance(d, dict) or "at" not in d or "p" not in d:', '            if "at" not in d or "p" not in d:'),
    ("J21 a-line-nested-too-deep-breaks-the-reader", ST,
     "            except (ValueError, RecursionError):\n                continue                       # not JSON",
     "            except ValueError:\n                continue                       # not JSON"),
    # (review 11) mutants the review found alive
    ("J16 today-is-the-utc-day", ST, "        today = now.astimezone(tz).date()", "        today = now.date()"),
    ("J17 a-settled-step-down-loses-the-owner", ST, "        if not (base.source == M.SOURCE_OWNER and n and label == base.label):\n            cur.source = M.SOURCE_AUTO", "        if True:\n            cur.source = M.SOURCE_AUTO"),
    # ── the owner's word: set_mood ──
    ("S1 a-word-that-is-no-mood-is-stored", IT, "        if target is None:\n            return _unknown_mood(word)", '        if target is None:\n            target = ("content", _M.ANCHOR_DIRECTION["content"])'),
    ("S2 a-word-that-is-no-manner-is-let-through", IT, "        if not man:\n            return _unknown_manner(man_word)", "        if not man:\n            pass"),
    ("S3 the-intensity-asked-for-is-ignored", IT,
     "            _before, after = state.set_owner_mood(label, direction, level, note=why, hold=",
     "            _before, after = state.set_owner_mood(label, direction, 2, note=why, hold="),
    # §4TX: the manner has its own level now (`man_level`)
    ("S4 the-manner-ignores-the-intensity", IT, "            _before, after = state.set_manner(man, man_level)", "            _before, after = state.set_manner(man, 2)"),
    ("S5 manner-none-takes-the-mood-off-too", IT, '            before, _after = state.set_manner("", 0)', "            before, _after = state.reset_mood()"),
    ("S6 an-intensity-three-anchor-reads-back-as-two", MD, "ANCHOR_MAGNITUDE = {1: 0.32, 2: 0.60, 3: 0.90}", "ANCHOR_MAGNITUDE = {1: 0.32, 2: 0.60, 3: 0.70}"),
    ("S7 grumpy-means-down", MD, '"grumpy": ("irritable", None)', '"grumpy": ("down", None)'),
    ("S8 neutral-is-no-clear-word", MD, 'CLEAR_WORDS = frozenset({"clear", "none", "neutral"})', 'CLEAR_WORDS = frozenset({"clear", "none"})'),
    # ── (review, MAJOR 1) set_mood is the owner's alone ──
    ("S12 any-request-may-set-the-mood", IT, "        if not owners:\n", "        if False:\n"),
    ("S13 a-truthy-answer-is-a-yes", IT, _OWNERS, "            owners = bool(mood_moves_on_this_request(context))\n"),
    ("S14 the-owner-check-fails-open", IT,
     '            return _mood_refusal("SYSTEM BLOCK: set_mood could not tell whether this is the owner\'s "\n'
     '                                 "request, so nothing was set.", "guard_unavailable"), False',
     "            owners = True"),
    ("S15 the-refusal-is-booked-as-outside-content", IT, '            return _mood_refusal(MOOD_OWNER_ONLY, "owner_only"), False', '            return _mood_refusal(MOOD_OWNER_ONLY, "untrusted_content"), False'),
    ("S16 a-probe-is-refused-not-dry-run", IT, "    if not probe:\n        try:\n            from ..core.agent import mood_moves_on_this_request", "    if True:\n        try:\n            from ..core.agent import mood_moves_on_this_request"),
    # (review r2, 2) a successful set_mood is a declared ok; (7) a probe that read outside content
    ("S17 a-successful-set-mood-is-a-plain-string", IT, '        return ToolOutcome.ok(". ".join(said) + ".")', '        return ". ".join(said) + "."'),
    ("S18 a-probe-that-read-a-page-gets-the-dry-run", IT, "    if ref is not None:\n        try:\n            pretty_log(\"Introspect\", \"set_mood refused — this request read outside content\"", "    if ref is not None and not probe:\n        try:\n            pretty_log(\"Introspect\", \"set_mood refused — this request read outside content\""),
    # ── the views ──
    ("V1 the-mood-story-is-a-write", IT, 'WRITE_ACTIONS = frozenset({"set_mood"})', 'WRITE_ACTIONS = frozenset({"set_mood", "mood"})'),
    ("V2 the-story-names-no-cause", IT, '        lines.append("Why: " + ("; ".join(why[-_MOOD_WHY_MAX:]) if why else "nothing on record") + ".")', '        lines.append("Why: nothing on record.")'),
    # (review 11) mutants the review found alive
    ("V11 why-shows-five-causes", IT, "_MOOD_WHY_MAX = 3", "_MOOD_WHY_MAX = 5"),
    ("V12 a-stamp-from-another-day-shows-no-date", IT, '    return local.strftime("%H:%M") if local.date() == today else local.strftime("%b %d %H:%M")', '    return local.strftime("%H:%M")'),
    ("V13 the-story-of-a-boredom-lists-old-events", IT, "    elif view.label == _M.BORED and view.cause == _M.quiet_cause(view.intensity):", "    elif False:"),
    # (review r2, 6) on a public surface: no causes
    ("V14 a-view-shows-the-cause-in-a-channel", IT, "        if _reply_is_public():\n            cause = MOOD_REASONS_PRIVATE         # the public view", "        if False:\n            cause = MOOD_REASONS_PRIVATE         # the public view"),
    ("V15 the-story-says-why-in-a-channel", IT, "    if public:\n        if not view.neutral:\n            lines.append(f\"Why: {MOOD_REASONS_PRIVATE}.\")", "    if False:\n        if not view.neutral:\n            lines.append(f\"Why: {MOOD_REASONS_PRIVATE}.\")"),
    ("V16 the-timeline-shows-its-causes-in-a-channel", IT, '    cause = str(row.get("cause") or "").strip() if notes else ""', '    cause = str(row.get("cause") or "").strip()'),
    ("V17 a-surface-that-cannot-be-read-is-private", IT, "    except Exception:  # noqa: BLE001\n        return True\n\n\ndef _mood_lines", "    except Exception:  # noqa: BLE001\n        return False\n\n\ndef _mood_lines"),
    ("V18 the-step-not-yet-written-shows-its-cause-in-a-channel", IT, '(f" — {view.cause}" if view.cause and not public else "")', '(f" — {view.cause}" if view.cause else "")'),
    ("V3 the-owners-note-is-shown-in-a-channel", IT, "        steps = [_mood_step(r, notes=not public) for r in shown]", "        steps = [_mood_step(r, notes=True) for r in shown]"),
    ("V5 the-view-line-drops-the-cause", IT, _VIEW, '        out.append(f"{indent}{head}: {label}, {n} of 3")\n'),
    ("V7 health-names-any-source", RT, '"source": "owner" if (_n and getattr(_m, "source", "") == "owner") else "auto",', '"source": str(getattr(_m, "source", "") or "auto"),'),
    ("V8 health-has-no-mood-object-without-a-reading", RT, _HEALTH, "    mood = None\n    try:\n"),
    ("V10 the-face-cools-for-an-irritable-mood", JS, "        case 'irritable': return 0.045;", "        case 'irritable': return -0.045;"),
    # ── the tone line: its text ──
    ("L1 level-three-may-not-name-the-mood", MD, '    3: "At 3 it is strong, and you may say outright that you are in this mood.",', '    3: "At 3 it is strong throughout.",'),
    # §4TX: the level sentence is picked into `level` (the remarked form after the first reply of a mood)
    ("L2 the-level-sentence-is-the-manners", MD, '                + level + " " + why + " " + _LIMITS)', '                + MANNER_LEVEL_SENTENCE[mn] + " " + why + " " + _LIMITS)'),
    ("L3 the-limits-leave-out-completeness-and-refusing", MD,
     'the same facts, numbers, tool use and completeness, "\n'
     '           "no new topics or asides; the mood changes no answer and no refusal. Code,',
     'the same facts, numbers and tool use, "\n'
     '           "no new topics or asides. Code,'),
    # L4/L5 (repaired): the manner line gained its gloss (`{gloss}`) before §4TX; the behaviour is unchanged
    ("L4 a-manner-alone-gives-no-line", MD, '    if man:\n        return (f"MANNER (set by the owner): {man}, {mn} of 3.{gloss}"', '    if False:\n        return (f"MANNER (set by the owner): {man}, {mn} of 3.{gloss}"'),
    ("L5 the-manner-is-left-out-of-a-mood-line", MD, '                + (f" MANNER (set by the owner): {man}, {mn} of 3.{gloss}" if man else "")', '                + ""'),
    # ── the tone line: who hears it, and where it sits ──
    ("H1 no-reply-carries-the-line", AG, '                    _ds_parts.append(("mood_tone", _mood_tone_block(self.context)))        # §4TC\n', ""),
    ("H2 a-member-does-not-hear-it", AG, _HEARD, '        if kind in ("owner", "probe", "job"):\n'),
    ("H3 a-probe-does-not-hear-it", AG, _HEARD, '        if kind in ("owner", "member", "job"):\n'),
    ("H5 a-scheduled-turn-does-not-hear-it", AG, _SCHED, "        return False\n"),
    ("H6 a-sub-agent-hears-it", AG, _SCHED, '        return kind == "background"\n'),
    ("H7 a-contained-copy-hears-it", AG, "    if is_contained_context(context):\n        return True\n    if getattr(context, \"owner_memory_isolated\"", "    if False:\n        return True\n    if getattr(context, \"owner_memory_isolated\""),
    ("H8 an-isolated-delegate-hears-it", AG, "    if getattr(context, \"owner_memory_isolated\", False) is True:\n        return True\n    return turn_origin(context) not in", "    if False:\n        return True\n    return turn_origin(context) not in"),
    ("H9 self-play-and-bench-runs-hear-it", AG, '    return turn_origin(context) not in ("user", ORIGIN_PROBE, "internal")', "    return False"),
    # (review 7) an anchored mood line that changed says so
    ("H11 a-changed-mood-line-does-not-say-it-replaces-the-old-one", AG, '        if k == "mood_tone" and base.get(k):', "        if False:"),
    ("H12 every-changed-section-says-updated", AG, '        if k == "mood_tone" and base.get(k):', "        if base.get(k):"),
    ("H13 a-changed-manner-is-not-marked", AG, '        if k == "mood_tone" and base.get(k):', '        if k == "mood_tone" and base.get(k) and "MANNER" not in t:'),
    ("H10 the-saved-head-carries-the-mood", AG, "        warmed_sys = (base_prompt + _STABLE_SKILL_INSTRUCTION).replace(", "        warmed_sys = (base_prompt + _STABLE_SKILL_INSTRUCTION + _mood_tone_block(self.context)).replace("),
]
PINS = ["tests/test_4tc_mood_engine.py", "tests/test_4tc_mood_hooks.py", "tests/test_4tc_set_mood.py",
        "tests/test_4tc_tone_line.py", "tests/test_4tb_set_mood.py",
        "tests/test_interface_face_signals_2026_09_11.py"]


def _tests():
    pat = (r"selfhood\|tool_introspect\|set_mood\|SelfModel\|SelfStateThread\|mood\|apply_human_label\|"
           r"_record_turn_trajectory\|_record_aborted_turn\|_heartbeat\|_promote_prior_turn_after_judge\|"
           r"_compose_injection\|_head_payload\|head_shape\|stamp_loop_breaker\|api_health\|"
           r"get_active_tool_definitions\|NarrativeSummariser")
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
